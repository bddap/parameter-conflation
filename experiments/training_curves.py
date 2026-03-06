"""
Unified experiment runner with per-epoch logging and incremental saves.

Supports multiple experiment types:
  1. combine_fn_sweep: vary f at fixed compression on one dataset
  2. compression_scaling: vary ratio at fixed f on one dataset
  3. scale_up: vary architecture width at fixed actual param budget

All results include per-epoch training curves.
Saves incrementally after each (config, seed) completes.
Resumes from partial results.

Usage:
    python experiments/training_curves.py exp1    # combine fn sweep, MNIST
    python experiments/training_curves.py exp2    # combine fn sweep, CIFAR
    python experiments/training_curves.py exp3    # compression scaling, MNIST
    python experiments/training_curves.py exp4    # scale up, MNIST
    python experiments/training_curves.py exp5    # compression scaling, CIFAR
    python experiments/training_curves.py all     # run all in priority order
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import json
import math

from virtual_params import VirtualLinear, VirtualConv2d, ArithMap, SinusoidalMap

SEEDS = [42, 137, 256]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp", "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp")


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Combine function registry
# ============================================================================

def get_combine_fns():
    """Return dict of name -> (num_terms, combine_fn)."""
    return {
        "identity (x=1)": (1,
            lambda a: a[0]),
        "add (x=2)": (2,
            lambda a: (a[0] + a[1]) / math.sqrt(2)),
        "multiply (x=2)": (2,
            lambda a: a[0] * a[1]),
        "hash_arith (x=3)": (3,
            lambda a: (a[0] * a[1] + a[2]) / math.sqrt(2)),
        "xor_inspired (x=3)": (3,
            lambda a: (a[0] * a[1] ** 2 + a[2]) / 2),
        "rotation (x=3)": (3,
            lambda a: a[0] * torch.cos(a[2]) - a[1] * torch.sin(a[2])),
        "wide_hash (x=5)": (5,
            lambda a: (a[0] * a[1] + a[2]) * (a[3] + a[4]) / 2),
        "deep_hash (x=7)": (7,
            lambda a: ((a[0]*a[1]+a[2]) * (a[3]*a[4]+a[5]) + a[6]) / math.sqrt(5)),
        "deeper_hash (x=11)": (11,
            lambda a: (((a[0]*a[1]+a[2]) * (a[3]*a[4]+a[5]) + a[6])
                       * (a[7]*a[8]+a[9]) + a[10]) / math.sqrt(11)),
    }


# ============================================================================
# Data loading
# ============================================================================

def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    return (DataLoader(train, batch_size=256, shuffle=True, num_workers=0),
            DataLoader(test, batch_size=512, shuffle=False, num_workers=0))


def load_cifar10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform_train)
    test = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform_test)
    return (DataLoader(train, batch_size=256, shuffle=True, num_workers=0),
            DataLoader(test, batch_size=512, shuffle=False, num_workers=0))


# ============================================================================
# Model definitions
# ============================================================================

class BaselineMLP(nn.Module):
    """Standard MLP with configurable hidden sizes."""
    def __init__(self, input_dim, hidden_sizes, output_dim):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.arch_name = f"MLP({','.join(map(str, hidden_sizes))})"

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class VirtualMLP(nn.Module):
    """MLP backed by virtual parameters."""
    def __init__(self, vpm, input_dim, hidden_sizes, output_dim):
        super().__init__()
        self.vpm = vpm
        self.layers = nn.ModuleList()
        prev = input_dim
        slot = 0
        for h in hidden_sizes:
            self.layers.append(VirtualLinear(vpm, prev, h, slot_id=slot))
            slot += 10
            prev = h
        self.output = VirtualLinear(vpm, prev, output_dim, slot_id=slot, gain=1.0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x)


class BaselineCNN(nn.Module):
    """Standard CNN for CIFAR-10."""
    def __init__(self, conv_channels=(32, 64), fc_hidden=256):
        super().__init__()
        c1, c2 = conv_channels
        self.conv1 = nn.Conv2d(3, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(c2 * 8 * 8, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 10)
        self.arch_name = f"CNN({c1},{c2},{fc_hidden})"

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class VirtualCNN(nn.Module):
    """CNN backed by virtual parameters for CIFAR-10."""
    def __init__(self, vpm, conv_channels=(32, 64), fc_hidden=256):
        super().__init__()
        c1, c2 = conv_channels
        self.vpm = vpm
        self.conv1 = VirtualConv2d(vpm, 3, c1, 3, padding=1, slot_id=0)
        self.conv2 = VirtualConv2d(vpm, c1, c2, 3, padding=1, slot_id=10)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = VirtualLinear(vpm, c2 * 8 * 8, fc_hidden, slot_id=20)
        self.fc2 = VirtualLinear(vpm, fc_hidden, 10, slot_id=30, gain=1.0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ============================================================================
# Training with per-epoch logging
# ============================================================================

def train_model(model, train_loader, test_loader, device, epochs, seed):
    """Train and return per-epoch curves."""
    set_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model = model.to(device)
    n_params = count_params(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    epoch_data = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0
        n_batches = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / n_batches

        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)
        test_acc = correct / total

        scheduler.step()
        epoch_data.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 5),
            "test_acc": round(test_acc, 4),
        })

    elapsed = time.time() - start
    best_acc = max(e["test_acc"] for e in epoch_data)
    last_acc = epoch_data[-1]["test_acc"]

    return {
        "num_params": n_params,
        "best_test_acc": best_acc,
        "last_test_acc": last_acc,
        "time_s": round(elapsed, 1),
        "epochs": epoch_data,
    }


# ============================================================================
# Incremental save/load
# ============================================================================

def load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"results": []}


def save_results(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def already_done(data, config_name, seed):
    for r in data["results"]:
        if r["config_name"] == config_name and r["seed"] == seed:
            return True
    return False


# ============================================================================
# Best baseline architecture search (analytical)
# ============================================================================

def best_mlp1_hidden(budget, input_dim=784, output_dim=10):
    """Largest 1-layer MLP within budget. Params = (input+1)*h + (h+1)*out."""
    # h*(input+1+output) + output = budget => h = (budget-output)/(input+1+output)
    h = (budget - output_dim) // (input_dim + 1 + output_dim)
    return max(h, 1)


def best_mlp2_hidden(budget, input_dim=784, output_dim=10):
    """Largest 2-layer square MLP within budget."""
    # h^2 + h*(input+2+output) + 2*output... approximate
    # Simplified: params ~ input*h + h + h*h + h + h*output + output
    #           = h*(input+1) + h^2 + h + h*output + output
    #           = h^2 + h*(input+2+output) + output
    a = 1
    b = input_dim + 2 + output_dim
    c = output_dim - budget
    disc = b * b - 4 * a * c
    if disc < 0:
        return 2
    h = int((-b + math.sqrt(disc)) / (2 * a))
    return max(h, 2)


def make_best_baseline_mnist(budget):
    """Return the best baseline MLP for a given param budget on MNIST."""
    candidates = []

    # 1-layer
    h = best_mlp1_hidden(budget)
    m = BaselineMLP(784, [h], 10)
    candidates.append((m.arch_name, m, count_params(m)))

    # 2-layer square
    h = best_mlp2_hidden(budget)
    if h >= 2:
        m = BaselineMLP(784, [h, h], 10)
        candidates.append((m.arch_name, m, count_params(m)))

    # Pick the one with most params within budget
    candidates = [(n, m, p) for n, m, p in candidates if p <= budget * 1.1]
    if not candidates:
        h = best_mlp1_hidden(budget)
        return f"MLP({h})", BaselineMLP(784, [h], 10)

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0][0], candidates[0][1]


def make_best_baseline_cifar(budget):
    """Return the best baseline CNN for a given param budget on CIFAR-10.

    Varies conv channel counts and FC hidden size to fit within budget.
    """
    best = None
    best_p = 0
    for c1 in [4, 8, 16, 32]:
        for c2 in [8, 16, 32, 64]:
            for fc in [32, 64, 128, 256]:
                m = BaselineCNN((c1, c2), fc)
                p = count_params(m)
                if p <= budget * 1.1 and p > best_p:
                    best = (m.arch_name, m)
                    best_p = p
    if best is None:
        return "CNN(4,8,32)", BaselineCNN((4, 8), 32)
    return best


# ============================================================================
# Experiment definitions
# ============================================================================

def run_experiment(experiment_fn, output_file, description):
    """Generic runner with resume support."""
    path = os.path.join(RESULTS_DIR, output_file)
    data = load_results(path)
    print(f"\n{'='*70}")
    print(description)
    print(f"Output: {path}")
    existing = len(data["results"])
    if existing > 0:
        print(f"Resuming: {existing} results already saved")
    print(f"{'='*70}\n")

    experiment_fn(data, path)

    # Final summary
    print(f"\n--- Summary ---")
    configs_seen = {}
    for r in data["results"]:
        name = r["config_name"]
        if name not in configs_seen:
            configs_seen[name] = []
        configs_seen[name].append(r["best_test_acc"])

    print(f"  {'Config':<30} {'Seeds':>5} {'Best Acc (mean +/- std)':>25}")
    print(f"  {'-'*65}")
    for name, accs in configs_seen.items():
        n = len(accs)
        mean = sum(accs) / n
        std = (sum((a - mean)**2 for a in accs) / max(n - 1, 1)) ** 0.5
        print(f"  {name:<30} {n:>5} {mean:.4f} +/- {std:.4f}")


def exp1_combine_fn_sweep_mnist(data, path):
    """Experiment 1: Combine fn sweep on MNIST at 1/50."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader, test_loader = load_mnist()

    full_params = 269_322  # MLP 784->256->256->10
    ratio = 50
    budget = full_params // ratio
    epochs = 20

    data["experiment"] = "exp1_combine_fn_sweep_mnist"
    data["ratio"] = ratio
    data["budget"] = budget

    combine_fns = get_combine_fns()

    # ArithMap configs
    for fn_name, (num_terms, combine_fn) in combine_fns.items():
        for seed in SEEDS:
            if already_done(data, fn_name, seed):
                print(f"  [skip] {fn_name} seed={seed}")
                continue

            print(f"  [run] {fn_name} seed={seed}")
            set_seed(seed)
            vpm = ArithMap(num_actual=budget, num_terms=num_terms, combine_fn=combine_fn)
            model = VirtualMLP(vpm, 784, [256, 256], 10)
            result = train_model(model, train_loader, test_loader, device, epochs, seed)
            result["config_name"] = fn_name
            result["seed"] = seed
            print(f"    -> best={result['best_test_acc']:.4f} ({result['time_s']:.0f}s)")

            data["results"].append(result)
            save_results(path, data)

    # Sinusoidal
    fn_name = "sinusoidal (K=8)"
    for seed in SEEDS:
        if already_done(data, fn_name, seed):
            print(f"  [skip] {fn_name} seed={seed}")
            continue

        print(f"  [run] {fn_name} seed={seed}")
        set_seed(seed)
        vpm = SinusoidalMap(num_actual=budget, num_terms=8)
        model = VirtualMLP(vpm, 784, [256, 256], 10)
        result = train_model(model, train_loader, test_loader, device, epochs, seed)
        result["config_name"] = fn_name
        result["seed"] = seed
        print(f"    -> best={result['best_test_acc']:.4f} ({result['time_s']:.0f}s)")

        data["results"].append(result)
        save_results(path, data)

    # Baseline
    fn_name = "baseline"
    for seed in SEEDS:
        if already_done(data, fn_name, seed):
            print(f"  [skip] {fn_name} seed={seed}")
            continue

        print(f"  [run] {fn_name} seed={seed}")
        arch_name, model = make_best_baseline_mnist(budget)
        result = train_model(model, train_loader, test_loader, device, epochs, seed)
        result["config_name"] = fn_name
        result["arch_name"] = arch_name
        result["seed"] = seed
        print(f"    -> {arch_name} best={result['best_test_acc']:.4f} ({result['time_s']:.0f}s)")

        data["results"].append(result)
        save_results(path, data)


def exp2_combine_fn_sweep_cifar(data, path):
    """Experiment 2: Combine fn sweep on CIFAR-10 at 1/50."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader, test_loader = load_cifar10()

    full_params = count_params(BaselineCNN())
    ratio = 50
    budget = full_params // ratio
    epochs = 20

    data["experiment"] = "exp2_combine_fn_sweep_cifar"
    data["ratio"] = ratio
    data["budget"] = budget

    # Test a subset: identity, hash_arith, deep_hash, deeper_hash, sinusoidal, baseline
    test_fns = {k: v for k, v in get_combine_fns().items()
                if k in ("identity (x=1)", "hash_arith (x=3)", "deep_hash (x=7)",
                         "deeper_hash (x=11)", "multiply (x=2)")}

    for fn_name, (num_terms, combine_fn) in test_fns.items():
        for seed in SEEDS:
            if already_done(data, fn_name, seed):
                print(f"  [skip] {fn_name} seed={seed}")
                continue

            print(f"  [run] {fn_name} seed={seed}")
            set_seed(seed)
            vpm = ArithMap(num_actual=budget, num_terms=num_terms, combine_fn=combine_fn)
            model = VirtualCNN(vpm)
            result = train_model(model, train_loader, test_loader, device, epochs, seed)
            result["config_name"] = fn_name
            result["seed"] = seed
            print(f"    -> best={result['best_test_acc']:.4f} ({result['time_s']:.0f}s)")

            data["results"].append(result)
            save_results(path, data)

    # Sinusoidal
    fn_name = "sinusoidal (K=8)"
    for seed in SEEDS:
        if already_done(data, fn_name, seed):
            print(f"  [skip] {fn_name} seed={seed}")
            continue

        print(f"  [run] {fn_name} seed={seed}")
        set_seed(seed)
        vpm = SinusoidalMap(num_actual=budget, num_terms=8)
        model = VirtualCNN(vpm)
        result = train_model(model, train_loader, test_loader, device, epochs, seed)
        result["config_name"] = fn_name
        result["seed"] = seed
        print(f"    -> best={result['best_test_acc']:.4f} ({result['time_s']:.0f}s)")

        data["results"].append(result)
        save_results(path, data)

    # Baseline
    fn_name = "baseline"
    for seed in SEEDS:
        if already_done(data, fn_name, seed):
            print(f"  [skip] {fn_name} seed={seed}")
            continue

        print(f"  [run] {fn_name} seed={seed}")
        arch_name, model = make_best_baseline_cifar(budget)
        result = train_model(model, train_loader, test_loader, device, epochs, seed)
        result["config_name"] = fn_name
        result["arch_name"] = arch_name
        result["seed"] = seed
        print(f"    -> {arch_name} best={result['best_test_acc']:.4f} ({result['time_s']:.0f}s)")

        data["results"].append(result)
        save_results(path, data)


def exp3_compression_scaling_mnist(data, path):
    """Experiment 3: Vary compression ratio on MNIST."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader, test_loader = load_mnist()

    full_params = 269_322
    ratios = [4, 10, 25, 50, 100, 200, 500]
    epochs = 20

    data["experiment"] = "exp3_compression_scaling_mnist"
    data["ratios"] = ratios

    # Test configs: identity, hash_arith, deep_hash + baseline per ratio
    test_fns = {k: v for k, v in get_combine_fns().items()
                if k in ("identity (x=1)", "hash_arith (x=3)", "deep_hash (x=7)")}

    for ratio in ratios:
        budget = full_params // ratio
        print(f"\n--- 1/{ratio} compression (M={budget:,}) ---")

        for fn_name, (num_terms, combine_fn) in test_fns.items():
            config_name = f"{fn_name} @ 1/{ratio}"
            for seed in SEEDS:
                if already_done(data, config_name, seed):
                    print(f"  [skip] {config_name} seed={seed}")
                    continue

                print(f"  [run] {config_name} seed={seed}")
                set_seed(seed)
                vpm = ArithMap(num_actual=budget, num_terms=num_terms, combine_fn=combine_fn)
                model = VirtualMLP(vpm, 784, [256, 256], 10)
                result = train_model(model, train_loader, test_loader, device, epochs, seed)
                result["config_name"] = config_name
                result["ratio"] = ratio
                result["budget"] = budget
                result["seed"] = seed
                print(f"    -> best={result['best_test_acc']:.4f} ({result['time_s']:.0f}s)")

                data["results"].append(result)
                save_results(path, data)

        # Baseline for this ratio
        config_name = f"baseline @ 1/{ratio}"
        for seed in SEEDS:
            if already_done(data, config_name, seed):
                print(f"  [skip] {config_name} seed={seed}")
                continue

            print(f"  [run] {config_name} seed={seed}")
            arch_name, model = make_best_baseline_mnist(budget)
            result = train_model(model, train_loader, test_loader, device, epochs, seed)
            result["config_name"] = config_name
            result["arch_name"] = arch_name
            result["ratio"] = ratio
            result["budget"] = budget
            result["seed"] = seed
            print(f"    -> {arch_name} best={result['best_test_acc']:.4f} ({result['time_s']:.0f}s)")

            data["results"].append(result)
            save_results(path, data)


def exp4_scale_up_mnist(data, path):
    """Experiment 4: Scale architecture UP beyond normal, fixed actual params."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader, test_loader = load_mnist()

    budget = 5386  # same as 1/50 of the 256-wide model
    epochs = 20
    hidden_sizes = [128, 256, 512, 1024, 2048]

    data["experiment"] = "exp4_scale_up_mnist"
    data["budget"] = budget

    combine_fns = get_combine_fns()
    # Use deep_hash as the main mapping
    fn_name_base = "deep_hash (x=7)"
    num_terms, combine_fn = combine_fns[fn_name_base]

    for H in hidden_sizes:
        config_name = f"virtual H={H}"
        virtual_params = 784 * H + H + H * H + H + H * 10 + 10
        compression = virtual_params / budget

        for seed in SEEDS:
            if already_done(data, config_name, seed):
                print(f"  [skip] {config_name} seed={seed}")
                continue

            print(f"  [run] {config_name} (virtual={virtual_params:,}, compression={compression:.0f}:1) seed={seed}")
            set_seed(seed)
            vpm = ArithMap(num_actual=budget, num_terms=num_terms, combine_fn=combine_fn)
            model = VirtualMLP(vpm, 784, [H, H], 10)
            result = train_model(model, train_loader, test_loader, device, epochs, seed)
            result["config_name"] = config_name
            result["hidden"] = H
            result["virtual_params"] = virtual_params
            result["compression_ratio"] = round(compression, 1)
            result["seed"] = seed
            print(f"    -> best={result['best_test_acc']:.4f} ({result['time_s']:.0f}s)")

            data["results"].append(result)
            save_results(path, data)

    # Non-conflated baselines at same widths (where feasible)
    for H in hidden_sizes:
        config_name = f"baseline-full H={H}"
        m = BaselineMLP(784, [H, H], 10)
        full_p = count_params(m)

        for seed in SEEDS:
            if already_done(data, config_name, seed):
                print(f"  [skip] {config_name} seed={seed}")
                continue

            print(f"  [run] {config_name} ({full_p:,} independent params) seed={seed}")
            model = BaselineMLP(784, [H, H], 10)
            result = train_model(model, train_loader, test_loader, device, epochs, seed)
            result["config_name"] = config_name
            result["hidden"] = H
            result["num_independent_params"] = full_p
            result["seed"] = seed
            print(f"    -> best={result['best_test_acc']:.4f} ({result['time_s']:.0f}s)")

            data["results"].append(result)
            save_results(path, data)

    # Param-matched baseline (best arch with ~budget params)
    config_name = "baseline-matched"
    for seed in SEEDS:
        if already_done(data, config_name, seed):
            print(f"  [skip] {config_name} seed={seed}")
            continue

        print(f"  [run] {config_name} seed={seed}")
        arch_name, model = make_best_baseline_mnist(budget)
        result = train_model(model, train_loader, test_loader, device, epochs, seed)
        result["config_name"] = config_name
        result["arch_name"] = arch_name
        result["seed"] = seed
        print(f"    -> {arch_name} best={result['best_test_acc']:.4f} ({result['time_s']:.0f}s)")

        data["results"].append(result)
        save_results(path, data)


def exp5_compression_scaling_cifar(data, path):
    """Experiment 5: Vary compression ratio on CIFAR-10."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader, test_loader = load_cifar10()

    full_params = count_params(BaselineCNN())
    ratios = [4, 10, 25, 50, 100]
    epochs = 20

    data["experiment"] = "exp5_compression_scaling_cifar"
    data["ratios"] = ratios

    test_fns = {k: v for k, v in get_combine_fns().items()
                if k in ("identity (x=1)", "deep_hash (x=7)")}

    for ratio in ratios:
        budget = full_params // ratio
        print(f"\n--- 1/{ratio} compression (M={budget:,}) ---")

        for fn_name, (num_terms, combine_fn) in test_fns.items():
            config_name = f"{fn_name} @ 1/{ratio}"
            for seed in SEEDS:
                if already_done(data, config_name, seed):
                    print(f"  [skip] {config_name} seed={seed}")
                    continue

                print(f"  [run] {config_name} seed={seed}")
                set_seed(seed)
                vpm = ArithMap(num_actual=budget, num_terms=num_terms, combine_fn=combine_fn)
                model = VirtualCNN(vpm)
                result = train_model(model, train_loader, test_loader, device, epochs, seed)
                result["config_name"] = config_name
                result["ratio"] = ratio
                result["budget"] = budget
                result["seed"] = seed
                print(f"    -> best={result['best_test_acc']:.4f} ({result['time_s']:.0f}s)")

                data["results"].append(result)
                save_results(path, data)

        config_name = f"baseline @ 1/{ratio}"
        for seed in SEEDS:
            if already_done(data, config_name, seed):
                print(f"  [skip] {config_name} seed={seed}")
                continue

            print(f"  [run] {config_name} seed={seed}")
            arch_name, model = make_best_baseline_cifar(budget)
            result = train_model(model, train_loader, test_loader, device, epochs, seed)
            result["config_name"] = config_name
            result["arch_name"] = arch_name
            result["ratio"] = ratio
            result["budget"] = budget
            result["seed"] = seed
            print(f"    -> {arch_name} best={result['best_test_acc']:.4f} ({result['time_s']:.0f}s)")

            data["results"].append(result)
            save_results(path, data)


# ============================================================================
# Main
# ============================================================================

EXPERIMENTS = {
    "exp1": ("exp1_combine_fn_sweep_mnist.json",
             "Exp 1: Combine fn sweep — MNIST 1/50",
             exp1_combine_fn_sweep_mnist),
    "exp2": ("exp2_combine_fn_sweep_cifar.json",
             "Exp 2: Combine fn sweep — CIFAR-10 1/50",
             exp2_combine_fn_sweep_cifar),
    "exp3": ("exp3_compression_scaling_mnist.json",
             "Exp 3: Compression ratio scaling — MNIST",
             exp3_compression_scaling_mnist),
    "exp4": ("exp4_scale_up_mnist.json",
             "Exp 4: Scale UP — MNIST, fixed M, vary width",
             exp4_scale_up_mnist),
    "exp5": ("exp5_compression_scaling_cifar.json",
             "Exp 5: Compression ratio scaling — CIFAR-10",
             exp5_compression_scaling_cifar),
}

PRIORITY_ORDER = ["exp1", "exp4", "exp3", "exp2", "exp5"]


def main():
    if len(sys.argv) < 2:
        print("Usage: python experiments/training_curves.py <exp1|exp2|exp3|exp4|exp5|all>")
        print("\nExperiments:")
        for key in PRIORITY_ORDER:
            fname, desc, _ = EXPERIMENTS[key]
            print(f"  {key}: {desc}")
        print(f"  all: run all in priority order")
        sys.exit(1)

    target = sys.argv[1]

    if target == "all":
        for key in PRIORITY_ORDER:
            fname, desc, fn = EXPERIMENTS[key]
            run_experiment(fn, fname, desc)
    elif target in EXPERIMENTS:
        fname, desc, fn = EXPERIMENTS[target]
        run_experiment(fn, fname, desc)
    else:
        print(f"Unknown experiment: {target}")
        print(f"Valid options: {', '.join(EXPERIMENTS.keys())}, all")
        sys.exit(1)


if __name__ == "__main__":
    main()
