"""
Extreme compression experiment on MNIST — rigorous version.

Compares virtual-param models against the BEST baseline architecture
at each param budget. Tests multiple baseline architectures:
  - 2-layer MLP: 784->h->h->10 (square)
  - 1-layer MLP: 784->h->10
  - 2-layer tapered: 784->h1->h2->10 with h1=4*h2

Reports mean +/- std over 5 seeds. Reports both best-epoch and
last-epoch test accuracy.
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

from virtual_params import VirtualLinear, SinusoidalMap, HashArithMap, DeepHashMap

SEEDS = [42, 137, 256, 1337, 9999]


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Model definitions
# ============================================================================

class MLP1(nn.Module):
    """1-hidden-layer MLP: 784 -> h -> 10."""
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        self.arch_name = f"MLP1(h={hidden})"

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MLP2(nn.Module):
    """2-hidden-layer MLP: 784 -> h -> h -> 10."""
    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)
        self.arch_name = f"MLP2(h={hidden})"

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MLP2Tapered(nn.Module):
    """2-hidden-layer tapered MLP: 784 -> h1 -> h2 -> 10."""
    def __init__(self, h1, h2):
        super().__init__()
        self.fc1 = nn.Linear(784, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 10)
        self.arch_name = f"MLP2T({h1},{h2})"

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class VirtualMLP(nn.Module):
    def __init__(self, vpm, hidden=256):
        super().__init__()
        self.vpm = vpm
        self.fc1 = VirtualLinear(vpm, 784, hidden, slot_id=0)
        self.fc2 = VirtualLinear(vpm, hidden, hidden, slot_id=10)
        self.fc3 = VirtualLinear(vpm, hidden, 10, slot_id=20, gain=1.0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================================
# Architecture search — analytical + a few candidates per budget
# ============================================================================

def best_mlp1_for_budget(budget):
    """Find largest 1-layer MLP within budget. Params = 795*h + 10."""
    h = (budget - 10) // 795
    return max(h, 1)


def best_mlp2_for_budget(budget):
    """Find largest 2-layer square MLP within budget. Params = h^2 + 796h + 10."""
    disc = 796**2 + 4 * (budget - 10)
    if disc < 0:
        return 2
    h = int((-796 + math.sqrt(disc)) / 2)
    return max(h, 2)


def best_tapered_for_budget(budget, taper_ratio=4):
    """Find largest 2-layer tapered MLP within budget, h1 = taper_ratio * h2.
    Params = 784*h1 + h1 + h1*h2 + h2 + 10*h2 + 10
           = h1*(785 + h2) + h2*11 + 10
    With h1 = taper_ratio * h2:
           = taper_ratio*h2*(785 + h2) + h2*11 + 10
           = taper_ratio*h2^2 + (785*taper_ratio + 11)*h2 + 10

    Returns (h1, h2) or None if budget is too small.
    """
    a = taper_ratio
    b = 785 * taper_ratio + 11
    c = 10 - budget
    disc = b**2 - 4 * a * c
    if disc < 0:
        return None
    h2 = int((-b + math.sqrt(disc)) / (2 * a))
    if h2 < 2:
        return None
    h1 = taper_ratio * h2
    return h1, h2


def _make_factory(model):
    """Create a callable that reconstructs a model with the same architecture."""
    if isinstance(model, MLP1):
        h = model.fc1.out_features
        return lambda: MLP1(h)
    elif isinstance(model, MLP2Tapered):
        h1 = model.fc1.out_features
        h2 = model.fc2.out_features
        return lambda h1=h1, h2=h2: MLP2Tapered(h1, h2)
    elif isinstance(model, MLP2):
        h = model.fc1.out_features
        return lambda h=h: MLP2(h)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def get_baseline_candidates(budget):
    """Return a list of (name, model) baseline candidates within ~budget params."""
    candidates = []

    # 1-layer MLP
    h = best_mlp1_for_budget(budget)
    if h >= 1:
        m = MLP1(h)
        candidates.append((m.arch_name, m, count_params(m)))

    # 2-layer square MLP
    h = best_mlp2_for_budget(budget)
    if h >= 2:
        m = MLP2(h)
        candidates.append((m.arch_name, m, count_params(m)))

    # 2-layer tapered (4:1 ratio)
    result = best_tapered_for_budget(budget, taper_ratio=4)
    if result is not None:
        h1, h2 = result
        m = MLP2Tapered(h1, h2)
        candidates.append((m.arch_name, m, count_params(m)))

    # 2-layer tapered (2:1 ratio)
    result = best_tapered_for_budget(budget, taper_ratio=2)
    if result is not None:
        h1, h2 = result
        m = MLP2Tapered(h1, h2)
        candidates.append((m.arch_name, m, count_params(m)))

    return candidates


# ============================================================================
# Training
# ============================================================================

def train_model(model, name, train_loader, test_loader, device, epochs, seed, quiet=False):
    """Train a model and return results."""
    set_seed(seed)
    # Re-initialize weights with this seed for fairness
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model = model.to(device)
    n_params = count_params(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    last_acc = 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

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
        acc = correct / total
        best_acc = max(best_acc, acc)
        last_acc = acc
        scheduler.step()

    elapsed = time.time() - start
    if not quiet:
        print(f"    {name} ({n_params:,}p): best={best_acc:.4f} last={last_acc:.4f} ({elapsed:.0f}s)")

    return {"name": name, "num_params": n_params,
            "best_test_acc": round(best_acc, 4),
            "last_test_acc": round(last_acc, 4),
            "time_s": round(elapsed, 1)}


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp", "data")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False, num_workers=0)

    full_params = count_params(MLP2(256))
    epochs = 20
    ratios = [50, 100, 200, 500]

    # Phase 1: Find best baseline architecture per budget
    print("=" * 70)
    print("PHASE 1: Baseline Architecture Search (seed=42)")
    print("=" * 70)

    # Store (result_dict, model_factory) per ratio. model_factory is a callable
    # that returns a fresh model instance — avoids fragile string parsing.
    best_baselines = {}
    for ratio in ratios:
        budget = full_params // ratio
        print(f"\n--- Budget {budget:,} (1/{ratio} compression) ---")
        candidates = get_baseline_candidates(budget)
        best_result = {"name": "None", "num_params": 0, "best_test_acc": 0.0, "last_test_acc": 0.0, "time_s": 0.0}
        best_acc = -1.0
        best_factory = None
        for name, model, p in candidates:
            # Capture model class and args for later reconstruction
            factory = _make_factory(model)
            r = train_model(model, name, train_loader, test_loader, device, epochs, seed=42)
            if r["best_test_acc"] > best_acc:
                best_acc = r["best_test_acc"]
                best_result = r
                best_factory = factory
        assert best_factory is not None, f"No baseline candidates for budget {budget}"
        best_baselines[ratio] = (best_result, best_factory)
        print(f"  => Winner: {best_result['name']} ({best_result['num_params']:,}p, acc={best_result['best_test_acc']:.4f})")

    # Phase 2: Multi-seed evaluation of virtual methods + winning baselines
    print("\n" + "=" * 70)
    print("PHASE 2: Multi-seed evaluation (5 seeds)")
    print("=" * 70)

    all_results = {}
    for ratio in ratios:
        budget = full_params // ratio
        print(f"\n--- 1/{ratio} compression (M={budget:,}) ---")

        ratio_results = {"deephash": [], "hasharith": [], "sinusoidal": [], "baseline": []}
        baseline_result, baseline_factory = best_baselines[ratio]
        winner_name = baseline_result["name"]

        for seed in SEEDS:
            print(f"  Seed {seed}:")

            # DeepHash
            set_seed(seed)
            vpm = DeepHashMap(num_actual=budget)
            model = VirtualMLP(vpm, hidden=256)
            r = train_model(model, f"DeepHash", train_loader, test_loader, device, epochs, seed)
            ratio_results["deephash"].append(r)

            # HashArith
            set_seed(seed)
            vpm = HashArithMap(num_actual=budget)
            model = VirtualMLP(vpm, hidden=256)
            r = train_model(model, f"HashArith", train_loader, test_loader, device, epochs, seed)
            ratio_results["hasharith"].append(r)

            # Sinusoidal
            set_seed(seed)
            vpm = SinusoidalMap(num_actual=budget, num_terms=8)
            model = VirtualMLP(vpm, hidden=256)
            r = train_model(model, f"Sinusoidal", train_loader, test_loader, device, epochs, seed)
            ratio_results["sinusoidal"].append(r)

            # Reconstruct winning baseline from factory
            model = baseline_factory()
            r = train_model(model, f"Best-Baseline", train_loader, test_loader, device, epochs, seed)
            ratio_results["baseline"].append(r)

        all_results[ratio] = ratio_results

    # Phase 3: Summary
    print("\n" + "=" * 90)
    print("RESULTS (mean +/- std over 5 seeds)")
    print("=" * 90)

    summary_rows = []
    for ratio in ratios:
        rr = all_results[ratio]
        winner_name = best_baselines[ratio][0]["name"]
        winner_params = best_baselines[ratio][0]["num_params"]
        budget = full_params // ratio
        print(f"\n  1/{ratio} compression | budget={budget:,} | baseline={winner_name} ({winner_params:,}p)")
        print(f"  {'Method':<20} {'Params':>8} {'Best Acc (mean +/- std)':>25} {'Last Acc (mean +/- std)':>25}")
        print(f"  {'-' * 80}")

        for method in ["deephash", "hasharith", "sinusoidal", "baseline"]:
            accs_best = [r["best_test_acc"] for r in rr[method]]
            accs_last = [r["last_test_acc"] for r in rr[method]]
            n = len(accs_best)
            mean_b = sum(accs_best) / n
            std_b = (sum((a - mean_b)**2 for a in accs_best) / (n - 1)) ** 0.5 if n > 1 else 0.0
            mean_l = sum(accs_last) / n
            std_l = (sum((a - mean_l)**2 for a in accs_last) / (n - 1)) ** 0.5 if n > 1 else 0.0
            params = rr[method][0]["num_params"]
            label = method if method != "baseline" else f"baseline({winner_name})"
            print(f"  {label:<20} {params:>8,}  {mean_b:.4f} +/- {std_b:.4f}          {mean_l:.4f} +/- {std_l:.4f}")
            summary_rows.append({
                "ratio": ratio, "method": method, "params": params,
                "baseline_arch": winner_name if method == "baseline" else None,
                "best_mean": round(mean_b, 4), "best_std": round(std_b, 4),
                "last_mean": round(mean_l, 4), "last_std": round(std_l, 4),
            })

    # Delta analysis
    print(f"\n--- Delta vs Best Baseline (best-epoch mean) ---")
    for ratio in ratios:
        rr = all_results[ratio]
        baseline_mean = sum(r["best_test_acc"] for r in rr["baseline"]) / len(rr["baseline"])
        for method in ["deephash", "hasharith", "sinusoidal"]:
            method_mean = sum(r["best_test_acc"] for r in rr[method]) / len(rr[method])
            delta = method_mean - baseline_mean
            marker = " ***" if delta > 0 else ""
            print(f"  1/{ratio:<4} {method:<15}: {delta:+.4f}{marker}")

    # Save
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "extreme_compression_v2.json"), "w") as f:
        json.dump({
            "baselines": {str(k): v[0] for k, v in best_baselines.items()},
            "summary": summary_rows,
            "seeds": SEEDS,
        }, f, indent=2)
    print(f"\nResults saved to {os.path.join(results_dir, 'extreme_compression_v2.json')}")


if __name__ == "__main__":
    main()
