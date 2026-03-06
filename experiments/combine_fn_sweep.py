"""
Combine function sweep — how does the choice of f affect virtual param quality?

Tests various ArithMap combine_fns on MNIST at 1/50 compression.
Each variant uses the same VirtualMLP architecture (784->256->256->10)
backed by ArithMap with different (num_terms, combine_fn) pairs.

The key question: does the number of actual params per virtual param (x)
matter, and does the specific nonlinearity matter?

Variants tested:
    x=1: identity         a[0]
    x=2: multiply         a[0]*a[1]
    x=2: add              (a[0]+a[1]) / sqrt(2)
    x=3: hash_arith       (a[0]*a[1]+a[2]) / sqrt(2)
    x=3: xor_inspired     (a[0]*a[1]^2+a[2]) / 2
    x=3: rotation         a[0]*cos(a[2]) - a[1]*sin(a[2])
    x=5: wide_hash        (a[0]*a[1]+a[2])*(a[3]+a[4]) / sqrt(6)
    x=7: deep_hash        ((a[0]*a[1]+a[2])*(a[3]*a[4]+a[5])+a[6]) / sqrt(5)
    x=11: deeper_hash     3-level: ((l1)*(l2)+a[7])*((l3)+a[10])+... 

Also runs the best baseline architecture for comparison.

Reports mean +/- std over 5 seeds.
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

from virtual_params import VirtualLinear, ArithMap

SEEDS = [42, 137, 256, 1337, 9999]


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Combine function registry
# ============================================================================

def make_combine_fns():
    """Return list of (name, num_terms, combine_fn) tuples."""
    return [
        # x=1: pure repetition, no mixing
        ("identity (x=1)", 1,
         lambda a: a[0]),

        # x=2: additive, linear mixing
        ("add (x=2)", 2,
         lambda a: (a[0] + a[1]) / math.sqrt(2)),

        # x=2: multiplicative, nonlinear
        ("multiply (x=2)", 2,
         lambda a: a[0] * a[1]),

        # x=3: multiply + add (HashArith)
        ("hash_arith (x=3)", 3,
         lambda a: (a[0] * a[1] + a[2]) / math.sqrt(2)),

        # x=3: asymmetric nonlinearity
        ("xor_inspired (x=3)", 3,
         lambda a: (a[0] * a[1] ** 2 + a[2]) / 2),

        # x=3: rotation-based
        ("rotation (x=3)", 3,
         lambda a: a[0] * torch.cos(a[2]) - a[1] * torch.sin(a[2])),

        # x=5: wider single-level
        # Var((a0*a1+a2)*(a3+a4)) = Var(t1)*Var(t2) = 2*2 = 4
        ("wide_hash (x=5)", 5,
         lambda a: (a[0] * a[1] + a[2]) * (a[3] + a[4]) / 2),

        # x=7: two-level deep (DeepHash)
        ("deep_hash (x=7)", 7,
         lambda a: ((a[0]*a[1] + a[2]) * (a[3]*a[4] + a[5]) + a[6]) / math.sqrt(5)),

        # x=11: three-level deep
        # Level 1: t1 = a0*a1+a2, t2 = a3*a4+a5, t3 = a6*a7+a8
        # Level 2: u1 = t1*t2+a9
        # Level 3: virtual = u1*t3+a10
        # Var(t) = 2, Var(t1*t2) = 4, Var(u1) = 4+1 = 5
        # Var(u1*t3) = 5*2 = 10, Var(result) = 10+1 = 11
        ("deeper_hash (x=11)", 11,
         lambda a: (((a[0]*a[1]+a[2]) * (a[3]*a[4]+a[5]) + a[6])
                    * (a[7]*a[8]+a[9]) + a[10]) / math.sqrt(11)),
    ]


def verify_combine_fns():
    """Sanity-check that all combine_fns produce roughly unit variance."""
    print("Verifying combine_fn output variances...")
    fns = make_combine_fns()
    for name, num_terms, fn in fns:
        torch.manual_seed(0)
        m = ArithMap(num_actual=5000, num_terms=num_terms, combine_fn=fn)
        v = m.get_virtual((50000,), slot_id=0)
        var = v.var().item()
        status = "OK" if 0.5 < var < 2.0 else "WARNING"
        print(f"  {name:<25} var={var:.4f}  [{status}]")
    print()


# ============================================================================
# Models
# ============================================================================

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


# ============================================================================
# Training
# ============================================================================

def train_model(model, name, train_loader, test_loader, device, epochs, seed, quiet=False):
    """Train a model and return results."""
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

    # Verify combine_fns before running
    verify_combine_fns()

    # Setup
    ratio = 50
    full_params = 269_322  # MLP2(256): 784*256+256 + 256*256+256 + 256*10+10
    budget = full_params // ratio
    epochs = 20
    combine_fns = make_combine_fns()

    print("=" * 80)
    print(f"COMBINE FUNCTION SWEEP — MNIST 1/{ratio} compression (M={budget:,})")
    print(f"Architecture: 784->256->256->10, {len(SEEDS)} seeds, {epochs} epochs")
    print("=" * 80)

    all_results = {}

    # Run each combine_fn variant
    for fn_name, num_terms, combine_fn in combine_fns:
        print(f"\n--- {fn_name} ---")
        results = []
        for seed in SEEDS:
            set_seed(seed)
            vpm = ArithMap(num_actual=budget, num_terms=num_terms, combine_fn=combine_fn)
            model = VirtualMLP(vpm, hidden=256)
            r = train_model(model, fn_name, train_loader, test_loader, device, epochs, seed)
            results.append(r)
        all_results[fn_name] = results

    # Baseline: best small architecture (MLP1 with h=6 won at 1/50 previously)
    print(f"\n--- baseline MLP1(h=6) ---")
    baseline_results = []
    for seed in SEEDS:
        model = MLP1(hidden=6)
        r = train_model(model, "MLP1(h=6)", train_loader, test_loader, device, epochs, seed)
        baseline_results.append(r)
    all_results["baseline MLP1(h=6)"] = baseline_results

    # Summary
    print("\n" + "=" * 90)
    print(f"RESULTS — 1/{ratio} compression (M={budget:,}) — mean +/- std over {len(SEEDS)} seeds")
    print("=" * 90)
    print(f"  {'Method':<28} {'x':>3} {'Params':>8} {'Best Acc (mean +/- std)':>25}")
    print(f"  {'-' * 70}")

    summary_rows = []
    for fn_name in list(all_results.keys()):
        results = all_results[fn_name]
        accs = [r["best_test_acc"] for r in results]
        n = len(accs)
        mean = sum(accs) / n
        std = (sum((a - mean)**2 for a in accs) / (n - 1)) ** 0.5 if n > 1 else 0.0
        params = results[0]["num_params"]

        # Extract x from name
        if "(x=" in fn_name:
            x_str = fn_name.split("(x=")[1].rstrip(")")
        else:
            x_str = "-"

        print(f"  {fn_name:<28} {x_str:>3} {params:>8,}  {mean:.4f} +/- {std:.4f}")
        summary_rows.append({
            "method": fn_name, "num_terms": x_str, "params": params,
            "best_mean": round(mean, 4), "best_std": round(std, 4),
            "individual_accs": accs,
        })

    # Save
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "combine_fn_sweep.json")
    with open(results_path, "w") as f:
        json.dump({
            "ratio": ratio,
            "budget": budget,
            "epochs": epochs,
            "seeds": SEEDS,
            "summary": summary_rows,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
