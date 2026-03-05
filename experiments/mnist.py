"""
MNIST experiment: compare standard MLPs vs virtual-parameter MLPs.

Models:
  1. Baseline (full): MLP with standard parameters (~267K params)
  2. Virtual (sinusoidal): Same architecture, backed by M actual params via SinusoidalMap
  3. Virtual (hash_arith): Same architecture, backed by M actual params via HashArithMap
  4. Baseline (small): Smaller MLP with ~M params (to compare against "just use fewer params")

All models: 784 -> 256 -> 256 -> 10
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

from virtual_params import VirtualLinear, SinusoidalMap, HashArithMap


def count_params(model):
    """Count total learnable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class BaselineMLP(nn.Module):
    """Standard MLP with full parameters."""
    def __init__(self, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class VirtualMLP(nn.Module):
    """MLP backed by virtual parameters from a VirtualParameterMap."""
    def __init__(self, vpm, hidden=256):
        super().__init__()
        self.vpm = vpm
        # Assign non-overlapping slot_ids: each layer needs 2 (weight + bias)
        self.fc1 = VirtualLinear(vpm, 784, hidden, slot_id=0)
        self.fc2 = VirtualLinear(vpm, hidden, hidden, slot_id=100)
        self.fc3 = VirtualLinear(vpm, hidden, 10, slot_id=200)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    return total_loss / total, correct / total


def run_experiment(model, name, train_loader, test_loader, device,
                   epochs=20, lr=1e-3):
    model = model.to(device)
    n_params = count_params(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"Actual learnable params: {n_params:,}")
    print(f"{'='*60}")

    results = {
        "name": name,
        "num_params": n_params,
        "epochs": [],
    }

    best_test_acc = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        best_test_acc = max(best_test_acc, test_acc)

        epoch_data = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "test_loss": round(test_loss, 4),
            "test_acc": round(test_acc, 4),
        }
        results["epochs"].append(epoch_data)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    elapsed = time.time() - start_time
    results["best_test_acc"] = round(best_test_acc, 4)
    results["training_time_s"] = round(elapsed, 1)

    print(f"  Best test accuracy: {best_test_acc:.4f}")
    print(f"  Training time: {elapsed:.1f}s")

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp", "data")
    train_data = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=2)

    hidden = 256
    epochs = 20
    lr = 1e-3

    all_results = []

    # --- 1. Baseline (full) ---
    model = BaselineMLP(hidden=hidden)
    full_params = count_params(model)
    results = run_experiment(model, "Baseline (full)", train_loader, test_loader, device, epochs, lr)
    all_results.append(results)

    # --- Virtual param experiments at different compression ratios ---
    for ratio in [4, 10, 20, 50]:
        num_actual = full_params // ratio

        # 2. Virtual (sinusoidal)
        vpm_sin = SinusoidalMap(num_actual=num_actual, num_terms=8)
        model = VirtualMLP(vpm_sin, hidden=hidden)
        results = run_experiment(
            model, f"Virtual sinusoidal (1/{ratio}, M={num_actual:,})",
            train_loader, test_loader, device, epochs, lr
        )
        all_results.append(results)

        # 3. Virtual (hash_arith)
        vpm_hash = HashArithMap(num_actual=num_actual)
        model = VirtualMLP(vpm_hash, hidden=hidden)
        results = run_experiment(
            model, f"Virtual hash_arith (1/{ratio}, M={num_actual:,})",
            train_loader, test_loader, device, epochs, lr
        )
        all_results.append(results)

        # 4. Baseline (small) — matched param count
        # Find hidden size that gives ~num_actual params
        # params ≈ 784*h + h + h*h + h + h*10 + 10 = h*(784+h+11) + h + 10
        # Solve approximately
        small_h = int((-795 + math.sqrt(795**2 + 4 * num_actual)) / 2)
        small_h = max(small_h, 4)
        model = BaselineMLP(hidden=small_h)
        results = run_experiment(
            model, f"Baseline small (h={small_h}, ~{count_params(model):,} params)",
            train_loader, test_loader, device, epochs, lr
        )
        all_results.append(results)

    # Save results
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tmp", "mnist_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"MNIST SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<50} {'Params':>10} {'Best Acc':>10} {'Time':>8}")
    print(f"{'-'*80}")
    for r in all_results:
        print(f"{r['name']:<50} {r['num_params']:>10,} {r['best_test_acc']:>10.4f} {r['training_time_s']:>7.1f}s")


if __name__ == "__main__":
    main()
