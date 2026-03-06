"""
CIFAR-10 experiment: compare standard CNN vs virtual-parameter CNN.

Models:
  1. Baseline (full): Small CNN with standard parameters
  2. Virtual (sinusoidal): Same architecture, backed by M actual params
  3. Virtual (hash_arith): Same architecture, backed by M actual params
  4. Baseline (small): Smaller CNN with ~M params

Architecture: Conv(3->32,3) -> Conv(32->64,3) -> Conv(64->64,3) -> FC(1024->256) -> FC(256->10)
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

from virtual_params import VirtualLinear, VirtualConv2d, SinusoidalMap, HashArithMap


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class BaselineCNN(nn.Module):
    def __init__(self, ch1=32, ch2=64, ch3=64, fc_hidden=256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, ch1, 3, padding=1)
        self.conv2 = nn.Conv2d(ch1, ch2, 3, padding=1)
        self.conv3 = nn.Conv2d(ch2, ch3, 3, padding=1)
        # After 3 max-pools of 2x2: 32->16->8->4, so 4*4*ch3
        self.fc1 = nn.Linear(4 * 4 * ch3, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class VirtualCNN(nn.Module):
    def __init__(self, vpm, ch1=32, ch2=64, ch3=64, fc_hidden=256):
        super().__init__()
        self.vpm = vpm
        # Slot IDs spaced apart to avoid collisions
        self.conv1 = VirtualConv2d(vpm, 3, ch1, 3, padding=1, slot_id=0)
        self.conv2 = VirtualConv2d(vpm, ch1, ch2, 3, padding=1, slot_id=100)
        self.conv3 = VirtualConv2d(vpm, ch2, ch3, 3, padding=1, slot_id=200)
        self.fc1 = VirtualLinear(vpm, 4 * 4 * ch3, fc_hidden, slot_id=300)
        self.fc2 = VirtualLinear(vpm, fc_hidden, 10, slot_id=400, gain=1.0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


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
                   epochs=30, lr=1e-3):
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

    # Data with standard augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp", "data")
    train_data = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=2)

    epochs = 30
    lr = 1e-3

    all_results = []

    # --- 1. Baseline (full) ---
    model = BaselineCNN()
    full_params = count_params(model)
    results = run_experiment(model, "Baseline (full)", train_loader, test_loader, device, epochs, lr)
    all_results.append(results)

    # --- Virtual param experiments at different compression ratios ---
    for ratio in [4, 10, 20]:
        num_actual = full_params // ratio

        # Virtual (sinusoidal)
        vpm_sin = SinusoidalMap(num_actual=num_actual, num_terms=8)
        model = VirtualCNN(vpm_sin)
        results = run_experiment(
            model, f"Virtual sinusoidal (1/{ratio}, M={num_actual:,})",
            train_loader, test_loader, device, epochs, lr
        )
        all_results.append(results)

        # Virtual (hash_arith)
        vpm_hash = HashArithMap(num_actual=num_actual)
        model = VirtualCNN(vpm_hash)
        results = run_experiment(
            model, f"Virtual hash_arith (1/{ratio}, M={num_actual:,})",
            train_loader, test_loader, device, epochs, lr
        )
        all_results.append(results)

        # Baseline small — reduce channels proportionally
        scale = 1.0 / math.sqrt(ratio)
        small_ch1 = max(int(32 * scale), 4)
        small_ch2 = max(int(64 * scale), 4)
        small_ch3 = max(int(64 * scale), 4)
        small_fc = max(int(256 * scale), 4)
        model = BaselineCNN(ch1=small_ch1, ch2=small_ch2, ch3=small_ch3, fc_hidden=small_fc)
        results = run_experiment(
            model, f"Baseline small (1/{ratio}, ~{count_params(model):,} params)",
            train_loader, test_loader, device, epochs, lr
        )
        all_results.append(results)

    # Save results
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tmp", "cifar10_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"CIFAR-10 SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<50} {'Params':>10} {'Best Acc':>10} {'Time':>8}")
    print(f"{'-'*80}")
    for r in all_results:
        print(f"{r['name']:<50} {r['num_params']:>10,} {r['best_test_acc']:>10.4f} {r['training_time_s']:>7.1f}s")


if __name__ == "__main__":
    main()
