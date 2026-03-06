"""
Combined experiment runner for parameter conflation.

Runs MNIST, CIFAR-10, and sequence modeling experiments with reduced
epoch counts for CPU feasibility. Produces a unified results summary.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import time
import json
import math
import urllib.request

from virtual_params import VirtualLinear, VirtualConv2d, SinusoidalMap, HashArithMap

SEED = 42


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Training utilities
# ============================================================================

def train_epoch_classify(model, loader, optimizer, device):
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


def eval_classify(model, loader, device):
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


def train_epoch_seq(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    return total_loss / total_tokens


def eval_seq(model, loader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    return total_loss / total_tokens


def run_classify(model, name, train_loader, test_loader, device, epochs=10, lr=1e-3):
    set_seed(SEED)
    model = model.to(device)
    n_params = count_params(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"  {name} ({n_params:,} params)...", end="", flush=True)
    start = time.time()
    best_acc = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch_classify(model, train_loader, optimizer, device)
        test_loss, test_acc = eval_classify(model, test_loader, device)
        scheduler.step()
        best_acc = max(best_acc, test_acc)
        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4), "test_loss": round(test_loss, 4),
            "test_acc": round(test_acc, 4)
        })

    elapsed = time.time() - start
    print(f" best_acc={best_acc:.4f} ({elapsed:.0f}s)")
    return {"name": name, "num_params": n_params, "best_test_acc": round(best_acc, 4),
            "training_time_s": round(elapsed, 1), "epochs": history}


def run_seq(model, name, train_loader, test_loader, device, epochs=10, lr=1e-3):
    set_seed(SEED)
    model = model.to(device)
    n_params = count_params(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"  {name} ({n_params:,} params)...", end="", flush=True)
    start = time.time()
    best_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch_seq(model, train_loader, optimizer, device)
        test_loss = eval_seq(model, test_loader, device)
        scheduler.step()
        best_loss = min(best_loss, test_loss)
        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "test_loss": round(test_loss, 4),
            "train_bpc": round(train_loss / math.log(2), 4),
            "test_bpc": round(test_loss / math.log(2), 4),
        })

    elapsed = time.time() - start
    best_bpc = best_loss / math.log(2)
    print(f" best_bpc={best_bpc:.4f} ({elapsed:.0f}s)")
    return {"name": name, "num_params": n_params, "best_test_bpc": round(best_bpc, 4),
            "best_test_loss": round(best_loss, 4),
            "training_time_s": round(elapsed, 1), "epochs": history}


# ============================================================================
# Model definitions
# ============================================================================

class BaselineMLP(nn.Module):
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
    def __init__(self, vpm, hidden=256):
        super().__init__()
        self.vpm = vpm
        # Slot IDs spaced by 10 (each layer uses 2: weight + bias)
        self.fc1 = VirtualLinear(vpm, 784, hidden, slot_id=0)           # ReLU follows: gain=2.0 (default)
        self.fc2 = VirtualLinear(vpm, hidden, hidden, slot_id=10)       # ReLU follows: gain=2.0 (default)
        self.fc3 = VirtualLinear(vpm, hidden, 10, slot_id=20, gain=1.0) # Output layer: gain=1.0

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class BaselineCNN(nn.Module):
    def __init__(self, ch1=32, ch2=64, ch3=64, fc_hidden=256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, ch1, 3, padding=1)
        self.conv2 = nn.Conv2d(ch1, ch2, 3, padding=1)
        self.conv3 = nn.Conv2d(ch2, ch3, 3, padding=1)
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
        self.conv1 = VirtualConv2d(vpm, 3, ch1, 3, padding=1, slot_id=0)
        self.conv2 = VirtualConv2d(vpm, ch1, ch2, 3, padding=1, slot_id=10)
        self.conv3 = VirtualConv2d(vpm, ch2, ch3, 3, padding=1, slot_id=20)
        self.fc1 = VirtualLinear(vpm, 4 * 4 * ch3, fc_hidden, slot_id=30)
        self.fc2 = VirtualLinear(vpm, fc_hidden, 10, slot_id=40, gain=1.0)  # Output layer

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


class CharModel(nn.Module):
    """Char-level GRU: embedding -> GRU cell (manual, from Linear layers) -> output.

    Uses a manual GRU cell so that virtual-param version is a fair comparison
    (same architecture, same ops, just different weight source).
    """
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.ih = nn.Linear(embed_dim, 3 * hidden_dim)
        self.hh = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        batch, seq_len = x.shape
        embeds = self.embedding(x)
        h = torch.zeros(batch, self.hidden_dim, device=x.device)
        outputs = []
        for t in range(seq_len):
            x_t = embeds[:, t, :]
            gi = self.ih(x_t)
            gh = self.hh(h)
            r_i, z_i, n_i = gi.chunk(3, dim=-1)
            r_h, z_h, n_h = gh.chunk(3, dim=-1)
            r = torch.sigmoid(r_i + r_h)
            z = torch.sigmoid(z_i + z_h)
            n = torch.tanh(n_i + r * n_h)
            h = (1 - z) * n + z * h
            outputs.append(h)
        h_seq = torch.stack(outputs, dim=1)
        return self.output_proj(h_seq)


class VirtualCharModel(nn.Module):
    """Same GRU architecture as CharModel but with virtual params for Linear layers."""
    def __init__(self, vpm, vocab_size, embed_dim=16, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.ih = VirtualLinear(vpm, embed_dim, 3 * hidden_dim, slot_id=0)
        self.hh = VirtualLinear(vpm, hidden_dim, 3 * hidden_dim, slot_id=10)
        self.output_proj = VirtualLinear(vpm, hidden_dim, vocab_size, slot_id=20, gain=1.0)

    def forward(self, x):
        batch, seq_len = x.shape
        embeds = self.embedding(x)
        h = torch.zeros(batch, self.hidden_dim, device=x.device)
        outputs = []
        for t in range(seq_len):
            x_t = embeds[:, t, :]
            gi = self.ih(x_t)
            gh = self.hh(h)
            r_i, z_i, n_i = gi.chunk(3, dim=-1)
            r_h, z_h, n_h = gh.chunk(3, dim=-1)
            r = torch.sigmoid(r_i + r_h)
            z = torch.sigmoid(z_i + z_h)
            n = torch.tanh(n_i + r * n_h)
            h = (1 - z) * n + z * h
            outputs.append(h)
        h_seq = torch.stack(outputs, dim=1)
        return self.output_proj(h_seq)


class CharDataset(Dataset):
    """Character-level dataset with shared vocabulary. Returns (x, y) sequences."""
    def __init__(self, data_tensor, seq_len=32):
        self.data = data_tensor
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


def mlp_hidden_for_budget(budget):
    """Solve h^2 + 796*h + 10 = budget for h (BaselineMLP param count formula).

    BaselineMLP(hidden=h) has:
      fc1: 784*h + h = 785*h
      fc2: h*h + h
      fc3: 10*h + 10
      total: h^2 + 796*h + 10
    """
    # h = (-796 + sqrt(796^2 + 4*(budget - 10))) / 2
    discriminant = 796 ** 2 + 4 * (budget - 10)
    if discriminant < 0:
        return 4
    h = int((-796 + math.sqrt(discriminant)) / 2)
    return max(h, 4)


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp", "data")
    os.makedirs(data_dir, exist_ok=True)

    all_results = {}
    ratios = [4, 10, 50]

    # ======================== MNIST ========================
    print("\n" + "=" * 70)
    print("MNIST EXPERIMENT")
    print("=" * 70)

    mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(data_dir, train=True, download=True, transform=mnist_transform)
    test_data = datasets.MNIST(data_dir, train=False, download=True, transform=mnist_transform)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False, num_workers=0)

    mnist_results = []
    epochs_mnist = 15

    # Baseline full
    set_seed(SEED)
    model = BaselineMLP(hidden=256)
    full_params = count_params(model)
    mnist_results.append(run_classify(model, "Baseline (full)", train_loader, test_loader, device, epochs_mnist))

    for ratio in ratios:
        num_actual = full_params // ratio

        # Virtual sinusoidal
        set_seed(SEED)
        vpm = SinusoidalMap(num_actual=num_actual, num_terms=8)
        model = VirtualMLP(vpm, hidden=256)
        mnist_results.append(run_classify(model, f"Sinusoidal 1/{ratio}", train_loader, test_loader, device, epochs_mnist))

        # Virtual hash_arith
        set_seed(SEED)
        vpm = HashArithMap(num_actual=num_actual)
        model = VirtualMLP(vpm, hidden=256)
        mnist_results.append(run_classify(model, f"HashArith 1/{ratio}", train_loader, test_loader, device, epochs_mnist))

        # Baseline small — matched param count
        set_seed(SEED)
        small_h = mlp_hidden_for_budget(num_actual)
        model = BaselineMLP(hidden=small_h)
        mnist_results.append(run_classify(model, f"Baseline small 1/{ratio}", train_loader, test_loader, device, epochs_mnist))

    all_results["mnist"] = mnist_results

    # ======================== CIFAR-10 ========================
    print("\n" + "=" * 70)
    print("CIFAR-10 EXPERIMENT")
    print("=" * 70)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262))
    ])
    train_data = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)

    cifar_results = []
    epochs_cifar = 15

    set_seed(SEED)
    model = BaselineCNN()
    full_params = count_params(model)
    cifar_results.append(run_classify(model, "Baseline (full)", train_loader, test_loader, device, epochs_cifar))

    for ratio in ratios:
        num_actual = full_params // ratio

        set_seed(SEED)
        vpm = SinusoidalMap(num_actual=num_actual, num_terms=8)
        model = VirtualCNN(vpm)
        cifar_results.append(run_classify(model, f"Sinusoidal 1/{ratio}", train_loader, test_loader, device, epochs_cifar))

        set_seed(SEED)
        vpm = HashArithMap(num_actual=num_actual)
        model = VirtualCNN(vpm)
        cifar_results.append(run_classify(model, f"HashArith 1/{ratio}", train_loader, test_loader, device, epochs_cifar))

        # Baseline small: scale channels to approximately match param count
        # Use binary search for accurate param matching
        best_scale = 1.0 / math.sqrt(ratio)
        for trial_scale in [best_scale * f for f in [0.8, 0.9, 1.0, 1.1, 1.2]]:
            sc1 = max(int(32 * trial_scale), 4)
            sc2 = max(int(64 * trial_scale), 4)
            sc3 = max(int(64 * trial_scale), 4)
            sfc = max(int(256 * trial_scale), 4)
            trial_model = BaselineCNN(ch1=sc1, ch2=sc2, ch3=sc3, fc_hidden=sfc)
            trial_params = count_params(trial_model)
            if abs(trial_params - num_actual) < abs(count_params(BaselineCNN(
                    ch1=max(int(32 * best_scale), 4), ch2=max(int(64 * best_scale), 4),
                    ch3=max(int(64 * best_scale), 4), fc_hidden=max(int(256 * best_scale), 4)
            )) - num_actual):
                best_scale = trial_scale

        sc1 = max(int(32 * best_scale), 4)
        sc2 = max(int(64 * best_scale), 4)
        sc3 = max(int(64 * best_scale), 4)
        sfc = max(int(256 * best_scale), 4)
        set_seed(SEED)
        model = BaselineCNN(ch1=sc1, ch2=sc2, ch3=sc3, fc_hidden=sfc)
        cifar_results.append(run_classify(model, f"Baseline small 1/{ratio}", train_loader, test_loader, device, epochs_cifar))

    all_results["cifar10"] = cifar_results

    # ======================== SEQUENCE ========================
    print("\n" + "=" * 70)
    print("SEQUENCE MODELING EXPERIMENT")
    print("=" * 70)

    # Download tiny shakespeare
    text_path = os.path.join(data_dir, "shakespeare.txt")
    if not os.path.exists(text_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, text_path)
    with open(text_path, "r") as f:
        text = f.read()

    text = text[:100000]

    # Build vocabulary ONCE on the full text, then encode and split
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    data_encoded = torch.tensor([char2idx[c] for c in text], dtype=torch.long)

    split = int(len(data_encoded) * 0.9)
    seq_len = 32
    embed_dim = 16
    hidden_dim = 128  # smaller than sequence.py for CPU speed

    train_dataset = CharDataset(data_encoded[:split], seq_len=seq_len)
    test_dataset = CharDataset(data_encoded[split:], seq_len=seq_len)
    print(f"  Text: {len(text):,} chars, vocab={vocab_size}, seq_len={seq_len}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    seq_results = []
    epochs_seq = 15

    set_seed(SEED)
    model = CharModel(vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)
    full_params = count_params(model)
    embed_param_count = vocab_size * embed_dim
    linear_param_count = full_params - embed_param_count
    print(f"  Full model: {full_params:,} total ({embed_param_count:,} embed + {linear_param_count:,} linear)")

    seq_results.append(run_seq(model, "Baseline (full)", train_loader, test_loader, device, epochs_seq))

    for ratio in ratios:
        # Target: total params roughly = full_params / ratio
        # Virtual model: embed_param_count + num_actual
        # Baseline small: embed_param_count + small_linear
        # So match on: num_actual ≈ (full_params / ratio) - embed_param_count
        target_total = full_params // ratio
        num_actual = max(target_total - embed_param_count, 100)

        # Virtual (sinusoidal)
        set_seed(SEED)
        vpm = SinusoidalMap(num_actual=num_actual, num_terms=8)
        model = VirtualCharModel(vpm, vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)
        seq_results.append(run_seq(model, f"Sinusoidal 1/{ratio}", train_loader, test_loader, device, epochs_seq))

        # Virtual (hash_arith)
        set_seed(SEED)
        vpm = HashArithMap(num_actual=num_actual)
        model = VirtualCharModel(vpm, vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)
        seq_results.append(run_seq(model, f"HashArith 1/{ratio}", train_loader, test_loader, device, epochs_seq))

        # Baseline small: shrink hidden_dim to match target_total params
        # GRU CharModel params = embed + ih + hh + output_proj
        #   embed = vocab * embed_dim
        #   ih = embed_dim * 3h + 3h
        #   hh = h * 3h + 3h
        #   output_proj = h * vocab + vocab
        #   linear = 3h*(embed_dim + h + 2) + vocab*(h + 1)
        #   total = embed + linear
        # Solve: 3h^2 + (3*embed_dim + 6 + vocab)*h + (vocab + embed - target_total) = 0
        a_coeff = 3
        b_coeff = 3 * embed_dim + 6 + vocab_size
        c_coeff = vocab_size + embed_param_count - target_total
        disc = b_coeff ** 2 - 4 * a_coeff * c_coeff
        if disc > 0:
            small_h = int((-b_coeff + math.sqrt(disc)) / (2 * a_coeff))
        else:
            small_h = 8
        small_h = max(small_h, 8)

        set_seed(SEED)
        model = CharModel(vocab_size, embed_dim=embed_dim, hidden_dim=small_h)
        seq_results.append(run_seq(model, f"Baseline small 1/{ratio}", train_loader, test_loader, device, epochs_seq))

    all_results["sequence"] = seq_results

    # ======================== SAVE & SUMMARIZE ========================
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp")
    with open(os.path.join(results_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\n" + "=" * 85)
    print("FULL RESULTS SUMMARY")
    print("=" * 85)

    for task_name, results in all_results.items():
        metric_key = "best_test_acc" if task_name != "sequence" else "best_test_bpc"
        metric_label = "Best Acc" if task_name != "sequence" else "Best BPC"
        print(f"\n--- {task_name.upper()} ---")
        print(f"  {'Model':<35} {'Params':>10} {metric_label:>10} {'Time':>8}")
        print(f"  {'-' * 67}")
        for r in results:
            val = r.get(metric_key, "N/A")
            if isinstance(val, float):
                val = f"{val:.4f}"
            print(f"  {r['name']:<35} {r['num_params']:>10,} {val:>10} {r['training_time_s']:>7.1f}s")

    print(f"\nResults saved to {os.path.join(results_dir, 'all_results.json')}")


if __name__ == "__main__":
    main()
