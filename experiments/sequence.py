"""
Sequence modeling experiment: character-level next-char prediction.

Uses a simple single-layer LSTM-style setup built from Linear layers,
so we can swap in VirtualLinear layers cleanly.

We'll generate training data from a deterministic source (repeated Shakespeare-like
text) to keep things simple and reproducible without external downloads.

Models:
  1. Baseline (full): Standard char-level model
  2. Virtual (sinusoidal): Same architecture, virtual params
  3. Virtual (hash_arith): Same architecture, virtual params
  4. Baseline (small): Smaller model with matched param count
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import json
import math
import urllib.request

from virtual_params import VirtualLinear, SinusoidalMap, HashArithMap


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- Dataset ---

def get_text_data(data_dir):
    """Download a small text corpus (Shakespeare)."""
    filepath = os.path.join(data_dir, "shakespeare.txt")
    if not os.path.exists(filepath):
        os.makedirs(data_dir, exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, filepath)
    with open(filepath, "r") as f:
        return f.read()


class CharDataset(Dataset):
    def __init__(self, data_tensor, seq_len=64):
        """Character-level dataset from pre-encoded tensor.

        Args:
            data_tensor: LongTensor of character indices (encoded with a shared vocabulary)
            seq_len: context window length
        """
        self.seq_len = seq_len
        self.data = data_tensor

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


# --- Models ---

class CharModel(nn.Module):
    """
    Simple character-level language model.
    Embedding -> 2-layer GRU-like (using Linear) -> output projection.

    We use a manual GRU-like cell built from Linear layers so that the
    virtual-param version is a fair comparison.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # GRU-like gates: input + hidden -> 3*hidden (reset, update, candidate)
        self.ih = nn.Linear(embed_dim, 3 * hidden_dim)
        self.hh = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: [batch, seq_len]
        batch, seq_len = x.shape
        embeds = self.embedding(x)  # [batch, seq_len, embed_dim]

        h = torch.zeros(batch, self.hidden_dim, device=x.device)
        outputs = []

        for t in range(seq_len):
            x_t = embeds[:, t, :]
            gates_i = self.ih(x_t)
            gates_h = self.hh(h)

            r_i, z_i, n_i = gates_i.chunk(3, dim=-1)
            r_h, z_h, n_h = gates_h.chunk(3, dim=-1)

            r = torch.sigmoid(r_i + r_h)
            z = torch.sigmoid(z_i + z_h)
            n = torch.tanh(n_i + r * n_h)
            h = (1 - z) * n + z * h
            outputs.append(h)

        h_seq = torch.stack(outputs, dim=1)  # [batch, seq_len, hidden]
        logits = self.output_proj(h_seq)  # [batch, seq_len, vocab]
        return logits


class VirtualCharModel(nn.Module):
    """Same architecture as CharModel but with virtual parameters for Linear layers."""
    def __init__(self, vpm, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.vpm = vpm
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Embedding keeps its own params (relatively small, and discrete lookup
        # doesn't work well with continuous virtual params)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.ih = VirtualLinear(vpm, embed_dim, 3 * hidden_dim, slot_id=0)
        self.hh = VirtualLinear(vpm, hidden_dim, 3 * hidden_dim, slot_id=100)
        self.output_proj = VirtualLinear(vpm, hidden_dim, vocab_size, slot_id=200, gain=1.0)

    def forward(self, x):
        batch, seq_len = x.shape
        embeds = self.embedding(x)

        h = torch.zeros(batch, self.hidden_dim, device=x.device)
        outputs = []

        for t in range(seq_len):
            x_t = embeds[:, t, :]
            gates_i = self.ih(x_t)
            gates_h = self.hh(h)

            r_i, z_i, n_i = gates_i.chunk(3, dim=-1)
            r_h, z_h, n_h = gates_h.chunk(3, dim=-1)

            r = torch.sigmoid(r_i + r_h)
            z = torch.sigmoid(z_i + z_h)
            n = torch.tanh(n_i + r * n_h)
            h = (1 - z) * n + z * h
            outputs.append(h)

        h_seq = torch.stack(outputs, dim=1)
        logits = self.output_proj(h_seq)
        return logits


# --- Training ---

def train_epoch(model, loader, optimizer, device):
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


def evaluate(model, loader, device):
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

    best_test_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_loss = evaluate(model, test_loader, device)
        scheduler.step()

        best_test_loss = min(best_test_loss, test_loss)

        # bits per character
        train_bpc = train_loss / math.log(2)
        test_bpc = test_loss / math.log(2)

        epoch_data = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "test_loss": round(test_loss, 4),
            "train_bpc": round(train_bpc, 4),
            "test_bpc": round(test_bpc, 4),
        }
        results["epochs"].append(epoch_data)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: train_bpc={train_bpc:.4f} test_bpc={test_bpc:.4f}")

    elapsed = time.time() - start_time
    best_bpc = best_test_loss / math.log(2)
    results["best_test_loss"] = round(best_test_loss, 4)
    results["best_test_bpc"] = round(best_bpc, 4)
    results["training_time_s"] = round(elapsed, 1)

    print(f"  Best test BPC: {best_bpc:.4f}")
    print(f"  Training time: {elapsed:.1f}s")

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp", "data")
    text = get_text_data(data_dir)
    print(f"Text length: {len(text):,} chars")

    # Build shared vocabulary from full text, then encode and split
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    data_encoded = torch.tensor([char2idx[c] for c in text], dtype=torch.long)

    split = int(len(data_encoded) * 0.9)
    seq_len = 64
    train_dataset = CharDataset(data_encoded[:split], seq_len=seq_len)
    test_dataset = CharDataset(data_encoded[split:], seq_len=seq_len)

    print(f"Vocab size: {vocab_size}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    embed_dim = 64
    hidden_dim = 128
    epochs = 20
    lr = 1e-3

    all_results = []

    # 1. Baseline (full)
    model = CharModel(vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)
    full_params = count_params(model)
    results = run_experiment(model, "Baseline (full)", train_loader, test_loader, device, epochs, lr)
    all_results.append(results)

    # Virtual param experiments
    # Only count the Linear layer params for compression target
    # (embedding params stay separate)
    embed_params = vocab_size * embed_dim
    linear_params = full_params - embed_params
    print(f"\nFull model: {full_params:,} total, {embed_params:,} embedding, {linear_params:,} linear")

    for ratio in [4, 10, 20]:
        num_actual = linear_params // ratio

        # Virtual (sinusoidal)
        vpm_sin = SinusoidalMap(num_actual=num_actual, num_terms=8)
        model = VirtualCharModel(vpm_sin, vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)
        results = run_experiment(
            model, f"Virtual sinusoidal (1/{ratio}, M={num_actual:,})",
            train_loader, test_loader, device, epochs, lr
        )
        all_results.append(results)

        # Virtual (hash_arith)
        vpm_hash = HashArithMap(num_actual=num_actual)
        model = VirtualCharModel(vpm_hash, vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)
        results = run_experiment(
            model, f"Virtual hash_arith (1/{ratio}, M={num_actual:,})",
            train_loader, test_loader, device, epochs, lr
        )
        all_results.append(results)

        # Baseline small
        scale = 1.0 / math.sqrt(ratio)
        small_hidden = max(int(hidden_dim * scale), 8)
        model = CharModel(vocab_size, embed_dim=embed_dim, hidden_dim=small_hidden)
        results = run_experiment(
            model, f"Baseline small (h={small_hidden}, ~{count_params(model):,} params)",
            train_loader, test_loader, device, epochs, lr
        )
        all_results.append(results)

    # Save results
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tmp", "sequence_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"SEQUENCE MODELING SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<50} {'Params':>10} {'Best BPC':>10} {'Time':>8}")
    print(f"{'-'*80}")
    for r in all_results:
        print(f"{r['name']:<50} {r['num_params']:>10,} {r['best_test_bpc']:>10.4f} {r['training_time_s']:>7.1f}s")


if __name__ == "__main__":
    main()
