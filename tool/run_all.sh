#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Parameter Conflation Experiments ==="
echo "Running from: $ROOT"
echo ""

cd "$ROOT"

echo ">>> MNIST Experiment"
nix-shell shell.nix --run "python3 experiments/mnist.py"
echo ""

echo ">>> CIFAR-10 Experiment"
nix-shell shell.nix --run "python3 experiments/cifar10.py"
echo ""

echo ">>> Sequence Modeling Experiment"
nix-shell shell.nix --run "python3 experiments/sequence.py"
echo ""

echo "=== All experiments complete ==="
echo "Results in tmp/*_results.json"
