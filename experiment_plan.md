# Experiment Plan

## Questions to Answer

1. **Which combine_fn is best?** How does the choice of f affect accuracy?
   Does more mixing (higher x) always help?
2. **Training dynamics**: Do virtual models converge differently than baselines?
   Faster/slower? Do they plateau or keep improving?
3. **Dataset dependence**: Does the combine_fn ranking hold across MNIST and CIFAR?
4. **Compression scaling**: How does accuracy degrade as compression increases?
   Is the curve smooth or is there a cliff?
5. **Scaling UP**: What happens when virtual params >> what a normal model would
   use? E.g. a 784->1024->1024->10 MLP backed by M actual params. Does extra
   width help even though the actual param count is fixed?

## Combine Functions Under Test

| Name             | x  | f(a)                                        | norm    |
|------------------|----|---------------------------------------------|---------|
| identity         | 1  | a[0]                                        | 1       |
| add              | 2  | a[0]+a[1]                                   | sqrt(2) |
| multiply         | 2  | a[0]*a[1]                                   | 1       |
| hash_arith       | 3  | a[0]*a[1]+a[2]                              | sqrt(2) |
| xor_inspired     | 3  | a[0]*a[1]^2+a[2]                            | 2       |
| rotation         | 3  | a[0]*cos(a[2])-a[1]*sin(a[2])              | 1       |
| wide_hash        | 5  | (a[0]*a[1]+a[2])*(a[3]+a[4])               | 2       |
| deep_hash        | 7  | (a[0]*a[1]+a[2])*(a[3]*a[4]+a[5])+a[6]    | sqrt(5) |
| deeper_hash      | 11 | 3-level version of deep_hash                | sqrt(11)|
| sinusoidal (K=8) | 8  | weighted sinusoidal basis (separate class)  | auto    |

## Experiments

All experiments log per-epoch train loss and test accuracy.
Results saved incrementally (after each run completes).
3 seeds per config (42, 137, 256) — enough for error bars, practical on CPU.

### Experiment 1: Combine Function Sweep (MNIST)

**Goal**: Which f is best? Does x matter?

- Dataset: MNIST
- Architecture: 784->256->256->10
- Compression: 1/50 (budget ~5,386 actual params)
- Epochs: 20
- Configs: all 10 combine functions + baseline MLP1(h=6)

Estimated time: ~10 configs x 3 seeds x ~3 min = ~90 min

### Experiment 2: Combine Function Sweep (CIFAR-10)

**Goal**: Does the ranking hold on a harder task?

- Dataset: CIFAR-10
- Architecture: Conv(32)->Conv(64)->FC(256)->FC(10) (same as existing)
- Compression: 1/50
- Epochs: 20
- Configs: top 4 from Exp 1 + identity + baseline

Estimated time: ~6 configs x 3 seeds x ~5 min = ~90 min

### Experiment 3: Compression Ratio Scaling

**Goal**: How does accuracy vs compression curve look per combine_fn?

- Dataset: MNIST
- Architecture: 784->256->256->10
- Compression: 1/4, 1/10, 1/25, 1/50, 1/100, 1/200, 1/500
- Epochs: 20
- Configs: top 3 combine fns from Exp 1 + identity + best baseline per budget
- Seeds: 3

Estimated time: ~5 configs x 7 ratios x 3 seeds x ~3 min = ~315 min (~5 hrs)

### Experiment 4: Scaling UP (virtual params > normal)

**Goal**: What happens when we make the virtual model BIGGER than a standard
model would normally be?

Instead of compressing (N virtual from M < N actual), we test:
- Fixed actual param budget M
- Vary architecture width: hidden = 256, 512, 1024, 2048
- Compare against non-conflated baselines with same width but all params independent

This answers: does extra width help even when actual params are fixed?
And the baseline comparison answers: how much of the benefit is width vs
param independence?

- Dataset: MNIST
- Compression: fixed M = 5,386 (same as 1/50 of the 256-wide model)
- Architectures:
  - Virtual: 784->H->H->10, H in {128, 256, 512, 1024, 2048}
  - Baseline-full: same arch with all params independent (for H=128,256 only —
    larger would be too many params to be meaningful as a "baseline")
  - Baseline-matched: best architecture with ~5,386 independent params
- Combine fn: best from Exp 1 (likely deep_hash)
- Epochs: 20
- Seeds: 3

Estimated time: ~8 configs x 3 seeds x ~5 min = ~120 min

### Experiment 5: Compression Ratio Scaling (CIFAR-10)

Same as Exp 3 but on CIFAR. Only run if CPU time permits.

- Dataset: CIFAR-10
- Compression: 1/4, 1/10, 1/25, 1/50, 1/100
- Configs: top 2 combine fns + baseline
- Seeds: 3

Estimated time: ~3 configs x 5 ratios x 3 seeds x ~5 min = ~225 min (~4 hrs)

## Output Format

Each experiment produces a JSON file in tmp/ with structure:
```json
{
  "experiment": "combine_fn_sweep_mnist",
  "configs": [...],
  "results": [
    {
      "config_name": "deep_hash (x=7)",
      "seed": 42,
      "num_params": 5386,
      "epochs": [
        {"epoch": 1, "train_loss": 0.45, "test_acc": 0.87},
        ...
      ],
      "best_test_acc": 0.93,
      "last_test_acc": 0.92
    },
    ...
  ]
}
```

## Priority Order

1. Exp 1 (combine fn sweep, MNIST) — answers the core x question
2. Exp 4 (scaling up) — most novel question
3. Exp 3 (compression ratio scaling) — fills in the curve
4. Exp 2 (combine fn sweep, CIFAR) — generalization check
5. Exp 5 (compression scaling, CIFAR) — nice to have

## Implementation Notes

- Single unified runner script: experiments/training_curves.py
- Incremental saves: write results after each (config, seed) completes
- Per-epoch logging: record train_loss and test_acc every epoch
- Resume support: skip (config, seed) pairs already in the output file
- Shared data loading: load dataset once, reuse across configs
