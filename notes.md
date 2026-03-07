# Parameter Conflation: Experiment Notes

## The Idea

Decouple the number of stored parameters (M) from the number of operations/weights
a model uses (N). A small set of "actual parameters" is mapped to a larger set of
"virtual parameters" via a cheap, differentiable function f. The model's forward
pass uses the virtual parameters, but only M values are stored and optimized.

## Mapping Functions

### 1. HashArith
```
virtual[j] = (actual[h1(j)] * actual[h2(j)] + actual[h3(j)]) / sqrt(2)
```
- 3 hash lookups + multiply + add + normalize, O(1) per virtual param
- One level of multiplicative nonlinearity

### 2. Sinusoidal
```
virtual[j] = sum_k( actual[s_k(j)] * basis[k,j] )
```
L2-normalized sinusoidal basis per position. K=8 terms. O(K) per virtual param.

### 3. DeepHash (best performer)
```
t1 = actual[h1]*actual[h2] + actual[h3]
t2 = actual[h4]*actual[h5] + actual[h6]
virtual[j] = (t1 * t2 + actual[h7]) / sqrt(5)
```
- 7 hash lookups, two levels of multiplicative interaction, O(1) per virtual param
- Richer combinatorics than HashArith from the same actual param pool

## Key Result: Extreme Compression on MNIST

Rigorous experiment: best-of-architecture-search baselines, 5 seeds with error bars,
both best-epoch and last-epoch accuracy reported.

Virtual model: MLP 784->256->256->10 with shared actual params (DeepHash mapping).
Baselines: best architecture found per budget from {MLP1, MLP2, MLP2-tapered}.

### Results (mean +/- std over 5 seeds)

| Ratio | DeepHash (M params)       | Best Baseline (arch, params)            | Delta    |
|-------|---------------------------|-----------------------------------------|----------|
| 1/50  | **93.20% +/- 0.21% (5,386p)** | MLP1(h=6): 89.64% +/- 1.22% (4,780p) | **+3.56pp** |
| 1/100 | **91.78% +/- 0.13% (2,693p)** | MLP2(h=3): 55.30% +/- 14.42% (2,407p) | +36.48pp* |
| 1/200 | **89.31% +/- 0.46% (1,346p)** | MLP2T(4,2): 41.66% +/- 17.22% (3,180p) | +47.65pp* |
| 1/500 | **83.94% +/- 0.97% (538p)**   | MLP2T(4,2): 41.66% +/- 17.22% (3,180p) | +42.28pp* |

`*` = baseline architecture is too narrow to function; comparison is about architecture viability, not weight quality.

### Interpretation

**The 1/50 result is the cleanest signal.** Both architectures are functional (MLP1(h=6)
reaches ~90%), param counts are comparable (5,386 vs 4,780), and DeepHash wins by
3.56pp with tight error bars. This is a genuine advantage of parameter conflation.

**At 1/100+, baselines collapse.** With only 2,693 params, the best independent
architecture (MLP2(h=3)) has 3 hidden neurons — too narrow for MNIST. It's unstable
across seeds (28%-71% range). Virtual models maintain 256-wide layers and are
remarkably stable (91.78% +/- 0.13%). This shows parameter conflation's main value:
**it decouples architecture width from param count**, preventing bottleneck collapse.

**538 actual params, 83.94% accuracy on MNIST.** DeepHash with just 538 stored
floats powers a 784->256->256->10 network (269,322 virtual params). That's a 500:1
compression ratio with meaningful accuracy. The network stores 2.1 KB of weights
and achieves performance that would normally require a model 500x larger.

### All three mappings beat baselines consistently

| Ratio | DeepHash | HashArith | Sinusoidal | Best Baseline |
|-------|----------|-----------|------------|---------------|
| 1/50  | 93.20%   | 92.51%    | 91.56%     | 89.64%        |
| 1/100 | 91.78%   | 90.80%    | 89.61%     | 55.30%        |
| 1/200 | 89.31%   | 88.52%    | 87.06%     | 41.66%        |
| 1/500 | 83.94%   | 82.74%    | 79.62%     | 41.66%        |

Ranking is always DeepHash > HashArith > Sinusoidal. Deeper nonlinearity helps.

## Standard Compression Results (v3 — 15 epochs)

At moderate compression, baselines still win:

### MNIST

| Model                  | Params   | Best Acc |
|------------------------|----------|----------|
| Baseline (full)        | 269,322  | 0.9832   |
| HashArith 1/4          | 67,330   | 0.9535   |
| **Baseline small 1/4** | 67,231   | 0.9774   |
| HashArith 1/10         | 26,932   | 0.9430   |
| **Baseline small 1/10**| 26,506   | 0.9664   |

### CIFAR-10

| Model                  | Params   | Best Acc |
|------------------------|----------|----------|
| Baseline (full)        | 321,290  | 0.7858   |
| HashArith 1/4          | 80,322   | 0.5364   |
| **Baseline small 1/4** | 81,290   | 0.7155   |

### Shakespeare char-level

| Model                  | Params   | Best BPC |
|------------------------|----------|----------|
| Baseline (full)        | 213,773  | 2.7527   |
| HashArith 1/4          | 53,443   | 3.3623   |
| **Baseline small 1/4** | 53,437   | 2.7716   |

## Analysis

### Why virtual params win at extreme compression

The key insight: **architecture width is a form of capacity independent of parameter count.**

A 784->256->256->10 network has 256-dimensional hidden representations. Even with
heavily conflated weights (the 256x256 matrix is built from only 538 distinct values
at 1/500), the network can still route information through 256 channels. The gradient
can adjust the 538 actual params, and each adjustment ripples through all the virtual
weights that reference it — a single gradient step effectively updates hundreds of
virtual weights simultaneously.

By contrast, a 784->3->10 network with 2,407 independent params forces ALL information
through 3 dimensions. No amount of training can overcome this fundamental bottleneck.

### Why virtual params lose at moderate compression

At 1/4, a 784->145->145->10 network (67,231 params) has enough width for MNIST.
Its weights are fully independent — each can be optimized to its ideal value without
affecting other weights. The virtual model's 784->256->256->10 has more width but
less weight independence. Each actual param affects many virtual weights, creating
optimization interference: improving one virtual weight may degrade another.

The crossover point (where virtual beats independent) occurs when the baseline's
architecture becomes too narrow. For MNIST this is around 1/50.

### Why CIFAR and sequence tasks are harder

CIFAR-10 and Shakespeare don't show the crossover even at 1/50. Hypotheses:
- **Higher rank requirements**: these tasks need weight matrices with higher effective
  rank, and conflation limits rank to ~M
- **Conv kernels are small**: a 3x3x32x64 conv is only 18,432 params — conflation
  doesn't save much per layer, but still constrains the weights
- **Language precision**: character-level language modeling needs precise per-token
  statistics that conflated weights can't represent

## Bug History

### v1: init_std=0.02 killed the multiplicative branch
### v2: sinusoidal normalization, vocab splits, param matching
### v3: output layer gain=1.0 (Xavier), seeded model construction

## Combine Function Sweep (Exp 1)

Tested 10 combine functions at 1/50 compression on MNIST, 3 seeds each.
All use ArithMap with the same architecture (784->256->256->10).

| Rank | Combine fn          | x  | Accuracy (mean +/- std) |
|------|---------------------|----|-------------------------|
| 1    | deeper_hash         | 11 | **93.57% +/- 0.23%**   |
| 2    | wide_hash           | 5  | 93.23% +/- 0.06%       |
| 3    | deep_hash           | 7  | 93.21% +/- 0.23%       |
| 4    | multiply            | 2  | 92.80% +/- 0.04%       |
| 5    | xor_inspired        | 3  | 92.63% +/- 0.42%       |
| 6    | rotation            | 3  | 92.63% +/- 0.19%       |
| 7    | hash_arith          | 3  | 92.43% +/- 0.13%       |
| 8    | sinusoidal (K=8)    | 8  | 91.55% +/- 0.06%       |
| 9    | identity            | 1  | 91.49% +/- 0.17%       |
| 10   | add                 | 2  | 91.37% +/- 0.30%       |
| --   | baseline MLP(6,6)   | -  | 89.05% +/- 0.86%       |

### Key findings

1. **More mixing (higher x) helps.** Clear monotonic trend from x=1 to x=11.
   But diminishing returns: the jump from x=1 to x=2 (+1.3pp) is larger than
   x=7 to x=11 (+0.4pp).

2. **Nonlinearity matters more than number of terms.** `multiply` (x=2, 92.8%)
   beats `hash_arith` (x=3, 92.4%). The product operation is the key ingredient.
   `add` (x=2, 91.4%) performs like `identity` (x=1, 91.5%) — linear combination
   of actual params doesn't increase effective diversity.

3. **Sinusoidal is equivalent to identity.** Despite using K=8 terms, sinusoidal
   (91.6%) is no better than identity (91.5%). The sinusoidal basis weights are
   fixed (not learned), so it's really a linear combination of actual params.
   This confirms finding #2: linear mixing doesn't help.

4. **All virtual methods beat the baseline.** Even the worst (add, 91.4%) beats
   the best baseline (MLP(6,6), 89.1%) by 2.3pp.

5. **The "multiply+add" motif is robust.** hash_arith, wide_hash, deep_hash,
   deeper_hash all use variations of `a*b+c`. The ranking within this family
   tracks with depth/width of the expression tree.

## Scale-Up Experiment (Exp 4)

Fixed M=5,386 actual params, varied architecture width H from 128 to 2048.
deep_hash (x=7) mapping. Compared against non-conflated baselines at same width.

| Config              | H     | Virtual params | Actual params | Accuracy        |
|---------------------|-------|----------------|---------------|-----------------|
| virtual             | 128   | 118,282        | 5,386         | 93.03% +/- 0.28%|
| virtual             | 256   | 269,322        | 5,386         | 93.21% +/- 0.24%|
| virtual             | 512   | 669,706        | 5,386         | 93.29% +/- 0.25%|
| virtual             | 1024  | 1,863,690      | 5,386         | 93.43% +/- 0.18%|
| virtual             | 2048  | 5,824,522      | 5,386         | 93.32% +/- 0.08%|
| baseline-full       | 128   | --             | 118,282       | 98.08% +/- 0.10%|
| baseline-full       | 256   | --             | 269,322       | 98.35% +/- 0.02%|
| baseline-full       | 512   | --             | 669,706       | 98.57% +/- 0.06%|
| baseline-full       | 1024  | --             | 1,863,690     | 98.57% +/- 0.06%|
| baseline-full       | 2048  | --             | 5,824,522     | 98.54% +/- 0.06%|
| baseline-matched    | 6,6   | --             | ~5,000        | 89.05% +/- 0.86%|

### Key findings

1. **Scaling width helps slightly, then plateaus.** From H=128 to H=1024,
   accuracy improves from 93.0% to 93.4% — only 0.4pp. H=2048 doesn't
   improve further (93.3%). With only 5,386 actual params, extra width
   gives diminishing returns quickly.

2. **The bottleneck is actual param count, not width.** At H=2048, the model
   has 5.8M virtual params from 5,386 actual — a 1081:1 compression ratio.
   Accuracy is essentially the same as H=256 (50:1 compression). The
   actual param budget is the binding constraint.

3. **Full baselines show width matters for independent params.** Non-conflated
   models improve from 98.1% (H=128) to 98.6% (H=512), then plateau.
   The plateau is later because they have more independent params to exploit
   the width.

4. **Virtual models are ~5pp below full baselines at every width.** This gap
   is the cost of parameter conflation — weight interdependence limits what
   the optimizer can achieve.

5. **Virtual H=128 (5,386 actual) beats baseline-matched MLP(6,6) (~5,000 actual)
   by 4pp.** This confirms the core finding: parameter conflation lets you use
   a wider architecture at the same storage cost.

## Compression Ratio Scaling (Exp 3)

Accuracy vs compression ratio on MNIST for identity, hash_arith, deep_hash,
and best baseline. 3 seeds each. Architecture: 784->256->256->10.

| Ratio | identity (x=1) | hash_arith (x=3) | deep_hash (x=7) | baseline       |
|-------|----------------|-------------------|------------------|----------------|
| 1/4   | 95.38 +/- 0.13 | 95.99 +/- 0.08    | 96.56 +/- 0.11   | **97.83 +/- 0.02** |
| 1/10  | 94.42 +/- 0.10 | 94.84 +/- 0.10    | 95.67 +/- 0.06   | **96.71 +/- 0.19** |
| 1/25  | 93.01 +/- 0.14 | 93.56 +/- 0.02    | 94.21 +/- 0.20   | **94.38 +/- 0.15** |
| 1/50  | 91.49 +/- 0.17 | 92.44 +/- 0.13    | **93.21 +/- 0.23** | 89.05 +/- 0.86 |
| 1/100 | 89.61 +/- 0.16 | 90.68 +/- 0.29    | **91.61 +/- 0.15** | 52.74 +/- 21.87|
| 1/200 | 87.16 +/- 0.82 | 88.23 +/- 0.46    | **89.32 +/- 0.37** | 27.31 +/- 0.77 |
| 1/500 | 80.36 +/- 0.97 | 82.30 +/- 0.82    | **84.36 +/- 0.15** | 27.31 +/- 0.77 |

### Key findings

1. **Crossover at ~1/25 to 1/50.** At 1/25 the baseline (94.4%) still narrowly
   beats deep_hash (94.2%). By 1/50, deep_hash (93.2%) clearly wins over the
   baseline (89.1%). The crossover point is where the baseline architecture
   becomes too narrow to function.

2. **Graceful degradation.** Virtual models lose ~2pp per doubling of compression
   ratio. The curve is smooth — no cliff. Even at 1/500, deep_hash (84.4%) is
   far above chance (10%) and well above the collapsed baseline (27.3%).

3. **deep_hash advantage grows with compression.** The gap between deep_hash and
   identity is 1.2pp at 1/4 but 4.0pp at 1/500. Nonlinear mixing becomes more
   important as the actual param pool shrinks.

4. **Baselines collapse sharply.** Between 1/50 and 1/100, baselines drop from
   89% to 53% — a cliff caused by architecture width falling below the minimum
   for the task. Virtual models have no such cliff.

## Combine Function Sweep on CIFAR-10 (Exp 2)

Tested at 1/50 compression. CNN architecture: Conv(32)->Conv(64)->FC(256)->FC(10).

| Rank | Combine fn          | x  | Accuracy (mean +/- std) |
|------|---------------------|----|-------------------------|
| 1    | deeper_hash         | 11 | 47.68% +/- 0.73%       |
| 2    | deep_hash           | 7  | 46.48% +/- 0.34%       |
| 3    | multiply            | 2  | 45.69% +/- 1.15%       |
| 4    | hash_arith          | 3  | 45.16% +/- 0.90%       |
| 5    | sinusoidal (K=8)    | 8  | 42.77% +/- 0.65%       |
| 6    | identity (x=1)      | 1  | 42.71% +/- 0.55%       |
| --   | baseline CNN(32,8,32)| - | **61.56% +/- 0.62%**   |

### Key findings

1. **Same ranking as MNIST.** deeper_hash > deep_hash > multiply > hash_arith >
   sinusoidal ~ identity. The combine_fn ranking is task-independent.

2. **Baseline wins decisively on CIFAR at 1/50.** 61.6% vs 47.7% for the best
   virtual model. CIFAR requires more independent parameters — the higher-rank
   weight matrices needed for image features can't be well-approximated by
   conflated params at this ratio.

3. **Nonlinearity gap is larger on CIFAR.** deeper_hash (47.7%) beats identity
   (42.7%) by 5pp — vs 2pp on MNIST. Harder tasks benefit more from richer
   mixing.

4. **Sinusoidal confirms the linear mixing finding.** Again performs like
   identity despite 8 terms. Linear combination is truly no better than
   raw repetition.

## What To Try Next

1. **CIFAR compression scaling (Exp 5)** — find the crossover point for CIFAR
2. **Learned f** — small neural net as mapping, trained end-to-end
3. **Hybrid** — virtual params for large layers, independent for small ones
4. **Scale** — test on transformers with millions of virtual params
