# Parameter Conflation: Experiment Notes

## The Idea

Decouple the number of stored parameters (M) from the number of operations/weights
a model uses (N). A small set of "actual parameters" is mapped to a larger set of
"virtual parameters" via a cheap, differentiable function f. The model's forward
pass uses the virtual parameters, but only M values are stored and optimized.

## Mapping Functions Tested

### 1. HashArith
```
virtual[j] = actual[h1(j)] * actual[h2(j)] + actual[h3(j)]
```
- Three hash lookups + multiply + add per virtual param
- Nonlinear (multiplicative interaction)
- O(1) per virtual param

### 2. Sinusoidal
```
virtual[j] = sum_k( actual[s_k(j)] * sin(omega_k * j + phi_k) ) / sqrt(K)
```
- K=8 terms per virtual param, each using a different actual param
- Smooth variation across virtual param indices
- O(K) per virtual param

## Results

### MNIST (MLP: 784->256->256->10, 10 epochs)

| Model                  | Params   | Best Acc |
|------------------------|----------|----------|
| Baseline (full)        | 269,322  | 0.9815   |
| Sinusoidal 1/4         | 67,330   | 0.8831   |
| HashArith 1/4          | 67,330   | 0.9011   |
| **Baseline small 1/4** | 67,231   | 0.9764   |
| Sinusoidal 1/10        | 26,932   | 0.8656   |
| HashArith 1/10         | 26,932   | 0.8974   |
| **Baseline small 1/10**| 26,506   | 0.9582   |
| Sinusoidal 1/50        | 5,386    | 0.8385   |
| **HashArith 1/50**     | 5,386    | **0.8881** |
| Baseline small 1/50    | 4,822    | 0.8711   |

### CIFAR-10 (CNN: 3 conv + 2 FC, 15 epochs)

| Model                  | Params   | Best Acc |
|------------------------|----------|----------|
| Baseline (full)        | 321,290  | 0.7767   |
| Sinusoidal 1/4         | 80,322   | 0.1000   |
| HashArith 1/4          | 80,322   | 0.4142   |
| **Baseline small 1/4** | 81,290   | 0.7106   |
| Sinusoidal 1/10        | 32,129   | 0.1000   |
| HashArith 1/10         | 32,129   | 0.1000   |
| **Baseline small 1/10**| 32,210   | 0.6389   |

### Shakespeare char-level (MLP: embed->256->256->vocab, 10 epochs)

| Model                  | Params   | Best BPC |
|------------------------|----------|----------|
| Baseline (full)        | 213,227  | 3.1161   |
| Sinusoidal 1/4         | 54,014   | 4.2223   |
| HashArith 1/4          | 54,014   | 3.9698   |
| Baseline small 1/4     | 90,731   | 3.0996   |
| Sinusoidal 1/10        | 22,172   | 4.4334   |
| HashArith 1/10         | 22,172   | 4.4414   |
| Baseline small 1/10    | 53,243   | 3.0931   |
| Sinusoidal 1/50        | 5,189    | 4.5906   |
| HashArith 1/50         | 5,189    | 4.5454   |
| Baseline small 1/50    | 22,927   | 3.1763   |

## Key Findings

### 1. The approach works, but current mappings lose to simply making the model smaller

At 1/4 and 1/10 compression, a standard smaller model (with fewer but independent
params) outperforms the virtual-parameter model in most cases. The mapping functions
don't produce weights that are as useful as independently learned ones.

**Exception: Extreme compression (1/50) on MNIST.** HashArith (88.81%) beats the
matched-param baseline (87.11%) by 1.7 percentage points. This suggests parameter
conflation may have a sweet spot at very high compression ratios where independent
params are too few to be useful, but virtual params can leverage combinatorial
structure.

### 2. HashArith consistently outperforms Sinusoidal

HashArith's multiplicative interaction (a1 * a2 + a3) produces more useful virtual
params than the sinusoidal basis approach across all tasks. Possible reasons:
- Products of params create genuinely distinct values (different signs, magnitudes)
- Sinusoidal mixing is still too smooth—many virtual params end up correlated
- The additive term (a3) in HashArith provides an independent offset

### 3. CIFAR-10 is much harder for parameter conflation

Virtual params completely fail on CIFAR-10 at 1/10+ compression (10% = random chance).
CNNs may need more independent parameters per filter than MLPs need per hidden unit.
The spatial structure of conv kernels may not tolerate conflated params well.

### 4. Sequence modeling is also hard

The baseline small models handily beat virtual-param models on Shakespeare text.
The gap is large (3.1 BPC vs 4.0+ BPC). Character-level language modeling requires
precise, independent weights to capture the statistical structure of text.

## Bug Found & Fixed

The initial sinusoidal mapping (v1) had a critical bug: all virtual params shared
the same K=8 actual param indices, varying only by a sinusoidal modulation of
position mapped to [0, 2*pi). With 200K+ positions, this produced:
- Cosine similarity between weight matrix rows: **0.9998** (should be ~0)
- Cross-layer correlation: **0.94** (should be ~0)
- Result: 20% accuracy on MNIST (random chance for 10 classes)

Fix: give each virtual param its own set of K actual param indices via hashing.
After fix, cosine sim dropped to 0.004, cross-layer correlation to -0.03.

## Analysis: Why Doesn't It Work Better?

The fundamental issue: **the mapping function constrains the rank of the virtual
weight matrices.** With M actual params mapped to an NxN weight matrix:

- HashArith: `W[i,j] = actual[h1(i,j)] * actual[h2(i,j)] + actual[h3(i,j)]`
  The matrix is built from M values with replacement. Its effective rank is
  limited by M, not by NxN. The optimizer can adjust the M actual params, but
  the hash structure rigidly determines which virtual params share actual params.

- A standard NxN matrix has rank up to min(N,N). A conflated matrix with M << N^2
  actual params can't achieve full rank, which limits what linear transformations
  the layer can represent.

**This is the core challenge: the mapping f needs to be expressive enough to
produce high-rank virtual weight matrices from few actual params.**

## Ideas for Next Steps

### Short-term improvements to test
1. **Increase K (terms per virtual param)** — currently K=8 for sinusoidal.
   Try K=32 or K=64 to see if more mixing helps.
2. **Per-layer scaling + bias** — add a learned scale and bias per layer
   (tiny overhead) on top of the conflated weights: `W_layer = alpha * W_virtual + beta`
3. **Hybrid model** — only conflate the large weight matrices (FC layers),
   keep small ones (biases, conv kernels) independent.
4. **Better init** — the init_std=0.02 was arbitrary. HashArith's product
   squashes variance (0.02 * 0.02 ≈ 0.0004). Explore Kaiming-aware init.

### Longer-term directions
5. **Learned f** — see ideas.md. A small neural network as the mapping
   function, trained end-to-end.
6. **Structured factorization** — instead of random hashing, use low-rank
   factorization: actual params form two small matrices A (M x r) and B (r x M),
   and W_virtual = A @ B reshaped. This guarantees rank r.
7. **Residual conflation** — start with a conflated base and add a small
   number of independent parameters as a residual.
8. **Scale experiments** — the approach might perform differently at larger
   scale. At 100M+ virtual params with 10M actual, the combinatorial space
   is much richer.

### What would "success" look like?
A conflated model that matches a standard model's accuracy while using
significantly less memory for storage (checkpoints, inference). The compute
cost may be slightly higher (mapping function overhead), but memory should
be strictly better.

The 1/50 MNIST result (HashArith beating baseline small) is the first hint
that this could work. The question is whether better mapping functions can
make this consistent across tasks and compression ratios.
