# Virtual Parameters: Ideas for Later

## Learned Mapping Function f

Instead of using a fixed/structured mapping from actual params to virtual params,
make `f` itself a small learned neural network.

### Concept
- Actual parameters: a tensor of M values
- Virtual parameter j: `f_theta(j, actual_params)` where f_theta is a small MLP
- f_theta takes a virtual param index (or positional encoding of it) and outputs the
  virtual param value by attending to / combining actual params
- f_theta's own parameters (theta) are additional overhead, but could be very small

### Why this is interesting
- The mapping itself learns what combinations of actual params are useful
- Could discover non-obvious parameter reuse patterns
- More expressive than fixed hash-based or polynomial mappings
- Meta-learning flavor: the network learns how to construct its own weights

### Potential issues
- Computing f for every virtual param on every forward pass could be expensive
- Gradient flow through f adds complexity
- Might be hard to train (chicken-and-egg: f needs good actual params, actual params
  need good f)
- Could explore amortized approaches: compute f once per batch, cache virtual params

### Variants
1. **Tiny MLP**: f(index_embedding, actual_params) -> virtual_param_value
2. **Attention-based**: virtual param "queries" attend to actual params as "keys/values"
3. **Implicit neural representation**: treat the weight tensor as a continuous function,
   parameterized by a small network (related to neural implicit representations / NeRF
   for weights)

### Related work to check
- Neural Implicit Representations (NeRF, SIREN) applied to weight spaces
- Implicit Neural Representations for weight generation
- "Multiplicative interactions" papers
