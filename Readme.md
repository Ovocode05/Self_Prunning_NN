# Self-Pruning Neural Network

**Author:** Krrish Punj  
**Dataset:** CIFAR-10

---

## 1. Why L1 Penalty on Sigmoid Gates Encourages Sparsity

Each weight in a `PrunableLinear` layer is scaled by a gate value `g = sigmoid(s) ∈ (0, 1)`, where `s` is a learnable scalar. The total loss is:

```
Total Loss = CrossEntropy(y_pred, y_true) + λ · (Σ gᵢ / N)
```

The regularisation term is the normalised L1 norm of all gate values. Two properties make L1 the right choice:

**Constant gradient pressure.** The gradient of `|g|` with respect to `g` is ±1, regardless of how small `g` already is. The push toward zero never weakens as a gate approaches zero — unlike L2 whose gradient `2λg → 0` near zero, causing values to hover just above zero rather than reaching it exactly.

**Geometry of L1 minimisation.** The L1 ball has corners on the coordinate axes. The optimiser is geometrically attracted to those corners, which correspond to exactly-zero gate values. This is why L1 produces true zeros while L2 only produces small values.

**λ as a control knob.** A higher λ makes the sparsity term dominate, pruning more connections at the cost of accuracy. This tradeoff is the central result of this experiment.

---

## 2. Results

| Lambda (λ) | Test Accuracy | Sparsity Level (%) |
| :--------: | :-----------: | :----------------: |
|  0.00002   |     53.6%     |       12.0%        |
|  0.00010   |     54.1%     |       58.5%        |
|  0.00020   |     52.9%     |       77.2%        |
|  0.00500   |     42.0%     |       99.7%        |

**Key observation:** Sparsity increases from 12% to 77% with only a 1.2% drop in accuracy. The network retains most of its predictive capacity even with three quarters of its connections pruned — confirming that a large fraction of weights in a flat MLP on CIFAR-10 are redundant. Accuracy degrades meaningfully only at λ=0.005, where 99.7% of gates are suppressed and the model is effectively reduced to a handful of surviving connections.

---

## 3. Gate Value Distribution — Best Model (λ = 0.00010)

The best model is selected by highest test accuracy (54.1% at λ=0.00010), which also gives the most interpretable gate distribution at 58.5% sparsity.

![Gate Value Distribution]('/Distribution.png')

The histogram confirms the method is working as intended. The dominant spike at gate value ≈ 0 represents the pruned connections — gates driven to near-zero by the L1 penalty, contributing negligibly to the forward pass. The long tail extending toward 0.1 represents the surviving active connections the network identified as informative. The sharp boundary between the two populations, rather than a smooth unimodal distribution, is the signature of L1 regularisation — it forces a binary-like decision on each gate rather than uniformly shrinking all of them.

---

## 4. Implementation Notes

- **`PrunableLinear`** stores a `gate_scores` parameter of shape `(out_features, in_features)` matching `nn.Linear`'s weight convention. The forward pass computes `pruned_weight = weight × sigmoid(gate_scores)` and calls `F.linear`. Gradients flow through both tensors automatically.
- **Sparsity metric** counts gates whose sigmoid value falls below a threshold of 0.01 — measuring gate activity directly, not weight magnitude, since kaiming initialisation produces weights already small in absolute value independent of gate state.
- **Normalised sparsity loss** divides the summed gate values by total gate count, making λ comparable across different architectures and layer widths.
- **Early stopping** (patience=3) and `ReduceLROnPlateau` prevent over-training, which is particularly important at high λ where the sparsity loss can dominate and collapse accuracy if training continues past convergence.
