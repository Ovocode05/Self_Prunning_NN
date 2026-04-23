# Self-Pruning Neural Network

**Author:** Krrish Punj  
**Dataset:** CIFAR-10

---

## 1. Why L1 Penalty on Sigmoid Gates Encourages Sparsity ?

Each weight in a `PrunableLinear` layer is multiplied by a gate value `g = sigmoid(s) ∈ (0, 1)`, where `s` is a learnable scalar. The total loss is:

```
Total Loss = CrossEntropy(y_pred, y_true) + λ · (Σ gᵢ / N)
```

The regularisation term is the normalised L1 norm of all gate values. It's ability to make redundant weights completely 0 rather than just shrinking them. It applies a constant gradient penalty of abs(+-1) which keeps on shrinking the weights until the passive ones are 0 during the training process. More importantly, a higher λ means the sparsity term has more weight in the total loss, so the network prunes more aggressively. 

---

## 2. Results

| Lambda (λ) | Test Accuracy | Sparsity Level (%) |
| :--------: | :-----------: | :----------------: |
|  0.00002   |     53.6%     |       12.0%        |
|  0.00010   |     54.1%     |       58.5%        |
|  0.00020   |     52.9%     |       77.2%        |
|  0.00500   |     42.0%     |       99.7%        |

**Key observation:** Sparsity increases from 12% to 77% with only a 1.2% drop in accuracy and the network retains most of its predictive capacity, which confirmed that a large fraction of weights in a shallow MLP were redundant.

---

## 3. Gate Value Distribution

The model is selected by best accuracy and sparsity tradeoff balance (52% at λ=0.0002) which gave 77% sparsity.

![Gate Value Distribution](Distribution.png)

The histogram confirms the method is working as intended. The dominant spike at gate value ~ 0 represents the pruned connections, gates driven to near-zero by the L1 penalty, contributing negligibly to the forward pass.

---

## 4. Implementation Notes

- **`PrunableLinear`** stores a `gate_scores` parameter of shape `(out_features, in_features)` matching `nn.Linear`'s weight convention. The forward pass computes `pruned_weight = weight × sigmoid(gate_scores)` and calls `F.linear`. Gradients flow through both tensors automatically.
- **Normalised sparsity loss** divides the summed gate values by total gate count, making λ comparable across different architectures and layer widths.
- **Early stopping** (patience=3 epochs) prevent over-training, which is particularly important at high λ where the sparsity loss can dominate and collapse accuracy if training continues past convergence.
---

## 5. Setup 
```bash
git clone "https://github.com/Ovocode05/Self_Pruning_NN"
cd Self_Pruning_NN
```
```bash
pip install -r requirement.txt
```
> **Optional:** If you want to use a virtual environment first:
> ```bash
> python -m venv venv
> source venv/bin/activate  # Mac/Linux
> venv\Scripts\activate     # Windows
> ```

```bash
python Script.py
```
CIFAR-10 will download automatically into `./data` on first run (~170 MB). 

## 6. References
- [Precomputed mean and std for CIFAR-10 dataset](https://www.kaggle.com/code/fanbyprinciple/cifar10-explanation-with-pytorch)
