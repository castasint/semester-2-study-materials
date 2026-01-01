# 📋 DNN Midterm Formula Sheet
### Quick Reference Card | AIMLCZG511 | Sessions 1-8

---

## 🧠 PERCEPTRON

### Weighted Sum
```
z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = wᵀx
```

### Activation (Step/Sign)
```
ŷ = sign(z) = { +1 if z ≥ 0
              { -1 if z < 0
```

### Weight Update Rule
```
Δwᵢ = η(t - ŷ)xᵢ
wᵢ ← wᵢ + Δwᵢ
```

### XOR Perceptrons
- Single hidden layer: n perceptrons (n = inputs)
- Deep network: O(log n) perceptrons

---

## 📈 LINEAR REGRESSION

### Model
```
ŷ = wᵀx = w₀ + w₁x₁ + ... + wₐxₐ
ŷ = Xw  (vectorized)
```

### MSE Loss
```
J(w) = (1/2N) Σ(ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²
     = (1/2N) ||Xw - y||²
```

### Gradient
```
∇J = (1/N) Xᵀ(Xw - y)
```

### Update Rule
```
w ← w - η∇J
```

---

## 🎯 BINARY CLASSIFICATION

### Sigmoid
```
σ(z) = 1 / (1 + e⁻ᶻ)
σ'(z) = σ(z)(1 - σ(z))
```

### Prediction
```
ŷ = σ(wᵀx) = P(y=1|x)
class = 1 if ŷ ≥ 0.5 else 0
```

### Binary Cross-Entropy (BCE)
```
ℓ = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
J = -(1/N) Σ[y⁽ⁱ⁾log(ŷ⁽ⁱ⁾) + (1-y⁽ⁱ⁾)log(1-ŷ⁽ⁱ⁾)]
```

### Gradient (same form!)
```
∇ℓ = (ŷ - y)x
```

---

## 🎨 MULTI-CLASS CLASSIFICATION

### Softmax
```
ŷₖ = exp(zₖ) / Σⱼexp(zⱼ)
```

### Categorical Cross-Entropy (CCE)
```
ℓ = -Σₖ yₖ·log(ŷₖ)
```
(y is one-hot encoded)

---

## 🔥 ACTIVATION FUNCTIONS

| Function | Formula | Derivative | Range |
|----------|---------|------------|-------|
| **Sigmoid** | 1/(1+e⁻ᶻ) | σ(1-σ) | (0,1) |
| **Tanh** | (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ) | 1-tanh²(z) | (-1,1) |
| **ReLU** | max(0,z) | 1 if z>0, else 0 | [0,∞) |
| **Softmax** | eᶻⁱ/Σeᶻʲ | - | (0,1) |
| **Identity** | z | 1 | (-∞,∞) |

---

## 🌐 DFNN FORWARD PROPAGATION

### Per Layer
```
z⁽ℓ⁾ = h⁽ℓ⁻¹⁾W⁽ℓ⁾ + b⁽ℓ⁾
h⁽ℓ⁾ = σ(z⁽ℓ⁾)
```

### Initial & Final
```
h⁽⁰⁾ = x
ŷ = h⁽ᴸ⁾
```

---

## ⬅️ BACKPROPAGATION

### Output Layer Error
```
δ⁽ᴸ⁾ = (1/B)(Ŷ - Y)
```
(Works for MSE+Id, BCE+Sigmoid, CCE+Softmax)

### Hidden Layer Error
```
δ⁽ℓ⁾ = (δ⁽ℓ⁺¹⁾W⁽ℓ⁺¹⁾ᵀ) ⊙ σ'(z⁽ℓ⁾)
```

### Gradients
```
∂J/∂W⁽ℓ⁾ = (1/B) H⁽ℓ⁻¹⁾ᵀδ⁽ℓ⁾
∂J/∂b⁽ℓ⁾ = (1/B) 1ᵀδ⁽ℓ⁾
```

---

## 📊 PARAMETER COUNT

### With Bias
```
Total = Σ nₗ(nₗ₋₁ + 1)
```

### Without Bias
```
Total = Σ nₗ × nₗ₋₁
```

### Example: 784→256→128→10
```
Layer 1: 784×256 + 256 = 200,960
Layer 2: 256×128 + 128 = 32,896
Layer 3: 128×10 + 10 = 1,290
Total: 235,146
```

---

## 🖼️ CNN FORMULAS

### Output Size
```
output = ⌊(n + 2p - f) / s⌋ + 1
```
Where:
- n = input size
- p = padding
- f = filter/kernel size
- s = stride

### Common Cases
| Padding | Formula | Name |
|---------|---------|------|
| p = 0 | (n-f)/s + 1 | Valid |
| p = (f-1)/2 | n (if s=1) | Same |

### Pooling Output
```
output = ⌊(n - pool_size) / stride⌋ + 1
```

---

## 📏 CLASSIFICATION METRICS

### Confusion Matrix
```
              Predicted
             Pos    Neg
Actual Pos   TP     FN
       Neg   FP     TN
```

### Formulas
```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × (P × R) / (P + R)
```

---

## 🏗️ CNN ARCHITECTURES

### LeNet (1998)
```
Conv(5×5,6) → Pool → Conv(5×5,16) → Pool → FC → FC → Out
```

### AlexNet (2012)
- Large kernels (11×11, 5×5)
- ReLU, Dropout
- GPU training

### VGG (2014)
- Only 3×3 kernels
- Very deep (16-19 layers)
- 138M parameters

### ResNet (2015)
- Skip connections: output = F(x) + x
- 100+ layers possible
- Solves vanishing gradient

---

## 🔄 GRADIENT DESCENT VARIANTS

| Type | Updates per Epoch | Use Case |
|------|-------------------|----------|
| **Batch GD** | 1 (all data) | Small dataset |
| **SGD** | N (per sample) | Large dataset |
| **Mini-batch** | N/B (batches) | Standard |

### Common Batch Sizes
32, 64, 128, 256

---

## 🎛️ LOSS SUMMARY

| Task | Activation | Loss |
|------|------------|------|
| **Regression** | Identity | MSE |
| **Binary Class** | Sigmoid | BCE |
| **Multi-class** | Softmax | CCE |

---

## ⚡ QUICK TIPS

1. **Perceptron**: Only for linearly separable
2. **Sigmoid output**: For probabilities [0,1]
3. **ReLU hidden**: Avoids vanishing gradient
4. **Softmax**: Makes probabilities sum to 1
5. **BCE vs CCE**: Binary vs multi-class
6. **Transfer Learning**: Pre-trained → new task
7. **Skip connections**: Enable deep networks

---

## 🔢 NUMERICAL TIPS

### Sigmoid Values
```
σ(0) = 0.5
σ(1) ≈ 0.731
σ(2) ≈ 0.881
σ(-1) ≈ 0.269
σ(-2) ≈ 0.119
```

### Common e Values
```
e⁰ = 1
e¹ ≈ 2.718
e² ≈ 7.389
e³ ≈ 20.09
e⁻¹ ≈ 0.368
e⁻² ≈ 0.135
```

### log Values
```
log(0.5) ≈ -0.693
log(0.1) ≈ -2.303
log(0.9) ≈ -0.105
```

---

**🍀 Good luck with your exam!**
