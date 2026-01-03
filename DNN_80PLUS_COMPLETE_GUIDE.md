# 🎯 DNN EXAM - TARGET 80+ / 100 MARKS

> **AIMLCZG511 | Target: 80+ raw = 24+ scaled**
> **Exam: Jan 4th, 1:00 PM**

---

## 📊 MARKS BREAKDOWN FOR 80+

| Part Type | Total Available | Target | Strategy |
|-----------|-----------------|--------|----------|
| **Part A** (Numerical) | 30 marks | **28-30** | Master ALL calculations |
| **Part B** (Code) | 25 marks | **22-25** | Know all Python patterns |
| **Part C** (Metrics/Reasoning) | 27 marks | **20-22** | Metrics + clear explanations |
| **Part D** (Comparison) | 18 marks | **12-15** | Good conceptual answers |
| **Total** | 100 | **82-92** | ✅ 80+ achievable |

---

## 🔥 COMPLETE MASTERY GUIDE

### Q1: PERCEPTRON (20 marks) - Target: 18+

#### Q1A: Calculation (6M) - Target: 6/6 ⭐⭐⭐

**FORMULA:**
```
z = w₀x₀ + w₁x₁ + w₂x₂   (x₀ = 1 for bias)
ŷ = sign(z) = { +1 if z ≥ 0
              { -1 if z < 0
Error only if ŷ ≠ t
Δwᵢ = η(t - ŷ)xᵢ
wᵢ_new = wᵢ + Δwᵢ
```

**COMPLETE EXAMPLE - OR Gate (Bipolar):**

| x₁ | x₂ | t |
|----|----|----|
| -1 | -1 | -1 |
| -1 | 1 | 1 |
| 1 | -1 | 1 |
| 1 | 1 | 1 |

Initial: w = [0, 0, 0], η = 1

**Step 1:** x = [1, -1, -1], t = -1
- z = 0(1) + 0(-1) + 0(-1) = 0
- ŷ = sign(0) = +1
- Error! t - ŷ = -1 - 1 = -2
- Δw₀ = 1(-2)(1) = -2, Δw₁ = 1(-2)(-1) = 2, Δw₂ = 1(-2)(-1) = 2
- **w = [-2, 2, 2]**

**Step 2:** x = [1, -1, 1], t = 1
- z = -2(1) + 2(-1) + 2(1) = -2 - 2 + 2 = -2
- ŷ = sign(-2) = -1
- Error! t - ŷ = 1 - (-1) = 2
- Δw₀ = 2(1) = 2, Δw₁ = 2(-1) = -2, Δw₂ = 2(1) = 2
- **w = [0, 0, 4]**

**Step 3:** x = [1, 1, -1], t = 1
- z = 0(1) + 0(1) + 4(-1) = -4
- ŷ = sign(-4) = -1
- Error! t - ŷ = 2
- Δw₀ = 2, Δw₁ = 2, Δw₂ = -2
- **w = [2, 2, 2]**

**Step 4:** x = [1, 1, 1], t = 1
- z = 2 + 2 + 2 = 6
- ŷ = sign(6) = +1
- No error! w unchanged

**Final: w = [2, 2, 2]**

**Verification:**
- (-1,-1): z = 2-2-2 = -2 → -1 ✓
- (-1,1): z = 2-2+2 = 2 → +1 ✓
- (1,-1): z = 2+2-2 = 2 → +1 ✓
- (1,1): z = 2+2+2 = 6 → +1 ✓

---

#### Q1B: Linear Separability (4M) - Target: 4/4

**What is Linear Separability?**
- Two classes can be separated by a single straight line (2D) or hyperplane (nD)
- Perceptron can ONLY learn linearly separable patterns

**Limitation - XOR Problem:**
```
XOR Truth Table:
| x₁ | x₂ | XOR |
|----|----|----|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Visual:
  x₂
  1 | ●(1)    ○(0)
    |
  0 | ○(0)    ●(1)
    +-------------x₁
      0       1

NO SINGLE LINE CAN SEPARATE ● from ○!
```

**Solution: Multi-Layer Perceptron (MLP)**
- Add hidden layers with non-linear activation
- Hidden layer transforms input space
- XOR = (x₁ AND NOT x₂) OR (NOT x₁ AND x₂)
- Minimum: 2 hidden neurons for XOR

---

#### Q1C: Generalization vs Overfitting (5M) - Target: 5/5

**GENERALIZATION:**
- Model performs well on UNSEEN data (test set)
- Learning the underlying PATTERN, not memorizing examples
- Sign: Similar accuracy on train and test sets

**OVERFITTING:**
- Model memorizes training data
- Poor performance on new data
- Sign: High train accuracy, LOW test accuracy
- Large gap between train/test performance

**UNDERFITTING:**
- Model too simple
- Can't even fit training data well
- Sign: LOW accuracy on BOTH train and test

**Solutions to Overfitting:**
1. **More training data**
2. **Regularization** (L1, L2 penalty on weights)
3. **Dropout** (randomly disable neurons)
4. **Early stopping** (stop when validation loss increases)
5. **Reduce model complexity** (fewer layers/neurons)
6. **Data augmentation** (for images)

**Reasoning Question Pattern:**
```
Q: "Training accuracy keeps improving but validation drops after few epochs"
A: This is OVERFITTING. Model is memorizing training data.
   Solution: Use dropout, add regularization, or early stopping.

Q: "High training loss even after many epochs"
A: This is UNDERFITTING. Model lacks capacity.
   Solution: Increase model size, train longer, check for bugs.
```

---

#### Q1D: Code Completion (5M) - Target: 5/5

```python
import numpy as np

def train_perceptron(X, y, learning_rate=1.0, epochs=100):
    # Add bias term (column of 1s)
    X_bias = np.c_[np.ones(X.shape[0]), X]       # (1) np.ones
    
    # Initialize weights to zero
    w = np.zeros(X_bias.shape[1])                # (2) zeros
    
    for epoch in range(epochs):
        for i in range(len(X)):
            # Compute weighted sum
            z = np.dot(w, X_bias[i])             # (3) dot
            
            # Apply sign activation
            y_pred = 1 if z >= 0 else -1         # (4) 1, -1
            
            # Update weights if wrong
            if y_pred != y[i]:
                w = w + learning_rate * (y[i] - y_pred) * X_bias[i]  # (5)
    
    return w
```

**Common Blanks:**
- `np.ones(X.shape[0])` - bias column
- `np.zeros(...)` - initialize weights
- `np.dot(w, x)` or `w @ x` - weighted sum
- `1 if z >= 0 else -1` - sign activation (bipolar)
- `(y[i] - y_pred)` - error term

---

### Q2: LINEAR REGRESSION (20 marks) - Target: 18+

#### Q2A: Batch Gradient Descent (6M) - Target: 6/6 ⭐⭐⭐

**FORMULAS:**
```
Prediction:  ŷ = Xw
Error:       e = ŷ - y
MSE Loss:    J = (1/2N) × ||Xw - y||² = (1/2N) × Σ(ŷᵢ - yᵢ)²
Gradient:    ∇J = (1/N) × Xᵀ × (Xw - y) = (1/N) × Xᵀe
Update:      w_new = w - η × ∇J
```

**COMPLETE EXAMPLE:**

Data:
| x₁ | x₂ | y |
|----|-----|---|
| 1 | 1 | 3 |
| 2 | 2 | 6 |
| 3 | 1 | 5 |

Setup: w = [0, 0, 0], η = 0.1, N = 3

**Step 1: Design Matrix (with bias)**
```
X = [1  1  1]    y = [3]
    [1  2  2]        [6]
    [1  3  1]        [5]
```

**Step 2: Predictions**
```
ŷ = Xw = [1  1  1] × [0]   = [0]
         [1  2  2]   [0]     [0]
         [1  3  1]   [0]     [0]
```

**Step 3: Error**
```
e = ŷ - y = [0-3]   = [-3]
            [0-6]     [-6]
            [0-5]     [-5]
```

**Step 4: MSE Loss**
```
J = (1/2N) × Σeᵢ²
  = (1/6) × (9 + 36 + 25)
  = (1/6) × 70
  = 11.67
```

**Step 5: Gradient**
```
∇J = (1/N) × Xᵀe
   = (1/3) × [1  1  1] × [-3]
             [1  2  3]   [-6]
             [1  2  1]   [-5]

   = (1/3) × [1(-3) + 1(-6) + 1(-5)]
             [1(-3) + 2(-6) + 3(-5)]
             [1(-3) + 2(-6) + 1(-5)]

   = (1/3) × [-14]   = [-4.67]
             [-30]     [-10.0]
             [-20]     [-6.67]
```

**Step 6: Update**
```
w_new = w - η∇J
      = [0]   - 0.1 × [-4.67]
        [0]           [-10.0]
        [0]           [-6.67]

      = [0.467]
        [1.000]
        [0.667]
```

**Answer: w = [0.467, 1.0, 0.667] after 1 iteration**

---

#### Q2B: Code Completion (5M) - Target: 5/5

```python
def batch_gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    # Add bias column
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Initialize weights
    w = np.zeros(X_b.shape[1])                   # (1)
    
    n_samples = X.shape[0]
    
    for _ in range(epochs):
        # Forward pass
        y_pred = X_b @ w                          # (2) or np.dot(X_b, w)
        
        # Compute error
        error = y_pred - y                        # (3)
        
        # Compute gradient
        gradient = (1/n_samples) * X_b.T @ error  # (4)
        
        # Update weights
        w = w - learning_rate * gradient          # (5)
    
    return w
```

---

#### Q2C: RMSE Reasoning (6M) - Target: 5/6

**RMSE = √MSE = √[(1/N) × Σ(ŷᵢ - yᵢ)²]**

**Key Points:**
- RMSE is in SAME UNITS as target variable
- If predicting salary in $, RMSE = 5000 means average error is $5000
- Lower RMSE = Better model
- RMSE penalizes large errors more than MAE

**Comparison Example:**
```
Model A: MSE = 25,000,000  → RMSE = √25,000,000 = 5,000
Model B: RMSE = 4,500

Model B is better (lower error)
```

---

#### Q2D: Model Comparison (3M) - Target: 3/3

**Template Answer:**
```
Given: Model A has MSE = X, Model B has MSE = Y

If MSE_A > MSE_B:
  Model B is better because it has lower mean squared error,
  meaning its predictions are closer to actual values on average.

Consider also:
- Are they on same test set?
- Check for overfitting (compare train vs test MSE)
- Consider model complexity (prefer simpler if similar MSE)
```

---

### Q3: BINARY CLASSIFICATION (20 marks) - Target: 18+

#### Q3A: Sigmoid + Gradient (6M) - Target: 6/6 ⭐⭐⭐

**FORMULAS:**
```
Weighted sum:      z = wᵀx = Σwᵢxᵢ
Sigmoid:           ŷ = σ(z) = 1 / (1 + e⁻ᶻ)
BCE Loss:          ℓ = -[y log(ŷ) + (1-y) log(1-ŷ)]
Gradient:          ∇ℓ = (ŷ - y) × x
Update:            w_new = w - η × ∇ℓ
```

**SIGMOID VALUES (MEMORIZE!):**
```
σ(0) = 0.5
σ(1) = 0.731
σ(2) = 0.881
σ(-1) = 0.269
σ(-2) = 0.119
```

**COMPLETE EXAMPLE:**

Given: x = [1, 2, 3] (with bias), w = [0.5, -0.2, 0.3], y = 1, η = 0.5

**Step 1: Weighted Sum**
```
z = wᵀx = 0.5(1) + (-0.2)(2) + 0.3(3)
  = 0.5 - 0.4 + 0.9
  = 1.0
```

**Step 2: Sigmoid**
```
ŷ = σ(1.0) = 1/(1 + e⁻¹) = 1/(1 + 0.368) = 1/1.368 = 0.731
```

**Step 3: BCE Loss**
```
ℓ = -[y log(ŷ) + (1-y) log(1-ŷ)]
  = -[1 × log(0.731) + 0 × log(0.269)]
  = -log(0.731)
  = -(-0.313)
  = 0.313
```

**Step 4: Gradient**
```
∇ℓ = (ŷ - y) × x
   = (0.731 - 1) × [1, 2, 3]
   = -0.269 × [1, 2, 3]
   = [-0.269, -0.538, -0.807]
```

**Step 5: Update**
```
w_new = w - η × ∇ℓ
      = [0.5, -0.2, 0.3] - 0.5 × [-0.269, -0.538, -0.807]
      = [0.5 + 0.135, -0.2 + 0.269, 0.3 + 0.404]
      = [0.635, 0.069, 0.704]
```

---

#### Q3B: Code Completion (5M) - Target: 5/5

```python
def mini_batch_logistic(X, y, batch_size, learning_rate, epochs):
    w = np.zeros(X.shape[1])
    
    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            
            # Weighted sum
            z = X_batch @ w                           # (1)
            
            # Sigmoid activation
            y_pred = 1 / (1 + np.exp(-z))             # (2)
            
            # Error
            error = y_pred - y_batch                  # (3)
            
            # Gradient
            gradient = X_batch.T @ error / len(X_batch)  # (4)
            
            # Update
            w = w - learning_rate * gradient          # (5)
    
    return w
```

---

#### Q3C: Confusion Matrix Metrics (6M) - Target: 6/6 ⭐⭐

**CONFUSION MATRIX:**
```
              Predicted
             Pos    Neg
Actual Pos   TP     FN
       Neg   FP     TN
```

**FORMULAS:**
```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)
            "Of all predicted positive, how many are actually positive?"

Recall    = TP / (TP + FN)
            "Of all actual positive, how many did we find?"

F1 Score  = 2 × (Precision × Recall) / (Precision + Recall)
```

**EXAMPLE:**
Given: TP = 85, FP = 10, FN = 15, TN = 890

```
Accuracy  = (85 + 890) / (85 + 10 + 15 + 890) = 975/1000 = 97.5%
Precision = 85 / (85 + 10) = 85/95 = 89.5%
Recall    = 85 / (85 + 15) = 85/100 = 85%
F1        = 2 × (0.895 × 0.85) / (0.895 + 0.85) = 1.52/1.745 = 87.1%
```

**RECALL vs PRECISION IMPACT:**

| Scenario | Priority | Why |
|----------|----------|-----|
| Disease detection | HIGH RECALL | Can't miss sick patients (FN is dangerous) |
| Spam filter | HIGH PRECISION | Don't block good emails (FP is annoying) |
| Fraud detection | HIGH RECALL | Don't miss fraudsters |
| Search results | HIGH PRECISION | Show relevant results |

---

#### Q3D: Model Comparison (3M) - Target: 3/3

**Template:**
```
Given two models with different metrics:
- Model A: Higher accuracy, lower recall
- Model B: Lower accuracy, higher recall

For disease detection: Choose Model B (recall matters more)
For spam filtering: Choose Model A (precision matters more)

Always consider:
1. Business context (cost of FP vs FN)
2. Class imbalance (accuracy misleading if 99% one class)
3. F1 for balanced trade-off
```

---

### Q4: MULTI-CLASS (SOFTMAX) (20 marks) - Target: 18+

#### Q4A: Softmax + CCE (6M) - Target: 6/6 ⭐⭐⭐

**FORMULAS:**
```
Softmax:     ŷₖ = e^zₖ / Σⱼe^zⱼ

CCE Loss:    ℓ = -Σₖ yₖ log(ŷₖ)
             (with one-hot y, this = -log(ŷ_true_class))

Prediction:  argmax(ŷ)
```

**e VALUES (MEMORIZE!):**
```
e⁰ = 1
e⁰·⁵ = 1.649
e¹ = 2.718
e² = 7.389
e³ = 20.09
```

**COMPLETE EXAMPLE:**

Given: z = [2.0, 1.0, 0.5], True class = 0 (one-hot: [1, 0, 0])

**Step 1: Compute e^z**
```
e^2.0 = 7.389
e^1.0 = 2.718
e^0.5 = 1.649
```

**Step 2: Sum**
```
Sum = 7.389 + 2.718 + 1.649 = 11.756
```

**Step 3: Softmax Probabilities**
```
ŷ₀ = 7.389 / 11.756 = 0.628
ŷ₁ = 2.718 / 11.756 = 0.231
ŷ₂ = 1.649 / 11.756 = 0.140

Verify: 0.628 + 0.231 + 0.140 = 0.999 ≈ 1.0 ✓
```

**Step 4: CCE Loss**
```
ℓ = -[1×log(0.628) + 0×log(0.231) + 0×log(0.140)]
  = -log(0.628)
  = 0.465
```

**Step 5: Prediction**
```
Predicted class = argmax([0.628, 0.231, 0.140]) = 0
```

**Step 6: Correct?**
```
Predicted: 0, True: 0
CORRECT ✓
```

---

#### Q4B: Code Completion (5M) - Target: 5/5

```python
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # numerical stability
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)  # (1)

def categorical_cross_entropy(y_pred, y_true):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1-eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))  # (2)

def train_step(X, y, W, b, learning_rate):
    z = X @ W + b                              # (3)
    y_pred = softmax(z)
    
    dz = (y_pred - y) / len(X)                 # (4)
    dW = X.T @ dz                              # (5)
    db = np.mean(dz, axis=0)
    
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b
```

---

#### Q4C: Multi-class Metrics (6M) - Target: 5/6

**3-Class Confusion Matrix:**
```
              Predicted
             C0    C1    C2
Actual C0    50    5     5     (TP₀=50, FN₀=10)
       C1    3     45    2     (TP₁=45, FN₁=5)
       C2    2     5     43    (TP₂=43, FN₂=7)
```

**Per-Class Metrics:**
```
Class 0:
  Precision = 50 / (50+3+2) = 50/55 = 0.91
  Recall = 50 / (50+5+5) = 50/60 = 0.83

Class 1:
  Precision = 45 / (5+45+5) = 45/55 = 0.82
  Recall = 45 / (3+45+2) = 45/50 = 0.90

Class 2:
  Precision = 43 / (5+2+43) = 43/50 = 0.86
  Recall = 43 / (2+5+43) = 43/50 = 0.86
```

**Macro Average:**
```
Macro Precision = (0.91 + 0.82 + 0.86) / 3 = 0.86
Macro Recall = (0.83 + 0.90 + 0.86) / 3 = 0.86
```

---

### Q5: DFNN (20 marks) - Target: 18+

#### Q5A: Forward Propagation (6M) - Target: 6/6 ⭐⭐⭐

**FORMULAS:**
```
Layer ℓ:
  z⁽ℓ⁾ = h⁽ℓ⁻¹⁾ W⁽ℓ⁾ + b⁽ℓ⁾    (linear)
  h⁽ℓ⁾ = activation(z⁽ℓ⁾)      (non-linear)

ReLU:    max(0, z)
Sigmoid: 1 / (1 + e⁻ᶻ)

BCE Loss: -[y log(ŷ) + (1-y) log(1-ŷ)]
```

**COMPLETE EXAMPLE:**

Architecture: Input(2) → Hidden(2, ReLU) → Output(1, Sigmoid)

Given:
```
x = [0.5, 0.8]
W⁽¹⁾ = [[0.3, -0.2],    b⁽¹⁾ = [0.1, -0.1]
        [0.4,  0.5]]

W⁽²⁾ = [[0.6],          b⁽²⁾ = [0.2]
        [0.7]]
        
y = 1
```

**Layer 1:**
```
z⁽¹⁾ = x × W⁽¹⁾ + b⁽¹⁾
     = [0.5, 0.8] × [[0.3, -0.2],  + [0.1, -0.1]
                     [0.4,  0.5]]

     = [0.5×0.3 + 0.8×0.4, 0.5×(-0.2) + 0.8×0.5] + [0.1, -0.1]
     = [0.15 + 0.32, -0.1 + 0.4] + [0.1, -0.1]
     = [0.47, 0.30] + [0.1, -0.1]
     = [0.57, 0.20]

h⁽¹⁾ = ReLU([0.57, 0.20]) = [max(0,0.57), max(0,0.20)] = [0.57, 0.20]
```

**Layer 2:**
```
z⁽²⁾ = h⁽¹⁾ × W⁽²⁾ + b⁽²⁾
     = [0.57, 0.20] × [[0.6],  + [0.2]
                       [0.7]]

     = [0.57×0.6 + 0.20×0.7] + [0.2]
     = [0.342 + 0.14] + [0.2]
     = [0.482] + [0.2]
     = [0.682]

ŷ = σ(0.682) = 1/(1 + e⁻⁰·⁶⁸²) ≈ 0.664
```

**BCE Loss:**
```
ℓ = -[y log(ŷ) + (1-y) log(1-ŷ)]
  = -[1 × log(0.664) + 0]
  = -log(0.664)
  = 0.41
```

---

#### Q5B: Code Completion (5M) - Target: 5/5

```python
def forward_2layer(X, W1, b1, W2, b2):
    # Layer 1
    z1 = X @ W1 + b1                          # (1)
    h1 = np.maximum(0, z1)                    # (2) ReLU
    
    # Layer 2
    z2 = h1 @ W2 + b2                         # (3)
    y_pred = 1 / (1 + np.exp(-z2))            # (4) Sigmoid
    
    return y_pred                              # (5)
```

---

#### Q5C: DFNN Design (6M) - Target: 5/6

**Sentiment Classification (3 classes):**

```
ARCHITECTURE:
┌─────────────────────────────────────────┐
│ Input Layer: 100 neurons                │ ← Word embedding dimension
├─────────────────────────────────────────┤
│ Hidden Layer 1: 64 neurons, ReLU        │ ← Feature extraction
├─────────────────────────────────────────┤
│ Hidden Layer 2: 32 neurons, ReLU        │ ← Higher-level features
├─────────────────────────────────────────┤
│ Output Layer: 3 neurons, Softmax        │ ← Class probabilities
└─────────────────────────────────────────┘

CHOICES JUSTIFIED:
- ReLU in hidden: Avoids vanishing gradient, computationally efficient
- Softmax in output: Outputs probability distribution for multi-class
- Loss: Categorical Cross-Entropy
- Optimizer: Adam (adaptive learning rate)
- Regularization: Dropout (0.2-0.5) between layers
```

---

#### Q5D: Parameter Count (3M) - Target: 3/3 ⭐

**FORMULA:**
```
Parameters per layer = input_neurons × output_neurons + output_neurons
                     = nₗ₋₁ × nₗ + nₗ
                     = nₗ(nₗ₋₁ + 1)

Total = Σ over all layers
```

**EXAMPLE:** 100 → 64 → 32 → 3

```
Layer 1: 100 × 64 + 64 = 6,400 + 64 = 6,464
Layer 2: 64 × 32 + 32 = 2,048 + 32 = 2,080
Layer 3: 32 × 3 + 3 = 96 + 3 = 99

TOTAL: 6,464 + 2,080 + 99 = 8,643 parameters
```

**WITHOUT BIAS (if asked):**
```
Layer 1: 100 × 64 = 6,400
Layer 2: 64 × 32 = 2,048
Layer 3: 32 × 3 = 96
Total: 8,544 weights only
```

---

## 📋 SUMMARY: 80+ STRATEGY

### Time Allocation in Exam (2 hours):

| Question | Time | Target Marks |
|----------|------|--------------|
| Q1 | 20 min | 18/20 |
| Q2 | 20 min | 18/20 |
| Q3 | 25 min | 18/20 |
| Q4 | 25 min | 16/20 |
| Q5 | 25 min | 16/20 |
| Review | 5 min | - |

**Total Target: 86/100 = 25.8/30 scaled**

### What You MUST Perfect:
1. ✅ Perceptron weight update table (Q1A)
2. ✅ Gradient descent iteration (Q2A)
3. ✅ Sigmoid calculation (Q3A)
4. ✅ Softmax + CCE (Q4A)
5. ✅ DFNN forward pass (Q5A)
6. ✅ All Python code patterns (Part B)
7. ✅ Confusion matrix metrics (Q3C)
8. ✅ Parameter counting (Q5D)

---

## ⏰ YOUR DNN STUDY SCHEDULE

```
5:30 PM - 8:30 PM: DNN INTENSIVE

5:30-6:00: Q1 (Perceptron)
  ├── Do OR gate example COMPLETELY
  ├── Write code from memory
  └── Know overfitting definition

6:00-6:30: Q2 (Linear Regression)
  ├── Do GD iteration COMPLETELY
  ├── Write code from memory
  └── RMSE reasoning

6:30-7:00: Q3 (Binary Classification)
  ├── Sigmoid calculation
  ├── BCE Loss
  ├── Metrics: TP, FP, FN, TN
  └── Code

7:00-7:30: DINNER

7:30-8:00: Q4 (Softmax)
  ├── Calculate e^z for 3 classes
  ├── Normalize to get probabilities
  ├── CCE = -log(ŷ_true)
  └── Code

8:00-8:30: Q5 (DFNN)
  ├── Forward pass through 2 layers
  ├── ReLU = max(0, z)
  ├── Parameter counting formula
  └── Code
```

---

**With this guide, 80+ is VERY achievable! The key is to PRACTICE the calculations, not just read them.**
