# ⏰ 5-HOUR DNN MIDTERM CRASH COURSE

> **Based on EXACT Exam Pattern | 5 Questions × 20 marks = 100 marks**

---

## 📋 EXAM PATTERN ANALYSIS

| Q# | Topic | Part A (6M) | Part B (5M) | Part C (6M) | Part D (3M) |
|----|-------|-------------|-------------|-------------|-------------|
| **Q1** | Perceptron | Compute z, ŷ, update w | Python code fill-blanks | Linear separability + Overfitting | - |
| **Q2** | Linear Regression | Batch GD iteration | Python code fill-blanks | RMSE reasoning | Model comparison |
| **Q3** | Binary Classification | Sigmoid, gradient, update | Python code fill-blanks | Precision/Recall from confusion matrix | Model evaluation |
| **Q4** | Softmax Multi-class | Compute softmax + CCE | Python code fill-blanks | Confusion matrix metrics | Model comparison |
| **Q5** | DFNN | Forward propagation | Python code fill-blanks | Design a network | Parameter count |

---

## ⏱️ REVISED 5-HOUR PLAN (Aligned to Exam)

### HOUR 1: Q1 - Perceptron (0:00 - 1:00)

#### Part A Practice: Numerical (6 marks)
**You WILL be asked to:**
- Compute weighted sum z = w₀ + w₁x₁ + w₂x₂
- Compute output ŷ = sign(z)
- Update weights using: **Δwᵢ = η(t - ŷ)xᵢ**

**PRACTICE THIS NOW:**
```
Given: w₀=w₁=w₂=0, η=1, x₀=1 (bias)
NAND gate (bipolar: +1/-1):
(-1,-1)→+1, (-1,+1)→+1, (+1,-1)→+1, (+1,+1)→-1

Step | x₀ | x₁ | x₂ | t | z=Σwᵢxᵢ | ŷ=sign(z) | t-ŷ | Δw₀ | Δw₁ | Δw₂ | w₀ | w₁ | w₂
-----|----|----|----|----|---------|-----------|-----|-----|-----|-----|----|----|----
Init |    |    |    |    |         |           |     |     |     |     | 0  | 0  | 0
1    | 1  | -1 | -1 | +1 | 0       | +1        | 0   | 0   | 0   | 0   | 0  | 0  | 0
2    | 1  | -1 | +1 | +1 | 0       | +1        | 0   | 0   | 0   | 0   | 0  | 0  | 0
3    | 1  | +1 | -1 | +1 | 0       | +1        | 0   | 0   | 0   | 0   | 0  | 0  | 0
4    | 1  | +1 | +1 | -1 | 0       | +1        | -2  | -2  | -2  | -2  | -2 | -2 | -2
```

#### Part B Practice: Python Code (5 marks)
**MEMORIZE this template:**
```python
def perceptron_train(X, y, eta=1.0, epochs=100):
    X_bias = np.c_[np.ones(X.shape[0]), X]  # Add bias
    w = np.zeros(X_bias.shape[1])            # Init weights
    
    for epoch in range(epochs):
        for i in range(len(X)):
            z = np.dot(w, X_bias[i])         # Weighted sum
            y_pred = 1 if z >= 0 else -1     # Sign activation
            if y_pred != y[i]:
                w = w + eta * (y[i] - y_pred) * X_bias[i]  # Update
    return w
```

#### Part C: Theory (4+5 marks)
**Linear Separability:**
- Perceptron only works for linearly separable data
- XOR is NOT linearly separable → single perceptron fails
- Solution: MLP with hidden layer

**Overfitting vs Generalization:**
- Overfitting: Model memorizes training data, fails on new data
- Signs: High training accuracy, low test accuracy
- Solutions: More data, regularization, dropout, early stopping

---

### HOUR 2: Q2 - Linear Regression (1:00 - 2:00)

#### Part A Practice: Batch GD Iteration (6 marks)
**You WILL be asked to:**
1. Compute predictions ŷ = Xw
2. Calculate MSE loss
3. Compute gradient
4. Update weights

**PRACTICE THIS NOW:**
```
Data: (1,2), (2,4), (3,5)
X = [[1,1], [1,2], [1,3]], y = [2,4,5]ᵀ, w⁽⁰⁾ = [0,0]ᵀ, η = 0.1

Step 1: Predictions
ŷ = Xw = [[1,1],[1,2],[1,3]] × [0,0]ᵀ = [0,0,0]ᵀ

Step 2: Error
e = ŷ - y = [0-2, 0-4, 0-5]ᵀ = [-2,-4,-5]ᵀ

Step 3: MSE Loss
J = (1/2N) × (4+16+25) = (1/6) × 45 = 7.5

Step 4: Gradient
∇J = (1/N) × Xᵀe = (1/3) × [[1,1,1],[1,2,3]] × [-2,-4,-5]ᵀ
   = (1/3) × [-11, -23]ᵀ = [-3.67, -7.67]ᵀ

Step 5: Update
w⁽¹⁾ = w⁽⁰⁾ - η∇J = [0,0]ᵀ - 0.1×[-3.67,-7.67]ᵀ = [0.367, 0.767]ᵀ
```

#### Part B: Python Code (5 marks)
```python
def batch_gd(X, y, eta=0.01, epochs=1000):
    X_b = np.c_[np.ones(X.shape[0]), X]  # Add bias
    w = np.zeros(X_b.shape[1])           # Init weights
    
    for _ in range(epochs):
        y_pred = X_b @ w                 # Predictions
        error = y_pred - y               # Error
        gradient = (1/len(X)) * X_b.T @ error  # Gradient
        w = w - eta * gradient           # Update
    return w
```

#### Part C: RMSE Reasoning (3+3 marks)
- **RMSE = √MSE** - Same units as target variable
- Lower RMSE = better model
- Compare models by RMSE, not MSE (more interpretable)

---

### HOUR 3: Q3 - Binary Classification (2:00 - 3:00)

#### Part A Practice: Logistic Regression (6 marks)
**You WILL be asked to:**
1. Compute z = wᵀx
2. Predict probability ŷ = σ(z) = 1/(1+e⁻ᶻ)
3. Compute gradient
4. Update weights

**PRACTICE THIS NOW:**
```
Given: x = [1,2,3] (with bias), w = [0.5,-0.2,0.3], y = 1, η = 0.5

Step 1: Weighted sum
z = 0.5(1) + (-0.2)(2) + 0.3(3) = 0.5 - 0.4 + 0.9 = 1.0

Step 2: Sigmoid
ŷ = σ(1.0) = 1/(1+e⁻¹) = 1/1.368 = 0.731

Step 3: Gradient (BCE + Sigmoid gives this elegant form!)
∇ℓ = (ŷ - y) × x = (0.731 - 1) × [1,2,3] = [-0.269, -0.538, -0.807]

Step 4: Update
w_new = w - η∇ℓ = [0.5,-0.2,0.3] - 0.5×[-0.269,-0.538,-0.807]
      = [0.635, 0.069, 0.704]
```

#### Part B: Python Code (5 marks)
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_sgd(X, y, eta=0.1, epochs=100):
    X_b = np.c_[np.ones(X.shape[0]), X]
    w = np.zeros(X_b.shape[1])
    
    for _ in range(epochs):
        for i in range(len(X)):
            z = np.dot(w, X_b[i])
            y_pred = sigmoid(z)
            gradient = (y_pred - y[i]) * X_b[i]
            w = w - eta * gradient
    return w
```

#### Part C: Metrics from Confusion Matrix (3+3 marks)
```
Given: TP=85, FP=10, FN=15, TN=890

Accuracy = (TP+TN)/(TP+TN+FP+FN) = (85+890)/(85+10+15+890) = 975/1000 = 97.5%
Precision = TP/(TP+FP) = 85/(85+10) = 85/95 = 89.5%
Recall = TP/(TP+FN) = 85/(85+15) = 85/100 = 85%
```

**Precision vs Recall:**
- High **Precision** needed: Spam filter (don't mark good email as spam)
- High **Recall** needed: Disease detection (don't miss sick patients)

---

### HOUR 4: Q4 - Softmax Multi-class (3:00 - 4:00)

#### Part A Practice: Softmax + CCE (6 marks)
**You WILL be asked to:**
1. Compute softmax probabilities
2. Compute cross-entropy loss
3. Identify prediction (argmax)
4. Check if correct

**PRACTICE THIS NOW:**
```
Given: z = [2.5, 0.3, -1.2, 3.1, 0.7] for 5 classes, true class = "bird" (index 3)

Step 1: Exponentials
e^2.5 = 12.18, e^0.3 = 1.35, e^-1.2 = 0.30, e^3.1 = 22.20, e^0.7 = 2.01

Step 2: Sum
Σ = 12.18 + 1.35 + 0.30 + 22.20 + 2.01 = 38.04

Step 3: Softmax
ŷ = [12.18/38.04, 1.35/38.04, 0.30/38.04, 22.20/38.04, 2.01/38.04]
  = [0.32, 0.04, 0.01, 0.58, 0.05]

Step 4: Cross-Entropy (true class = 3, so y₃ = 1)
L = -log(ŷ₃) = -log(0.58) = 0.54

Step 5: Prediction
Predicted class = argmax(ŷ) = 3 (bird) ✓ CORRECT!
```

#### Part B: Python Code (5 marks)
```python
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Numerical stability
    return exp_z / np.sum(exp_z)

def train_softmax(X, y, W, eta=0.01):
    z = X @ W
    y_pred = softmax(z)
    gradient = (1/len(X)) * X.T @ (y_pred - y)  # One-hot y
    W = W - eta * gradient
    return W
```

#### Part C: Multi-class Metrics (3+3 marks)
For 3-class confusion matrix, compute per-class:
- Precision_k = TP_k / (TP_k + FP_k)
- Recall_k = TP_k / (TP_k + FN_k)
- Overall Accuracy = Correct / Total

---

### HOUR 5: Q5 - DFNN + Review (4:00 - 5:00)

#### Part A Practice: Forward Propagation (6 marks)
**You WILL be asked to do this step-by-step:**

```
Network: 2 inputs → 3 hidden (ReLU) → 2 outputs (Sigmoid)

Given: x = [1.0, 0.5]ᵀ, y = [1, 0]ᵀ
W⁽¹⁾ = [[0.5,0.2,-0.3],[0.8,0.1,-0.4]], b⁽¹⁾ = [0.1,-0.2,0.3]
W⁽²⁾ = [[0.4,-0.1],[0.6,0.2],[0.7,-0.3]], b⁽²⁾ = [0.2,-0.1]

LAYER 1:
z⁽¹⁾ = xᵀW⁽¹⁾ + b⁽¹⁾
     = [1.0,0.5] × [[0.5,0.2,-0.3],[0.8,0.1,-0.4]] + [0.1,-0.2,0.3]
     = [0.5+0.4, 0.2+0.05, -0.3-0.2] + [0.1,-0.2,0.3]
     = [0.9, 0.25, -0.5] + [0.1,-0.2,0.3]
     = [1.0, 0.05, -0.2]

h⁽¹⁾ = ReLU(z⁽¹⁾) = [max(0,1.0), max(0,0.05), max(0,-0.2)]
     = [1.0, 0.05, 0]

LAYER 2:
z⁽²⁾ = h⁽¹⁾W⁽²⁾ + b⁽²⁾
     = [1.0,0.05,0] × [[0.4,-0.1],[0.6,0.2],[0.7,-0.3]] + [0.2,-0.1]
     = [0.4+0.03+0, -0.1+0.01+0] + [0.2,-0.1]
     = [0.43, -0.09] + [0.2,-0.1]
     = [0.63, -0.19]

ŷ = σ(z⁽²⁾) = [σ(0.63), σ(-0.19)] = [0.65, 0.45]
```

#### Part B: Python Code (5 marks)
```python
def relu(z):
    return np.maximum(0, z)

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    h1 = relu(z1)
    z2 = h1 @ W2 + b2
    y_pred = sigmoid(z2)
    return y_pred
```

#### Part C: Design Network (6 marks)
**For sentiment classification (3 classes):**
- Input: 100-dim word embedding
- Hidden 1: 64 neurons, ReLU
- Hidden 2: 32 neurons, ReLU
- Output: 3 neurons, Softmax
- Loss: Categorical Cross-Entropy

#### Part D: Parameter Count (3 marks)
```
Network: 8 → 10 → 6 → 4 (NO biases)

Layer 1: 8 × 10 = 80
Layer 2: 10 × 6 = 60
Layer 3: 6 × 4 = 24
─────────────────────
Total: 164 parameters
```

---

## 📝 FINAL 15-MIN REVIEW CHECKLIST

### Formulas (MUST memorize)
- [ ] Perceptron: `Δw = η(t-ŷ)x`
- [ ] Gradient: `∇J = (1/N)Xᵀ(Xw-y)`
- [ ] Sigmoid: `σ(z) = 1/(1+e⁻ᶻ)`
- [ ] Softmax: `ŷₖ = eᶻᵏ/Σeᶻʲ`
- [ ] All gradients: `∂L/∂z = ŷ - y`
- [ ] Metrics: `Acc, Prec, Recall`

### Python Patterns (MUST know)
- [ ] `np.c_[np.ones(...), X]` - add bias
- [ ] `np.zeros(...)` - init weights
- [ ] `np.dot(w, x)` or `X @ w` - matrix multiply
- [ ] `np.maximum(0, z)` - ReLU
- [ ] `1/(1+np.exp(-z))` - sigmoid
- [ ] `np.exp(z)/np.sum(np.exp(z))` - softmax

### Common Values
```
σ(0) = 0.5    σ(1) ≈ 0.73    σ(-1) ≈ 0.27
e¹ ≈ 2.72    e² ≈ 7.39      log(0.5) ≈ -0.69
```

---

**Now aligned exactly with exam pattern! Good luck! 🍀**
