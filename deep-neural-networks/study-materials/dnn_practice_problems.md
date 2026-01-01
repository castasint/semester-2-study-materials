# 📝 DNN Midterm Practice Problems

> **Based on Actual Exam Patterns | AIMLCZG511 | Sessions 1-8**

---

# Section A: Perceptron (20 marks typical)

## Problem A1: Perceptron Learning Algorithm [6 marks]

**Question:** Consider the implementation of a perceptron for an **OR gate** using bipolar representation (input and output values are either +1 or -1).

**Given:**
- Initial weights: w₀ = w₁ = w₂ = 0, where w₀ is the bias and x₀ = 1
- Learning rate: η = 1
- Activation function: h = sign(z), where z = w₀ + w₁x₁ + w₂x₂
- Weight update rule: Δwᵢ = η(t - y)xᵢ

| x₁ | x₂ | Target (OR) |
|----|----|-------------|
| -1 | -1 | -1 |
| -1 | 1  | 1  |
| 1  | -1 | 1  |
| 1  | 1  | 1  |

**Tasks:**
A. Implement the perceptron learning algorithm for the above OR gate using one full epoch.
B. Show the computations for z, y, Δw, and updated weights for each input.
C. Write the final weight vector (w₀, w₁, w₂) after the epoch.
D. Verify that the perceptron correctly classifies all inputs with the final weights.

---

### Solution A1:

**Epoch 1:**

| Step | x₀ | x₁ | x₂ | t | z = Σwᵢxᵢ | y = sign(z) | t - y | Δw₀ | Δw₁ | Δw₂ | w₀ | w₁ | w₂ |
|------|----|----|----|----|-----------|-------------|-------|-----|-----|-----|----|----|---|
| Init | - | - | - | - | - | - | - | - | - | - | 0 | 0 | 0 |
| 1 | 1 | -1 | -1 | -1 | 0 | +1 | -2 | -2 | 2 | 2 | -2 | 2 | 2 |
| 2 | 1 | -1 | 1 | 1 | -2+(-2)+2=-2 | -1 | 2 | 2 | -2 | 2 | 0 | 0 | 4 |
| 3 | 1 | 1 | -1 | 1 | 0+0-4=-4 | -1 | 2 | 2 | 2 | -2 | 2 | 2 | 2 |
| 4 | 1 | 1 | 1 | 1 | 2+2+2=6 | +1 | 0 | 0 | 0 | 0 | 2 | 2 | 2 |

**Final weights after Epoch 1:** w₀ = 2, w₁ = 2, w₂ = 2

**Verification:**
| x₁ | x₂ | z = 2 + 2x₁ + 2x₂ | y = sign(z) | Target | Correct? |
|----|----|-------------------|-------------|--------|----------|
| -1 | -1 | 2 - 2 - 2 = -2 | -1 | -1 | ✓ |
| -1 | 1 | 2 - 2 + 2 = 2 | +1 | 1 | ✓ |
| 1 | -1 | 2 + 2 - 2 = 2 | +1 | 1 | ✓ |
| 1 | 1 | 2 + 2 + 2 = 6 | +1 | 1 | ✓ |

**All inputs correctly classified!**

---

## Problem A2: Linear Separability [4 marks]

**Question:**
A. Why would a single-layer perceptron fail to classify the XOR function? [1 mark]
B. How can a Multi-Layer Perceptron (MLP) overcome this limitation? [2 marks]
C. For an XOR function over 4 binary input variables, how many perceptrons are required for a single hidden layer network vs a deep network? [1 mark]

### Solution A2:

**A.** XOR is **not linearly separable** - no single hyperplane (line in 2D) can separate the positive and negative classes. A single perceptron can only learn linear decision boundaries.

**B.** MLP overcomes this by:
- **Hidden layers** create intermediate representations that transform the input space
- **Non-linear activation functions** allow learning non-linear decision boundaries
- For XOR: Hidden layer computes AND and OR-like features, output combines them

**C.** 
- **Single hidden layer:** 4 perceptrons in hidden layer (one for each pair separation)
- **Deep network:** O(log n) = O(log 4) ≈ 2 perceptrons with proper connectivity

---

## Problem A3: Perceptron Code Completion [5 marks]

**Question:** Fill in the blanks in the following perceptron training code:

```python
import numpy as np

def train_perceptron(X, y, learning_rate=1.0, epochs=100):
    # Add bias term
    X_bias = np.c_[_______(X.shape[0]), X]  # (1)
    
    # Initialize weights
    w = np._______(X_bias.shape[1])  # (2)
    
    for epoch in range(epochs):
        for i in range(len(X)):
            # Compute weighted sum
            z = np._______(w, X_bias[i])  # (3)
            
            # Apply activation function
            y_pred = _______ if z >= 0 else _______  # (4)
            
            # Update weights if prediction is wrong
            if y_pred != y[i]:
                w = w + learning_rate * _______ * X_bias[i]  # (5)
    
    return w
```

### Solution A3:

```python
import numpy as np

def train_perceptron(X, y, learning_rate=1.0, epochs=100):
    # Add bias term
    X_bias = np.c_[np.ones(X.shape[0]), X]  # (1) np.ones
    
    # Initialize weights
    w = np.zeros(X_bias.shape[1])  # (2) zeros
    
    for epoch in range(epochs):
        for i in range(len(X)):
            # Compute weighted sum
            z = np.dot(w, X_bias[i])  # (3) dot
            
            # Apply activation function
            y_pred = 1 if z >= 0 else -1  # (4) 1, -1 (or 0 for binary)
            
            # Update weights if prediction is wrong
            if y_pred != y[i]:
                w = w + learning_rate * (y[i] - y_pred) * X_bias[i]  # (5) (y[i] - y_pred)
    
    return w
```

---

# Section B: Linear Regression (20 marks typical)

## Problem B1: Batch Gradient Descent [6 marks]

**Question:** Consider a linear regression model for predicting house prices. Given:

| House | Size (x₁) | Bedrooms (x₂) | Price (y) |
|-------|-----------|---------------|-----------|
| 1 | 1 | 1 | 3 |
| 2 | 2 | 2 | 6 |
| 3 | 3 | 1 | 5 |

**Given:**
- Initial weights: w = [w₀, w₁, w₂]ᵀ = [0, 0, 0]ᵀ
- Learning rate: η = 0.1

**Tasks:**
A. Write the design matrix X (with bias column) and target vector y.
B. Compute the initial predictions ŷ⁽⁰⁾.
C. Calculate the MSE loss.
D. Compute the gradient ∇J.
E. Perform one weight update and show the new weights w⁽¹⁾.

### Solution B1:

**A. Design matrix and target:**
$$\mathbf{X} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 2 \\ 1 & 3 & 1 \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} 3 \\ 6 \\ 5 \end{bmatrix}$$

**B. Initial predictions:**
$$\hat{\mathbf{y}}^{(0)} = \mathbf{X}\mathbf{w}^{(0)} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 2 \\ 1 & 3 & 1 \end{bmatrix} \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$$

**C. MSE Loss:**
$$J = \frac{1}{2N}\sum_{i=1}^{N}(\hat{y}^{(i)} - y^{(i)})^2 = \frac{1}{6}[(0-3)^2 + (0-6)^2 + (0-5)^2]$$
$$J = \frac{1}{6}[9 + 36 + 25] = \frac{70}{6} = 11.67$$

**D. Gradient:**
$$\mathbf{e} = \hat{\mathbf{y}} - \mathbf{y} = \begin{bmatrix} -3 \\ -6 \\ -5 \end{bmatrix}$$

$$\nabla J = \frac{1}{N}\mathbf{X}^T\mathbf{e} = \frac{1}{3}\begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \\ 1 & 2 & 1 \end{bmatrix} \begin{bmatrix} -3 \\ -6 \\ -5 \end{bmatrix}$$

$$\nabla J = \frac{1}{3}\begin{bmatrix} -3-6-5 \\ -3-12-15 \\ -3-12-5 \end{bmatrix} = \frac{1}{3}\begin{bmatrix} -14 \\ -30 \\ -20 \end{bmatrix} = \begin{bmatrix} -4.67 \\ -10.0 \\ -6.67 \end{bmatrix}$$

**E. Weight update:**
$$\mathbf{w}^{(1)} = \mathbf{w}^{(0)} - \eta \nabla J = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix} - 0.1 \begin{bmatrix} -4.67 \\ -10.0 \\ -6.67 \end{bmatrix} = \begin{bmatrix} 0.467 \\ 1.0 \\ 0.667 \end{bmatrix}$$

---

## Problem B2: Regression Evaluation [3 marks]

**Question:** Two models predict annual salary:
- Model A: MSE = 25000000 (on scale of thousands)
- Model B: RMSE = 4500

Which model is better? Justify with proper unit analysis.

### Solution B2:

**Conversion:**
- Model A: MSE = 25,000,000 → RMSE = √25,000,000 = 5,000 (in same units as salary)
- Model B: RMSE = 4,500

**Analysis:**
- Model B (RMSE = 4,500) is better than Model A (RMSE = 5,000)
- RMSE is in the same units as the target variable (dollars)
- Lower RMSE means predictions are closer to actual values

---

## Problem B3: Regression Code Completion [5 marks]

```python
def batch_gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    # Add bias column
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Initialize weights
    w = np.zeros(________)  # (1)
    
    n_samples = X.shape[0]
    
    for _ in range(epochs):
        # Forward pass: predictions
        y_pred = ________  # (2)
        
        # Compute error
        error = ________  # (3)
        
        # Compute gradient
        gradient = (1/n_samples) * ________  # (4)
        
        # Update weights
        w = ________  # (5)
    
    return w
```

### Solution B3:

```python
def batch_gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    w = np.zeros(X_b.shape[1])  # (1)
    
    n_samples = X.shape[0]
    
    for _ in range(epochs):
        y_pred = X_b @ w  # or np.dot(X_b, w)  # (2)
        
        error = y_pred - y  # (3)
        
        gradient = (1/n_samples) * X_b.T @ error  # (4)
        
        w = w - learning_rate * gradient  # (5)
    
    return w
```

---

# Section C: Binary Classification (20 marks typical)

## Problem C1: Logistic Regression [6 marks]

**Question:** A binary classifier uses logistic regression with:
- x = [1, 2, 3] (with bias x₀ = 1 already included)
- w = [0.5, -0.2, 0.3]
- Target y = 1

**Tasks:**
A. Compute the weighted sum z = wᵀx
B. Calculate the predicted probability ŷ = σ(z)
C. Compute the binary cross-entropy loss
D. Calculate the gradient ∇ℓ for this single example
E. Update weights using η = 0.5

### Solution C1:

**A. Weighted sum:**
$$z = w^Tx = 0.5(1) + (-0.2)(2) + 0.3(3) = 0.5 - 0.4 + 0.9 = 1.0$$

**B. Predicted probability:**
$$\hat{y} = \sigma(1.0) = \frac{1}{1 + e^{-1}} = \frac{1}{1 + 0.368} = \frac{1}{1.368} = 0.731$$

**C. Cross-entropy loss:**
$$\ell = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$$
$$\ell = -[1 \cdot \log(0.731) + 0] = -(-0.313) = 0.313$$

**D. Gradient:**
$$\nabla\ell = (\hat{y} - y) \cdot x = (0.731 - 1) \cdot [1, 2, 3]$$
$$\nabla\ell = -0.269 \cdot [1, 2, 3] = [-0.269, -0.538, -0.807]$$

**E. Weight update:**
$$w_{new} = w - \eta \nabla\ell = [0.5, -0.2, 0.3] - 0.5[-0.269, -0.538, -0.807]$$
$$w_{new} = [0.5 + 0.135, -0.2 + 0.269, 0.3 + 0.404] = [0.635, 0.069, 0.704]$$

---

## Problem C2: Confusion Matrix Metrics [3 marks]

**Question:** A COVID-19 test gives: TP=85, FP=10, FN=15, TN=890

A. Calculate accuracy, precision, and recall.
B. Is high recall more important for this test? Explain.

### Solution C2:

**A. Metrics:**
$$\text{Accuracy} = \frac{85 + 890}{85 + 10 + 15 + 890} = \frac{975}{1000} = 97.5\%$$

$$\text{Precision} = \frac{85}{85 + 10} = \frac{85}{95} = 89.5\%$$

$$\text{Recall} = \frac{85}{85 + 15} = \frac{85}{100} = 85\%$$

**B. Importance of Recall:**
- Yes, high recall is critical for COVID-19 testing
- Missing positive cases (FN) is dangerous - infected patients go untreated and spread disease
- False positives (FP) are inconvenient but not life-threatening
- We want to minimize false negatives, which means maximizing recall

---

# Section D: Multi-class Classification (20 marks typical)

## Problem D1: Softmax and Cross-Entropy [5 marks]

**Question:** A 3-class classifier outputs logits:
$$z = [2.0, 1.0, 0.5]$$
True label is class 0 (one-hot: [1, 0, 0])

A. Compute softmax probabilities (round to 3 decimal places)
B. Compute the categorical cross-entropy loss
C. Is the model's prediction correct?

### Solution D1:

**A. Softmax:**
$$e^{2.0} = 7.389, \quad e^{1.0} = 2.718, \quad e^{0.5} = 1.649$$
$$\text{Sum} = 7.389 + 2.718 + 1.649 = 11.756$$
$$\hat{y} = \left[\frac{7.389}{11.756}, \frac{2.718}{11.756}, \frac{1.649}{11.756}\right] = [0.628, 0.231, 0.140]$$

**B. Cross-entropy:**
$$\ell = -\sum_{k=1}^{3} y_k \log(\hat{y}_k) = -[1 \cdot \log(0.628) + 0 + 0]$$
$$\ell = -\log(0.628) = 0.465$$

**C. Model prediction:**
- Predicted class = argmax([0.628, 0.231, 0.140]) = class 0
- True class = 0
- **Prediction is CORRECT** ✓

---

## Problem D2: Multi-class Code Completion [5 marks]

```python
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Stability trick
    return ________  # (1)

def categorical_cross_entropy(y_pred, y_true):
    # y_true is one-hot encoded
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1-eps)
    return ________  # (2)

def train_step(X, y, W, b, learning_rate):
    # Forward pass
    z = ________ + b  # (3)
    y_pred = softmax(z)
    
    # Compute gradient (output layer)
    dz = ________  # (4)
    dW = ________  # (5)
    db = np.mean(dz, axis=0)
    
    # Update
    W -= learning_rate * dW
    b -= learning_rate * db
    return W, b
```

### Solution D2:

```python
def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)  # (1)

def categorical_cross_entropy(y_pred, y_true):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1-eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))  # (2)

def train_step(X, y, W, b, learning_rate):
    z = X @ W + b  # (3)
    y_pred = softmax(z)
    
    dz = (y_pred - y) / len(X)  # (4) - normalized error
    dW = X.T @ dz  # (5)
    db = np.mean(dz, axis=0)
    
    W -= learning_rate * dW
    b -= learning_rate * db
    return W, b
```

---

# Section E: Deep Feedforward Networks (20 marks typical)

## Problem E1: Forward Propagation [6 marks]

**Question:** Consider a 2-layer DFNN with:
- Input: x = [0.5, 0.8]ᵀ
- Hidden layer: 2 neurons, ReLU activation
- Output layer: 1 neuron, Sigmoid activation

**Parameters:**
$$W^{(1)} = \begin{bmatrix} 0.3 & -0.2 \\ 0.4 & 0.5 \end{bmatrix}, \quad b^{(1)} = \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix}$$

$$W^{(2)} = \begin{bmatrix} 0.6 \\ 0.7 \end{bmatrix}, \quad b^{(2)} = [0.2]$$

Compute the output step by step.

### Solution E1:

**Layer 1 - Pre-activation:**
$$z^{(1)} = W^{(1)T}x + b^{(1)}$$
$$z^{(1)} = \begin{bmatrix} 0.3 & 0.4 \\ -0.2 & 0.5 \end{bmatrix} \begin{bmatrix} 0.5 \\ 0.8 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix}$$
$$z^{(1)} = \begin{bmatrix} 0.15 + 0.32 \\ -0.10 + 0.40 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix} = \begin{bmatrix} 0.57 \\ 0.20 \end{bmatrix}$$

**Layer 1 - ReLU Activation:**
$$h^{(1)} = \text{ReLU}(z^{(1)}) = \begin{bmatrix} \max(0, 0.57) \\ \max(0, 0.20) \end{bmatrix} = \begin{bmatrix} 0.57 \\ 0.20 \end{bmatrix}$$

**Layer 2 - Pre-activation:**
$$z^{(2)} = W^{(2)T}h^{(1)} + b^{(2)}$$
$$z^{(2)} = [0.6 \cdot 0.57 + 0.7 \cdot 0.20] + 0.2 = 0.342 + 0.14 + 0.2 = 0.682$$

**Layer 2 - Sigmoid Activation:**
$$\hat{y} = \sigma(0.682) = \frac{1}{1 + e^{-0.682}} = \frac{1}{1.505} = 0.664$$

**Final output: ŷ = 0.664**

---

## Problem E2: Compute Parameters [3 marks]

**Question:** A network has:
- 8 input neurons
- Hidden layer 1: 10 neurons
- Hidden layer 2: 6 neurons  
- Output: 4 neurons
- No biases included

Calculate total trainable weights.

### Solution E2:

$$\text{Weights} = (8 \times 10) + (10 \times 6) + (6 \times 4)$$
$$\text{Weights} = 80 + 60 + 24 = 164$$

**Total trainable weights: 164**

---

## Problem E3: DFNN Design [6 marks]

**Question:** Design a Deep Feedforward Neural Network for sentiment classification with 3 classes (positive, negative, neutral). The input is a 100-dimensional word embedding vector.

Specify:
A. Number of layers and neurons per layer
B. Activation functions for each layer
C. Output layer configuration
D. Appropriate loss function

### Solution E3:

**A. Architecture:**
```
Input Layer: 100 neurons (word embedding)
Hidden Layer 1: 64 neurons
Hidden Layer 2: 32 neurons
Output Layer: 3 neurons (one per class)
```

**B. Activation Functions:**
- Hidden Layer 1: ReLU (efficient, avoids vanishing gradients)
- Hidden Layer 2: ReLU
- Output Layer: Softmax (for probability distribution over 3 classes)

**C. Output Layer Configuration:**
- 3 neurons with Softmax activation
- Each neuron outputs probability for one class
- Probabilities sum to 1

**D. Loss Function:**
- **Categorical Cross-Entropy** (for multi-class classification)
- If using sparse labels (not one-hot): Sparse Categorical Cross-Entropy

---

## Problem E4: Activation Function Comparison [5 marks]

**Question:**
A. A deep network for character recognition uses tanh in hidden layers and trains slowly. Why? [2 marks]
B. Why does switching to ReLU improve convergence? [2 marks]
C. When would sigmoid still be preferred? [1 mark]

### Solution E4:

**A. Why tanh causes slow training:**
- **Vanishing gradient problem**: tanh(z) ∈ (-1, 1), and derivative tanh'(z) = 1 - tanh²(z)
- For |z| > 2, gradient becomes very small (near 0)
- In deep networks, small gradients multiply → gradients vanish
- Early layers receive almost no gradient → weights don't update

**B. Why ReLU improves convergence:**
- ReLU'(z) = 1 for z > 0, maintaining gradient flow
- No vanishing gradient for positive activations
- Computationally efficient (just max operation)
- Creates sparse activations (some neurons output 0)
- Gradients flow directly for active neurons

**C. When sigmoid is preferred:**
- **Binary classification output layer** - outputs probability in [0,1]
- When output must represent probability
- Not recommended for hidden layers due to vanishing gradients

---

# Section F: Convolutional Neural Networks (20 marks typical)

## Problem F1: Convolution Computation [4 marks]

**Question:** Given a 5×5 input and 3×3 kernel:

```
Input:                  Kernel:
1  2  3  4  5          1  0 -1
2  3  4  5  6          1  0 -1
3  4  5  6  7          1  0 -1
4  5  6  7  8
5  6  7  8  9
```

Compute the top-left value of the output feature map (valid convolution, stride=1).

### Solution F1:

Top-left 3×3 region of input:
```
1  2  3
2  3  4
3  4  5
```

Element-wise multiplication and sum:
$$\text{output}[0,0] = 1(1) + 2(0) + 3(-1) + 2(1) + 3(0) + 4(-1) + 3(1) + 4(0) + 5(-1)$$
$$= 1 - 3 + 2 - 4 + 3 - 5 = -6$$

---

## Problem F2: Output Size Calculation [3 marks]

**Question:** Input image: 64×64×3. Apply:
1. Conv layer: 32 filters of 5×5, stride=1, padding=2
2. Max pooling: 2×2, stride=2
3. Conv layer: 64 filters of 3×3, stride=1, padding=1

What's the output size after each layer?

### Solution F2:

**After Conv1:**
$$\text{size} = \left\lfloor\frac{64 + 2(2) - 5}{1}\right\rfloor + 1 = 64$$
Output: **64×64×32**

**After MaxPool:**
$$\text{size} = \left\lfloor\frac{64 - 2}{2}\right\rfloor + 1 = 32$$
Output: **32×32×32**

**After Conv2:**
$$\text{size} = \left\lfloor\frac{32 + 2(1) - 3}{1}\right\rfloor + 1 = 32$$
Output: **32×32×64**

---

## Problem F3: CNN Architectures [6 marks]

**Question:**
A. What is the key innovation in ResNet? Why does it help? [2 marks]
B. How does VGG differ from AlexNet in its use of convolutions? [2 marks]
C. What is transfer learning? When should you use fine-tuning vs feature extraction? [2 marks]

### Solution F3:

**A. ResNet Innovation:**
- **Skip connections (residual connections)**: Output = F(x) + x
- Instead of learning H(x), network learns residual F(x) = H(x) - x
- Benefits:
  - Gradients flow directly through skip connections
  - Enables training very deep networks (100+ layers)
  - If identity is optimal, F(x) = 0 is easy to learn

**B. VGG vs AlexNet:**
- AlexNet uses large kernels (11×11, 5×5)
- VGG uses only 3×3 kernels throughout
- Two 3×3 convs have same receptive field as one 5×5, but:
  - Fewer parameters: 2(3×3) = 18 vs 5×5 = 25
  - More non-linearities (two ReLUs vs one)
  - Better feature learning with same receptive field

**C. Transfer Learning:**
- Using pre-trained model (trained on large dataset) for new task
- **Feature Extraction**: Freeze pre-trained layers, only train new classifier
  - Use when: Small dataset, similar domain
- **Fine-tuning**: Unfreeze some/all layers, train with small learning rate
  - Use when: Larger dataset or different domain

---

## Problem F4: CNN Code [5 marks]

**Question:** Complete the Keras CNN code for MNIST:

```python
model = Sequential([
    Conv2D(_______, kernel_size=_______, activation='relu', 
           input_shape=(28, 28, 1)),  # (1), (2)
    MaxPooling2D(pool_size=_______),  # (3)
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    _______(),  # (4) - flatten
    Dense(128, activation='relu'),
    Dense(_______, activation=_______)  # (5) - output layer
])
```

### Solution F4:

```python
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu',  # (1) 32, (2) (3, 3)
           input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),  # (3) (2, 2)
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),  # (4)
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # (5) 10, 'softmax'
])
```

---

# Section G: Conceptual Questions

## Problem G1: Overfitting and Generalization [4 marks]

**Question:** You train a network on 20 samples with high training loss, then train on 10,000 samples. Is this the right approach? Explain.

### Solution G1:

**No, this approach is incorrect.**

**Problem Analysis:**
- High training loss on 20 samples indicates model can't even fit training data
- This is **underfitting**, not overfitting
- Adding more data won't help - model lacks capacity or has other issues

**Correct Solutions:**
1. Increase model complexity (more layers/neurons)
2. Decrease regularization
3. Check for bugs in code
4. Use different architecture
5. Ensure data is properly preprocessed

**When more data helps:** When training loss is low but validation loss is high (overfitting)

---

## Problem G2: Weight Initialization [3 marks]

**Question:** Why is proper weight initialization important? Name one method.

### Solution G2:

**Importance:**
- Poor initialization → vanishing/exploding gradients
- Weights too small → signals shrink, gradients vanish
- Weights too large → signals explode, training unstable
- Symmetric initialization → neurons learn same features

**Methods:**
- **Xavier/Glorot**: W ~ N(0, 1/n_in) - good for tanh/sigmoid
- **He**: W ~ N(0, 2/n_in) - good for ReLU
- **Random small values**: W ~ N(0, 0.01)

---

## Problem G3: Data Shuffling [3 marks]

**Question:** Training set has all dog images first, then all cat images. Should you shuffle before training? Why?

### Solution G3:

**Yes, shuffling is necessary.**

**Reasons:**
1. Without shuffling, model sees only dogs for first half of each epoch
2. Gradients will be biased towards one class at a time
3. Model may "forget" previous class before seeing it again
4. Mini-batches should have diverse examples for stable gradients

**Best Practice:**
- Shuffle at the beginning of each epoch
- Ensures each mini-batch has balanced class representation

---

## Problem G4: Regularization [3 marks]

**Question:** Training accuracy keeps improving but validation accuracy drops after few epochs on a small medical dataset (500 samples). Suggest a technique and justify.

### Solution G4:

**Problem:** Classic overfitting - model memorizes training data

**Suggested Technique: Dropout**

**Justification:**
- Randomly drops neurons during training
- Prevents co-adaptation of neurons
- Acts as ensemble of sub-networks
- Very effective for small datasets
- Easy to implement (add Dropout layers)

**Alternatives:**
- Data augmentation (for images)
- Early stopping
- Weight decay (L2 regularization)
- Reduce model size

---

# Answer Key Summary

| Problem | Key Answer Points |
|---------|-------------------|
| A1 | Final weights: w₀=2, w₁=2, w₂=2 for OR gate |
| A2 | XOR not linearly separable, MLP adds hidden layers + non-linearity |
| B1 | Gradient: [-4.67, -10, -6.67], Update shows weight changes |
| B2 | Model B (RMSE=4500) < Model A (RMSE=5000) |
| C1 | z=1.0, ŷ=0.731, loss=0.313 |
| C2 | Accuracy=97.5%, Precision=89.5%, Recall=85% |
| D1 | ŷ=[0.628, 0.231, 0.140], loss=0.465 |
| E1 | Final output: ŷ = 0.664 |
| E2 | Total weights: 164 |
| F1 | Top-left output: -6 |
| F2 | 64×64×32 → 32×32×32 → 32×32×64 |
| G1 | High train loss = underfitting, need more capacity not data |
| G3 | Must shuffle to ensure balanced mini-batches |
