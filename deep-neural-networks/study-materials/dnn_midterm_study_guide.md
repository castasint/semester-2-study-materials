# 📚 Deep Neural Networks (AIMLCZG511) - Midterm Study Guide

> **Comprehensive study material for Sessions 1-8 | Closed Book Exam | 30% Weightage | 2 Hours**

---

## 📋 Table of Contents

1. [Session 1: Introduction & Fundamentals](#session-1-introduction--fundamentals)
2. [Session 2: Perceptron](#session-2-perceptron)
3. [Session 3: Linear NN for Regression](#session-3-linear-nn-for-regression)
4. [Session 4: Linear NN for Classification](#session-4-linear-nn-for-classification)
5. [Session 5: Deep Feedforward Neural Networks (DFNN/MLP)](#session-5-deep-feedforward-neural-networks)
6. [Sessions 6-8: Convolutional Neural Networks (CNNs)](#sessions-6-8-convolutional-neural-networks)
7. [Essential Formulas Quick Reference](#essential-formulas-quick-reference)
8. [Practice Problems with Solutions](#practice-problems-with-solutions)
9. [Python Code Templates](#python-code-templates)

---

# Session 1: Introduction & Fundamentals

## 1.1 What is Deep Learning?

### Definition
Deep Learning is a subset of Machine Learning that uses neural networks with multiple layers (depth) to learn hierarchical representations of data.

### Why Deep Learning?
| Traditional ML | Deep Learning |
|---------------|---------------|
| Requires manual feature engineering | Automatic feature extraction |
| Limited by feature quality | Learns complex patterns |
| Works well on small datasets | Excels with large datasets |
| Interpretable | Often "black box" |

### Key Applications
- **Computer Vision**: Image classification, object detection, face recognition
- **Natural Language Processing**: Translation, sentiment analysis, chatbots
- **Speech Recognition**: Wake word detection, voice assistants
- **Healthcare**: Medical image diagnosis, drug discovery

## 1.2 Neural Network Fundamentals

### Biological vs Artificial Neurons

| Biological Neuron | Artificial Neuron |
|-------------------|-------------------|
| Dendrites receive signals | Inputs (x₁, x₂, ..., xₙ) |
| Cell body processes signals | Weighted sum computation |
| Axon transmits output | Activation function output |
| Synapses connect neurons | Weights (w₁, w₂, ..., wₙ) |
| ~10¹⁰ neurons in brain | Configurable architecture |

### Artificial Neuron Mathematical Model

```
           x₁ ──w₁──┐
           x₂ ──w₂──┼──▶ Σ ──▶ f(z) ──▶ ŷ
           x₃ ──w₃──┤
            1 ──w₀──┘ (bias)

z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = wᵀx + b
ŷ = f(z)  where f is the activation function
```

### The Four Components of Machine Learning

1. **Data**: Training examples with features and labels
2. **Model**: Neural network architecture
3. **Objective Function**: Loss/cost function to minimize
4. **Optimization Algorithm**: Method to update weights (e.g., gradient descent)

## 1.3 Key Concepts

### Supervised Learning
- Training with labeled data pairs (x, y)
- Goal: Learn mapping f: X → Y
- Examples: Classification, Regression

### Training Pipeline
1. Forward propagation (compute predictions)
2. Calculate loss
3. Backward propagation (compute gradients)
4. Update weights

---

# Session 2: Perceptron

## 2.1 Perceptron Basics

### Definition
The **Perceptron** is the simplest form of artificial neuron, introduced by Rosenblatt in 1958. It's a binary classifier that makes predictions based on a linear combination of inputs.

### Architecture

```
        x₁ ──w₁──┐
        x₂ ──w₂──┼──▶ Σ ──▶ sign(z) ──▶ ŷ ∈ {-1, +1} or {0, 1}
         1 ──w₀──┘ (bias)
```

### Mathematical Formulation

**Weighted Sum:**
$$z = w_0 + w_1x_1 + w_2x_2 + \ldots + w_nx_n = \mathbf{w}^T\mathbf{x}$$

**Activation Function (Sign/Step):**
$$\hat{y} = \text{sign}(z) = \begin{cases} +1 & \text{if } z \geq 0 \\ -1 & \text{if } z < 0 \end{cases}$$

Or with binary outputs:
$$\hat{y} = h(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}$$

## 2.2 Perceptron Learning Algorithm

### Algorithm Steps

```
1. Initialize weights: w₀ = w₁ = w₂ = ... = wₙ = 0
2. For each training example (x, t):
   a. Compute output: z = Σ wᵢxᵢ
   b. Apply activation: ŷ = sign(z)
   c. If ŷ ≠ t (prediction is wrong):
      Update weights: wᵢ ← wᵢ + η(t - ŷ)xᵢ
3. Repeat until convergence or max epochs
```

### Weight Update Rule

$$\Delta w_i = \eta (t - \hat{y}) x_i$$
$$w_i \leftarrow w_i + \Delta w_i$$

Where:
- η = learning rate (typically 0.1 to 1)
- t = target/expected output
- ŷ = predicted output
- xᵢ = input feature (x₀ = 1 for bias)

## 2.3 Perceptron for Logic Gates

### AND Gate (Binary: 0, 1)

| x₁ | x₂ | t (AND) |
|----|----|----|
| 0  | 0  | 0  |
| 0  | 1  | 0  |
| 1  | 0  | 0  |
| 1  | 1  | 1  |

**Possible weights:** w₀ = -1.5, w₁ = 1, w₂ = 1

### OR Gate (Binary: 0, 1)

| x₁ | x₂ | t (OR) |
|----|----|----|
| 0  | 0  | 0  |
| 0  | 1  | 1  |
| 1  | 0  | 1  |
| 1  | 1  | 1  |

**Possible weights:** w₀ = -0.5, w₁ = 1, w₂ = 1

### NAND Gate (Bipolar: -1, +1)

| x₁ | x₂ | t (NAND) |
|----|----|----------|
| -1 | -1 | 1        |
| -1 | 1  | 1        |
| 1  | -1 | 1        |
| 1  | 1  | -1       |

## 2.4 Worked Example: NAND Gate (Bipolar)

**Given:**
- Initial weights: w₀ = w₁ = w₂ = 0
- Learning rate: η = 1
- Bias input: x₀ = 1

**Epoch 1:**

| Step | x₀ | x₁ | x₂ | t | z = Σwᵢxᵢ | ŷ = sign(z) | Error (t-ŷ) | Δw₀ | Δw₁ | Δw₂ | New w₀ | New w₁ | New w₂ |
|------|----|----|----|----|-----------|-------------|-------------|-----|-----|-----|--------|--------|--------|
| 1 | 1 | -1 | -1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 1 | -1 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 1 | 1 | -1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 1 | 1 | 1 | -1 | 0 | 1 | -2 | -2 | -2 | -2 | -2 | -2 | -2 |

**Final weights after Epoch 1:** w₀ = -2, w₁ = -2, w₂ = -2

**Verification:**
- (-1,-1): z = -2 + 2 + 2 = 2 > 0 → ŷ = +1 ✓
- (-1,+1): z = -2 + 2 - 2 = -2 < 0 → ŷ = -1 ✗ (needs more epochs)

## 2.5 Linear Separability

### Definition
Two classes are **linearly separable** if they can be separated by a hyperplane (line in 2D, plane in 3D).

### Perceptron Convergence Theorem
> If training data is linearly separable, the perceptron learning algorithm will converge in finite steps.

### XOR Problem - Why Perceptron Fails

| x₁ | x₂ | XOR |
|----|----|-----|
| 0  | 0  | 0   |
| 0  | 1  | 1   |
| 1  | 0  | 1   |
| 1  | 1  | 0   |

**XOR is NOT linearly separable!** No single line can separate the classes.

```
  x₂
  1 │  •(1)     ○(0)
    │
  0 │  ○(0)     •(1)
    └──────────────── x₁
       0        1
```

### Solution: Multi-Layer Perceptron (MLP)
- XOR can be represented as: XOR = (A AND NOT B) OR (NOT A AND B)
- Requires hidden layer with non-linear activation
- For XOR with single hidden layer: 2 perceptrons in hidden layer
- For n-input XOR function with shallow network: n perceptrons needed
- For n-input XOR function with deep network: O(log n) perceptrons

---

# Session 3: Linear NN for Regression

## 3.1 Regression Problem

### Definition
Regression predicts **continuous** values (how much? how many?)

### Examples
- House price prediction based on features
- Temperature forecasting
- Stock price prediction
- Energy consumption estimation

## 3.2 Linear Regression Model

### Single Neuron Architecture

```
        x₁ ──w₁──┐
        x₂ ──w₂──┼──▶ Σ ──▶ f(z)=z ──▶ ŷ ∈ ℝ
        ...      │
        xd ──wd──┤
         1 ──w₀──┘ (bias)
```

### Mathematical Formulation

**Prediction:**
$$\hat{y} = f(\mathbf{w}^T\mathbf{x}) = w_0 + w_1x_1 + w_2x_2 + \ldots + w_dx_d$$

**Identity Activation Function:**
$$f(z) = z$$

Properties:
- Output range: (-∞, +∞)
- Derivative: f'(z) = 1
- Allows predicting any real value

### Matrix Form

**Design Matrix:**
$$\mathbf{X} = \begin{bmatrix} 1 & x_1^{(1)} & \ldots & x_d^{(1)} \\ 1 & x_1^{(2)} & \ldots & x_d^{(2)} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_1^{(N)} & \ldots & x_d^{(N)} \end{bmatrix} \in \mathbb{R}^{N \times (d+1)}$$

**Vectorized Prediction:**
$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$$

## 3.3 Loss Function: Mean Squared Error (MSE)

### Per-Example Loss
$$\ell(\mathbf{w}; \mathbf{x}^{(i)}, y^{(i)}) = \frac{1}{2}(\hat{y}^{(i)} - y^{(i)})^2 = \frac{1}{2}(\mathbf{w}^T\mathbf{x}^{(i)} - y^{(i)})^2$$

### Total Loss (Cost Function)
$$J(\mathbf{w}) = \frac{1}{N}\sum_{i=1}^{N} \frac{1}{2}(\hat{y}^{(i)} - y^{(i)})^2 = \frac{1}{2N}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$$

### Why MSE?
- Differentiable everywhere
- Convex (single global minimum for linear models)
- Penalizes large errors more (quadratic growth)
- Maximum likelihood under Gaussian noise assumption

## 3.4 Gradient Descent

### Algorithm

1. Initialize weights: w⁽⁰⁾ = 0 (or random)
2. Repeat until convergence:
   - Compute gradient: ∇J(w⁽ᵗ⁾)
   - Update: w⁽ᵗ⁺¹⁾ = w⁽ᵗ⁾ - η∇J(w⁽ᵗ⁾)

### Gradient Computation

**For single example:**
$$\nabla_{\mathbf{w}} \ell = (\hat{y}^{(i)} - y^{(i)}) \mathbf{x}^{(i)}$$

**For batch (all examples):**
$$\nabla J(\mathbf{w}) = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}^{(i)} - y^{(i)})\mathbf{x}^{(i)} = \frac{1}{N}\mathbf{X}^T(\mathbf{X}\mathbf{w} - \mathbf{y})$$

### Update Rule
$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \frac{\eta}{N}\mathbf{X}^T(\mathbf{X}\mathbf{w}^{(t)} - \mathbf{y})$$

## 3.5 Worked Example: House Price Prediction

**Dataset:**
| Size (x₁) | Price (y) |
|-----------|-----------|
| 1 | 2 |
| 2 | 4 |
| 3 | 5 |

**Setup:**
- η = 0.1, N = 3
- X = [[1,1], [1,2], [1,3]], y = [2, 4, 5]ᵀ
- w⁽⁰⁾ = [0, 0]ᵀ

**Iteration 0:**
- Predictions: ŷ⁽⁰⁾ = Xw⁽⁰⁾ = [0, 0, 0]ᵀ
- Error: e = ŷ - y = [-2, -4, -5]ᵀ
- Loss: J = (1/6)(4 + 16 + 25) = 7.5
- Gradient: ∇J = (1/3)Xᵀe = (1/3)[−11, −23]ᵀ = [−3.67, −7.67]ᵀ
- Update: w⁽¹⁾ = [0, 0]ᵀ - 0.1[−3.67, −7.67]ᵀ = [0.367, 0.767]ᵀ

## 3.6 Evaluation Metrics for Regression

### Mean Squared Error (MSE)
$$MSE = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}^{(i)} - y^{(i)})^2$$

### Root Mean Squared Error (RMSE)
$$RMSE = \sqrt{MSE}$$
- Same units as target variable
- More interpretable

### Mean Absolute Error (MAE)
$$MAE = \frac{1}{N}\sum_{i=1}^{N}|\hat{y}^{(i)} - y^{(i)}|$$

### R² (Coefficient of Determination)
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(\hat{y}^{(i)} - y^{(i)})^2}{\sum(y^{(i)} - \bar{y})^2}$$
- R² = 1: Perfect fit
- R² = 0: Model is as good as predicting mean
- R² < 0: Model is worse than predicting mean

---

# Session 4: Linear NN for Classification

## 4.1 Classification Problem

### Definition
Classification predicts **discrete** categories (which class?)

### Types
1. **Binary Classification**: 2 classes (e.g., spam/not spam, positive/negative)
2. **Multi-class Classification**: K > 2 classes (e.g., digits 0-9, animal species)

## 4.2 Binary Classification (Logistic Regression)

### Architecture

```
        x₁ ──w₁──┐
        x₂ ──w₂──┼──▶ Σ ──▶ σ(z) ──▶ ŷ ∈ [0,1]
        ...      │         (sigmoid)
        xd ──wd──┤
         1 ──w₀──┘
```

### Sigmoid Activation Function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Properties:**
- Output range: (0, 1) → interpretable as probability
- σ(0) = 0.5
- Derivative: σ'(z) = σ(z)(1 - σ(z))

### Prediction

**Probability:**
$$\hat{y} = P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T\mathbf{x}}}$$

**Decision Rule:**
$$\text{Predicted class} = \begin{cases} 1 & \text{if } \hat{y} \geq 0.5 \\ 0 & \text{if } \hat{y} < 0.5 \end{cases}$$

## 4.3 Binary Cross-Entropy Loss

### Per-Example Loss
$$\ell(\mathbf{w}; \mathbf{x}^{(i)}, y^{(i)}) = -[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})]$$

### Total Loss
$$J(\mathbf{w}) = -\frac{1}{N}\sum_{i=1}^{N}[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})]$$

### Why Cross-Entropy?
- Heavily penalizes confident wrong predictions
- Convex for logistic regression
- Derived from maximum likelihood estimation

### Gradient
$$\nabla_{\mathbf{w}} J = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}^{(i)} - y^{(i)})\mathbf{x}^{(i)}$$

**Same elegant form as linear regression!**

## 4.4 SGD vs Batch Gradient Descent

| Aspect | Batch GD | SGD | Mini-batch SGD |
|--------|----------|-----|----------------|
| Examples per update | All N | 1 | B (e.g., 32) |
| Speed per iteration | Slow | Fast | Medium |
| Memory | High | Low | Medium |
| Convergence | Smooth | Noisy | Balanced |
| Generalization | - | Better | Good |

### Mini-Batch SGD Algorithm

```
for epoch = 1 to T:
    Shuffle dataset
    for each mini-batch B of size B:
        1. Forward pass: ŷ = σ(Xw)
        2. Compute error: e = ŷ - y
        3. Compute gradient: ∇J = (1/B)XᵀE
        4. Update weights: w ← w - η∇J
```

## 4.5 Worked Example: Binary Classification

**Dataset:** Study hours vs Pass/Fail
| Hours (x₁) | Pass (y) |
|------------|----------|
| 1 | 0 |
| 2 | 0 |
| 3 | 1 |
| 4 | 1 |

**Setup:** η = 0.5, w⁽⁰⁾ = [0, 0]ᵀ

**SGD Iteration 1 (example 1: x=[1,1], y=0):**
- z = [1, 1] · [0, 0]ᵀ = 0
- ŷ = σ(0) = 0.5
- Error = 0.5 - 0 = 0.5
- Gradient = 0.5 · [1, 1]ᵀ = [0.5, 0.5]ᵀ
- w⁽¹⁾ = [0, 0]ᵀ - 0.5[0.5, 0.5]ᵀ = [-0.25, -0.25]ᵀ

## 4.6 Multi-class Classification (Softmax)

### Architecture

```
        x₁ ──┐
        x₂ ──┼──▶ [Multiple Neurons] ──▶ Softmax ──▶ [ŷ₁, ŷ₂, ..., ŷₖ]
        ...  │
        xd ──┘
```

Each of K output neurons computes: zₖ = wₖᵀx

### Softmax Activation

$$\hat{y}_k = \text{softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}$$

**Properties:**
- Outputs sum to 1: Σŷₖ = 1
- All outputs in (0, 1)
- Interpretable as probabilities

### Categorical Cross-Entropy Loss

$$\ell = -\sum_{k=1}^{K}y_k\log(\hat{y}_k)$$

Where y is one-hot encoded:
- If true class is k, then y = [0, ..., 0, 1, 0, ..., 0] (1 at position k)

### Total Loss
$$J(\mathbf{W}) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}y_k^{(i)}\log(\hat{y}_k^{(i)})$$

## 4.7 Worked Example: Softmax

**Given:** Logits z = [2.5, 0.3, -1.2, 3.1, 0.7] for 5 classes

**Step 1: Compute exponentials**
- e^2.5 = 12.18
- e^0.3 = 1.35
- e^-1.2 = 0.30
- e^3.1 = 22.20
- e^0.7 = 2.01

**Step 2: Sum**
- Σ = 12.18 + 1.35 + 0.30 + 22.20 + 2.01 = 38.04

**Step 3: Softmax probabilities**
- ŷ = [12.18/38.04, 1.35/38.04, 0.30/38.04, 22.20/38.04, 2.01/38.04]
- ŷ ≈ [0.32, 0.04, 0.01, 0.58, 0.05]

**If true class is "bird" (class 4):**
- Cross-entropy loss = -log(0.58) ≈ 0.54

## 4.8 Evaluation Metrics for Classification

### Confusion Matrix

```
                    Predicted
                  Pos    Neg
Actual   Pos      TP     FN
         Neg      FP     TN
```

### Key Metrics

**Accuracy:**
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision (Positive Predictive Value):**
$$\text{Precision} = \frac{TP}{TP + FP}$$
- "Of all predicted positive, how many are actually positive?"
- High precision → few false positives

**Recall (Sensitivity, True Positive Rate):**
$$\text{Recall} = \frac{TP}{TP + FN}$$
- "Of all actual positive, how many did we catch?"
- High recall → few false negatives

**F1 Score:**
$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### When to Use What?
- **High Precision needed:** Spam detection (don't mark good emails as spam)
- **High Recall needed:** Disease detection (don't miss any sick patients)

---

# Session 5: Deep Feedforward Neural Networks

## 5.1 Motivation: The XOR Problem

Linear models cannot solve XOR because it's not linearly separable.

**Solution:** Add hidden layers with non-linear activations!

## 5.2 DFNN Architecture

```
x₁ ─┐       ┌─ h₁ ─┐       ┌─ ŷ₁
x₂ ─┼───────┼─ h₂ ─┼───────┼─ ŷ₂
... ┤       ├─ ... ┤       ├─ ...
xd ─┘       └─ hₙ ─┘       └─ ŷₖ

Input     Hidden Layer    Output
(d units)  (n units)      (K units)
```

### Notation
- L: Number of layers (excluding input)
- nₗ: Number of units in layer ℓ
- d = n₀: Input dimension
- K = nₗ: Output dimension
- W⁽ℓ⁾ ∈ ℝ^(nₗ₋₁ × nₗ): Weight matrix for layer ℓ
- b⁽ℓ⁾ ∈ ℝ^nₗ: Bias vector for layer ℓ

## 5.3 Forward Propagation

### Layer-by-Layer Computation

For each layer ℓ = 1, 2, ..., L:

**Initial condition:**
$$\mathbf{h}^{(0)} = \mathbf{x}$$

**Pre-activation (linear transformation):**
$$\mathbf{z}^{(\ell)} = \mathbf{h}^{(\ell-1)}\mathbf{W}^{(\ell)} + \mathbf{b}^{(\ell)}$$

**Activation (non-linear transformation):**
$$\mathbf{h}^{(\ell)} = \sigma^{(\ell)}(\mathbf{z}^{(\ell)})$$

**Final output:**
$$\hat{\mathbf{y}} = \mathbf{h}^{(L)}$$

### Vectorized for Mini-Batch
For mini-batch X_B ∈ ℝ^(B×d):

$$\mathbf{H}^{(0)} = \mathbf{X}_B$$
$$\mathbf{Z}^{(\ell)} = \mathbf{H}^{(\ell-1)}\mathbf{W}^{(\ell)} + \mathbf{1}_B\mathbf{b}^{(\ell)T}$$
$$\mathbf{H}^{(\ell)} = \sigma^{(\ell)}(\mathbf{Z}^{(\ell)})$$
$$\hat{\mathbf{Y}}_B = \mathbf{H}^{(L)}$$

## 5.4 Activation Functions

### ReLU (Rectified Linear Unit)
$$\text{ReLU}(z) = \max(0, z)$$
$$\text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}$$

**Pros:**
- Computationally efficient
- No vanishing gradient for positive inputs
- Sparse activation

**Cons:**
- "Dead neurons" for z < 0 (gradient = 0)

### Sigmoid
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

**Pros:** Output in (0,1), good for probabilities

**Cons:** Vanishing gradient, slow convergence

### Tanh
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
$$\tanh'(z) = 1 - \tanh^2(z)$$

**Pros:** Zero-centered output (-1, 1)

**Cons:** Still suffers from vanishing gradients

### Which to Use?
- **Hidden layers:** ReLU (or variants like Leaky ReLU)
- **Binary classification output:** Sigmoid
- **Multi-class output:** Softmax
- **Regression output:** Linear (identity)

## 5.5 Backward Propagation

### Key Insight
Use chain rule to compute gradients layer by layer, from output to input.

### Error Signal (δ)
$$\delta^{(\ell)} = \frac{\partial J}{\partial \mathbf{Z}^{(\ell)}}$$

### Output Layer Gradient
For common loss-activation pairs (MSE+Identity, BCE+Sigmoid, CCE+Softmax):
$$\delta^{(L)} = \frac{1}{B}(\hat{\mathbf{Y}} - \mathbf{Y})$$

### Backpropagation Through Hidden Layers
$$\delta^{(\ell)} = (\delta^{(\ell+1)}\mathbf{W}^{(\ell+1)T}) \odot \sigma'^{(\ell)}(\mathbf{Z}^{(\ell)})$$

Where ⊙ is element-wise multiplication.

### Parameter Gradients
$$\frac{\partial J}{\partial \mathbf{W}^{(\ell)}} = \frac{1}{B}\mathbf{H}^{(\ell-1)T}\delta^{(\ell)}$$
$$\frac{\partial J}{\partial \mathbf{b}^{(\ell)}} = \frac{1}{B}\mathbf{1}_B^T\delta^{(\ell)}$$

## 5.6 Worked Example: 2-Layer DFNN

**Architecture:** Input(2) → Hidden(3, ReLU) → Output(2, Sigmoid)

**Given:**
- x = [1.0, 0.5]ᵀ
- W⁽¹⁾ = [[0.5, 0.2, -0.3], [0.8, 0.1, -0.4]]
- b⁽¹⁾ = [0.1, -0.2, 0.3]
- W⁽²⁾ = [[0.4, -0.1], [0.6, 0.2], [0.7, -0.3]]
- b⁽²⁾ = [0.2, -0.1]
- y = [1, 0]ᵀ (target)

**Forward Pass:**

**Layer 1:**
$$\mathbf{z}^{(1)} = \mathbf{x}^T\mathbf{W}^{(1)} + \mathbf{b}^{(1)} = [0.7, -0.1, 0.2]$$
$$\mathbf{h}^{(1)} = \text{ReLU}(\mathbf{z}^{(1)}) = [0.7, 0.0, 0.2]$$

**Layer 2:**
$$\mathbf{z}^{(2)} = \mathbf{h}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = [0.6, -0.02]$$
$$\hat{\mathbf{y}} = \sigma(\mathbf{z}^{(2)}) = [0.646, 0.495]$$

**Loss (BCE):**
$$J = -\frac{1}{2}[1 \cdot \log(0.646) + 0 + 0 + 1 \cdot \log(0.505)] = 0.56$$

## 5.7 Computing Total Parameters

**Formula for layer ℓ:**
$$\text{Parameters}^{(\ell)} = n_{\ell-1} \times n_\ell + n_\ell = n_\ell(n_{\ell-1} + 1)$$

**Total parameters:**
$$\text{Total} = \sum_{\ell=1}^{L} n_\ell(n_{\ell-1} + 1)$$

**Example:** Network with architecture 784 → 256 → 128 → 10
- Layer 1: 784 × 256 + 256 = 200,960
- Layer 2: 256 × 128 + 128 = 32,896
- Layer 3: 128 × 10 + 10 = 1,290
- **Total: 235,146 parameters**

**Without biases:** Just multiply neurons in consecutive layers and sum.

## 5.8 Impact of Depth and Width

### Depth (Number of layers)
- More layers = more abstraction levels
- Enables learning hierarchical features
- Too deep → vanishing/exploding gradients, hard to train

### Width (Neurons per layer)
- More neurons = more capacity
- Can represent more complex functions
- Too wide → overfitting, computational cost

### Universal Approximation Theorem
A neural network with:
- Single hidden layer
- Sufficient number of neurons
- Non-linear activation
Can approximate any continuous function!

(But deeper networks are often more efficient)

---

# Sessions 6-8: Convolutional Neural Networks

## 6.1 Motivation

### Why CNNs for Images?
- **Fully connected networks fail** because:
  - Images have spatial structure
  - Too many parameters (e.g., 256×256×3 = 196,608 inputs!)
  - Don't capture translation invariance

### Key Properties CNNs Exploit
1. **Locality:** Nearby pixels are related
2. **Translation Invariance:** Features can appear anywhere
3. **Hierarchical Patterns:** Complex features = combinations of simple features

## 6.2 Convolution Operation

### 1D Convolution (Conceptual)
$$y[n] = \sum_{k} x[n-k] \cdot w[k]$$

### 2D Convolution for Images
$$y[i,j] = \sum_{m}\sum_{n} x[i+m, j+n] \cdot w[m,n]$$

### Example: 3×3 Kernel on 4×4 Image

```
Input (4×4):           Kernel (3×3):
┌─────────────┐        ┌───────────┐
│ 1  2  3  4 │        │ 1  0 -1  │
│ 5  6  7  8 │   *    │ 1  0 -1  │
│ 9  10 11 12│        │ 1  0 -1  │
│ 13 14 15 16│        └───────────┘
└─────────────┘

Output (2×2):
y[0,0] = 1×1 + 2×0 + 3×(-1) + 5×1 + 6×0 + 7×(-1) + 9×1 + 10×0 + 11×(-1)
       = 1 - 3 + 5 - 7 + 9 - 11 = -6
```

## 6.3 Padding and Stride

### Padding
Adding zeros around the image to control output size.

**Valid (no padding):** Output size = (n - f + 1) × (n - f + 1)
**Same (zero padding):** Output size = n × n (pad with p = (f-1)/2)

### Stride
Step size for moving the kernel.

**Output size formula:**
$$\text{output size} = \left\lfloor\frac{n + 2p - f}{s}\right\rfloor + 1$$

Where:
- n = input size
- f = kernel size
- p = padding
- s = stride

## 6.4 Pooling

### Max Pooling
Takes maximum value in each region.
- Reduces spatial dimensions
- Provides translation invariance
- No learnable parameters

### Average Pooling
Takes average value in each region.

### Example: 2×2 Max Pooling with stride 2

```
Input (4×4):           Output (2×2):
┌─────────────┐        ┌───────┐
│ 1  3 │ 2  4 │        │ 6 │ 8 │
│ 5  6 │ 7  8 │  →     ├───┼───┤
├──────┼──────┤        │14 │16 │
│ 9  10│11 12 │        └───────┘
│13 14 │15 16 │
└─────────────┘
```

## 6.5 CNN Layer Components

### Convolutional Layer
- Learnable kernels/filters
- Extract features (edges, textures, patterns)
- Parameters: W (kernel weights) + b (bias)

### Activation Layer
- Apply non-linearity (usually ReLU)
- Introduces non-linear decision boundaries

### Pooling Layer
- Reduce spatial dimensions
- Provide invariance
- No learnable parameters

### Fully Connected Layer
- Connect all neurons
- Typically at the end for classification

## 6.6 LeNet Architecture (First Successful CNN)

```
Input(32×32×1) → Conv(5×5, 6) → Pool(2×2) → Conv(5×5, 16) → Pool(2×2) 
             → FC(120) → FC(84) → Output(10)
```

| Layer | Type | Output Shape | Parameters |
|-------|------|--------------|------------|
| Input | - | 32×32×1 | 0 |
| C1 | Conv 5×5, 6 filters | 28×28×6 | 156 |
| S2 | Avg Pool 2×2 | 14×14×6 | 0 |
| C3 | Conv 5×5, 16 filters | 10×10×16 | 2416 |
| S4 | Avg Pool 2×2 | 5×5×16 | 0 |
| C5 | FC, 120 units | 120 | 48120 |
| F6 | FC, 84 units | 84 | 10164 |
| Output | FC, 10 units | 10 | 850 |

## 6.7 AlexNet (2012 - ImageNet Breakthrough)

### Key Innovations
- Much deeper than LeNet
- ReLU activation (instead of tanh)
- Dropout for regularization
- Data augmentation
- GPU training

### Architecture
```
Input(224×224×3) → Conv(11×11, 96, s=4) → Pool(3×3, s=2) 
→ Conv(5×5, 256) → Pool(3×3, s=2) 
→ Conv(3×3, 384) → Conv(3×3, 384) → Conv(3×3, 256) → Pool(3×3, s=2)
→ FC(4096) → Dropout → FC(4096) → Dropout → FC(1000)
```

## 6.8 VGGNet (2014)

### Key Insight
Use smaller (3×3) kernels but more layers.

Two 3×3 convolutions have same receptive field as one 5×5 but:
- Fewer parameters: 2×(3×3) = 18 vs 5×5 = 25
- More non-linearities

### VGG-16 Architecture

```
Input → [Conv3-64]×2 → Pool
      → [Conv3-128]×2 → Pool
      → [Conv3-256]×3 → Pool
      → [Conv3-512]×3 → Pool
      → [Conv3-512]×3 → Pool
      → FC-4096 → FC-4096 → FC-1000
```

Total: 138 million parameters

## 6.9 GoogLeNet/Inception (2014)

### Inception Block
Run multiple filter sizes in parallel, concatenate outputs:

```
        ┌─ 1×1 Conv ─────────┐
        │                    │
Input ──├─ 1×1 → 3×3 Conv ───┼── Concat → Output
        │                    │
        ├─ 1×1 → 5×5 Conv ───┤
        │                    │
        └─ Pool → 1×1 Conv ──┘
```

### 1×1 Convolution
- Reduces channel dimensions (bottleneck)
- Adds non-linearity
- Reduces computation

## 6.10 ResNet (2015) - Residual Networks

### The Problem
Very deep networks suffer from:
- Vanishing gradients
- Degradation problem (deeper ≠ better)

### Residual Block (Skip Connection)

```
        x ────────────────────┐
        │                     │ (identity)
        ↓                     │
    [Weight Layer]            │
        ↓                     │
      [ReLU]                  │
        ↓                     │
    [Weight Layer]            │
        ↓                     │
       (+) ←──────────────────┘
        ↓
      [ReLU]
        ↓
      F(x) + x
```

### Key Idea
Instead of learning H(x), learn F(x) = H(x) - x

Then: output = F(x) + x

If identity is optimal, F(x) = 0 is easy to learn!

### Benefits
- Enables training of 100+ layer networks
- Gradient flows directly through skip connections
- Won ImageNet 2015

## 6.11 Transfer Learning

### Concept
Use knowledge from pre-trained models on new tasks.

### Why Transfer Learning?
- Pre-trained on large datasets (ImageNet: 1.2M images)
- Early layers learn general features (edges, textures)
- Later layers learn task-specific features
- Requires less training data for new task

### Approaches

**1. Feature Extraction:**
- Freeze pre-trained layers
- Only train new output layer
- Fast, works with small datasets

**2. Fine-tuning:**
- Unfreeze some/all layers
- Train with small learning rate
- Better performance, needs more data

### Implementation Steps
1. Load pre-trained model (e.g., ResNet)
2. Replace final classification layer
3. Freeze base layers (optional)
4. Train on new dataset

---

# Essential Formulas Quick Reference

## Activation Functions

| Function | Formula | Derivative | Output Range |
|----------|---------|------------|--------------|
| Sigmoid | σ(z) = 1/(1+e⁻ᶻ) | σ(z)(1-σ(z)) | (0, 1) |
| Tanh | (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ) | 1 - tanh²(z) | (-1, 1) |
| ReLU | max(0, z) | 1 if z>0, 0 otherwise | [0, ∞) |
| Softmax | eᶻⁱ/Σeᶻʲ | - | (0, 1), sum=1 |
| Identity | z | 1 | (-∞, ∞) |

## Loss Functions

| Task | Loss Function | Formula |
|------|---------------|---------|
| Regression | MSE | (1/2N)Σ(ŷ-y)² |
| Binary Classification | BCE | -(1/N)Σ[y log(ŷ) + (1-y)log(1-ŷ)] |
| Multi-class | CCE | -(1/N)ΣΣ yₖ log(ŷₖ) |

## Gradient Descent

$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla J(\mathbf{w}^{(t)})$$

## Forward Propagation (DFNN)
$$\mathbf{z}^{(\ell)} = \mathbf{h}^{(\ell-1)}\mathbf{W}^{(\ell)} + \mathbf{b}^{(\ell)}$$
$$\mathbf{h}^{(\ell)} = \sigma(\mathbf{z}^{(\ell)})$$

## Backpropagation
$$\delta^{(L)} = \frac{1}{B}(\hat{\mathbf{Y}} - \mathbf{Y})$$
$$\delta^{(\ell)} = (\delta^{(\ell+1)}\mathbf{W}^{(\ell+1)T}) \odot \sigma'(\mathbf{z}^{(\ell)})$$
$$\nabla_{\mathbf{W}} J = \frac{1}{B}\mathbf{H}^{(\ell-1)T}\delta^{(\ell)}$$

## Convolution Output Size
$$\text{output} = \left\lfloor\frac{n + 2p - f}{s}\right\rfloor + 1$$

## Total Parameters (Fully Connected)
$$\text{Parameters} = \sum_{\ell=1}^{L} n_\ell(n_{\ell-1} + 1)$$
(Without bias: just n_ℓ × n_ℓ₋₁)

## Classification Metrics

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
$$\text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Recall} = \frac{TP}{TP + FN}$$
$$F1 = 2 \times \frac{P \times R}{P + R}$$

---

# Practice Problems with Solutions

## Problem 1: Perceptron Learning (5 marks)

**Question:** Implement perceptron learning for AND gate using bipolar representation. Initial weights: w₀=w₁=w₂=0, η=1.

**Solution:**

| x₀ | x₁ | x₂ | t | z | ŷ | Update? | Δw₀ | Δw₁ | Δw₂ | New w |
|----|----|----|---|---|---|---------|-----|-----|-----|-------|
| 1 | -1 | -1 | -1 | 0 | 1 | Yes | -2 | 2 | 2 | [-2,2,2] |
| 1 | -1 | 1 | -1 | -2 | -1 | No | 0 | 0 | 0 | [-2,2,2] |
| 1 | 1 | -1 | -1 | 2 | 1 | Yes | -2 | -2 | 2 | [-4,0,4] |
| 1 | 1 | 1 | 1 | 0 | 1 | No | 0 | 0 | 0 | [-4,0,4] |

Continue until convergence...

---

## Problem 2: Linear Regression Gradient Descent (6 marks)

**Question:** Given data points (1,2), (2,4), (3,5), perform one iteration of batch gradient descent with η=0.1, w⁽⁰⁾=[0,0]ᵀ.

**Solution:**

1. **Data matrices:**
   - X = [[1,1], [1,2], [1,3]]
   - y = [2, 4, 5]ᵀ

2. **Initial predictions:**
   - ŷ⁽⁰⁾ = Xw⁽⁰⁾ = [0, 0, 0]ᵀ

3. **Error:**
   - e = ŷ - y = [-2, -4, -5]ᵀ

4. **Gradient:**
   - ∇J = (1/3)Xᵀe = (1/3) × [1 1 1; 1 2 3] × [-2; -4; -5]
   - ∇J = (1/3) × [-11; -23] = [-3.67; -7.67]

5. **Update:**
   - w⁽¹⁾ = [0, 0]ᵀ - 0.1 × [-3.67; -7.67] = [0.367, 0.767]ᵀ

---

## Problem 3: Softmax and Cross-Entropy (5 marks)

**Question:** Given logits z = [2.0, 1.0, 0.1] for 3-class classifier:
a) Compute softmax probabilities
b) If true class is 1, compute cross-entropy loss

**Solution:**

a) **Softmax:**
   - e² = 7.39, e¹ = 2.72, e⁰·¹ = 1.11
   - Sum = 11.22
   - ŷ = [7.39/11.22, 2.72/11.22, 1.11/11.22] = [0.659, 0.242, 0.099]

b) **Cross-entropy (true class = 1):**
   - Loss = -log(0.659) = 0.417

---

## Problem 4: DFNN Forward Propagation (6 marks)

**Question:** Network: 2 inputs, 2 hidden (ReLU), 1 output (sigmoid).
- x = [0.5, 0.3]ᵀ
- W⁽¹⁾ = [[0.1, 0.2], [0.3, 0.4]], b⁽¹⁾ = [0.1, 0.1]
- W⁽²⁾ = [[0.5], [0.6]], b⁽²⁾ = [0.2]

**Solution:**

1. **Hidden layer pre-activation:**
   - z⁽¹⁾ = [0.5, 0.3] × [[0.1, 0.2], [0.3, 0.4]] + [0.1, 0.1]
   - z⁽¹⁾ = [0.05+0.09+0.1, 0.1+0.12+0.1] = [0.24, 0.32]

2. **Hidden layer activation (ReLU):**
   - h⁽¹⁾ = ReLU([0.24, 0.32]) = [0.24, 0.32]

3. **Output pre-activation:**
   - z⁽²⁾ = [0.24, 0.32] × [[0.5], [0.6]] + [0.2]
   - z⁽²⁾ = 0.12 + 0.192 + 0.2 = 0.512

4. **Output activation (Sigmoid):**
   - ŷ = σ(0.512) = 1/(1 + e⁻⁰·⁵¹²) ≈ 0.625

---

## Problem 5: CNN Output Size (3 marks)

**Question:** Input: 32×32×3, Conv: 5×5, 64 filters, stride=1, padding=2, followed by 2×2 max pooling. What's the output size?

**Solution:**

1. **After convolution:**
   - output = ⌊(32 + 2×2 - 5)/1⌋ + 1 = 32
   - Size: 32×32×64

2. **After pooling (2×2, stride 2):**
   - output = ⌊(32 - 2)/2⌋ + 1 = 16
   - **Final size: 16×16×64**

---

## Problem 6: Total Parameters (3 marks)

**Question:** Network: 8 inputs → 10 hidden → 6 hidden → 4 outputs (no biases). Calculate trainable weights.

**Solution:**
- Layer 1: 8 × 10 = 80
- Layer 2: 10 × 6 = 60
- Layer 3: 6 × 4 = 24
- **Total: 80 + 60 + 24 = 164 weights**

---

# Python Code Templates

## Perceptron Training

```python
import numpy as np

def perceptron_train(X, y, eta=1.0, epochs=100):
    """
    Train perceptron
    X: Input features (N x d)
    y: Labels (+1 or -1)
    """
    # Add bias term
    X_bias = np.c_[np.ones(X.shape[0]), X]
    w = np.zeros(X_bias.shape[1])
    
    for epoch in range(epochs):
        for i in range(len(X)):
            z = np.dot(w, X_bias[i])
            y_pred = np.sign(z) if z != 0 else 1
            if y_pred != y[i]:
                w += eta * (y[i] - y_pred) * X_bias[i]
    return w
```

## Linear Regression with Gradient Descent

```python
def linear_regression_gd(X, y, eta=0.01, epochs=1000):
    """
    Batch gradient descent for linear regression
    """
    X_bias = np.c_[np.ones(X.shape[0]), X]
    w = np.zeros(X_bias.shape[1])
    N = len(y)
    
    for epoch in range(epochs):
        y_pred = X_bias @ w
        error = y_pred - y
        gradient = (1/N) * X_bias.T @ error
        w -= eta * gradient
    return w
```

## Logistic Regression (Binary Classification)

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, eta=0.1, epochs=1000):
    """
    Binary classification with SGD
    """
    X_bias = np.c_[np.ones(X.shape[0]), X]
    w = np.zeros(X_bias.shape[1])
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(X))
        for i in indices:
            z = np.dot(w, X_bias[i])
            y_pred = sigmoid(z)
            gradient = (y_pred - y[i]) * X_bias[i]
            w -= eta * gradient
    return w
```

## Softmax Function

```python
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Numerical stability
    return exp_z / np.sum(exp_z)
```

## Simple DFNN Forward Pass

```python
def relu(z):
    return np.maximum(0, z)

def forward_pass(X, W1, b1, W2, b2):
    """
    2-layer network: Input -> ReLU -> Sigmoid
    """
    z1 = X @ W1 + b1
    h1 = relu(z1)
    z2 = h1 @ W2 + b2
    y_pred = sigmoid(z2)
    return y_pred, (z1, h1, z2)
```

## Computing Loss

```python
def mse_loss(y_pred, y_true):
    return 0.5 * np.mean((y_pred - y_true)**2)

def binary_cross_entropy(y_pred, y_true):
    eps = 1e-15  # Avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_pred, y_true):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
```

---

# Key Takeaways for Exam

## ✅ Concepts You Must Know

1. **Perceptron:** Learning algorithm, weight update rule, linear separability, XOR problem
2. **Regression:** MSE loss, gradient computation, batch GD algorithm
3. **Classification:** Sigmoid, softmax, BCE, CCE, SGD vs batch GD
4. **DFNN:** Forward pass, backward pass, chain rule, parameter counting
5. **CNNs:** Convolution, pooling, padding, stride, LeNet, AlexNet, VGG, ResNet
6. **Transfer Learning:** Feature extraction vs fine-tuning

## 🔑 Common Exam Patterns

1. **Numerical computations:** Perceptron iterations, gradient descent, softmax probabilities
2. **Code completion:** Fill in missing PyTorch/Keras code
3. **Architecture design:** Design network for specific task
4. **Reasoning questions:** Why use X loss? Why does Y fail?
5. **Metric calculations:** Precision, recall, F1 from confusion matrix

## ⚠️ Common Mistakes to Avoid

- Forgetting bias term in calculations
- Wrong matrix dimensions
- Confusing BCE with CCE
- Not normalizing softmax correctly
- Forgetting to apply activation functions

---

> 📝 **Best of luck with your midterm exam!** 🍀
