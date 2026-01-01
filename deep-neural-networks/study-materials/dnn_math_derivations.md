# 🧮 DNN Math Derivations Guide

> **Complete Mathematical Derivations for Exam | AIMLCZG511**

---

## Table of Contents

1. [Perceptron Weight Update Derivation](#1-perceptron-weight-update-derivation)
2. [Linear Regression Gradient Derivation](#2-linear-regression-gradient-derivation)
3. [Sigmoid Function Properties](#3-sigmoid-function-properties)
4. [Binary Cross-Entropy Gradient](#4-binary-cross-entropy-gradient)
5. [Softmax Function Derivation](#5-softmax-function-derivation)
6. [Categorical Cross-Entropy Gradient](#6-categorical-cross-entropy-gradient)
7. [Backpropagation Chain Rule](#7-backpropagation-chain-rule)
8. [ReLU Derivative](#8-relu-derivative)
9. [CNN Output Size Formula](#9-cnn-output-size-formula)
10. [Parameter Counting](#10-parameter-counting)

---

## 1. Perceptron Weight Update Derivation

### The Perceptron Model

**Forward Pass:**
$$z = \sum_{i=0}^{n} w_i x_i = w_0 + w_1 x_1 + w_2 x_2 + \ldots + w_n x_n$$

**Activation (Sign Function):**
$$\hat{y} = \text{sign}(z) = \begin{cases} +1 & \text{if } z \geq 0 \\ -1 & \text{if } z < 0 \end{cases}$$

### Error Signal
$$e = t - \hat{y}$$

Where:
- t = target output (+1 or -1)
- ŷ = predicted output

### Weight Update Rule

**Derivation:**
The goal is to move weights such that the output moves closer to the target.

If prediction is **correct** (t = ŷ):
$$e = t - \hat{y} = 0 \implies \Delta w_i = 0$$

If prediction is **wrong** (t ≠ ŷ):
- Case 1: t = +1, ŷ = -1 → e = +2 → need to **increase** wᵀx
- Case 2: t = -1, ŷ = +1 → e = -2 → need to **decrease** wᵀx

**Update formula:**
$$\Delta w_i = \eta \cdot e \cdot x_i = \eta (t - \hat{y}) x_i$$

$$w_i^{new} = w_i^{old} + \Delta w_i$$

### Worked Example

**NAND Gate (Bipolar: +1, -1)**

| x₁ | x₂ | t |
|----|----|---|
| -1 | -1 | +1 |
| -1 | +1 | +1 |
| +1 | -1 | +1 |
| +1 | +1 | -1 |

**Initial:** w₀ = w₁ = w₂ = 0, η = 1, x₀ = 1 (bias)

**Step 1:** x = [1, -1, -1], t = +1
- z = 0·1 + 0·(-1) + 0·(-1) = 0
- ŷ = sign(0) = +1
- e = +1 - (+1) = 0
- No update: w = [0, 0, 0]

**Step 2:** x = [1, -1, +1], t = +1
- z = 0·1 + 0·(-1) + 0·(+1) = 0
- ŷ = sign(0) = +1
- e = +1 - (+1) = 0
- No update: w = [0, 0, 0]

**Step 3:** x = [1, +1, -1], t = +1
- z = 0·1 + 0·(+1) + 0·(-1) = 0
- ŷ = sign(0) = +1
- e = +1 - (+1) = 0
- No update: w = [0, 0, 0]

**Step 4:** x = [1, +1, +1], t = -1
- z = 0·1 + 0·(+1) + 0·(+1) = 0
- ŷ = sign(0) = +1
- e = -1 - (+1) = -2
- **Update:** Δw = -2 · [1, 1, 1] = [-2, -2, -2]
- **New weights:** w = [0-2, 0-2, 0-2] = [-2, -2, -2]

---

## 2. Linear Regression Gradient Derivation

### The Model
$$\hat{y} = \mathbf{w}^T \mathbf{x} = w_0 + w_1 x_1 + \ldots + w_d x_d$$

### Mean Squared Error (MSE) Loss

**For single example:**
$$\ell(w) = \frac{1}{2}(\hat{y} - y)^2 = \frac{1}{2}(\mathbf{w}^T \mathbf{x} - y)^2$$

**For all examples (cost function):**
$$J(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{2}(\hat{y}^{(i)} - y^{(i)})^2 = \frac{1}{2N} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$$

### Gradient Derivation

**Step 1: Expand the loss**
$$J(\mathbf{w}) = \frac{1}{2N}(\mathbf{X}\mathbf{w} - \mathbf{y})^T(\mathbf{X}\mathbf{w} - \mathbf{y})$$

**Step 2: Apply chain rule**
Let $\mathbf{e} = \mathbf{X}\mathbf{w} - \mathbf{y}$ (error vector)

$$J = \frac{1}{2N} \mathbf{e}^T \mathbf{e} = \frac{1}{2N} \sum_{i=1}^{N} e_i^2$$

**Step 3: Differentiate w.r.t. w**

For a single example:
$$\frac{\partial \ell}{\partial w_j} = \frac{\partial}{\partial w_j}\left[\frac{1}{2}(\mathbf{w}^T\mathbf{x} - y)^2\right]$$

Using chain rule:
$$\frac{\partial \ell}{\partial w_j} = (\mathbf{w}^T\mathbf{x} - y) \cdot \frac{\partial}{\partial w_j}(\mathbf{w}^T\mathbf{x})$$

Since $\frac{\partial}{\partial w_j}(w_0 x_0 + w_1 x_1 + \ldots + w_j x_j + \ldots) = x_j$:

$$\frac{\partial \ell}{\partial w_j} = (\hat{y} - y) \cdot x_j$$

**In vector form:**
$$\nabla_{\mathbf{w}} \ell = (\hat{y} - y) \mathbf{x}$$

**For the full batch:**
$$\nabla_{\mathbf{w}} J = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)}) \mathbf{x}^{(i)} = \frac{1}{N} \mathbf{X}^T (\mathbf{X}\mathbf{w} - \mathbf{y})$$

### Update Rule
$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla_{\mathbf{w}} J = \mathbf{w}^{(t)} - \frac{\eta}{N} \mathbf{X}^T (\mathbf{X}\mathbf{w}^{(t)} - \mathbf{y})$$

---

## 3. Sigmoid Function Properties

### Definition
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### Key Properties

**Range:** $(0, 1)$

**Special values:**
- $\sigma(0) = 0.5$
- $\sigma(\infty) = 1$
- $\sigma(-\infty) = 0$

### Derivative Derivation

$$\sigma(z) = \frac{1}{1 + e^{-z}} = (1 + e^{-z})^{-1}$$

Using chain rule:
$$\frac{d\sigma}{dz} = -(1 + e^{-z})^{-2} \cdot (-e^{-z})$$

$$= \frac{e^{-z}}{(1 + e^{-z})^2}$$

**Simplification trick:**

Note that:
$$1 - \sigma(z) = 1 - \frac{1}{1 + e^{-z}} = \frac{e^{-z}}{1 + e^{-z}}$$

Therefore:
$$\sigma(z)(1 - \sigma(z)) = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \frac{e^{-z}}{(1 + e^{-z})^2}$$

**Final result:**
$$\boxed{\sigma'(z) = \sigma(z)(1 - \sigma(z))}$$

### Numerical Example
If $z = 1$:
- $\sigma(1) = \frac{1}{1 + e^{-1}} = \frac{1}{1.368} = 0.731$
- $\sigma'(1) = 0.731 \times (1 - 0.731) = 0.731 \times 0.269 = 0.197$

---

## 4. Binary Cross-Entropy Gradient

### Loss Function
$$\ell = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

Where: $\hat{y} = \sigma(z) = \sigma(\mathbf{w}^T\mathbf{x})$

### Gradient Derivation

We need $\frac{\partial \ell}{\partial w_j}$. Using chain rule:

$$\frac{\partial \ell}{\partial w_j} = \frac{\partial \ell}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w_j}$$

**Step 1: Compute $\frac{\partial \ell}{\partial \hat{y}}$**

$$\frac{\partial \ell}{\partial \hat{y}} = -\left[\frac{y}{\hat{y}} + (1-y) \cdot \frac{-1}{1-\hat{y}}\right] = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$$

$$= \frac{-y(1-\hat{y}) + (1-y)\hat{y}}{\hat{y}(1-\hat{y})} = \frac{-y + y\hat{y} + \hat{y} - y\hat{y}}{\hat{y}(1-\hat{y})} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$$

**Step 2: Compute $\frac{\partial \hat{y}}{\partial z}$**

$$\frac{\partial \hat{y}}{\partial z} = \sigma'(z) = \sigma(z)(1-\sigma(z)) = \hat{y}(1-\hat{y})$$

**Step 3: Compute $\frac{\partial z}{\partial w_j}$**

$$\frac{\partial z}{\partial w_j} = \frac{\partial}{\partial w_j}(\mathbf{w}^T\mathbf{x}) = x_j$$

**Step 4: Combine using chain rule**

$$\frac{\partial \ell}{\partial w_j} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})} \cdot \hat{y}(1-\hat{y}) \cdot x_j$$

The $\hat{y}(1-\hat{y})$ terms cancel:

$$\boxed{\frac{\partial \ell}{\partial w_j} = (\hat{y} - y) x_j}$$

**In vector form:**
$$\nabla_{\mathbf{w}} \ell = (\hat{y} - y) \mathbf{x}$$

> **Beautiful result:** Same form as linear regression gradient!

---

## 5. Softmax Function Derivation

### Definition

For K classes with logits $\mathbf{z} = [z_1, z_2, \ldots, z_K]^T$:

$$\hat{y}_k = \text{softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

### Properties
- All outputs sum to 1: $\sum_{k=1}^{K} \hat{y}_k = 1$
- Each output in $(0, 1)$: $0 < \hat{y}_k < 1$

### Softmax Derivative

We need $\frac{\partial \hat{y}_i}{\partial z_j}$

Let $S = \sum_{k=1}^{K} e^{z_k}$ (denominator)

**Case 1: i = j (same index)**

$$\hat{y}_i = \frac{e^{z_i}}{S}$$

Using quotient rule:
$$\frac{\partial \hat{y}_i}{\partial z_i} = \frac{e^{z_i} \cdot S - e^{z_i} \cdot e^{z_i}}{S^2} = \frac{e^{z_i}(S - e^{z_i})}{S^2}$$

$$= \frac{e^{z_i}}{S} \cdot \frac{S - e^{z_i}}{S} = \hat{y}_i (1 - \hat{y}_i)$$

**Case 2: i ≠ j (different index)**

$$\frac{\partial \hat{y}_i}{\partial z_j} = \frac{0 \cdot S - e^{z_i} \cdot e^{z_j}}{S^2} = -\frac{e^{z_i}}{S} \cdot \frac{e^{z_j}}{S} = -\hat{y}_i \hat{y}_j$$

**Combined (using Kronecker delta):**
$$\boxed{\frac{\partial \hat{y}_i}{\partial z_j} = \hat{y}_i (\delta_{ij} - \hat{y}_j)}$$

Where $\delta_{ij} = 1$ if $i=j$, else $0$.

---

## 6. Categorical Cross-Entropy Gradient

### Loss Function

For one-hot encoded target $\mathbf{y}$:
$$\ell = -\sum_{k=1}^{K} y_k \log(\hat{y}_k)$$

Since $\mathbf{y}$ is one-hot (only one $y_c = 1$ for true class $c$):
$$\ell = -\log(\hat{y}_c)$$

### Gradient w.r.t. Logits

We need $\frac{\partial \ell}{\partial z_j}$

$$\frac{\partial \ell}{\partial z_j} = -\sum_{k=1}^{K} y_k \cdot \frac{1}{\hat{y}_k} \cdot \frac{\partial \hat{y}_k}{\partial z_j}$$

$$= -\sum_{k=1}^{K} y_k \cdot \frac{1}{\hat{y}_k} \cdot \hat{y}_k(\delta_{kj} - \hat{y}_j)$$

$$= -\sum_{k=1}^{K} y_k (\delta_{kj} - \hat{y}_j)$$

$$= -\sum_{k=1}^{K} y_k \delta_{kj} + \sum_{k=1}^{K} y_k \hat{y}_j$$

Since $\sum_k y_k = 1$ (one-hot) and $\delta_{kj} = 1$ only when $k = j$:

$$= -y_j + \hat{y}_j \cdot 1 = \hat{y}_j - y_j$$

**Final result:**
$$\boxed{\frac{\partial \ell}{\partial z_j} = \hat{y}_j - y_j}$$

**In vector form:**
$$\nabla_{\mathbf{z}} \ell = \hat{\mathbf{y}} - \mathbf{y}$$

> **Amazing:** Same elegant form for all three (MSE, BCE, CCE) when paired with right activation!

---

## 7. Backpropagation Chain Rule

### Forward Pass Recap

For layer $\ell$:
$$\mathbf{z}^{(\ell)} = \mathbf{h}^{(\ell-1)} \mathbf{W}^{(\ell)} + \mathbf{b}^{(\ell)}$$
$$\mathbf{h}^{(\ell)} = \sigma^{(\ell)}(\mathbf{z}^{(\ell)})$$

### Error Signal Definition

$$\boldsymbol{\delta}^{(\ell)} \triangleq \frac{\partial J}{\partial \mathbf{z}^{(\ell)}}$$

### Output Layer Error

For Softmax + Cross-Entropy or Sigmoid + BCE:
$$\boldsymbol{\delta}^{(L)} = \hat{\mathbf{y}} - \mathbf{y}$$

### Backpropagation to Hidden Layers

**Chain rule derivation:**

$$\boldsymbol{\delta}^{(\ell)} = \frac{\partial J}{\partial \mathbf{z}^{(\ell)}}$$

Using chain rule: $\mathbf{z}^{(\ell)} \to \mathbf{h}^{(\ell)} \to \mathbf{z}^{(\ell+1)} \to J$

$$\boldsymbol{\delta}^{(\ell)} = \frac{\partial J}{\partial \mathbf{z}^{(\ell+1)}} \cdot \frac{\partial \mathbf{z}^{(\ell+1)}}{\partial \mathbf{h}^{(\ell)}} \cdot \frac{\partial \mathbf{h}^{(\ell)}}{\partial \mathbf{z}^{(\ell)}}$$

Since:
- $\frac{\partial J}{\partial \mathbf{z}^{(\ell+1)}} = \boldsymbol{\delta}^{(\ell+1)}$
- $\mathbf{z}^{(\ell+1)} = \mathbf{h}^{(\ell)} \mathbf{W}^{(\ell+1)} + \mathbf{b}^{(\ell+1)}$, so $\frac{\partial \mathbf{z}^{(\ell+1)}}{\partial \mathbf{h}^{(\ell)}} = \mathbf{W}^{(\ell+1)}$
- $\frac{\partial \mathbf{h}^{(\ell)}}{\partial \mathbf{z}^{(\ell)}} = \sigma'^{(\ell)}(\mathbf{z}^{(\ell)})$ (element-wise)

**Result:**
$$\boxed{\boldsymbol{\delta}^{(\ell)} = (\boldsymbol{\delta}^{(\ell+1)} \mathbf{W}^{(\ell+1)T}) \odot \sigma'^{(\ell)}(\mathbf{z}^{(\ell)})}$$

### Parameter Gradients

**Weight gradient:**
$$\frac{\partial J}{\partial \mathbf{W}^{(\ell)}} = \mathbf{h}^{(\ell-1)T} \boldsymbol{\delta}^{(\ell)}$$

**Bias gradient:**
$$\frac{\partial J}{\partial \mathbf{b}^{(\ell)}} = \sum_{\text{batch}} \boldsymbol{\delta}^{(\ell)}$$

---

## 8. ReLU Derivative

### Definition
$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

### Derivative
$$\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

**Note:** Technically undefined at $z = 0$, but we use 0 by convention.

### In Backpropagation

For hidden layer with ReLU:
$$\boldsymbol{\delta}^{(\ell)} = (\boldsymbol{\delta}^{(\ell+1)} \mathbf{W}^{(\ell+1)T}) \odot \mathbf{1}_{z^{(\ell)} > 0}$$

Where $\mathbf{1}_{z > 0}$ is the indicator function (1 where $z > 0$, 0 elsewhere).

### Why ReLU Helps

- Gradient is either 0 or 1 (no vanishing for positive inputs)
- Leads to sparse activations
- Computationally efficient

---

## 9. CNN Output Size Formula

### After Convolution

$$\text{output size} = \left\lfloor \frac{n + 2p - f}{s} \right\rfloor + 1$$

Where:
- $n$ = input size
- $p$ = padding
- $f$ = filter/kernel size
- $s$ = stride

### Common Cases

**Valid convolution (no padding):**
$$\text{output} = \left\lfloor \frac{n - f}{s} \right\rfloor + 1$$

**Same convolution (preserve size, stride=1):**
$$p = \frac{f - 1}{2}$$
$$\text{output} = n$$

### After Pooling

Same formula applies:
$$\text{output} = \left\lfloor \frac{n - \text{pool\_size}}{\text{stride}} \right\rfloor + 1$$

### Worked Example

**Input:** 32 × 32 × 3

**Layer 1:** Conv 5×5, 32 filters, stride=1, padding=0
$$\text{output} = \left\lfloor \frac{32 - 5}{1} \right\rfloor + 1 = 28$$
Output: 28 × 28 × 32

**Layer 2:** MaxPool 2×2, stride=2
$$\text{output} = \left\lfloor \frac{28 - 2}{2} \right\rfloor + 1 = 14$$
Output: 14 × 14 × 32

**Layer 3:** Conv 3×3, 64 filters, stride=1, padding=1
$$\text{output} = \left\lfloor \frac{14 + 2 - 3}{1} \right\rfloor + 1 = 14$$
Output: 14 × 14 × 64

---

## 10. Parameter Counting

### Fully Connected Layer

**With bias:**
$$\text{params} = n_{in} \times n_{out} + n_{out} = n_{out}(n_{in} + 1)$$

**Without bias:**
$$\text{params} = n_{in} \times n_{out}$$

### Convolutional Layer

**With bias:**
$$\text{params} = (f \times f \times c_{in}) \times c_{out} + c_{out}$$

Where:
- $f$ = filter size
- $c_{in}$ = input channels
- $c_{out}$ = output channels (number of filters)

**Without bias:**
$$\text{params} = f^2 \times c_{in} \times c_{out}$$

### Worked Example

**Network:** 8 → 10 → 6 → 4 (no biases)

**Weights:**
- Layer 1: $8 \times 10 = 80$
- Layer 2: $10 \times 6 = 60$
- Layer 3: $6 \times 4 = 24$
- **Total: 164**

**With biases:**
- Layer 1: $10 \times (8 + 1) = 90$
- Layer 2: $6 \times (10 + 1) = 66$
- Layer 3: $4 \times (6 + 1) = 28$
- **Total: 184**

---

## Quick Reference: All Gradients

| Model | Loss | Activation | Gradient w.r.t. z |
|-------|------|------------|-------------------|
| Linear Regression | MSE | Identity | $\hat{y} - y$ |
| Logistic Regression | BCE | Sigmoid | $\hat{y} - y$ |
| Softmax Classification | CCE | Softmax | $\hat{\mathbf{y}} - \mathbf{y}$ |
| Hidden Layer | - | ReLU | $\delta^{(\ell+1)} W^{(\ell+1)T} \odot \mathbf{1}_{z>0}$ |
| Hidden Layer | - | Sigmoid | $\delta^{(\ell+1)} W^{(\ell+1)T} \odot \sigma(z)(1-\sigma(z))$ |

---

**🧮 Master these derivations and you'll ace the math portion of your exam!**
