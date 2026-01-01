# Session 4: Neural Networks and Neural Language Modeling
## AIMLCZG530 - Natural Language Processing

---

# 1. Why Neural Language Models?

## 1.1 Limitations of N-gram Models

| Limitation | Description |
|------------|-------------|
| **No generalization** | "cat" and "dog" are unrelated |
| **Sparsity** | Most n-grams never seen |
| **Fixed context** | Only n-1 words of history |
| **No semantics** | Words are just symbols |

## 1.2 Neural LM Advantages

| Advantage | Description |
|-----------|-------------|
| **Word embeddings** | Similar words have similar vectors |
| **Generalization** | "cat sat" → can predict "dog sat" |
| **Long context** | RNN/Transformer can use more history |
| **Shared parameters** | Efficient representation |

---

# 2. Feed-Forward Neural Networks

## 2.1 Basic Architecture

```
Input Layer → Hidden Layer(s) → Output Layer
    x            h = f(Wx + b)      y = softmax(Vh)
```

## 2.2 Components

### Neurons
- Receive weighted inputs
- Apply activation function
- Produce output

### Layers
| Layer | Function |
|-------|----------|
| **Input** | Receives features |
| **Hidden** | Learns representations |
| **Output** | Produces predictions |

### Activation Functions

| Function | Formula | Properties |
|----------|---------|------------|
| **Sigmoid** | σ(x) = 1/(1+e^-x) | Output: (0,1) |
| **Tanh** | tanh(x) = (e^x - e^-x)/(e^x + e^-x) | Output: (-1,1) |
| **ReLU** | max(0, x) | Simple, effective |
| **Softmax** | e^xᵢ / Σe^xⱼ | Probability distribution |

## 2.3 Forward Propagation

**Step-by-step for single hidden layer**:

```
1. Input: x (n-dimensional)
2. Hidden: h = σ(W₁x + b₁)
3. Output: y = softmax(W₂h + b₂)
```

## 2.4 Loss Functions

### Cross-Entropy Loss (for classification)
```
L = -Σ yᵢ log(ŷᵢ)
```

### Negative Log-Likelihood (for LM)
```
L = -log P(wₜ | context)
```

---

# 3. Neural Language Model Architecture

## 3.1 Basic Feed-Forward LM

**Proposed by Bengio et al. (2003)**

```
┌─────────────────────────────────────────────┐
│                  Softmax                     │ → P(next word)
│                 (V units)                    │
├─────────────────────────────────────────────┤
│                Hidden Layer                  │
│                 (h units)                    │
│              h = tanh(Wx + b)               │
├─────────────────────────────────────────────┤
│           Concatenated Embeddings            │
│              [e₁; e₂; e₃; e₄]               │
├─────────────────────────────────────────────┤
│              Lookup Table (E)                │
│           (Embedding matrix V×d)             │
├─────────────────────────────────────────────┤
│            Input: Context words              │
│           (one-hot or indices)               │
└─────────────────────────────────────────────┘
```

## 3.2 Architecture Details

### Input Representation
- **Context window**: n-1 previous words
- **One-hot vectors**: V-dimensional (vocabulary size)

### Embedding Layer
- **Lookup table**: E (V × d matrix)
- **Word → embedding**: eᵢ = E[wᵢ]
- **Concatenation**: x = [e₁; e₂; ...; eₙ₋₁]

### Hidden Layer
```
h = tanh(W · x + b)
```
- W: Weight matrix (h × (n-1)×d)
- b: Bias vector
- h: Hidden units (typically 50-200)

### Output Layer
```
y = softmax(V · h + c)
```
- V: Output weight matrix (|V| × h)
- Softmax: Converts to probability distribution

## 3.3 Mathematical Formulation

**Complete model**:
```
P(wₜ | wₜ₋₁, wₜ₋₂, ..., wₜ₋ₙ₊₁) = softmax(V · tanh(W · [e₁;...;eₙ₋₁] + b) + c)
```

---

# 4. Training Neural Language Models

## 4.1 Training Objective

**Maximize log-likelihood**:
```
J = Σₜ log P(wₜ | wₜ₋₁, ..., wₜ₋ₙ₊₁)
```

**Equivalently, minimize cross-entropy loss**:
```
L = -Σₜ log P(wₜ | context)
```

## 4.2 Backpropagation

**Compute gradients**:
1. Forward pass: compute predictions
2. Compute loss
3. Backward pass: compute gradients ∂L/∂θ
4. Update parameters: θ = θ - η∇L

## 4.3 Stochastic Gradient Descent (SGD)

```
For each mini-batch:
    1. Forward pass
    2. Compute loss
    3. Compute gradients
    4. Update: θ = θ - η·∇L
```

## 4.4 Embedding Learning

**Two strategies**:

| Strategy | Description | Pros/Cons |
|----------|-------------|-----------|
| **Pre-trained** | Use Word2Vec, GloVe | Less data needed |
| **Joint training** | Learn with LM | Task-specific |
| **Frozen** | Don't update embeddings | Faster, less flexible |
| **Fine-tuned** | Update embeddings | Better, more parameters |

---

# 5. Why Neural LMs Work Better

## 5.1 Semantic Generalization

**N-gram model**:
- Seen: "The cat sat on the mat"
- Unseen: "The dog sat on the rug"
- P(sat | dog) = 0 (never seen!)

**Neural model**:
- v_cat ≈ v_dog (similar embeddings)
- Can generalize: "dog sat" is probable too!

## 5.2 Shared Representations

```
Similar contexts → Similar embeddings → Similar predictions
```

## 5.3 Smooth Probability Estimates

- No zero probabilities (smoothing built-in)
- Continuous output space
- Gradual transitions

---

# 6. Advanced Architectures (Brief Overview)

## 6.1 Recurrent Neural Networks (RNN)

```
hₜ = tanh(W_hh·hₜ₋₁ + W_xh·xₜ + b)
yₜ = softmax(W_hy·hₜ)
```

**Advantage**: Variable-length context

## 6.2 LSTM (Long Short-Term Memory)

- Addresses vanishing gradient
- Gates: forget, input, output
- Cell state for long-term memory

## 6.3 Transformer

- Self-attention mechanism
- Parallel processing
- Foundation for BERT, GPT

---

# 7. Comparison: N-gram vs Neural LM

| Aspect | N-gram LM | Neural LM |
|--------|-----------|-----------|
| Context | Fixed (n-1) | Variable |
| Parameters | O(V^n) | O(V×d + d²) |
| Sparsity | Major issue | Not an issue |
| Training | Counting | Gradient descent |
| Generalization | None | Embedding-based |
| Interpretability | High | Low |
| Computation | Fast lookup | Matrix multiply |
| Memory | Large tables | Network weights |

---

# 8. Key Equations

| Concept | Equation |
|---------|----------|
| Hidden layer | h = tanh(Wx + b) |
| Output | y = softmax(Vh + c) |
| Softmax | P(i) = e^zᵢ / Σe^zⱼ |
| Cross-entropy | L = -Σ yᵢ log(ŷᵢ) |
| SGD update | θ = θ - η∇L |

---

# 📝 Practice Questions

## Q1. Compare N-gram and Neural Language Models on:
a) Handling unseen word combinations
b) Memory requirements
c) Training complexity

## Q2. For a 4-gram neural LM with:
- Vocabulary size V = 10,000
- Embedding dimension d = 100
- Hidden layer size h = 200

How many parameters in the embedding layer?

## Q3. Why does a neural LM generalize better than an N-gram model?

## Q4. Explain the role of the softmax function in neural language models.

## Q5. What is the advantage of using pre-trained embeddings vs learning them jointly?

---

*Reference: Session 4 - Neural Networks and Neural Language Modeling*
