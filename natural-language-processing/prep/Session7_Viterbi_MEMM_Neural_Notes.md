# Session 7: Statistical, ML, and Neural Models for POS Tagging
## AIMLCZG530 - Natural Language Processing

---

# 1. Three Fundamental HMM Problems

| Problem | Question | Algorithm |
|---------|----------|-----------|
| **Likelihood** | P(observations \| model) | Forward Algorithm |
| **Decoding** | Best state sequence | Viterbi Algorithm |
| **Learning** | Estimate parameters | Baum-Welch (EM) |

For POS tagging, we focus on **Decoding** (Viterbi).

---

# 2. The Forward Algorithm

## 2.1 Purpose
Compute probability of observation sequence: P(O | λ)

## 2.2 Intuition
Sum over all possible state sequences (too expensive to enumerate)

## 2.3 Forward Variable
```
αₜ(j) = P(o₁, o₂, ..., oₜ, qₜ = j | λ)
```

Probability of seeing observations o₁...oₜ AND being in state j at time t.

## 2.4 Algorithm

**Initialization (t = 1)**:
```
α₁(j) = π(j) × B(j, o₁)
```

**Recursion (t > 1)**:
```
αₜ(j) = [Σᵢ αₜ₋₁(i) × A(i,j)] × B(j, oₜ)
```

**Termination**:
```
P(O | λ) = Σⱼ αₜ(j)
```

---

# 3. The Viterbi Algorithm

## 3.1 Purpose
Find the most likely sequence of hidden states (POS tags).

## 3.2 Key Insight
Use **dynamic programming** instead of enumerating all paths.

**Complexity**: O(N² × T)
- N = number of states (tags)
- T = sequence length

## 3.3 Viterbi Variable

```
Vₜ(j) = max P(q₁, q₂, ..., qₜ = j, o₁, o₂, ..., oₜ | λ)
```

Probability of best path ending in state j at time t.

## 3.4 Algorithm Steps

### Step 1: Initialization (t = 1)
```
V₁(j) = π(j) × B(j, o₁)
bp₁(j) = 0  (no backpointer for first state)
```

### Step 2: Recursion (t = 2 to T)
```
Vₜ(j) = max[Vₜ₋₁(i) × A(i,j)] × B(j, oₜ)
bpₜ(j) = argmax[Vₜ₋₁(i) × A(i,j)]
```

### Step 3: Termination
```
Best final state: q*ₜ = argmax Vₜ(j)
```

### Step 4: Backtracking
```
q*ₜ₋₁ = bpₜ(q*ₜ)
q*ₜ₋₂ = bpₜ₋₁(q*ₜ₋₁)
... and so on
```

## 3.5 Complete Example

**Sentence**: "I run"
**States**: PRP, VB, NN

**Parameters**:
```
Start probabilities:
π(PRP) = 0.6, π(VB) = 0.2, π(NN) = 0.2

Transition matrix:
       PRP   VB   NN
PRP    0.1   0.6   0.3
VB     0.2   0.2   0.6
NN     0.3   0.4   0.3

Emission probabilities:
P("I" | PRP) = 0.9, P("I" | VB) = 0, P("I" | NN) = 0
P("run" | PRP) = 0, P("run" | VB) = 0.4, P("run" | NN) = 0.2
```

### Step 1: Initialization (word = "I")

```
V₁(PRP) = π(PRP) × P("I"|PRP) = 0.6 × 0.9 = 0.54
V₁(VB) = π(VB) × P("I"|VB) = 0.2 × 0 = 0
V₁(NN) = π(NN) × P("I"|NN) = 0.2 × 0 = 0
```

### Step 2: Recursion (word = "run")

**For VB**:
```
From PRP: 0.54 × 0.6 = 0.324
From VB: 0 × 0.2 = 0
From NN: 0 × 0.4 = 0
Max = 0.324 (from PRP)
V₂(VB) = 0.324 × P("run"|VB) = 0.324 × 0.4 = 0.1296
bp₂(VB) = PRP
```

**For NN**:
```
From PRP: 0.54 × 0.3 = 0.162
From VB: 0 × 0.6 = 0
From NN: 0 × 0.3 = 0
Max = 0.162 (from PRP)
V₂(NN) = 0.162 × P("run"|NN) = 0.162 × 0.2 = 0.0324
bp₂(NN) = PRP
```

**For PRP**:
```
Max = 0.054 × 0 = 0
```

### Step 3: Termination
```
Best final state: VB (0.1296 > 0.0324 > 0)
```

### Step 4: Backtracking
```
q₂* = VB
q₁* = bp₂(VB) = PRP
```

**Result**: PRP → VB ("I" = pronoun, "run" = verb)

### Viterbi Table

| State | "I" | "run" | Backpointer |
|-------|-----|-------|-------------|
| PRP | 0.54 | 0 | - |
| VB | 0 | 0.1296 | PRP |
| NN | 0 | 0.0324 | PRP |

---

# 4. Log Probabilities in Viterbi

## 4.1 Problem
Products of small probabilities → underflow

## 4.2 Solution
Work in log space:
```
log(a × b) = log(a) + log(b)
```

## 4.3 Log Viterbi

**Initialization**:
```
log V₁(j) = log π(j) + log B(j, o₁)
```

**Recursion**:
```
log Vₜ(j) = max[log Vₜ₋₁(i) + log A(i,j)] + log B(j, oₜ)
```

**Example**:
```
log V₁(PRP) = log(0.6) + log(0.9) = -0.22 + (-0.05) = -0.27
log V₂(VB) = log(0.54) + log(0.6) + log(0.4) = -0.27 + (-0.22) + (-0.40) = -0.89
```

---

# 5. Maximum Entropy Markov Model (MEMM)

## 5.1 HMM Limitations

| Limitation | Description |
|------------|-------------|
| **Limited features** | Only word identity |
| **Independence** | Observations independent given states |
| **Generative** | Models P(word \| tag) not P(tag \| word) |

## 5.2 MEMM: Discriminative Approach

**Key difference**:
- HMM: P(word | tag) × P(tag | prev_tag) — Generative
- MEMM: P(tag | word, prev_tag, features) — Discriminative

## 5.3 Features in MEMM

| Feature Type | Examples |
|--------------|----------|
| **Current word** | word = "running" |
| **Previous tag** | prev_tag = VB |
| **Word suffix** | suffix = "-ing" |
| **Word prefix** | prefix = "un-" |
| **Capitalization** | is_capitalized = True |
| **Contains digit** | has_digit = False |
| **Previous word** | prev_word = "is" |
| **Next word** | next_word = "fast" |
| **Word shape** | shape = "Xxxxx" |

## 5.4 Maximum Entropy Model

**Formula**:
```
P(tag | features) = exp(Σᵢ wᵢ × fᵢ) / Z
```

Where:
- wᵢ = learned weights
- fᵢ = feature functions (0 or 1)
- Z = normalization constant

## 5.5 Example Features

For word "unhappiness" with previous tag = JJ:

| Feature | Value |
|---------|-------|
| word = "unhappiness" | 1 |
| prev_tag = JJ | 1 |
| suffix = "-ness" | 1 |
| prefix = "un-" | 1 |
| length > 8 | 1 |
| is_capitalized | 0 |

## 5.6 MEMM vs HMM

| Aspect | HMM | MEMM |
|--------|-----|------|
| **Type** | Generative | Discriminative |
| **Models** | P(word\|tag) | P(tag\|word, features) |
| **Features** | Word only | Arbitrary overlapping |
| **Training** | Count-based | Gradient-based |
| **Inference** | Viterbi | Modified Viterbi |

---

# 6. Bidirectionality

## 6.1 Problem with Left-to-Right

HMM and MEMM only use **left context**.

**Example**: "I saw her duck"
- Without right context: "duck" could be noun or verb
- With right context: If next word is "fly" → verb, if next is "." → likely noun

## 6.2 Bidirectional Models

**Approach**: Use both left AND right context

**Methods**:
1. **Bi-LSTM**: Process sequence in both directions
2. **Transformers**: Self-attention sees all positions
3. **CRF layer**: Considers whole sequence

---

# 7. Neural Network Models for POS Tagging

## 7.1 Basic Neural Tagger

```
Word → Embedding → Feed-forward NN → Softmax → Tag
```

**Limitation**: Only considers current word

## 7.2 RNN-based Tagger

```
[w₁, w₂, w₃, ...] → [e₁, e₂, e₃, ...] → RNN → [h₁, h₂, h₃, ...] → [t₁, t₂, t₃, ...]
```

**Advantage**: Captures sequential context

## 7.3 Bi-LSTM Tagger

```
Forward LSTM:  → → → →
              h₁ h₂ h₃ h₄
              ↓  ↓  ↓  ↓
Words:        w₁ w₂ w₃ w₄
              ↑  ↑  ↑  ↑
              h₁ h₂ h₃ h₄
Backward LSTM: ← ← ← ←

Combined: [h_forward; h_backward] → Dense → Softmax → Tag
```

**Accuracy**: ~97%

## 7.4 Bi-LSTM-CRF

**Why CRF layer?**
- Ensures valid tag sequences
- Models tag transitions globally
- Better than independent softmax per position

**Architecture**:
```
Words → Embeddings → Bi-LSTM → CRF → Tags
```

**Accuracy**: ~97.5%

## 7.5 Transformer-based (BERT)

**Process**:
1. Tokenize with WordPiece
2. Pass through BERT encoder
3. Add classification head
4. Fine-tune on POS data

**Accuracy**: ~98-99%

---

# 8. Comparison of POS Tagging Methods

| Method | Accuracy | Features | Context |
|--------|----------|----------|---------|
| **Rule-based** | ~90% | Rules | Limited |
| **HMM** | ~95% | Word only | Left (n-gram) |
| **MEMM** | ~96% | Rich features | Left |
| **Bi-LSTM** | ~97% | Embeddings | Bidirectional |
| **Bi-LSTM-CRF** | ~97.5% | Embeddings + CRF | Bidirectional |
| **BERT** | ~98.5% | Contextualized | Full sentence |

---

# 9. Handling Unknown Words

## 9.1 The OOV Problem
Words not seen in training have no emission probabilities.

## 9.2 Solutions

| Solution | Description |
|----------|-------------|
| **Suffix rules** | "-ing" → often VBG |
| **Word shape** | "McDonals" → NNP |
| **Subword embeddings** | FastText, BPE |
| **Character-level models** | Char-CNN, Char-LSTM |

---

# 10. Key Formulas

| Algorithm | Key Formula |
|-----------|-------------|
| Viterbi Init | V₁(j) = π(j) × B(j, o₁) |
| Viterbi Recursion | Vₜ(j) = max[Vₜ₋₁(i) × A(i,j)] × B(j, oₜ) |
| Log Viterbi | log Vₜ = max[log Vₜ₋₁ + log A] + log B |
| MaxEnt | P(y\|x) = exp(Σwf) / Z |
| Forward | αₜ(j) = [Σᵢ αₜ₋₁(i) × A(i,j)] × B(j, oₜ) |

---

# 📝 Practice Questions

## Q1. Viterbi Calculation
Given:
- Start: π(N)=0.5, π(V)=0.5
- Trans: P(V|N)=0.5, P(N|V)=0.3
- Emit: P("fish"|N)=0.2, P("fish"|V)=0.3

Fill Viterbi table for "fish fish".

## Q2. Why does MEMM use P(tag|word) instead of P(word|tag)?

## Q3. What features would you use for word "unhappily" preceded by "was"?

## Q4. Compare Bi-LSTM and Bi-LSTM-CRF. Why add CRF?

## Q5. Calculate log Viterbi score:
- log V₁(N) = -2
- log A(N,V) = -0.5
- log B(V, word) = -1.2

---

*Reference: Session 7 - Statistical, ML and Neural Models of POS Tagging*
