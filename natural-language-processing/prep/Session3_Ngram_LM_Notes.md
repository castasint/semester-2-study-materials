# Session 3: N-gram Language Modeling
## AIMLCZG530 - Natural Language Processing

---

# 1. Language Models - Introduction

## 1.1 What is a Language Model?

A **Language Model** assigns probabilities to sequences of words.

**Two key tasks**:
1. **P(W)**: Probability of a sentence/sequence
2. **P(wₙ | w₁...wₙ₋₁)**: Probability of next word given history

## 1.2 Why Language Models Matter

| Application | How LM Helps |
|-------------|--------------|
| **Speech Recognition** | Choose most likely word sequence |
| **Machine Translation** | Select fluent output |
| **Spelling Correction** | Rank correction candidates |
| **Text Generation** | Predict next words |
| **Autocomplete** | Suggest completions |

## 1.3 Formal Definition

For a sentence W = w₁, w₂, ..., wₙ:
```
P(W) = P(w₁, w₂, ..., wₙ)
```

---

# 2. N-Grams

## 2.1 Chain Rule of Probability

**Exact computation**:
```
P(w₁, w₂, ..., wₙ) = P(w₁) × P(w₂|w₁) × P(w₃|w₁,w₂) × ... × P(wₙ|w₁...wₙ₋₁)
```

**Problem**: Need to estimate P(wₙ | w₁...wₙ₋₁) - history can be very long!

## 2.2 Markov Assumption

**Key Insight**: Approximate by looking at only last (n-1) words

```
P(wₙ | w₁...wₙ₋₁) ≈ P(wₙ | wₙ₋ₖ₊₁...wₙ₋₁)
```

## 2.3 Types of N-grams

| N | Name | Formula | Context |
|---|------|---------|---------|
| 1 | Unigram | P(wₙ) | No context |
| 2 | Bigram | P(wₙ \| wₙ₋₁) | 1 previous word |
| 3 | Trigram | P(wₙ \| wₙ₋₂, wₙ₋₁) | 2 previous words |
| 4 | 4-gram | P(wₙ \| wₙ₋₃, wₙ₋₂, wₙ₋₁) | 3 previous words |

## 2.4 Estimating N-gram Probabilities (MLE)

**Maximum Likelihood Estimation**:

### Bigram
```
P(wₙ | wₙ₋₁) = Count(wₙ₋₁, wₙ) / Count(wₙ₋₁)
```

### Trigram
```
P(wₙ | wₙ₋₂, wₙ₋₁) = Count(wₙ₋₂, wₙ₋₁, wₙ) / Count(wₙ₋₂, wₙ₋₁)
```

## 2.5 Complete Example

**Corpus**:
```
<s> I am happy </s>
<s> I am learning NLP </s>
<s> I love NLP </s>
```

**Counts**:
- Count("I") = 3
- Count("I", "am") = 2
- Count("I", "love") = 1
- Count("am") = 2
- Count("am", "happy") = 1
- Count("am", "learning") = 1

**Bigram Probabilities**:
```
P(am | I) = Count(I, am) / Count(I) = 2/3 = 0.667
P(love | I) = Count(I, love) / Count(I) = 1/3 = 0.333
P(happy | am) = Count(am, happy) / Count(am) = 1/2 = 0.5
P(learning | am) = Count(am, learning) / Count(am) = 1/2 = 0.5
```

## 2.6 Sentence Probability

**Example**: P("I am happy")

```
P(<s> I am happy </s>) = P(I|<s>) × P(am|I) × P(happy|am) × P(</s>|happy)
```

---

# 3. Generalization and the Zero Problem

## 3.1 The Sparsity Problem

**Training corpus**: "I saw a dog"
**Test sentence**: "I saw a cat"

P(cat | a) = Count(a, cat) / Count(a) = 0/1 = **0**

**Result**: Entire sentence probability becomes 0!

## 3.2 Why This Happens

- Language is **creative** - infinite possible sentences
- Any finite corpus will miss many valid n-grams
- **Unseen n-grams** get probability 0

## 3.3 Solutions Overview

| Technique | Approach |
|-----------|----------|
| **Smoothing** | Add counts to unseen events |
| **Backoff** | Use lower-order n-gram if higher-order unseen |
| **Interpolation** | Combine multiple n-gram orders |

---

# 4. Smoothing Techniques

## 4.1 Laplace (Add-1) Smoothing

**Idea**: Add 1 to all counts

**Formula**:
```
P_Laplace(wₙ | wₙ₋₁) = [Count(wₙ₋₁, wₙ) + 1] / [Count(wₙ₋₁) + V]
```

Where V = vocabulary size

**Example**:
- Count("dog", "runs") = 0
- Count("dog") = 100
- V = 10,000

```
P_MLE(runs | dog) = 0/100 = 0

P_Laplace(runs | dog) = (0 + 1) / (100 + 10,000) = 1/10,100 ≈ 0.0001
```

**Problem**: Steals too much probability from seen events

## 4.2 Add-k Smoothing

**Generalization**: Add k instead of 1 (k < 1)

```
P_Add-k(wₙ | wₙ₋₁) = [Count(wₙ₋₁, wₙ) + k] / [Count(wₙ₋₁) + k×V]
```

## 4.3 Linear Interpolation

**Idea**: Combine unigram, bigram, trigram with weights

**Formula**:
```
P(wₙ|wₙ₋₂,wₙ₋₁) = λ₁×P(wₙ|wₙ₋₂,wₙ₋₁) + λ₂×P(wₙ|wₙ₋₁) + λ₃×P(wₙ)
```

**Constraints**: λ₁ + λ₂ + λ₃ = 1

**Example**:
- λ₁ = 0.6, λ₂ = 0.3, λ₃ = 0.1
- P(runs | dog, the) = 0.01 (trigram)
- P(runs | the) = 0.05 (bigram)
- P(runs) = 0.001 (unigram)

```
P_interp = 0.6×0.01 + 0.3×0.05 + 0.1×0.001
         = 0.006 + 0.015 + 0.0001
         = 0.0211
```

## 4.4 Backoff

**Idea**: Use higher-order n-gram if available, otherwise "back off" to lower order

**Algorithm**:
```
if Count(wₙ₋₂, wₙ₋₁, wₙ) > 0:
    use P_trigram(wₙ | wₙ₋₂, wₙ₋₁)
elif Count(wₙ₋₁, wₙ) > 0:
    use α × P_bigram(wₙ | wₙ₋₁)
else:
    use α × α × P_unigram(wₙ)
```

Where α is a discount factor

---

# 5. Stupid Backoff

## 5.1 Motivation
- Used at **web scale** (billions of words)
- Google's solution for very large corpora
- Simple and effective

## 5.2 Formula

```
S(wᵢ | wᵢ₋ₖ₊₁...wᵢ₋₁) = 
    Count(wᵢ₋ₖ₊₁...wᵢ) / Count(wᵢ₋ₖ₊₁...wᵢ₋₁)    if count > 0
    0.4 × S(wᵢ | wᵢ₋ₖ₊₂...wᵢ₋₁)                   otherwise
```

## 5.3 Key Properties

| Property | Description |
|----------|-------------|
| **No normalization** | Not a proper probability distribution |
| **Works well at scale** | Effective for large datasets |
| **Simple** | Easy to implement |
| **Fixed backoff weight** | Always multiply by 0.4 |

## 5.4 Example

Calculate S(happy | am, I):

```
If Count(I, am, happy) > 0:
    S = Count(I, am, happy) / Count(I, am)
Else:
    S = 0.4 × S(happy | am)
    
    If Count(am, happy) > 0:
        S = 0.4 × [Count(am, happy) / Count(am)]
    Else:
        S = 0.4 × 0.4 × S(happy)
        S = 0.16 × [Count(happy) / Total_words]
```

---

# 6. Evaluating Language Models

## 6.1 Extrinsic Evaluation
- Evaluate on downstream task (MT, ASR)
- **Pros**: Real-world performance
- **Cons**: Expensive, task-specific

## 6.2 Intrinsic Evaluation: Perplexity

### Definition
Perplexity measures how "surprised" the model is by test data.

### Formula
```
PP(W) = P(w₁, w₂, ..., wₙ)^(-1/N)
      = ⁿ√(1 / P(W))
```

For bigram model:
```
PP(W) = [∏ᵢ P(wᵢ | wᵢ₋₁)]^(-1/N)
```

### Interpretation

| Perplexity | Interpretation |
|------------|----------------|
| Lower | Better model |
| Higher | Worse model |
| = k | Like choosing uniformly from k words |

### Intuition
- PP = 10 means model is as confused as randomly picking from 10 words
- "Branching factor" of the model

## 6.3 Perplexity Calculation Example

**Sentence**: "I love NLP" (N = 3 words)

**Probabilities**:
- P(I) = 0.1
- P(love | I) = 0.2
- P(NLP | love) = 0.05

**Calculate**:
```
P(sentence) = 0.1 × 0.2 × 0.05 = 0.001

PP = (0.001)^(-1/3)
   = (1/0.001)^(1/3)
   = (1000)^(1/3)
   = 10
```

Perplexity = 10 → Model choosing from ~10 equally likely words

## 6.4 Log Perplexity

To avoid numerical underflow:
```
log PP = -1/N × Σᵢ log P(wᵢ | history)
```

## 6.5 Typical Perplexity Values

| Model | Perplexity (WSJ corpus) |
|-------|-------------------------|
| Unigram | ~962 |
| Bigram | ~170 |
| Trigram | ~109 |
| Neural LM | ~50-80 |

---

# 7. Practical Considerations

## 7.1 Unknown Words (OOV)

**Problem**: Words not in vocabulary

**Solutions**:
1. **<UNK> token**: Replace rare words with special token
2. **Threshold**: Words appearing < k times → <UNK>
3. **Character-level**: Handle at character level

## 7.2 Sentence Boundaries

Add special tokens:
- **<s>**: Start of sentence
- **</s>**: End of sentence

**Example**:
```
"I love NLP" → "<s> I love NLP </s>"
```

## 7.3 Log Probabilities

**Problem**: Probability products become very small

**Solution**: Work in log space
```
log P(w₁...wₙ) = Σᵢ log P(wᵢ | history)
```

## 7.4 Choosing N

| N | Pros | Cons |
|---|------|------|
| 1 (Unigram) | Many counts | No context |
| 2 (Bigram) | Some context | Limited history |
| 3 (Trigram) | Better context | Sparse counts |
| 4+ | More context | Very sparse |

**Trade-off**: More context vs. more reliable estimates

---

# 8. N-gram vs Neural Language Models

| Aspect | N-gram | Neural LM |
|--------|--------|-----------|
| **Context** | Fixed (n-1 words) | Variable/Long |
| **Parameters** | O(V^n) | O(V × d) |
| **Generalization** | None (exact match) | Embedding-based |
| **Training** | Counting | Gradient descent |
| **Sparsity** | Major problem | Not an issue |
| **Interpretability** | High | Low |
| **Speed** | Fast | Slower |

---

# 9. Key Formulas Summary

| Concept | Formula |
|---------|---------|
| Bigram MLE | `P(wₙ\|wₙ₋₁) = C(wₙ₋₁,wₙ) / C(wₙ₋₁)` |
| Laplace | `P = (C+1) / (N+V)` |
| Interpolation | `P = λ₁P₃ + λ₂P₂ + λ₃P₁` |
| Stupid Backoff | `S = 0.4 × S(lower-order)` |
| Perplexity | `PP = P(W)^(-1/N)` |
| Log Perplexity | `log PP = -1/N × Σ log P(wᵢ)` |

---

# 📝 Practice Questions

## Q1. Bigram Probability
Corpus: "I am happy. I am sad. I love happy."
Calculate: P(happy | am), P(sad | am), P(am | I)

## Q2. Laplace Smoothing
- Count("the", "cat") = 5
- Count("the") = 100
- V = 20,000
Calculate P_Laplace(cat | the)

## Q3. Perplexity
Sentence: "dogs run fast" (3 words)
P(dogs) = 0.01, P(run|dogs) = 0.1, P(fast|run) = 0.05
Calculate perplexity.

## Q4. Linear Interpolation
λ₁=0.5, λ₂=0.3, λ₃=0.2
P(runs|dog,the) = 0.02, P(runs|the) = 0.04, P(runs) = 0.001
Calculate interpolated probability.

## Q5. Stupid Backoff
Count(I, love, NLP) = 0
Count(love, NLP) = 10
Count(love) = 200
Calculate S(NLP | love, I)

---

*Reference: Session 3 - N-gram Language Modeling*
