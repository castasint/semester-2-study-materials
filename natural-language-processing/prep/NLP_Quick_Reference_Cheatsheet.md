# 📋 NLP Quick Reference Cheatsheet
## AIMLCZG530 - Last Minute Revision Guide

---

# 🔢 Essential Formulas

## Module 2: N-gram & Language Models

| Formula | Equation |
|---------|----------|
| **Bigram Probability** | `P(wₙ|wₙ₋₁) = Count(wₙ₋₁,wₙ) / Count(wₙ₋₁)` |
| **Laplace Smoothing** | `P = (Count + 1) / (N + V)` |
| **Perplexity** | `PP = P(W)^(-1/N) = ⁿ√(1/P(W))` |
| **Interpolation** | `P = λ₁P₃ + λ₂P₂ + λ₃P₁` where `Σλ = 1` |

**Perplexity Interpretation:** Lower = Better. PP of 10 means model choosing from ~10 words.

---

## Module 4: Vector Semantics

| Formula | Equation |
|---------|----------|
| **Term Frequency** | `TF = 1 + log₁₀(count)` |
| **Inverse Doc Freq** | `IDF = log₁₀(N / df)` |
| **TF-IDF** | `TF-IDF = TF × IDF` |
| **Cosine Similarity** | `cos(A,B) = (A·B) / (‖A‖ × ‖B‖)` |
| **Euclidean Distance** | `d = √Σ(aᵢ - bᵢ)²` |
| **Document Centroid** | `D = (v₁ + v₂ + ... + vₙ) / n` |

**Key Insight:** Words appearing in ALL documents have IDF = 0 (not discriminative!)

---

## Module 4: Word Embeddings

| Formula | Equation |
|---------|----------|
| **Word Analogy** | `v_target = v_a* - v_a + v_b` |
| **SGNS Error** | `Error = σ(v·u) - y` (y=1 for positive, 0 for negative) |
| **Vector Update** | `v_new = v_old - η × Error × u` |

**Skip-gram vs CBOW:**
- Skip-gram: target → context (better for rare words)
- CBOW: context → target (faster training)

---

## Module 5 & 6: HMM & Viterbi

| Formula | Equation |
|---------|----------|
| **Transition Prob** | `P(tagᵢ|tagᵢ₋₁) = Count(tagᵢ₋₁→tagᵢ) / Count(tagᵢ₋₁)` |
| **Emission Prob** | `P(word|tag) = Count(tag,word) / Count(tag)` |
| **HMM Score** | `Score = Transition × Emission` |
| **Viterbi Init** | `V₁(j) = π(j) × P(o₁|j)` |
| **Viterbi Recursion** | `Vₜ(j) = max[Vₜ₋₁(i) × P(j|i)] × P(oₜ|j)` |

**Viterbi Complexity:** O(N² × T) where N = states, T = sequence length

---

# 📝 Key Concepts Summary

## Types of Ambiguity
| Type | Example |
|------|---------|
| Structural | "Flying planes can be dangerous" |
| Lexical (Polysemy) | "bank" = river/financial |
| Grammatical | "can" = modal/verb/noun |

## Levels of Language Analysis
```
Pragmatic → Discourse → Semantic → Syntactic → Lexical → Morphological
```

## HMM Components
- **Hidden States:** POS tags (what we want to find)
- **Observations:** Words (what we see)
- **Transition:** P(tag | prev_tag)
- **Emission:** P(word | tag)

## HMM vs MEMM
| Aspect | HMM | MEMM |
|--------|-----|------|
| Type | Generative P(word\|tag) | Discriminative P(tag\|word) |
| Features | Limited | Rich, overlapping |

---

# ⚡ Quick Calculation Steps

## TF-IDF Calculation
```
1. TF = 1 + log₁₀(word_count_in_doc)
2. IDF = log₁₀(total_docs / docs_with_word)
3. TF-IDF = TF × IDF
```

## Cosine Similarity
```
1. Dot product: Σ(aᵢ × bᵢ)
2. Magnitude A: √Σ(aᵢ²)
3. Magnitude B: √Σ(bᵢ²)
4. Cosine = dot / (mag_A × mag_B)
```

## Perplexity
```
1. P(sentence) = P(w₁) × P(w₂|w₁) × P(w₃|w₂) × ...
2. N = number of words
3. PP = P(sentence)^(-1/N)
```

## Word2Vec Update (Skip-gram SGNS)
```
1. Error = sigmoid(v·u) - y
2. Gradient = Error × u
3. v_new = v_old - learning_rate × Gradient
```

## Viterbi Algorithm
```
1. INIT: V₁(tag) = π(tag) × P(word₁|tag)
2. RECURSE: Vₜ(j) = max[Vₜ₋₁(i) × Trans(i→j)] × Emit(j)
3. BACKTRACK: Follow backpointers from max final state
```

## HMM Disambiguation
```
1. For each candidate tag:
   Score = P(tag|prev_tag) × P(word|tag)
2. Choose tag with highest score
```

---

# 🎯 Common Exam Patterns

## Q1 (4 marks): Introduction
- List 4 NLP applications
- Identify ambiguity types
- Explain language levels

## Q2 (4 marks): N-gram & Perplexity
- Calculate bigram probabilities
- Apply Laplace smoothing
- Calculate perplexity

## Q3 (4 marks): Neural LM & LLM
- Compare N-gram vs Neural LM
- Explain pre-training vs fine-tuning
- Zero-shot vs Few-shot prompting

## Q4 (4 marks): Vector Semantics
- TF-IDF calculation
- Cosine similarity
- Document embeddings

## Q5 (5 marks): Word Embeddings
- Skip-gram training pairs
- Word2Vec update calculation
- Word analogy (parallelogram)

## Q6 (4 marks): HMM POS Tagging
- Transition probabilities
- HMM disambiguation
- Hidden vs Observed states

## Q7 (5 marks): Viterbi & MEMM
- Viterbi table construction
- Backtracking
- HMM vs MEMM comparison

---

# ⚠️ Common Mistakes to Avoid

| Mistake | Correct Approach |
|---------|------------------|
| Forgetting +1 in TF | TF = 1 + log₁₀(count) |
| Wrong order in HMM | Score = Transition × Emission |
| Skipping magnitude in cosine | Always calculate both ‖A‖ and ‖B‖ |
| Wrong perplexity power | PP = P^(-1/N), not P^(1/N) |
| Viterbi: forgetting emission | Vₜ = max[...] × P(obs\|state) |

---

# 📊 Visual Quick Reference (13 Diagrams)

## Module 1: Introduction
![Levels of Language Analysis](./images/language_levels.png)
![Types of Ambiguity](./images/ambiguity_types.png)

## Module 2: Vector Semantics & Embeddings
![TF-IDF Calculation](./images/tfidf.png)
![Cosine Similarity](./images/cosine.png)
![Word2Vec Skip-gram](./images/word2vec.png)
![CBOW Architecture](./images/cbow.png)
![Word Analogy](./images/analogy.png)

## Module 3: N-gram Language Models
![N-gram Windows](./images/ngram.png)
![Smoothing Techniques](./images/smoothing.png)
![Perplexity](./images/perplexity.png)

## Module 4: Neural Language Models
![Neural LM Architecture](./images/neural_lm.png)

## Module 5 & 6: POS Tagging
![HMM for POS](./images/hmm.png)
![Viterbi Trellis](./images/viterbi.png)

---

**Good luck! 🎓 Remember: Show all steps for partial credit!**
