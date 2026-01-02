# 📇 NLP MIDSEM FLASHCARDS
## Quick Reference for Last 30 Minutes

---

# 🔴 QUESTION 1: INTRODUCTION (4 Marks)

## 4 NLP Applications
1. **Machine Translation** - Google Translate
2. **Sentiment Analysis** - Positive/Negative detection
3. **NER** - Finding names, places, organizations
4. **Question Answering** - Siri, Alexa

## 6 Levels (Bottom to Top)
**M**orphological → **L**exical → **S**yntactic → **S**emantic → **D**iscourse → **P**ragmatic

## 3 Types of Ambiguity
- **Structural**: "I saw man with telescope" (who has it?)
- **Lexical**: "bank" (river/financial)
- **Grammatical**: "can" (modal/verb/noun)

---

# 🟠 QUESTION 2: N-GRAM & PERPLEXITY (4 Marks)

## Bigram
```
P(word | prev) = C(prev, word) / C(prev)
```

## Laplace Smoothing
```
P = (C + 1) / (N + V)     V = vocab size
```

## Perplexity
```
PP = P(sentence)^(-1/N)   Lower = Better!
```

**Example**: P = 0.04, N = 3 → PP = (25)^(1/3) ≈ 2.92

---

# 🟡 QUESTION 3: NEURAL LM & LLM (4 Marks)

## N-gram vs Neural LM
| N-gram | Neural |
|--------|--------|
| Fixed context | Long context |
| No generalization | Embeddings help |
| Sparsity problem | No sparsity |

## Prompting Types
- **Zero-shot**: No examples
- **One-shot**: 1 example
- **Few-shot**: Multiple examples
- **Chain-of-Thought**: Step by step reasoning

---

# 🟢 QUESTION 4: VECTOR SEMANTICS (4 Marks)

## TF-IDF
```
TF = 1 + log₁₀(count)      ← Don't forget the 1!
IDF = log₁₀(N / df)
TF-IDF = TF × IDF
```

## Cosine Similarity
```
cos = (A·B) / (||A|| × ||B||)
||A|| = √(a₁² + a₂² + ...)
```

**Example**: A = [2,1,0,2], B = [1,1,2,1]
- Dot = 5, ||A|| = 3, ||B|| = √7 = 2.65
- cos = 5/(3×2.65) = 0.63

---

# 🔵 QUESTION 5: WORD EMBEDDINGS (5 Marks)

## Word Analogy
```
v_Queen = v_King - v_Man + v_Woman
```

## Skip-gram vs CBOW
- **Skip-gram**: target → context (better for rare)
- **CBOW**: context → target (faster)

## Word2Vec Update
```
Error = σ(v·u) - y    (y=1 positive, y=0 negative)
v_new = v_old - η × Error × u
```
- Positive pair → vectors CLOSER
- Negative pair → vectors APART

---

# 🟣 QUESTION 6: HMM POS TAGGING (4 Marks)

## HMM Components
- Hidden = Tags (NN, VB, DT)
- Observed = Words
- Transition = P(tag | prev_tag)
- Emission = P(word | tag)

## HMM Disambiguation
```
Score = P(tag | prev) × P(word | tag)
Choose HIGHEST score!
```

**Example**: "flies" after NN
- Score(NN) = 0.3 × 0.02 = 0.006
- Score(VBZ) = 0.4 × 0.05 = 0.020 ← Winner!

---

# ⚫ QUESTION 7: VITERBI & MEMM (5 Marks)

## Viterbi 3 Steps
1. **INIT**: V₁(j) = π(j) × P(word₁|j)
2. **RECURSE**: Vₜ(j) = max[Vₜ₋₁(i) × A(i,j)] × B(j,wordₜ)
3. **BACKTRACK**: Follow pointers from max final

## HMM vs MEMM
| HMM | MEMM |
|-----|------|
| Generative | Discriminative |
| P(word\|tag) | P(tag\|word, features) |
| Limited features | Rich features |

---

# ⚠️ DON'T FORGET!

| Formula | Key Point |
|---------|-----------|
| TF | **1** + log₁₀(count) |
| PP | Power is **NEGATIVE**: -1/N |
| HMM | Transition **×** Emission |
| Viterbi | Don't forget **emission** at end |
| Cosine | Calculate **BOTH** magnitudes |

---

# 🏆 EXAM ORDER
1. Q4+Q5 (Vector + Embedding) - 9 marks
2. Q6+Q7 (HMM + Viterbi) - 9 marks
3. Q2 (N-gram) - 4 marks
4. Q3 (LLM) - 4 marks
5. Q1 (Theory) - 4 marks

---

**🍀 Show ALL steps = Partial credit!**
