# NLP Regular Exam Dec 2025 - Questions Only

## Page 1

### Q1 (4 Marks) - Module 1: Introduction

(a) List any four applications of NLP with brief explanation. (2 marks)

(b) Identify the type of ambiguity in the following sentences: (2 marks)
   - "I saw her duck"
   - "The professor said on Monday he would give an exam"

---

### Q2 (4 Marks) - Module 2: Language Models

Given corpus:
```
<s> I want to eat Chinese food </s>
<s> I want to eat Italian food </s>  
<s> I want Chinese food </s>
```

(a) Calculate P(Chinese | want) (1 mark)

(b) Calculate P(to | want) (1 mark)

(c) Calculate perplexity of "I want to eat Chinese food" given: (2 marks)
   - P(I | <s>) = 0.67
   - P(want | I) = 1.0
   - P(to | want) = 0.67
   - P(eat | to) = 1.0
   - P(Chinese | eat) = 0.5
   - P(food | Chinese) = 1.0

---

## Page 2

### Q3 (4 Marks) - Module 3: Neural LM and LLM

(a) Explain Pre-training and Fine-tuning in context of LLMs. (2 marks)

(b) Differentiate between Zero-shot and Few-shot prompting. Give example of Few-shot for sentiment classification. (2 marks)

---

### Q4 (4 Marks) - Module 4: Vector Semantics

(a) Calculate TF-IDF for term "data" in Document 1: (2 marks)
   - "data" appears 6 times in Document 1
   - Total documents N = 10,000
   - Documents containing "data" = 100
   - Use TF = 1 + log₁₀(tf), IDF = log₁₀(N/df)

(b) Calculate Cosine Similarity between: (2 marks)
   - A = [1, 2, 3]
   - B = [2, 3, 4]

---

### Q5 (5 Marks) - Module 4: Word Embeddings

(a) In Skip-gram, for sentence "The quick brown fox" with center word "quick" and window size 1, list training pairs. (2 marks)

(b) Given vectors: (1.5 marks)
   - v(man) = [1.0, 0.5]
   - v(woman) = [1.0, 1.0]
   - v(king) = [2.0, 1.0]
   
   Calculate v(queen) using analogy "man:woman :: king:queen"

(c) Skip-gram Negative Sampling update: (1.5 marks)
   - v = [0.5, 0.5]
   - u = [1.0, 0.0]
   - σ(v·u) = 0.62
   - Positive pair (y = 1)
   - η = 0.1
   
   Calculate updated v'

---

## Page 3

### Q6 (4 Marks) - Module 5: POS Tagging

Given counts:
- C(DT) = 100, C(NN) = 200, C(VB) = 150
- C(DT, NN) = 80, C(NN, VB) = 60
- C("the", DT) = 40, C("book", NN) = 20, C("book", VB) = 10

(a) Calculate P(NN|DT) and P(VB|NN) (2 marks)

(b) For word "book" after DT, determine which tag (NN or VB) is more likely using HMM. 
    Given: P(NN|DT) = 0.8, P(VB|DT) = 0.1 (2 marks)

---

### Q7 (5 Marks) - Module 6: Viterbi Algorithm

Find best tag sequence for "The man runs" using Viterbi.

States: DT, NN, VB

Start probabilities:
- π(DT) = 0.6, π(NN) = 0.3, π(VB) = 0.1

Transition probabilities:
|      | DT  | NN  | VB  |
|------|-----|-----|-----|
| DT   | 0.1 | 0.8 | 0.1 |
| NN   | 0.1 | 0.3 | 0.6 |
| VB   | 0.2 | 0.5 | 0.3 |

Emission probabilities:
| Word | P(·|DT) | P(·|NN) | P(·|VB) |
|------|---------|---------|---------|
| The  | 0.7     | 0.02    | 0.01    |
| man  | 0.01    | 0.4     | 0.02    |
| runs | 0.01    | 0.1     | 0.6     |

Show Viterbi table with backpointers and find best path.

---

**Total: 30 Marks**
