# Session 2: Vector Semantics and Word Embeddings
## AIMLCZG530 - Natural Language Processing

---

# 1. Lexical Semantics

## 1.1 What is Lexical Semantics?
- Study of word meanings and relationships between words
- How words relate to concepts, objects, and other words

## 1.2 Lexical Relations

| Relation | Definition | Example |
|----------|------------|---------|
| **Synonymy** | Similar meaning | happy ≈ joyful |
| **Antonymy** | Opposite meaning | hot ↔ cold |
| **Hypernymy** | Is-a relationship (broader) | dog → animal |
| **Hyponymy** | Is-a relationship (narrower) | animal ← dog |
| **Meronymy** | Part-of relationship | wheel → car |
| **Holonymy** | Has-part relationship | car → wheel |

## 1.3 Word Senses
- **Polysemy**: One word, multiple related meanings
  - "bank" - financial institution, river bank
- **Homonymy**: Same form, unrelated meanings
  - "bat" - flying mammal, sports equipment

---

# 2. Vector Semantics

## 2.1 The Distributional Hypothesis

> **"You shall know a word by the company it keeps"** - J.R. Firth (1957)

**Core Idea**: Words appearing in similar contexts have similar meanings.

**Example**:
- "I had orange juice for breakfast"
- "I had apple juice for breakfast"
- Since "orange" and "apple" appear in identical contexts → they're semantically similar

## 2.2 Why Vectors for Meaning?

| Traditional | Vector-based |
|-------------|--------------|
| Dictionary definitions | Learned from data |
| Discrete symbols | Continuous representations |
| No similarity measure | Cosine similarity |
| Manual creation | Automatic learning |

## 2.3 Types of Vector Representations

### Sparse Vectors (Count-based)
- High-dimensional (vocabulary size ~50,000+)
- Mostly zeros
- Examples: Term-Document Matrix, TF-IDF

### Dense Vectors (Embeddings)
- Low-dimensional (50-300 dimensions)
- Learned representations
- Examples: Word2Vec, GloVe, FastText

---

# 3. Words and Vectors

## 3.1 Term-Document Matrix

**Example Corpus**:
- D1: "I love NLP"
- D2: "NLP is fun"
- D3: "I love fun"

**Matrix**:
|        | D1 | D2 | D3 |
|--------|----|----|----| 
| I      | 1  | 0  | 1  |
| love   | 1  | 0  | 1  |
| NLP    | 1  | 1  | 0  |
| is     | 0  | 1  | 0  |
| fun    | 0  | 1  | 1  |

Each column = document vector
Each row = word across documents

## 3.2 Term-Term (Co-occurrence) Matrix

**Word represented by its context words**

|        | I | love | NLP | is | fun |
|--------|---|------|-----|----|-----|
| I      | 0 | 2    | 1   | 0  | 1   |
| love   | 2 | 0    | 1   | 0  | 1   |
| NLP    | 1 | 1    | 0   | 1  | 1   |

Each row = word vector based on co-occurrence

## 3.3 Problems with Raw Counts

1. **Frequency bias**: Common words dominate
2. **High dimensionality**: Vocabulary size
3. **Sparsity**: Most entries are zero
4. **No weighting**: All co-occurrences equal

---

# 4. TF-IDF (Term Frequency - Inverse Document Frequency)

## 4.1 Intuition
- **TF**: Words appearing often in a document are important for that document
- **IDF**: Words appearing in few documents are more discriminative

## 4.2 Formulas

### Term Frequency (TF)
```
tf(t, d) = 1 + log₁₀(count(t, d))    if count > 0
         = 0                          if count = 0
```

OR (raw frequency):
```
tf(t, d) = count(t, d)
```

### Inverse Document Frequency (IDF)
```
idf(t) = log₁₀(N / df(t))
```
Where:
- N = Total number of documents
- df(t) = Number of documents containing term t

### TF-IDF Weight
```
w(t, d) = tf(t, d) × idf(t)
```

## 4.3 Complete Example

**Corpus**: N = 1,000,000 documents

| Word | Frequency in Doc | Documents containing word |
|------|-----------------|---------------------------|
| "the" | 100 | 1,000,000 |
| "algorithm" | 5 | 10,000 |
| "neural" | 3 | 5,000 |

**Calculations**:

For "the":
```
tf = 1 + log₁₀(100) = 1 + 2 = 3
idf = log₁₀(1,000,000 / 1,000,000) = log₁₀(1) = 0
TF-IDF = 3 × 0 = 0  ← Common word, no discriminative value!
```

For "algorithm":
```
tf = 1 + log₁₀(5) = 1 + 0.699 = 1.699
idf = log₁₀(1,000,000 / 10,000) = log₁₀(100) = 2
TF-IDF = 1.699 × 2 = 3.398
```

For "neural":
```
tf = 1 + log₁₀(3) = 1 + 0.477 = 1.477
idf = log₁₀(1,000,000 / 5,000) = log₁₀(200) = 2.301
TF-IDF = 1.477 × 2.301 = 3.399
```

## 4.4 Key Insight
- **High TF-IDF**: Frequent in document, rare in corpus → Important for this document
- **Low TF-IDF**: Either rare in document OR common everywhere → Not discriminative

---

# 5. Document Similarity

## 5.1 Cosine Similarity

**Formula**:
```
cos(A, B) = (A · B) / (||A|| × ||B||)
          = Σ(aᵢ × bᵢ) / (√Σaᵢ² × √Σbᵢ²)
```

**Interpretation**:
| Value | Meaning |
|-------|---------|
| 1 | Identical direction |
| 0 | Orthogonal (no similarity) |
| -1 | Opposite directions |

## 5.2 Step-by-Step Example

**Vectors**: A = [2, 1, 0, 3], B = [1, 0, 2, 2]

**Step 1: Dot Product**
```
A · B = (2×1) + (1×0) + (0×2) + (3×2) = 2 + 0 + 0 + 6 = 8
```

**Step 2: Magnitudes**
```
||A|| = √(4 + 1 + 0 + 9) = √14 ≈ 3.742
||B|| = √(1 + 0 + 4 + 4) = √9 = 3
```

**Step 3: Cosine Similarity**
```
cos(A, B) = 8 / (3.742 × 3) = 8 / 11.226 ≈ 0.713
```

## 5.3 Euclidean Distance

**Formula**:
```
d(A, B) = √Σ(aᵢ - bᵢ)²
```

**Example**: A = [1, 2], B = [4, 6]
```
d = √[(4-1)² + (6-2)²] = √[9 + 16] = √25 = 5
```

## 5.4 Cosine vs Euclidean

| Aspect | Cosine | Euclidean |
|--------|--------|-----------|
| Measures | Angle between vectors | Distance between points |
| Scale | Independent of magnitude | Affected by magnitude |
| Text | Preferred for text | Better for continuous data |
| Range | -1 to 1 | 0 to ∞ |

---

# 6. Word2Vec

## 6.1 Key Innovation
- Learn **dense**, **low-dimensional** word vectors
- Vectors capture **semantic meaning**
- Based on **prediction** not counting

## 6.2 Two Architectures

### Skip-gram
- **Input**: Target word
- **Output**: Predict context words
- Better for **rare words**
- Slower training

### CBOW (Continuous Bag of Words)
- **Input**: Context words
- **Output**: Predict target word
- Better for **frequent words**
- Faster training

## 6.3 Skip-gram in Detail

### Architecture
```
Input (One-hot)    Hidden Layer    Output (Softmax)
    V × 1    →      V × d     →       V × 1
              W (V×d)        W' (d×V)
```

Where:
- V = Vocabulary size
- d = Embedding dimension (typically 50-300)

### Training Data Creation
**Sentence**: "The quick brown fox jumps"
**Window size**: 2

Target: "brown"
Context words: "The", "quick", "fox", "jumps"

**Training pairs**:
- (brown, The)
- (brown, quick)
- (brown, fox)
- (brown, jumps)

### Objective Function
Maximize probability of context given target:
```
J = Σ log P(context | target)
```

Where:
```
P(wₒ | wᵢ) = exp(u_wₒ · v_wᵢ) / Σexp(u_w · v_wᵢ)
```

## 6.4 Negative Sampling

### Problem
- Softmax over entire vocabulary is expensive
- O(V) per training example

### Solution
Instead of predicting all words, train binary classifier:
- **Positive pair**: (target, actual_context) → Label = 1
- **Negative pairs**: (target, random_words) → Label = 0

### Objective
```
J = log σ(uₒ · vᵢ) + Σₖ log σ(-uₖ · vᵢ)
```

Where:
- σ = sigmoid function
- k = number of negative samples (typically 5-20)

## 6.5 Skip-gram Gradient Update

**Given**:
- Target: "cat", Context: "meow" (Positive, y = 1)
- v_cat = [0.5, 0.5], u_meow = [1.0, 0.0]
- P = σ(v · u) = 0.62
- Learning rate η = 0.1

**Update**:
```
Error = P - y = 0.62 - 1 = -0.38
Gradient = Error × u = -0.38 × [1.0, 0.0] = [-0.38, 0.0]
v_new = v - η × Gradient = [0.5, 0.5] - 0.1 × [-0.38, 0.0]
v_new = [0.538, 0.5]
```

---

# 7. CBOW (Continuous Bag of Words)

## 7.1 Architecture

```
Context words (one-hot)
      ↓
   [w₁, w₂, w₃, w₄]
      ↓
Average/Sum embeddings
      ↓
   Hidden layer (d)
      ↓
   Softmax (V)
      ↓
   Target word
```

## 7.2 Key Differences from Skip-gram

| Aspect | Skip-gram | CBOW |
|--------|-----------|------|
| Input | Single target | Multiple context |
| Output | Multiple context | Single target |
| Training speed | Slower | Faster |
| Rare words | Better | Worse |
| Data efficiency | Less data needed | More data needed |

## 7.3 When to Use

- **Skip-gram**: Smaller datasets, care about rare words
- **CBOW**: Large datasets, frequent words matter, speed is important

---

# 8. GloVe (Global Vectors)

## 8.1 Key Idea
- Combines count-based and prediction-based methods
- Uses **global co-occurrence statistics**
- Learns from **ratios of co-occurrence probabilities**

## 8.2 Co-occurrence Matrix
Build matrix X where:
```
Xᵢⱼ = count of word j appearing in context of word i
```

## 8.3 Objective Function
```
J = Σᵢⱼ f(Xᵢⱼ) × (vᵢᵀuⱼ + bᵢ + b̃ⱼ - log(Xᵢⱼ))²
```

Where:
- f(x) = weighting function (reduces impact of very frequent pairs)
- v, u = word vectors
- b = bias terms

## 8.4 Comparison: Word2Vec vs GloVe

| Aspect | Word2Vec | GloVe |
|--------|----------|-------|
| Type | Prediction-based | Count + Prediction |
| Training | Local context windows | Global statistics |
| Algorithm | SGD on windows | Weighted least squares |
| Speed | Slower | Faster (single pass) |
| Performance | Similar | Similar |

---

# 9. Word Analogies

## 9.1 Parallelogram Method

**Famous example**:
```
King - Man + Woman ≈ Queen
```

**Formula**:
```
v_target = v_a* - v_a + v_b
```

Where:
- a : a* :: b : b*
- v_a* - v_a captures the relationship

## 9.2 Types of Analogies

| Type | Example |
|------|---------|
| **Gender** | King:Queen :: Man:Woman |
| **Country-Capital** | France:Paris :: Japan:Tokyo |
| **Comparative** | Big:Bigger :: Small:Smaller |
| **Tense** | Walk:Walked :: Run:Ran |

## 9.3 Calculation Example

**Analogy**: Man:King :: Woman:?

**Given**:
- v_Man = [0.5, 0.3, 0.1]
- v_King = [0.8, 0.6, 0.2]
- v_Woman = [0.4, 0.7, 0.3]

**Calculate**:
```
v_? = v_King - v_Man + v_Woman
    = [0.8, 0.6, 0.2] - [0.5, 0.3, 0.1] + [0.4, 0.7, 0.3]
    = [0.7, 1.0, 0.4]
```

Find word closest to [0.7, 1.0, 0.4] → "Queen"

---

# 10. Visualizing Embeddings

## 10.1 Dimensionality Reduction

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
- Non-linear projection to 2D/3D
- Preserves local structure
- Good for visualization

### PCA (Principal Component Analysis)
- Linear projection
- Preserves variance
- Faster than t-SNE

## 10.2 What Well-trained Embeddings Show

1. **Semantic clusters**: Animals group together, countries group together
2. **Analogical relationships**: Parallel lines for analogies
3. **Hierarchical structure**: Hypernyms/hyponyms nearby

## 10.3 Bias in Embeddings

**Problem**: Embeddings learn biases from training data

**Examples**:
- "Doctor" closer to "Man" than "Woman"
- "Nurse" closer to "Woman" than "Man"

**Solutions**:
- Debiasing algorithms
- Balanced training data
- Post-processing projections

---

# 11. Key Formulas Summary

| Concept | Formula |
|---------|---------|
| TF | `1 + log₁₀(count)` |
| IDF | `log₁₀(N / df)` |
| TF-IDF | `TF × IDF` |
| Cosine | `(A·B) / (‖A‖ × ‖B‖)` |
| Euclidean | `√Σ(aᵢ - bᵢ)²` |
| Word Analogy | `v_target = v_a* - v_a + v_b` |
| SGNS Error | `Error = σ(v·u) - y` |
| Vector Update | `v_new = v - η × Error × u` |

---

# 📝 Practice Questions

## Q1. Calculate TF-IDF
Word "deep" appears 10 times in document D1.
Total documents N = 500.
"deep" appears in 25 documents.

## Q2. Calculate Cosine Similarity
A = [1, 2, 3], B = [2, 4, 6]

## Q3. Skip-gram Training Pairs
Sentence: "I love natural language processing"
Window size: 2
Target: "natural"
List all training pairs.

## Q4. Word2Vec Update
Target: "code", Context: "python" (y=1)
v_code = [0.2, 0.8], u_python = [0.6, 0.4]
σ(v·u) = 0.55, η = 0.1
Calculate updated v_code.

## Q5. Word Analogy
France:Paris :: Germany:?
v_France = [0.3, 0.5], v_Paris = [0.4, 0.7], v_Germany = [0.35, 0.45]
Calculate the target vector.

---

*Reference: Session 2 - Vector Semantics and Embeddings*
