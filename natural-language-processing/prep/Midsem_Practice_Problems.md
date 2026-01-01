# 📝 NLP MIDSEM PRACTICE PROBLEMS
## With Complete Step-by-Step Solutions

**Exam Structure**: 7 Questions, 30 Marks Total

---

# Q1: Module 1 - Introduction (4 Marks)

## Problem 1.1
**List any 4 applications of NLP and briefly explain each.** (4 marks)

### Solution:
1. **Machine Translation**: Converting text from one language to another (e.g., English to Hindi). Uses encoder-decoder architectures and attention mechanisms.

2. **Sentiment Analysis**: Determining if text expresses positive, negative, or neutral opinion. Used in product reviews, social media monitoring.

3. **Named Entity Recognition (NER)**: Identifying and classifying entities like person names, locations, organizations in text.

4. **Question Answering**: Systems that answer questions in natural language. Example: Virtual assistants like Siri, Alexa.

---

## Problem 1.2
**Identify the type of ambiguity in each sentence:**
a) "I saw the man with the telescope"
b) "Time flies like an arrow"
c) "The bank was steep"

### Solution:
a) **Structural Ambiguity**: Two parse trees possible
   - "I used telescope to see man" OR "Man had telescope"

b) **Structural + Lexical Ambiguity**: 
   - "flies" can be noun or verb
   - Multiple parse structures possible

c) **Lexical Ambiguity (Polysemy)**:
   - "bank" = river bank OR financial institution

---

## Problem 1.3
**List the 6 levels of language analysis in order from lowest to highest.**

### Solution:
1. **Morphological** - Word parts (prefixes, suffixes)
2. **Lexical** - Word categories (POS tags)
3. **Syntactic** - Sentence structure (grammar)
4. **Semantic** - Word and sentence meaning
5. **Discourse** - Multi-sentence relationships
6. **Pragmatic** - Context and speaker intent

---

# Q2: Module 2 - Language Models (4 Marks)

## Problem 2.1: Bigram Probability
**Given corpus:**
```
<s> I love NLP </s>
<s> I love coding </s>
<s> NLP is fun </s>
```

**Calculate:**
a) P(love | I)
b) P(NLP | love)
c) P(is | NLP)

### Solution:
**Count from corpus:**
- Count(I) = 2
- Count(I, love) = 2
- Count(love) = 2
- Count(love, NLP) = 1
- Count(love, coding) = 1
- Count(NLP) = 2
- Count(NLP, is) = 1

**Calculations:**
```
a) P(love | I) = C(I, love) / C(I) = 2/2 = 1.0

b) P(NLP | love) = C(love, NLP) / C(love) = 1/2 = 0.5

c) P(is | NLP) = C(NLP, is) / C(NLP) = 1/2 = 0.5
```

---

## Problem 2.2: Laplace Smoothing
**Given:**
- Count("the", "cat") = 0
- Count("the") = 50
- Vocabulary size V = 10,000

**Calculate P(cat | the) using Laplace smoothing.**

### Solution:
```
P_Laplace(cat | the) = (Count(the, cat) + 1) / (Count(the) + V)
                     = (0 + 1) / (50 + 10,000)
                     = 1 / 10,050
                     = 0.0000995
                     ≈ 9.95 × 10⁻⁵
```

---

## Problem 2.3: Perplexity Calculation
**Given bigram model probabilities for sentence "I love NLP" (N=3):**
- P(I | <s>) = 0.4
- P(love | I) = 0.5
- P(NLP | love) = 0.2

**Calculate the perplexity.**

### Solution:
**Step 1: Calculate sentence probability**
```
P(sentence) = P(I|<s>) × P(love|I) × P(NLP|love)
            = 0.4 × 0.5 × 0.2
            = 0.04
```

**Step 2: Calculate perplexity**
```
PP = P(sentence)^(-1/N)
   = (0.04)^(-1/3)
   = (1/0.04)^(1/3)
   = (25)^(1/3)
   = ³√25
   ≈ 2.92
```

**Answer: Perplexity ≈ 2.92**

---

## Problem 2.4: Perplexity with More Words
**Sentence**: "the cat sat" (N = 3)
- P(the) = 0.1
- P(cat | the) = 0.05
- P(sat | cat) = 0.04

**Calculate perplexity.**

### Solution:
```
P(sentence) = 0.1 × 0.05 × 0.04 = 0.0002

PP = (0.0002)^(-1/3)
   = (1/0.0002)^(1/3)
   = (5000)^(1/3)
   = ³√5000
   ≈ 17.1
```

**Answer: Perplexity ≈ 17.1** (Higher PP = model more uncertain)

---

# Q3: Module 3 - Neural LM & LLM (4 Marks)

## Problem 3.1
**Compare N-gram and Neural Language Models (any 4 points).**

### Solution:
| Aspect | N-gram LM | Neural LM |
|--------|-----------|-----------|
| Context | Fixed (n-1 words) | Variable/Long |
| Generalization | None (exact match) | Embedding-based |
| Sparsity | Major issue | Not a problem |
| Training | Counting | Gradient descent |

---

## Problem 3.2
**Write prompts for sentiment classification using:**
a) Zero-shot
b) One-shot
c) Few-shot

### Solution:
**a) Zero-shot:**
```
Classify the sentiment as Positive or Negative.
Review: "This product exceeded my expectations!"
Sentiment:
```

**b) One-shot:**
```
Classify the sentiment as Positive or Negative.

Review: "Terrible quality, waste of money."
Sentiment: Negative

Review: "This product exceeded my expectations!"
Sentiment:
```

**c) Few-shot:**
```
Classify the sentiment as Positive or Negative.

Review: "Terrible quality, waste of money."
Sentiment: Negative

Review: "Absolutely loved it!"
Sentiment: Positive

Review: "Not worth the price."
Sentiment: Negative

Review: "This product exceeded my expectations!"
Sentiment:
```

---

## Problem 3.3
**Explain Chain-of-Thought prompting with an example.**

### Solution:
**Chain-of-Thought (CoT)** prompts the model to show step-by-step reasoning.

**Example:**
```
Q: A store has 23 apples. They sell 15 and receive 27 more. 
   How many apples do they have now?

A: Let me solve this step by step.
   Step 1: Started with 23 apples
   Step 2: Sold 15 apples: 23 - 15 = 8 apples
   Step 3: Received 27 more: 8 + 27 = 35 apples
   Answer: 35 apples
```

**Benefit**: Improves accuracy for complex reasoning tasks.

---

# Q4: Module 4 - Vector Semantics (4 Marks)

## Problem 4.1: TF-IDF Calculation
**Given:**
- Document D1 contains word "neural" 8 times
- Total documents N = 1000
- "neural" appears in 40 documents

**Calculate TF-IDF for "neural" in D1.**

### Solution:
**Step 1: Calculate TF**
```
TF = 1 + log₁₀(count)
   = 1 + log₁₀(8)
   = 1 + 0.903
   = 1.903
```

**Step 2: Calculate IDF**
```
IDF = log₁₀(N / df)
    = log₁₀(1000 / 40)
    = log₁₀(25)
    = 1.398
```

**Step 3: Calculate TF-IDF**
```
TF-IDF = TF × IDF
       = 1.903 × 1.398
       = 2.66
```

**Answer: TF-IDF = 2.66**

---

## Problem 4.2: TF-IDF for Multiple Terms
**Documents**: N = 500

| Word | Count in D1 | Documents containing word |
|------|-------------|---------------------------|
| "machine" | 5 | 100 |
| "learning" | 10 | 250 |
| "the" | 20 | 500 |

**Calculate TF-IDF for each word.**

### Solution:

**For "machine":**
```
TF = 1 + log₁₀(5) = 1 + 0.699 = 1.699
IDF = log₁₀(500/100) = log₁₀(5) = 0.699
TF-IDF = 1.699 × 0.699 = 1.19
```

**For "learning":**
```
TF = 1 + log₁₀(10) = 1 + 1 = 2.0
IDF = log₁₀(500/250) = log₁₀(2) = 0.301
TF-IDF = 2.0 × 0.301 = 0.60
```

**For "the":**
```
TF = 1 + log₁₀(20) = 1 + 1.301 = 2.301
IDF = log₁₀(500/500) = log₁₀(1) = 0  ← Common word!
TF-IDF = 2.301 × 0 = 0
```

**Key Insight**: "the" has TF-IDF = 0 because it appears in ALL documents (not discriminative)

---

## Problem 4.3: Cosine Similarity
**Calculate cosine similarity between:**
- Document A: [2, 1, 0, 2]
- Document B: [1, 1, 2, 1]

### Solution:

**Step 1: Dot Product**
```
A · B = (2×1) + (1×1) + (0×2) + (2×1)
      = 2 + 1 + 0 + 2
      = 5
```

**Step 2: Magnitude of A**
```
||A|| = √(2² + 1² + 0² + 2²)
      = √(4 + 1 + 0 + 4)
      = √9
      = 3
```

**Step 3: Magnitude of B**
```
||B|| = √(1² + 1² + 2² + 1²)
      = √(1 + 1 + 4 + 1)
      = √7
      = 2.646
```

**Step 4: Cosine Similarity**
```
cos(A, B) = (A · B) / (||A|| × ||B||)
          = 5 / (3 × 2.646)
          = 5 / 7.938
          = 0.63
```

**Answer: Cosine Similarity = 0.63**

---

## Problem 4.4: Euclidean Distance
**Same vectors: A = [2, 1, 0, 2], B = [1, 1, 2, 1]**

### Solution:
```
d(A, B) = √[(2-1)² + (1-1)² + (0-2)² + (2-1)²]
        = √[1 + 0 + 4 + 1]
        = √6
        = 2.45
```

**Answer: Euclidean Distance = 2.45**

---

# Q5: Module 4 - Word Embedding (5 Marks)

## Problem 5.1: Word Analogy
**Given vectors:**
- v_Man = [0.5, 0.3, 0.2]
- v_Woman = [0.4, 0.6, 0.3]
- v_King = [0.8, 0.4, 0.5]

**Find v_Queen using analogy: Man:King :: Woman:?**

### Solution:
**Formula**: v_Queen = v_King - v_Man + v_Woman

**Calculation:**
```
v_Queen = [0.8, 0.4, 0.5] - [0.5, 0.3, 0.2] + [0.4, 0.6, 0.3]
        = [0.8-0.5+0.4, 0.4-0.3+0.6, 0.5-0.2+0.3]
        = [0.7, 0.7, 0.6]
```

**Answer: v_Queen = [0.7, 0.7, 0.6]**

---

## Problem 5.2: Word2Vec Skip-gram Update
**Given:**
- Target word: "code", Context word: "python" (Positive pair, y = 1)
- v_code = [0.2, 0.6]
- u_python = [0.5, 0.3]
- σ(v · u) = 0.55 (sigmoid output)
- Learning rate η = 0.1

**Calculate updated v_code.**

### Solution:

**Step 1: Calculate Error**
```
Error = σ(v · u) - y
      = 0.55 - 1
      = -0.45
```

**Step 2: Calculate Gradient**
```
Gradient = Error × u_python
         = -0.45 × [0.5, 0.3]
         = [-0.225, -0.135]
```

**Step 3: Update v_code**
```
v_new = v_old - η × Gradient
      = [0.2, 0.6] - 0.1 × [-0.225, -0.135]
      = [0.2, 0.6] - [-0.0225, -0.0135]
      = [0.2 + 0.0225, 0.6 + 0.0135]
      = [0.2225, 0.6135]
```

**Answer: v_code_new = [0.2225, 0.6135]**

---

## Problem 5.3: Skip-gram Training Pairs
**Sentence**: "I love natural language processing"
**Window size**: 2

**Generate all training pairs for target word "natural".**

### Solution:
Context window of 2 means 2 words on each side.

**Target: "natural"**
**Context words**: "I", "love" (left 2), "language", "processing" (right 2)

**Training pairs:**
1. (natural, I)
2. (natural, love)
3. (natural, language)
4. (natural, processing)

---

## Problem 5.4: Another Word2Vec Update
**Given:**
- Target: "dog", Context: "cat" (Negative pair, y = 0)
- v_dog = [0.4, 0.8]
- u_cat = [0.6, 0.2]
- v · u = 0.4×0.6 + 0.8×0.2 = 0.24 + 0.16 = 0.40
- σ(0.40) = 0.60
- η = 0.2

**Calculate updated v_dog.**

### Solution:
```
Error = σ(v · u) - y = 0.60 - 0 = 0.60

Gradient = Error × u = 0.60 × [0.6, 0.2] = [0.36, 0.12]

v_new = v_old - η × Gradient
      = [0.4, 0.8] - 0.2 × [0.36, 0.12]
      = [0.4, 0.8] - [0.072, 0.024]
      = [0.328, 0.776]
```

**Answer: v_dog_new = [0.328, 0.776]**

**Note**: For negative pairs, vectors move APART (v decreased)

---

# Q6: Module 5 - POS Tagging (4 Marks)

## Problem 6.1: HMM Probabilities
**Given corpus with POS tags:**
```
DT NN VB IN DT NN
DT JJ NN VBZ RB
DT NN VB DT NN
```

**Calculate:**
a) P(NN | DT)
b) P(VB | NN)

### Solution:

**Count tags:**
- Count(DT) = 5
- Count(NN) = 5
- Count(DT → NN) = 3
- Count(DT → JJ) = 1
- Count(NN → VB) = 2
- Count(NN → VBZ) = 1

**Calculations:**
```
a) P(NN | DT) = Count(DT → NN) / Count(DT) = 3/5 = 0.6

b) P(VB | NN) = Count(NN → VB) / Count(NN) = 2/5 = 0.4
```

---

## Problem 6.2: HMM Disambiguation
**Sentence**: "Time flies"
**Previous tag for "Time"**: NN
**Candidate tags for "flies"**: NN, VBZ

**Given:**
- P(NN | NN) = 0.3
- P(VBZ | NN) = 0.4
- P("flies" | NN) = 0.02
- P("flies" | VBZ) = 0.05

**Which tag should "flies" get?**

### Solution:

**Score for NN:**
```
Score(NN) = P(NN | NN) × P("flies" | NN)
          = 0.3 × 0.02
          = 0.006
```

**Score for VBZ:**
```
Score(VBZ) = P(VBZ | NN) × P("flies" | VBZ)
           = 0.4 × 0.05
           = 0.020
```

**Answer: VBZ wins (0.020 > 0.006)**
"flies" should be tagged as VBZ (verb, 3rd person singular)

---

## Problem 6.3: Another HMM Disambiguation
**Word**: "book"
**Previous tag**: VB
**Candidates**: NN, VB

**Given:**
- P(NN | VB) = 0.5
- P(VB | VB) = 0.1
- P("book" | NN) = 0.04
- P("book" | VB) = 0.03

### Solution:
```
Score(NN) = 0.5 × 0.04 = 0.020
Score(VB) = 0.1 × 0.03 = 0.003
```

**Answer: NN wins (0.020 > 0.003)**

---

# Q7: Module 6 - Viterbi & MEMM (5 Marks)

## Problem 7.1: Complete Viterbi Calculation
**Sentence**: "The dog runs"
**States**: DT, NN, VBZ

**Given probabilities:**

**Start probabilities:**
- π(DT) = 0.6, π(NN) = 0.3, π(VBZ) = 0.1

**Transition:**
|     | DT  | NN  | VBZ |
|-----|-----|-----|-----|
| DT  | 0.1 | 0.7 | 0.2 |
| NN  | 0.1 | 0.2 | 0.7 |
| VBZ | 0.3 | 0.6 | 0.1 |

**Emission:**
- P("The" | DT) = 0.8, P("The" | NN) = 0.01, P("The" | VBZ) = 0.01
- P("dog" | DT) = 0.01, P("dog" | NN) = 0.3, P("dog" | VBZ) = 0.02
- P("runs" | DT) = 0.01, P("runs" | NN) = 0.05, P("runs" | VBZ) = 0.4

### Solution:

**Step 1: Initialization (word = "The")**
```
V₁(DT) = π(DT) × P("The"|DT) = 0.6 × 0.8 = 0.48
V₁(NN) = π(NN) × P("The"|NN) = 0.3 × 0.01 = 0.003
V₁(VBZ) = π(VBZ) × P("The"|VBZ) = 0.1 × 0.01 = 0.001
```

**Step 2: Recursion (word = "dog")**

For NN:
```
From DT: 0.48 × 0.7 = 0.336 ← max
From NN: 0.003 × 0.2 = 0.0006
From VBZ: 0.001 × 0.6 = 0.0006

V₂(NN) = 0.336 × P("dog"|NN) = 0.336 × 0.3 = 0.1008
bp₂(NN) = DT
```

For DT:
```
Max from prev = 0.48 × 0.1 = 0.048
V₂(DT) = 0.048 × 0.01 = 0.00048
bp₂(DT) = DT
```

For VBZ:
```
From DT: 0.48 × 0.2 = 0.096 ← max
V₂(VBZ) = 0.096 × 0.02 = 0.00192
bp₂(VBZ) = DT
```

**Step 3: Recursion (word = "runs")**

For VBZ:
```
From DT: 0.00048 × 0.2 = 0.000096
From NN: 0.1008 × 0.7 = 0.07056 ← max
From VBZ: 0.00192 × 0.1 = 0.000192

V₃(VBZ) = 0.07056 × P("runs"|VBZ) = 0.07056 × 0.4 = 0.0282
bp₃(VBZ) = NN
```

**Viterbi Table:**

| State | "The" | "dog" | "runs" | Backpointer |
|-------|-------|-------|--------|-------------|
| DT | 0.48 | 0.00048 | - | - |
| NN | 0.003 | 0.1008 | - | DT |
| VBZ | 0.001 | 0.00192 | 0.0282 | NN |

**Step 4: Backtracking**
```
Best final: VBZ (0.0282)
bp₃(VBZ) = NN
bp₂(NN) = DT
```

**Answer: DT → NN → VBZ**
("The" = Determiner, "dog" = Noun, "runs" = Verb)

---

## Problem 7.2: Compare HMM vs MEMM

### Solution:
| Aspect | HMM | MEMM |
|--------|-----|------|
| **Model Type** | Generative | Discriminative |
| **Probability** | P(word \| tag) | P(tag \| word, features) |
| **Features** | Word identity only | Rich overlapping features |
| **Unknown words** | Problematic | Better handling |
| **Inference** | Viterbi | Modified Viterbi |

---

## Problem 7.3: MEMM Features
**List features you would use for the word "unhappily" preceded by "was".**

### Solution:
1. word = "unhappily"
2. prev_word = "was"
3. prev_tag = VBD (verb past)
4. suffix = "-ly" (adverb indicator)
5. prefix = "un-" (negation)
6. contains_hyphen = No
7. is_capitalized = No
8. word_length > 8 = Yes
9. next_word features (if available)

---

# 📊 QUICK FORMULA REFERENCE

```
TF = 1 + log₁₀(count)
IDF = log₁₀(N / df)
TF-IDF = TF × IDF

Cosine = (A·B) / (||A|| × ||B||)

Perplexity = P(sentence)^(-1/N)

Bigram = C(w₁,w₂) / C(w₁)
Laplace = (C+1) / (N+V)

HMM Score = Transition × Emission
Viterbi = max[V(prev) × A] × B

Word Analogy = v_b - v_a + v_a*
Word2Vec Update = v - η × (σ(v·u) - y) × u
```

---

**🍀 Good luck with your exam!**
