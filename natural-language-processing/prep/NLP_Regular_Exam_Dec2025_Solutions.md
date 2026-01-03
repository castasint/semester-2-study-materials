# 📝 NLP REGULAR EXAM DEC 2025 - EXACT SOLUTIONS
## Solved from the Actual Exam Paper

---

# EXAM PAPER TRANSCRIPTION & SOLUTIONS

---

## Q1: Introduction to NLP (4 Marks)

### 📋 EXACT QUESTION:

> **(a)** List four applications of Natural Language Processing (NLP) with a brief explanation of each. (2 marks)
>
> **(b)** Identify the type of ambiguity present in the following sentences: (2 marks)
> - (i) "I saw her duck"
> - (ii) "The professor said on Monday he would give an exam"

---

### ✅ SOLUTION:

#### (a) Four NLP Applications:

1. **Machine Translation**
   - Converting text from one language to another automatically
   - Example: Google Translate converting English to Hindi

2. **Sentiment Analysis**
   - Determining the emotional tone (positive/negative/neutral) in text
   - Example: Analyzing customer reviews to gauge product satisfaction

3. **Named Entity Recognition (NER)**
   - Identifying and classifying named entities (persons, organizations, locations)
   - Example: Extracting "Microsoft" as Organization from news articles

4. **Question Answering**
   - Systems that automatically answer questions posed in natural language
   - Example: Virtual assistants like Siri answering "What's the weather today?"

#### (b) Ambiguity Types:

| Sentence | Ambiguity Type | Explanation |
|----------|----------------|-------------|
| "I saw her duck" | **Lexical + Structural** | "duck" can be noun (bird) or verb (bend down). Also: "her duck" vs "her" + "duck" |
| "The professor said on Monday he would give an exam" | **Structural (Attachment)** | Did the professor SAY this on Monday? Or is the EXAM on Monday? |

---

## Q2: Language Models (4 Marks)

### 📋 EXACT QUESTION:

> Given the following corpus:
> ```
> <s> I want to eat Chinese food </s>
> <s> I want to eat Italian food </s>
> <s> I want Chinese food </s>
> ```
>
> **(a)** Calculate the bigram probability P(Chinese | want) (1 mark)
>
> **(b)** Calculate P(to | want) (1 mark)
>
> **(c)** If P(Chinese food | want to eat) = 0.5 and P(Italian food | want to eat) = 0.5, calculate the perplexity of the sentence "I want to eat Chinese food" given:
> - P(I | \<s\>) = 0.67
> - P(want | I) = 1.0
> - P(to | want) = 0.67
> - P(eat | to) = 1.0
> - P(Chinese | eat) = 0.5
> - P(food | Chinese) = 1.0
> (2 marks)

---

### ✅ SOLUTION:

#### (a) P(Chinese | want):
```
Count occurrences of "want":
   Sentence 1: "want" appears - followed by "to"
   Sentence 2: "want" appears - followed by "to"
   Sentence 3: "want" appears - followed by "Chinese"
   Count(want) = 3

Count "want Chinese" pairs:
   Only in Sentence 3: "want Chinese"
   Count(want, Chinese) = 1

P(Chinese | want) = Count(want, Chinese) / Count(want)
                  = 1 / 3
                  = 0.33 ✓
```

#### (b) P(to | want):
```
Count(want, to) = 2 (Sentences 1 and 2)
Count(want) = 3

P(to | want) = 2 / 3 = 0.67 ✓
```

#### (c) Perplexity:
```
Step 1: Calculate P(sentence)
   P = P(I|<s>) × P(want|I) × P(to|want) × P(eat|to) × P(Chinese|eat) × P(food|Chinese)
   P = 0.67 × 1.0 × 0.67 × 1.0 × 0.5 × 1.0
   P = 0.67 × 0.67 × 0.5
   P = 0.2245

Step 2: N = 6 words (I, want, to, eat, Chinese, food)

Step 3: Perplexity
   PP = P^(-1/N)
      = (0.2245)^(-1/6)
      = (1/0.2245)^(1/6)
      = (4.454)^(1/6)
      = 1.28 ✓

Low perplexity indicates the model predicts this sentence well!
```

---

## Q3: Neural Language Models and LLM (4 Marks)

### 📋 EXACT QUESTION:

> **(a)** Explain the concepts of Pre-training and Fine-tuning in the context of Large Language Models (LLMs). (2 marks)
>
> **(b)** What is the difference between Zero-shot and Few-shot prompting? Write an example of Few-shot prompting for sentiment classification. (2 marks)

---

### ✅ SOLUTION:

#### (a) Pre-training and Fine-tuning:

| Aspect | Pre-training | Fine-tuning |
|--------|--------------|-------------|
| **Definition** | Training on massive unlabeled data to learn general language patterns | Adapting pre-trained model on smaller task-specific labeled data |
| **Data** | Billions of tokens (Wikipedia, books, web) | Thousands of labeled examples |
| **Task** | Self-supervised (predict next word, masked word) | Supervised (classification, QA, etc.) |
| **Goal** | Learn grammar, facts, reasoning | Specialize for specific application |
| **Time** | Weeks/months on many GPUs | Hours/days on single GPU |

**Example:**
- Pre-training: GPT learns language from internet text
- Fine-tuning: GPT fine-tuned on medical data → Medical chatbot

#### (b) Zero-shot vs Few-shot:

| Aspect | Zero-shot | Few-shot |
|--------|-----------|----------|
| **Examples given** | None | Multiple examples |
| **Prompt format** | Just instruction + query | Examples + query |
| **When to use** | Simple/familiar tasks | Complex/new tasks |

**Few-shot Example for Sentiment Classification:**
```
Classify the sentiment as Positive or Negative.

Review: "This product broke after one day. Terrible!"
Sentiment: Negative

Review: "Absolutely love it! Best purchase ever!"
Sentiment: Positive

Review: "Waste of money, very disappointed."
Sentiment: Negative

Review: "The quality exceeded my expectations."
Sentiment:
```
*Expected output: Positive*

---

## Q4: Vector Semantics (4 Marks)

### 📋 EXACT QUESTION:

> **(a)** Calculate the TF-IDF weight for the term "data" in Document 1 given: (2 marks)
> - Term "data" appears 6 times in Document 1
> - Total number of documents N = 10,000
> - Number of documents containing "data" = 100
> - Use: TF = 1 + log₁₀(tf), IDF = log₁₀(N/df)
>
> **(b)** Given two document vectors A = [1, 2, 3] and B = [2, 3, 4], calculate the Cosine Similarity between them. (2 marks)

---

### ✅ SOLUTION:

#### (a) TF-IDF Calculation:
```
Given: tf = 6, N = 10,000, df = 100

Step 1: Calculate TF
   TF = 1 + log₁₀(6)
      = 1 + 0.778
      = 1.778 ✓

Step 2: Calculate IDF
   IDF = log₁₀(N / df)
       = log₁₀(10,000 / 100)
       = log₁₀(100)
       = 2.0 ✓

Step 3: Calculate TF-IDF
   TF-IDF = TF × IDF
          = 1.778 × 2.0
          = 3.556 ✓
```

#### (b) Cosine Similarity:
```
A = [1, 2, 3], B = [2, 3, 4]

Step 1: Dot Product
   A · B = (1×2) + (2×3) + (3×4)
         = 2 + 6 + 12
         = 20 ✓

Step 2: Magnitude of A
   |A| = √(1² + 2² + 3²)
       = √(1 + 4 + 9)
       = √14
       = 3.742 ✓

Step 3: Magnitude of B
   |B| = √(2² + 3² + 4²)
       = √(4 + 9 + 16)
       = √29
       = 5.385 ✓

Step 4: Cosine Similarity
   cos(A, B) = (A · B) / (|A| × |B|)
             = 20 / (3.742 × 5.385)
             = 20 / 20.15
             = 0.993 ✓

Very high similarity! Documents have very similar topic distribution.
```

---

## Q5: Word Embeddings (5 Marks)

### 📋 EXACT QUESTION:

> **(a)** In Skip-gram Word2Vec, what are the input and output for training? If the sentence is "The quick brown fox" and center word is "quick" with window size 1, list the training pairs. (2 marks)
>
> **(b)** Given word vectors:
> - v(man) = [1.0, 0.5]
> - v(woman) = [1.0, 1.0]
> - v(king) = [2.0, 1.0]
>
> Using the word analogy relationship "man is to woman as king is to queen", calculate v(queen). (1.5 marks)
>
> **(c)** In Skip-gram with Negative Sampling, given:
> - Target word vector v = [0.5, 0.5]
> - Context word vector u = [1.0, 0.0]
> - σ(v · u) = 0.62
> - This is a positive pair (y = 1)
> - Learning rate η = 0.1
>
> Calculate the updated target vector v'. (1.5 marks)

---

### ✅ SOLUTION:

#### (a) Skip-gram Training Pairs:
```
Sentence: "The quick brown fox"
Center word: "quick"
Window size: 1 (one word left, one word right)

Input: "quick" (center word)
Output: Context words within window

Context words: "The" (left), "brown" (right)

Training pairs generated:
   (quick, The)
   (quick, brown)
```

#### (b) Word Analogy:
```
Relationship: man : woman :: king : queen
Formula: v(queen) = v(king) - v(man) + v(woman)

Given:
   v(man) = [1.0, 0.5]
   v(woman) = [1.0, 1.0]
   v(king) = [2.0, 1.0]

Calculation:
   v(queen) = [2.0, 1.0] - [1.0, 0.5] + [1.0, 1.0]
   
   Position 1: 2.0 - 1.0 + 1.0 = 2.0
   Position 2: 1.0 - 0.5 + 1.0 = 1.5

   v(queen) = [2.0, 1.5] ✓
```

#### (c) Skip-gram Update:
```
Given: v = [0.5, 0.5], u = [1.0, 0.0], σ(v·u) = 0.62, y = 1, η = 0.1

Step 1: Calculate Error
   Error = σ(v · u) - y
         = 0.62 - 1
         = -0.38 ✓

Step 2: Calculate Gradient
   Gradient = Error × u
            = -0.38 × [1.0, 0.0]
            = [-0.38, 0.0] ✓

Step 3: Update v
   v' = v - η × Gradient
      = [0.5, 0.5] - 0.1 × [-0.38, 0.0]
      = [0.5, 0.5] - [-0.038, 0.0]
      = [0.5 + 0.038, 0.5 + 0.0]
      = [0.538, 0.5] ✓

The vector moved CLOSER to context (correct for positive pair!)
```

---

## Q6: POS Tagging and HMM (4 Marks)

### 📋 EXACT QUESTION:

> Given a tagged corpus with the following counts:
> - C(DT) = 100, C(NN) = 200, C(VB) = 150
> - C(DT, NN) = 80, C(NN, VB) = 60
> - C("the", DT) = 40, C("book", NN) = 20, C("book", VB) = 10
>
> **(a)** Calculate the transition probabilities P(NN|DT) and P(VB|NN). (2 marks)
>
> **(b)** For the word "book" following a DT, determine which tag (NN or VB) is more likely using HMM disambiguation. Given P(NN|DT) = 0.8 and P(VB|DT) = 0.1. (2 marks)

---

### ✅ SOLUTION:

#### (a) Transition Probabilities:
```
P(NN | DT) = C(DT, NN) / C(DT)
           = 80 / 100
           = 0.8 ✓

P(VB | NN) = C(NN, VB) / C(NN)
           = 60 / 200
           = 0.3 ✓
```

#### (b) HMM Disambiguation:
```
Word: "book" after DT
Candidates: NN or VB

First, calculate emission probabilities:
   P("book" | NN) = C("book", NN) / C(NN) = 20 / 200 = 0.1
   P("book" | VB) = C("book", VB) / C(VB) = 10 / 150 = 0.067

Now calculate HMM scores:
   Score(NN) = P(NN | DT) × P("book" | NN)
             = 0.8 × 0.1
             = 0.08 ✓

   Score(VB) = P(VB | DT) × P("book" | VB)
             = 0.1 × 0.067
             = 0.0067 ✓

Comparison: 0.08 > 0.0067

Winner: NN (Noun) ✓

"book" after "the" should be tagged as NN (e.g., "the book")
```

---

## Q7: Viterbi Algorithm (5 Marks)

### 📋 EXACT QUESTION:

> Use the Viterbi algorithm to find the most probable tag sequence for the sentence "The man runs"
>
> **Given:**
> - States: DT, NN, VB
> - Start probabilities: π(DT) = 0.6, π(NN) = 0.3, π(VB) = 0.1
>
> **Transition probabilities:**
> |  | DT | NN | VB |
> |--|----|----|-----|
> | DT | 0.1 | 0.8 | 0.1 |
> | NN | 0.1 | 0.3 | 0.6 |
> | VB | 0.2 | 0.5 | 0.3 |
>
> **Emission probabilities:**
> | Word | P(·|DT) | P(·|NN) | P(·|VB) |
> |------|---------|---------|---------|
> | The | 0.7 | 0.02 | 0.01 |
> | man | 0.01 | 0.4 | 0.02 |
> | runs | 0.01 | 0.1 | 0.6 |
>
> Show complete Viterbi table with backpointers and trace back the best path.

---

### ✅ SOLUTION:

#### STEP 1: Initialization ("The")
```
V₁(DT) = π(DT) × P("The" | DT) = 0.6 × 0.7 = 0.42 ← BEST
V₁(NN) = π(NN) × P("The" | NN) = 0.3 × 0.02 = 0.006
V₁(VB) = π(VB) × P("The" | VB) = 0.1 × 0.01 = 0.001
```

#### STEP 2: Recursion ("man")

**For NN:**
```
From DT: V₁(DT) × P(NN|DT) = 0.42 × 0.8 = 0.336 ← MAX
From NN: V₁(NN) × P(NN|NN) = 0.006 × 0.3 = 0.0018
From VB: V₁(VB) × P(NN|VB) = 0.001 × 0.5 = 0.0005

V₂(NN) = 0.336 × P("man"|NN) = 0.336 × 0.4 = 0.1344
Backpointer: DT
```

**For DT:**
```
From DT: 0.42 × 0.1 = 0.042 ← MAX
V₂(DT) = 0.042 × 0.01 = 0.00042
```

**For VB:**
```
From DT: 0.42 × 0.1 = 0.042
From NN: 0.006 × 0.6 = 0.0036
Max = 0.042
V₂(VB) = 0.042 × 0.02 = 0.00084
```

#### STEP 3: Recursion ("runs")

**For VB:**
```
From DT: V₂(DT) × P(VB|DT) = 0.00042 × 0.1 = 0.000042
From NN: V₂(NN) × P(VB|NN) = 0.1344 × 0.6 = 0.08064 ← MAX
From VB: V₂(VB) × P(VB|VB) = 0.00084 × 0.3 = 0.000252

V₃(VB) = 0.08064 × P("runs"|VB) = 0.08064 × 0.6 = 0.04838
Backpointer: NN
```

**For NN:**
```
From NN: 0.1344 × 0.3 = 0.04032 ← MAX
V₃(NN) = 0.04032 × 0.1 = 0.004032
```

#### VITERBI TABLE:

| State | "The" | "man" | "runs" |
|-------|-------|-------|--------|
| **DT** | **0.42** | 0.00042 | - |
| **NN** | 0.006 | **0.1344** ←DT | 0.004032 |
| **VB** | 0.001 | 0.00084 | **0.04838** ←NN |

#### STEP 4: Backtracking
```
Best final state: VB (0.04838) ← HIGHEST

Trace back:
   "runs" → VB, backpointer = NN
   "man" → NN, backpointer = DT
   "The" → DT (start)

Path: DT → NN → VB
```

#### FINAL ANSWER:

| Word | Tag | Meaning |
|------|-----|---------|
| The | DT | Determiner |
| man | NN | Noun |
| runs | VB | Verb |

**Best tag sequence: DT → NN → VB** ✓

---

# 📊 QUICK REFERENCE

## Log Values Used:
```
log₁₀(6) = 0.778
log₁₀(100) = 2.0
```

## Key Formulas:
```
Bigram:    P(w|prev) = C(prev,w) / C(prev)
Perplexity: PP = P^(-1/N)
TF:        TF = 1 + log₁₀(count)
IDF:       IDF = log₁₀(N/df)
Cosine:    cos = (A·B) / (|A|×|B|)
Analogy:   Queen = King - Man + Woman
Word2Vec:  v' = v - η × (σ-y) × u
HMM:       Score = Trans × Emit
Viterbi:   V(j) = max[V(prev)×Trans] × Emit
```

---

**Good luck with your exam!** 🍀
