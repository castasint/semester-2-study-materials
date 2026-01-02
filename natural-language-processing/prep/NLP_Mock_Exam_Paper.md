# 📝 NLP MIDSEM - SAMPLE MOCK PAPER
## Created for Makeup Exam Practice | 30 Marks | 2 Hours

---

# Q1: Introduction to NLP (4 Marks)

---

### Question 1a: Applications (2 marks)

**List any 4 real-world applications of NLP and provide a one-line description for each.**

---

### Question 1b: Ambiguity (2 marks)

**Identify the type of ambiguity in each of the following sentences:**

a) "She saw the man with binoculars"

b) "The chicken is ready to eat"

c) "Lead can be harmful" (Lead = metal or Lead = to guide)

---

# Q2: Language Models (4 Marks)

---

### Question 2a: Bigram Probability (2 marks)

**Given the following corpus:**
```
<s> The cat sat on the mat </s>
<s> The dog sat on the floor </s>
<s> The cat ran fast </s>
```

**Calculate:**
- P(sat | cat)
- P(on | sat)

---

### Question 2b: Perplexity (2 marks)

**A bigram language model gives the following probabilities for the sentence "The cat ran":**
- P(The | start) = 0.5
- P(cat | The) = 0.3
- P(ran | cat) = 0.2

**Calculate the perplexity of this sentence. (N = 3 words)**

---

# Q3: Neural Language Models & LLM (4 Marks)

---

### Question 3a: Prompting (2 marks)

**Write a Few-shot prompt to classify movie reviews as "Positive" or "Negative".**

Include at least 2 examples and one test query.

---

### Question 3b: Comparison (2 marks)

**Compare N-gram Language Models and Neural Language Models on the following parameters:**
1. Handling of unseen word combinations
2. Context window size

---

# Q4: Vector Semantics (4 Marks)

---

### Question 4a: TF-IDF Calculation (2 marks)

**Given:**
- Word "algorithm" appears **10 times** in Document D1
- Total documents in corpus N = **2000**
- "algorithm" appears in **50 documents**

**Calculate the TF-IDF weight for "algorithm" in D1.**

Use: TF = 1 + log₁₀(count), IDF = log₁₀(N/df)

---

### Question 4b: Cosine Similarity (2 marks)

**Calculate the cosine similarity between the following document vectors:**
- Document A = [3, 0, 4]
- Document B = [0, 3, 4]

---

# Q5: Word Embeddings (5 Marks)

---

### Question 5a: Skip-gram Training Pairs (1 mark)

**Given the sentence: "Natural language processing is fun"**

If the target word is "processing" and window size is 2, list all the training pairs generated for Skip-gram.

---

### Question 5b: Word Analogy (2 marks)

**Given the following word vectors:**
- Tokyo = [0.7, 0.2, 0.5]
- Japan = [0.6, 0.4, 0.3]
- Paris = [0.8, 0.3, 0.4]

**Using the analogy "Tokyo is to Japan as Paris is to France", calculate the vector for France.**

---

### Question 5c: Word2Vec Update (2 marks)

**In Skip-gram with Negative Sampling:**
- Target word: "neural", Context word: "network" (Positive pair, y = 1)
- v(neural) = [0.3, 0.7]
- u(network) = [0.4, 0.2]
- Model predicts: σ(v · u) = 0.58
- Learning rate η = 0.2

**Calculate the updated vector v(neural)_new.**

---

# Q6: POS Tagging & HMM (4 Marks)

---

### Question 6a: HMM Probabilities (2 marks)

**Given the following tagged corpus:**
```
DT NN VBZ JJ
DT JJ NN VBZ
DT NN VBZ DT NN
```

**Calculate:**
- P(NN | DT)
- P(VBZ | NN)

---

### Question 6b: HMM Disambiguation (2 marks)

**Disambiguate the word "light" in the sentence "...turn on the light"**

Previous tag: DT (Determiner)

Candidate tags for "light": NN (Noun), JJ (Adjective), VB (Verb)

**Given:**
- P(NN | DT) = 0.6
- P(JJ | DT) = 0.3
- P(VB | DT) = 0.1
- P("light" | NN) = 0.04
- P("light" | JJ) = 0.02
- P("light" | VB) = 0.01

**Which tag should "light" receive?**

---

# Q7: Viterbi Algorithm (5 Marks)

---

### Question 7: Complete Viterbi Table (5 marks)

**Tag the sentence "A dog barks" using the Viterbi algorithm.**

**States:** DT (Determiner), NN (Noun), VBZ (Verb)

**Start Probabilities (π):**
- π(DT) = 0.5
- π(NN) = 0.3
- π(VBZ) = 0.2

**Transition Probabilities:**

| From → To | DT | NN | VBZ |
|-----------|-----|-----|-----|
| DT | 0.1 | 0.8 | 0.1 |
| NN | 0.2 | 0.3 | 0.5 |
| VBZ | 0.4 | 0.5 | 0.1 |

**Emission Probabilities:**

| Word | P(. given DT) | P(. given NN) | P(. given VBZ) |
|------|---------------|---------------|----------------|
| A | 0.7 | 0.02 | 0.01 |
| dog | 0.02 | 0.4 | 0.03 |
| barks | 0.01 | 0.08 | 0.5 |

**Requirements:**
1. Calculate the Viterbi values for each word and state
2. Show the backpointers
3. Perform backtracking to find the best tag sequence

---

---

# 📝 SOLUTIONS

---

## Q1 Solutions

### 1a: Applications
1. **Machine Translation** - Converting text from one language to another (Google Translate)
2. **Sentiment Analysis** - Detecting emotional tone in text (analyzing Twitter sentiment)
3. **Speech Recognition** - Converting spoken words to text (Siri, Alexa)
4. **Text Summarization** - Creating concise summaries of long documents

### 1b: Ambiguity Types
a) **Structural Ambiguity** - Who has binoculars? She or the man?
b) **Structural Ambiguity** - Is the chicken ready to eat food OR ready to be eaten?
c) **Lexical Ambiguity (Homonymy)** - Lead (metal) vs Lead (to guide) - different words, same spelling

---

## Q2 Solutions

### 2a: Bigram Probability
```
Count(cat) = 2 (appears in sentence 1 and 3)
Count(cat, sat) = 1
P(sat | cat) = 1/2 = 0.5

Count(sat) = 2
Count(sat, on) = 2
P(on | sat) = 2/2 = 1.0
```

### 2b: Perplexity
```
P(sentence) = P(The) × P(cat|The) × P(ran|cat)
            = 0.5 × 0.3 × 0.2
            = 0.03

PP = P(sentence)^(-1/N)
   = (0.03)^(-1/3)
   = (1/0.03)^(1/3)
   = (33.33)^(1/3)
   = 3.22
```

---

## Q3 Solutions

### 3a: Few-shot Prompt
```
Classify the following movie reviews as Positive or Negative.

Review: "This movie was absolutely terrible. Waste of time."
Sentiment: Negative

Review: "Amazing! Best film I've seen this year!"
Sentiment: Positive

Review: "The acting was superb and the plot kept me engaged."
Sentiment:
```

### 3b: Comparison
| Parameter | N-gram LM | Neural LM |
|-----------|-----------|-----------|
| Unseen combinations | Cannot handle (P=0 without smoothing) | Generalizes via embeddings |
| Context window | Fixed (n-1 words) | Variable/Long (especially Transformers) |

---

## Q4 Solutions

### 4a: TF-IDF
```
TF = 1 + log₁₀(10) = 1 + 1 = 2

IDF = log₁₀(2000/50) = log₁₀(40) = 1.602

TF-IDF = 2 × 1.602 = 3.204
```

### 4b: Cosine Similarity
```
A = [3, 0, 4], B = [0, 3, 4]

Dot product = 3×0 + 0×3 + 4×4 = 0 + 0 + 16 = 16

|A| = √(9 + 0 + 16) = √25 = 5
|B| = √(0 + 9 + 16) = √25 = 5

Cosine = 16 / (5 × 5) = 16/25 = 0.64
```

---

## Q5 Solutions

### 5a: Skip-gram Training Pairs
Target: "processing", Window: 2
Context words: "language", "is", "Natural" (if considering left beyond), "fun"

Training pairs:
- (processing, Natural) if window allows
- (processing, language)
- (processing, is)
- (processing, fun)

### 5b: Word Analogy
```
Tokyo:Japan :: Paris:France
France = Japan - Tokyo + Paris

France = [0.6, 0.4, 0.3] - [0.7, 0.2, 0.5] + [0.8, 0.3, 0.4]
       = [0.6-0.7+0.8, 0.4-0.2+0.3, 0.3-0.5+0.4]
       = [0.7, 0.5, 0.2]
```

### 5c: Word2Vec Update
```
Error = σ(v·u) - y = 0.58 - 1 = -0.42

Gradient = Error × u = -0.42 × [0.4, 0.2] = [-0.168, -0.084]

v_new = v_old - η × Gradient
      = [0.3, 0.7] - 0.2 × [-0.168, -0.084]
      = [0.3, 0.7] - [-0.0336, -0.0168]
      = [0.3336, 0.7168]
```

---

## Q6 Solutions

### 6a: HMM Probabilities
```
Count(DT) = 4
Count(DT → NN) = 2
Count(DT → JJ) = 1

P(NN | DT) = 2/4 = 0.5

Count(NN) = 4
Count(NN → VBZ) = 2

P(VBZ | NN) = 2/4 = 0.5
```

### 6b: HMM Disambiguation
```
Score(NN) = P(NN|DT) × P("light"|NN) = 0.6 × 0.04 = 0.024
Score(JJ) = P(JJ|DT) × P("light"|JJ) = 0.3 × 0.02 = 0.006
Score(VB) = P(VB|DT) × P("light"|VB) = 0.1 × 0.01 = 0.001

Highest: 0.024 → NN (Noun)

"light" should be tagged as NN (Noun)
```

---

## Q7 Solution: Full Viterbi

### STEP 1: Initialization ("A")
```
V₁(DT) = π(DT) × P("A"|DT) = 0.5 × 0.7 = 0.35 ← HIGHEST
V₁(NN) = π(NN) × P("A"|NN) = 0.3 × 0.02 = 0.006
V₁(VBZ) = π(VBZ) × P("A"|VBZ) = 0.2 × 0.01 = 0.002
```

### STEP 2: Recursion ("dog")

**For NN:**
```
From DT: 0.35 × P(NN|DT) = 0.35 × 0.8 = 0.28 ← MAX
From NN: 0.006 × P(NN|NN) = 0.006 × 0.3 = 0.0018
From VBZ: 0.002 × P(NN|VBZ) = 0.002 × 0.5 = 0.001

V₂(NN) = 0.28 × P("dog"|NN) = 0.28 × 0.4 = 0.112
Backpointer: DT
```

**For DT:**
```
From DT: 0.35 × 0.1 = 0.035 ← MAX
V₂(DT) = 0.035 × 0.02 = 0.0007
```

**For VBZ:**
```
From DT: 0.35 × 0.1 = 0.035 ← MAX
V₂(VBZ) = 0.035 × 0.03 = 0.00105
```

### STEP 3: Recursion ("barks")

**For VBZ:**
```
From DT: 0.0007 × 0.1 = 0.00007
From NN: 0.112 × 0.5 = 0.056 ← MAX
From VBZ: 0.00105 × 0.1 = 0.000105

V₃(VBZ) = 0.056 × P("barks"|VBZ) = 0.056 × 0.5 = 0.028
Backpointer: NN
```

**For NN:**
```
From NN: 0.112 × 0.3 = 0.0336 ← MAX
V₃(NN) = 0.0336 × 0.08 = 0.002688
```

### VITERBI TABLE:

| State | "A" | "dog" | "barks" |
|-------|-----|-------|---------|
| DT | **0.35** | 0.0007 | - |
| NN | 0.006 | **0.112** ←DT | 0.002688 |
| VBZ | 0.002 | 0.00105 | **0.028** ←NN |

### STEP 4: Backtracking
```
Best final: VBZ (0.028)
Backpointer₃(VBZ) → NN
Backpointer₂(NN) → DT

Sequence: DT → NN → VBZ
```

### Final Answer:

| Word | Tag |
|------|-----|
| A | DT (Determiner) |
| dog | NN (Noun) |
| barks | VBZ (Verb) |

---

# 📊 ADDITIONAL PRACTICE QUESTIONS

---

## Extra Q2 Variation: Laplace Smoothing

**Given:**
- Count("the", "elephant") = 0
- Count("the") = 200
- Vocabulary size V = 50,000

**Calculate P(elephant | the) using Laplace smoothing.**

**Solution:**
```
P = (Count + 1) / (N + V)
  = (0 + 1) / (200 + 50000)
  = 1 / 50200
  = 0.0000199
```

---

## Extra Q3 Variation: Chain-of-Thought

**Write a Chain-of-Thought prompt for solving: "If a train travels 120 km in 2 hours, what is its speed?"**

**Solution:**
```
Q: If a train travels 120 km in 2 hours, what is its speed?

A: Let me solve this step by step.

Step 1: Recall the formula
        Speed = Distance / Time

Step 2: Identify values
        Distance = 120 km
        Time = 2 hours

Step 3: Calculate
        Speed = 120 / 2 = 60 km/hour

Answer: The train's speed is 60 km/hour.
```

---

## Extra Q4 Variation: Euclidean Distance

**Calculate Euclidean distance between A = [1, 4] and B = [5, 1]**

**Solution:**
```
d = √[(5-1)² + (1-4)²]
  = √[16 + 9]
  = √25
  = 5
```

---

## Extra Q5 Variation: Negative Sampling Update

**Given a NEGATIVE pair (dog, pizza), y = 0:**
- v(dog) = [0.5, 0.5]
- u(pizza) = [0.4, 0.6]
- σ(v·u) = 0.55
- η = 0.1

**Calculate updated v(dog).**

**Solution:**
```
Error = σ - y = 0.55 - 0 = 0.55 (positive for negative pair)

Gradient = 0.55 × [0.4, 0.6] = [0.22, 0.33]

v_new = [0.5, 0.5] - 0.1 × [0.22, 0.33]
      = [0.5 - 0.022, 0.5 - 0.033]
      = [0.478, 0.467]

Note: Vectors moved APART (as expected for negative pair)
```

---

## Extra Q6 Variation: Emission Probability

**Given corpus:**
```
The/DT cat/NN sat/VB
The/DT dog/NN ran/VB
A/DT cat/NN jumped/VB
```

**Calculate P("cat" | NN)**

**Solution:**
```
Count(NN) = 3 (cat, dog, cat)
Count(NN, "cat") = 2

P("cat" | NN) = 2/3 = 0.667
```

---

## Extra Q7 Variation: MEMM Features

**List 4 features that an MEMM can use but an HMM cannot for tagging the word "running":**

**Solution:**
1. Word suffix = "-ing" (verb indicator)
2. Previous word = "was" (suggests verb)
3. Word shape = lowercase
4. Next word = "fast" (adverb often follows verb)

---

# 🎯 QUICK REFERENCE: Log Values

```
log₁₀(1) = 0
log₁₀(2) = 0.301
log₁₀(3) = 0.477
log₁₀(4) = 0.602
log₁₀(5) = 0.699
log₁₀(6) = 0.778
log₁₀(7) = 0.845
log₁₀(8) = 0.903
log₁₀(10) = 1.0
log₁₀(20) = 1.301
log₁₀(25) = 1.398
log₁₀(40) = 1.602
log₁₀(50) = 1.699
log₁₀(100) = 2.0
```

---

**Good luck! Practice these and you're ready for 25+!** 🎯
