# 📚 NLP MIDSEM - COMPLETE STUDY GUIDE
## Actual Questions + Theory + Formula + Step-by-Step Solutions

---

# Q1: INTRODUCTION TO NLP (4 Marks)

---

## 📋 ACTUAL EXAM QUESTION

> **Q1a (2 marks):** List any 4 real-world applications of NLP with a brief description for each.
>
> **Q1b (2 marks):** Identify the type of ambiguity in the following sentences:
> - "I saw the man with the telescope"
> - "The bank was flooded"
> - "Flying planes can be dangerous"

---

## 🧠 THEORY: What is NLP?

**Definition:**
Natural Language Processing (NLP) is a field of AI that gives machines the ability to read, understand, and derive meaning from human languages. It bridges the gap between human communication and computer understanding.

**Two Main Components:**
1. **NLU (Natural Language Understanding)**: Converting human language → machine representation
2. **NLG (Natural Language Generation)**: Converting machine representation → human language

---

## 🧠 THEORY: NLP Applications

| Application | What it does | Real-world Example |
|-------------|--------------|-------------------|
| **Machine Translation** | Converts text between languages | Google Translate: "Hello" → "नमस्ते" |
| **Sentiment Analysis** | Detects emotion in text | Twitter analyzing if tweets are happy/angry |
| **Named Entity Recognition** | Finds names, places, organizations | "Apple is in California" → Apple=ORG, California=LOC |
| **Question Answering** | Answers questions from text | Siri: "What's the weather?" |
| **Text Summarization** | Creates short summary of long text | News article → 2 line summary |
| **Speech Recognition** | Converts spoken words to text | Alexa, Google Assistant |
| **Chatbots** | Conversational AI agents | Customer service bots |
| **POS Tagging** | Labels words with grammar roles | "dog" → Noun, "runs" → Verb |

---

## 🧠 THEORY: Types of Ambiguity

**Why is NLP hard?** Because language is AMBIGUOUS - one sentence can mean multiple things!

### 1. Structural Ambiguity (Syntactic)
**Definition:** Same words can be grouped differently, giving different parse trees and meanings.

**Example:** "I saw the man with the telescope"
- **Parse 1:** I [saw the man] [with the telescope] → I used the telescope to see
- **Parse 2:** I saw [the man with the telescope] → The man had the telescope

### 2. Lexical Ambiguity (Word meaning)
**Definition:** Same word has multiple meanings (Polysemy or Homonymy)

**Example:** "The bank was flooded"
- **Meaning 1:** River bank (edge of river) was flooded with water
- **Meaning 2:** Financial bank (institution) was flooded with customers

### 3. Grammatical Ambiguity (POS)
**Definition:** Same word can be assigned different parts of speech

**Example:** "Flying planes can be dangerous"
- **Parse 1:** [Flying planes] can be dangerous → The planes that are flying
- **Parse 2:** [Flying] [planes] can be dangerous → The act of flying planes

---

## ✅ ANSWER TO EXAM QUESTION

### Q1a: 4 NLP Applications

1. **Machine Translation**: Converting text from one language to another
   - Example: Google Translate converting English to Hindi

2. **Sentiment Analysis**: Determining emotional tone of text
   - Example: Analyzing product reviews to detect customer satisfaction

3. **Named Entity Recognition (NER)**: Identifying entities like names, places, organizations
   - Example: Extracting "Apple" as Organization and "California" as Location from news

4. **Question Answering**: Systems that answer questions in natural language
   - Example: Siri answering "What's the weather today?"

### Q1b: Ambiguity Types

| Sentence | Type | Explanation |
|----------|------|-------------|
| "I saw the man with the telescope" | **Structural Ambiguity** | Two parse trees possible: I used telescope OR man had telescope |
| "The bank was flooded" | **Lexical Ambiguity** | "bank" = river bank OR financial institution |
| "Flying planes can be dangerous" | **Structural Ambiguity** | Planes that fly OR the act of flying planes |

---

# Q2: LANGUAGE MODELS (4 Marks)

---

## 📋 ACTUAL EXAM QUESTION

> **Q2a (2 marks):** Given the following corpus, calculate P(love given I) and P(NLP given love):
> ```
> <s> I love NLP </s>
> <s> I love machine learning </s>
> <s> NLP is fun </s>
> ```
>
> **Q2b (2 marks):** Calculate the perplexity of the sentence "I love NLP" given:
> - P(I given start) = 0.4
> - P(love given I) = 0.5
> - P(NLP given love) = 0.2

---

## 🧠 THEORY: What is a Language Model?

**Definition:**
A Language Model assigns PROBABILITIES to sequences of words. It answers: "How likely is this sentence?"

**Why useful?**
- Speech recognition: "recognize speech" vs "wreck a nice beach" (same sound, different probability)
- Autocomplete: "I want to" → probably "go", "eat", "sleep"
- Machine translation: Choose more natural-sounding translation

---

## 📐 FORMULA: Bigram Probability

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   P(word | prev) = Count(prev, word) / Count(prev)         │
│                                                             │
│   Numerator: How many times did we see "prev word" together│
│   Denominator: How many times did we see "prev"            │
│                                                             │
│   ASSUMPTION: Only previous word matters (Markov)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📐 FORMULA: Laplace Smoothing

**Problem:** What if Count(prev, word) = 0? Then P = 0, and entire sentence P = 0!

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   P(word | prev) = (Count(prev,word) + 1) / (Count(prev) + V)│
│                                                             │
│   V = Vocabulary size (total unique words)                  │
│                                                             │
│   WHY: Add 1 to pretend we saw everything at least once    │
│   WHY +V: To make probabilities still sum to 1             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📐 FORMULA: Perplexity

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   PP = P(sentence)^(-1/N)  =  (1/P)^(1/N)                  │
│                                                             │
│   P = P(w1) × P(w2|w1) × P(w3|w2) × ...                    │
│   N = number of words                                       │
│                                                             │
│   INTERPRETATION:                                           │
│   - Lower PP = Better model                                 │
│   - PP of 10 = model choosing from ~10 equally likely words│
│                                                             │
│   WHY -1/N? Normalize for length + inverse (low P → high PP)|
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ ANSWER TO EXAM QUESTION

### Q2a: Bigram Probabilities

**Given corpus:**
```
<s> I love NLP </s>
<s> I love machine learning </s>
<s> NLP is fun </s>
```

**Calculate P(love | I):**
```
Step 1: Count occurrences of "I"
   Sentence 1: I appears once
   Sentence 2: I appears once
   Sentence 3: I doesn't appear
   Count(I) = 2

Step 2: Count occurrences of "I love" together
   Sentence 1: "I love" ✓
   Sentence 2: "I love" ✓
   Count(I, love) = 2

Step 3: Apply formula
   P(love | I) = Count(I, love) / Count(I)
               = 2 / 2
               = 1.0 ✓
```

**Calculate P(NLP | love):**
```
Count(love) = 2
Count(love, NLP) = 1  (only in sentence 1)

P(NLP | love) = 1 / 2 = 0.5 ✓
```

### Q2b: Perplexity

**Given:** P(I|start)=0.4, P(love|I)=0.5, P(NLP|love)=0.2, N=3

```
Step 1: Calculate P(sentence)
   P = P(I) × P(love|I) × P(NLP|love)
     = 0.4 × 0.5 × 0.2
     = 0.04

Step 2: Calculate inverse
   1/P = 1/0.04 = 25

Step 3: Take N-th root (N=3)
   PP = 25^(1/3) = ³√25 = 2.92 ✓

Interpretation: The model is choosing from ~3 words at each step (good!)
```

---

# Q3: NEURAL LANGUAGE MODELS & LLM (4 Marks)

---

## 📋 ACTUAL EXAM QUESTION

> **Q3a (2 marks):** Explain the difference between Zero-shot and Few-shot prompting. Provide an example of Few-shot prompting.
>
> **Q3b (2 marks):** Write a Chain-of-Thought prompt for solving a math problem.

---

## 🧠 THEORY: Pre-training vs Fine-tuning

| Aspect | Pre-training | Fine-tuning |
|--------|--------------|-------------|
| **Data** | Massive (all of Wikipedia, internet) | Small task-specific |
| **Task** | Predict next word | Specific task (sentiment, QA) |
| **Goal** | Learn general language patterns | Specialize for specific use |
| **Time** | Weeks/months | Hours/days |

**Analogy:** Pre-training = General school education, Fine-tuning = Medical specialization

---

## 🧠 THEORY: Prompting Types

### Zero-shot
No examples given, just instructions.

### Few-shot
Multiple examples provided before the query.

### Chain-of-Thought (CoT)
Ask model to show step-by-step reasoning.

---

## ✅ ANSWER TO EXAM QUESTION

### Q3a: Zero-shot vs Few-shot

| Aspect | Zero-shot | Few-shot |
|--------|-----------|----------|
| **Definition** | No examples given, just instruction | Multiple examples given before query |
| **Format** | Task description → Query | Examples → Query |
| **When to use** | Simple tasks model already knows | Complex/new tasks |

**Zero-shot Example:**
```
Classify the sentiment as Positive or Negative.
Review: "This product is amazing!"
Sentiment:
```

**Few-shot Example:**
```
Classify the sentiment as Positive or Negative.

Review: "Terrible quality, broke on day 1!"
Sentiment: Negative

Review: "Best purchase ever! Highly recommend."
Sentiment: Positive

Review: "This product is amazing!"
Sentiment:
```

### Q3b: Chain-of-Thought Prompt

```
Q: A store has 47 apples. They sell 23 and receive 18 more.
   How many apples do they have now?

A: Let me solve this step by step.

Step 1: Start with initial count
        Initial apples = 47

Step 2: Subtract sold apples
        After selling: 47 - 23 = 24 apples

Step 3: Add received apples
        After receiving: 24 + 18 = 42 apples

Answer: 42 apples
```

**Why CoT works:** Forces the model to show intermediate reasoning, reducing errors on complex problems.

---

# Q4: VECTOR SEMANTICS (4 Marks)

---

## 📋 ACTUAL EXAM QUESTION

> **Q4a (2 marks):** Calculate TF-IDF for word "neural" in Document D1 given:
> - Word "neural" appears 8 times in D1
> - Total documents N = 1000
> - "neural" appears in 40 documents
>
> Use: TF = 1 + log₁₀(count), IDF = log₁₀(N/df)
>
> **Q4b (2 marks):** Calculate cosine similarity between:
> - Document A = [2, 1, 0, 2]
> - Document B = [1, 1, 2, 1]

---

## 🧠 THEORY: Distributional Hypothesis

**Key Idea:** "You shall know a word by the company it keeps" - J.R. Firth

Words that appear in similar contexts have similar meanings.
- "cat" and "dog" both appear near: pet, fur, food, cute
- Therefore "cat" and "dog" are semantically similar

---

## 📐 FORMULA: TF-IDF

```
┌─────────────────────────────────────────────────────────────┐
│ TERM FREQUENCY (TF)                                         │
│                                                             │
│   TF = 1 + log₁₀(count)                                    │
│                                                             │
│   count = times word appears in document                    │
│   WHY log? Diminishing returns (10x occurrences ≠ 10x more) │
│   WHY +1? Prevent negative values (log of <1 is negative)  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ INVERSE DOCUMENT FREQUENCY (IDF)                            │
│                                                             │
│   IDF = log₁₀(N / df)                                      │
│                                                             │
│   N = total documents                                       │
│   df = documents containing this word                       │
│                                                             │
│   If word in ALL docs: IDF = log(1) = 0 (useless word!)    │
│   If word in FEW docs: IDF is HIGH (discriminating word!)  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ TF-IDF = TF × IDF                                           │
│                                                             │
│   High TF-IDF = Important keyword for this document         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📐 FORMULA: Cosine Similarity

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                    A · B           a₁b₁ + a₂b₂ + ...       │
│   cos(θ) = ───────────────── = ─────────────────────────── │
│             |A| × |B|         √(Σaᵢ²) × √(Σbᵢ²)           │
│                                                             │
│   INTERPRETATION:                                           │
│   - cos = 1: Identical direction (same content)            │
│   - cos = 0: Perpendicular (unrelated)                     │
│   - cos = -1: Opposite (never happens with word counts)    │
│                                                             │
│   WHY cosine, not Euclidean?                               │
│   Cosine ignores magnitude (length of document)            │
│   A 100-word and 1000-word doc about "dogs" are similar    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ ANSWER TO EXAM QUESTION

### Q4a: TF-IDF Calculation

**Given:** count=8, N=1000, df=40

```
Step 1: Calculate TF
   TF = 1 + log₁₀(8)
      = 1 + 0.903
      = 1.903 ✓

Step 2: Calculate IDF
   IDF = log₁₀(N / df)
       = log₁₀(1000 / 40)
       = log₁₀(25)
       = 1.398 ✓

Step 3: Calculate TF-IDF
   TF-IDF = TF × IDF
          = 1.903 × 1.398
          = 2.66 ✓
```

### Q4b: Cosine Similarity

**Given:** A = [2, 1, 0, 2], B = [1, 1, 2, 1]

```
Step 1: Calculate Dot Product (A · B)
   A · B = (2×1) + (1×1) + (0×2) + (2×1)
         = 2 + 1 + 0 + 2
         = 5 ✓

Step 2: Calculate |A| (magnitude of A)
   |A| = √(2² + 1² + 0² + 2²)
       = √(4 + 1 + 0 + 4)
       = √9
       = 3 ✓

Step 3: Calculate |B| (magnitude of B)
   |B| = √(1² + 1² + 2² + 1²)
       = √(1 + 1 + 4 + 1)
       = √7
       = 2.646 ✓

Step 4: Calculate Cosine Similarity
   cos(A, B) = (A · B) / (|A| × |B|)
             = 5 / (3 × 2.646)
             = 5 / 7.938
             = 0.63 ✓

Interpretation: 0.63 = Moderately similar documents
```

---

# Q5: WORD EMBEDDINGS (5 Marks)

---

## 📋 ACTUAL EXAM QUESTION

> **Q5a (2 marks):** Given vectors:
> - Man = [0.5, 0.3, 0.2]
> - Woman = [0.4, 0.6, 0.3]
> - King = [0.8, 0.4, 0.5]
>
> Find Queen using analogy: King:Man :: Queen:Woman
>
> **Q5b (3 marks):** In Skip-gram with Negative Sampling:
> - Target: "code", Context: "python" (Positive pair, y = 1)
> - v(code) = [0.2, 0.6]
> - u(python) = [0.5, 0.3]
> - σ(v · u) = 0.55
> - Learning rate η = 0.1
>
> Calculate the updated vector v(code)_new.

---

## 🧠 THEORY: Word2Vec

**Goal:** Learn dense vector representations where similar words have similar vectors

**Two architectures:**
1. **Skip-gram:** Given center word → predict context words
2. **CBOW:** Given context words → predict center word

**Key insight:** Words in similar contexts get similar vectors!

---

## 📐 FORMULA: Word Analogy

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   King - Man + Woman ≈ Queen                                │
│                                                             │
│   General pattern: A is to B as C is to ?                  │
│   ? = C - A + B                                             │
│                                                             │
│   WHY: Vector arithmetic captures semantic relationships    │
│   "King" and "Queen" differ only in gender                 │
│   Subtracting "Man" removes male component                 │
│   Adding "Woman" adds female component                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📐 FORMULA: Word2Vec Skip-gram Update

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Calculate Error                                     │
│                                                             │
│   Error = σ(v · u) - y                                     │
│                                                             │
│   σ = sigmoid (outputs 0-1 probability)                    │
│   v · u = dot product of target and context vectors        │
│   y = 1 for real pair, 0 for fake pair                    │
│                                                             │
│   If y=1 (real): Error is negative when σ < 1             │
│   If y=0 (fake): Error is positive when σ > 0             │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ STEP 2: Update Vector                                       │
│                                                             │
│   v(new) = v(old) - η × Error × u                          │
│                                                             │
│   η = learning rate                                         │
│                                                             │
│   For positive pair (Error negative):                       │
│      v moves TOWARD u (vectors get closer)                 │
│                                                             │
│   For negative pair (Error positive):                       │
│      v moves AWAY from u (vectors get farther)             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ ANSWER TO EXAM QUESTION

### Q5a: Word Analogy

**Given:** Man=[0.5,0.3,0.2], Woman=[0.4,0.6,0.3], King=[0.8,0.4,0.5]

```
Formula: Queen = King - Man + Woman

Calculate element by element:
   Position 1: 0.8 - 0.5 + 0.4 = 0.7
   Position 2: 0.4 - 0.3 + 0.6 = 0.7
   Position 3: 0.5 - 0.2 + 0.3 = 0.6

Queen = [0.7, 0.7, 0.6] ✓
```

### Q5b: Word2Vec Update

**Given:** v(code)=[0.2,0.6], u(python)=[0.5,0.3], σ=0.55, y=1, η=0.1

```
Step 1: Calculate Error
   Error = σ(v · u) - y
         = 0.55 - 1
         = -0.45 ✓

   (Negative because we under-predicted - should be closer to 1)

Step 2: Calculate Gradient
   Gradient = Error × u
            = -0.45 × [0.5, 0.3]
            = [-0.225, -0.135] ✓

Step 3: Update v(code)
   v(new) = v(old) - η × Gradient
          = [0.2, 0.6] - 0.1 × [-0.225, -0.135]
          = [0.2, 0.6] - [-0.0225, -0.0135]
          = [0.2 + 0.0225, 0.6 + 0.0135]
          = [0.2225, 0.6135] ✓

Result: v(code) moved CLOSER to u(python) (correct for positive pair!)
```

---

# Q6: POS TAGGING & HMM (4 Marks)

---

## 📋 ACTUAL EXAM QUESTION

> **Q6a (2 marks):** Given tagged corpus:
> ```
> DT NN VB IN DT NN
> DT JJ NN VBZ RB
> DT NN VB DT NN
> ```
> Calculate P(NN | DT) and P(VB | NN)
>
> **Q6b (2 marks):** Word "flies" appears after tag NN. Possible tags: NN, VBZ
>
> Given:
> - P(NN | NN) = 0.3, P(VBZ | NN) = 0.4
> - P("flies" | NN) = 0.02, P("flies" | VBZ) = 0.05
>
> Which tag should "flies" get?

---

## 🧠 THEORY: Hidden Markov Model

**The Problem:** Given a sentence, assign the correct POS tag to each word

**Why HMM?**
- Tags are HIDDEN (we can't see them directly)
- Words are OBSERVED (we can see them)
- We use probabilistic clues to infer hidden tags

**HMM Components:**

| Component | Symbol | What it represents |
|-----------|--------|-------------------|
| Hidden States | Q | POS tags (NN, VB, DT...) |
| Observations | O | Words we see |
| Transition | A | P(tag | previous tag) |
| Emission | B | P(word | tag) |
| Start | π | P(tag | START) |

---

## 📐 FORMULA: HMM Disambiguation

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Score(tag) = P(tag | prev_tag) × P(word | tag)           │
│                ─────────────────   ─────────────            │
│                   TRANSITION         EMISSION               │
│                                                             │
│   Transition: How likely is this tag after the previous?   │
│   Emission: How likely is this word to be this tag?        │
│                                                             │
│   DECISION: Pick the tag with HIGHEST score                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ ANSWER TO EXAM QUESTION

### Q6a: HMM Probabilities

**Given corpus:**
```
DT NN VB IN DT NN
DT JJ NN VBZ RB
DT NN VB DT NN
```

**Calculate P(NN | DT):**
```
Count(DT) = 5 (appears 5 times total)
Count(DT → NN) = 3 (DT followed by NN appears 3 times)
   Line 1: DT→NN ✓
   Line 2: DT→JJ (no)
   Line 3: DT→NN ✓, DT→NN ✓

P(NN | DT) = 3 / 5 = 0.6 ✓
```

**Calculate P(VB | NN):**
```
Count(NN) = 5
Count(NN → VB) = 2
   Line 1: NN→VB ✓
   Line 3: NN→VB ✓

P(VB | NN) = 2 / 5 = 0.4 ✓
```

### Q6b: HMM Disambiguation

**Word "flies" after NN. Options: NN or VBZ?**

```
Score(NN) = P(NN | NN) × P("flies" | NN)
          = 0.3 × 0.02
          = 0.006

Score(VBZ) = P(VBZ | NN) × P("flies" | VBZ)
           = 0.4 × 0.05
           = 0.020

Comparison: 0.020 > 0.006

Winner: VBZ ✓

In "Time flies", the word "flies" is a VERB (3rd person singular).
```

---

# Q7: VITERBI ALGORITHM (5 Marks)

---

## 📋 ACTUAL EXAM QUESTION

> **Q7 (5 marks):** Complete the Viterbi table for the sentence "The dog runs"
>
> **States:** DT, NN, VBZ
>
> **Start Probabilities (π):** DT=0.6, NN=0.3, VBZ=0.1
>
> **Transition Probabilities:**
> | From → To | DT | NN | VBZ |
> |-----------|-----|-----|-----|
> | DT | 0.1 | 0.7 | 0.2 |
> | NN | 0.1 | 0.2 | 0.7 |
> | VBZ | 0.3 | 0.6 | 0.1 |
>
> **Emission Probabilities:**
> | Word | P(given DT) | P(given NN) | P(given VBZ) |
> |------|---------|---------|----------|
> | The | 0.8 | 0.01 | 0.01 |
> | dog | 0.01 | 0.3 | 0.02 |
> | runs | 0.01 | 0.05 | 0.4 |
>
> Show the Viterbi table with backpointers and find the best tag sequence.

---

## 🧠 THEORY: Why Viterbi?

**Problem:** For N words with T possible tags each:
- Total possible sequences = T^N
- For 10 words, 40 tags: 40^10 = 10 quadrillion possibilities!

**Solution:** Viterbi uses DYNAMIC PROGRAMMING
- At each word, only keep the best path TO each state
- Complexity: O(T² × N) instead of O(T^N)

---

## 📐 FORMULA: Viterbi Algorithm

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: INITIALIZATION (First word)                         │
│                                                             │
│   V₁(tag) = π(tag) × P(word₁ | tag)                        │
│                                                             │
│   Calculate for EACH possible tag                          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ STEP 2: RECURSION (Each subsequent word)                    │
│                                                             │
│   Vₜ(j) = max[Vₜ₋₁(i) × P(j|i)] × P(wordₜ | j)            │
│           └── try all prev states, keep max ──┘            │
│                                                             │
│   Store backpointer: which previous state gave the max     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ STEP 3: BACKTRACKING                                        │
│                                                             │
│   1. Find state with highest final score                   │
│   2. Follow backpointers backwards                         │
│   3. Reverse to get tag sequence                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ ANSWER TO EXAM QUESTION

### STEP 1: INITIALIZATION ("The")

```
V₁(DT) = π(DT) × P("The"|DT) = 0.6 × 0.8 = 0.48 ← HIGHEST
V₁(NN) = π(NN) × P("The"|NN) = 0.3 × 0.01 = 0.003
V₁(VBZ) = π(VBZ) × P("The"|VBZ) = 0.1 × 0.01 = 0.001
```

### STEP 2: RECURSION ("dog")

**For NN (try all previous states):**
```
From DT: V₁(DT) × P(NN|DT) = 0.48 × 0.7 = 0.336 ← MAX
From NN: V₁(NN) × P(NN|NN) = 0.003 × 0.2 = 0.0006
From VBZ: V₁(VBZ) × P(NN|VBZ) = 0.001 × 0.6 = 0.0006

V₂(NN) = 0.336 × P("dog"|NN) = 0.336 × 0.3 = 0.1008
Backpointer₂(NN) = DT
```

**For DT:**
```
Max = 0.48 × 0.1 = 0.048
V₂(DT) = 0.048 × P("dog"|DT) = 0.048 × 0.01 = 0.00048
```

**For VBZ:**
```
From DT: 0.48 × 0.2 = 0.096 ← MAX
V₂(VBZ) = 0.096 × P("dog"|VBZ) = 0.096 × 0.02 = 0.00192
```

### STEP 3: RECURSION ("runs")

**For VBZ:**
```
From DT: V₂(DT) × P(VBZ|DT) = 0.00048 × 0.2 = 0.000096
From NN: V₂(NN) × P(VBZ|NN) = 0.1008 × 0.7 = 0.07056 ← MAX
From VBZ: V₂(VBZ) × P(VBZ|VBZ) = 0.00192 × 0.1 = 0.000192

V₃(VBZ) = 0.07056 × P("runs"|VBZ) = 0.07056 × 0.4 = 0.02822
Backpointer₃(VBZ) = NN
```

**For NN:**
```
From NN: 0.1008 × 0.2 = 0.02016 ← MAX
V₃(NN) = 0.02016 × P("runs"|NN) = 0.02016 × 0.05 = 0.001008
```

### VITERBI TABLE:

| State | "The" | "dog" | "runs" |
|-------|-------|-------|--------|
| **DT** | **0.48** | 0.00048 | - |
| **NN** | 0.003 | **0.1008** ←DT | 0.001008 |
| **VBZ** | 0.001 | 0.00192 | **0.02822** ←NN |

### STEP 4: BACKTRACKING

```
Best final state: VBZ (0.02822) ← HIGHEST

Trace back:
   "runs" → VBZ (score = 0.02822)
      ↑ backpointer points to NN
   "dog" → NN (backpointer points to DT)
      ↑ backpointer points to DT
   "The" → DT (score = 0.48)

FINAL ANSWER: DT → NN → VBZ
```

### Final Tagging:

| Word | Tag | Meaning |
|------|-----|---------|
| The | DT | Determiner |
| dog | NN | Noun |
| runs | VBZ | Verb (3rd person singular) |

---

# 📊 QUICK REFERENCE: All Formulas

| Topic | Formula |
|-------|---------|
| **Bigram** | P(w given prev) = C(prev,w) / C(prev) |
| **Laplace** | P = (C+1) / (N+V) |
| **Perplexity** | PP = (1/P)^(1/N) |
| **TF** | TF = 1 + log₁₀(count) |
| **IDF** | IDF = log₁₀(N/df) |
| **Cosine** | cos = (A·B) / (mag A × mag B) |
| **Analogy** | Queen = King - Man + Woman |
| **Word2Vec** | v(new) = v(old) - η × (σ-y) × u |
| **HMM Score** | Score = Transition × Emission |
| **Viterbi** | V(j) = max[V(prev)×Trans] × Emit |

---

# 📝 LOG VALUES TO MEMORIZE

```
log₁₀(1) = 0       log₁₀(8) = 0.903
log₁₀(2) = 0.301   log₁₀(10) = 1.0
log₁₀(3) = 0.477   log₁₀(25) = 1.398
log₁₀(4) = 0.602   log₁₀(40) = 1.602
log₁₀(5) = 0.699   log₁₀(100) = 2.0
```

---

**Good luck with your makeup exam!** 🍀
