# 📚 NLP Mid-Semester Exam Preparation Guide
## AIMLCZG530 - Natural Language Processing

---

## 📋 Exam Structure (30 Marks Total)

Based on the announcement, the midsem exam follows this pattern:

| Question | Module | Topic | Marks |
|----------|--------|-------|-------|
| Q1 | Module 1 | Introduction & Applications | 4 |
| Q2 | Module 2 | Language Models (Problem-based) | 4 |
| Q3 | Module 3 | Neural Language Models & LLM (Problem/Application) | 4 |
| Q4 | Module 4 | Vector Semantics (Problem-based) | 4 |
| Q5 | Module 4 | Word Embedding (Problem-based) | 5 |
| Q6 | Module 5 | POS Tagging (Problem-based) | 4 |
| Q7 | Module 6 | Statistical, ML & Neural POS Models (Problem/Application) | 5 |

**Duration:** 2 hours  
**Type:** Closed Book  
**Syllabus:** Contact Sessions 1 to 8

---

# 📖 Module-wise Content Summary

---

# 🖼️ Visual Concept Diagrams

Below are visual explanations of key NLP concepts covered in the exam (Sessions 1-8):

---

## 📗 Module 1: Introduction to NLP

### Levels of Language Analysis
![Six levels from Morphological to Pragmatic as a hierarchy](./images/language_levels.png)

### Types of Ambiguity
![Structural, Lexical, and Grammatical ambiguity with examples](./images/ambiguity_types.png)

---

## 📘 Module 2: Vector Semantics & Word Embeddings

### TF-IDF Calculation
![Step-by-step TF-IDF calculation with formulas and example](./images/tfidf.png)

### Cosine Similarity
![Vector space with two vectors showing angle and cosine formula](./images/cosine.png)

### Word2Vec Skip-gram Architecture
![Skip-gram neural network: target word → context words](./images/word2vec.png)

### Word2Vec CBOW Architecture
![CBOW: context words → target word prediction](./images/cbow.png)

### Word Analogy (Parallelogram Method)
![King-Queen-Man-Woman analogy in vector space](./images/analogy.png)

---

## 📙 Module 3: N-gram Language Models

### N-gram Sliding Windows
![Unigram, bigram, trigram with probability calculations](./images/ngram.png)

### Smoothing Techniques
![Laplace, Interpolation, Backoff, and Stupid Backoff methods](./images/smoothing.png)

### Perplexity Metric
![Perplexity as branching factor - comparing low vs high perplexity](./images/perplexity.png)

---

## 📕 Module 4: Neural Language Models & LLMs

### Neural Language Model Architecture
![Neural LM with embedding layer, hidden layer, and softmax output](./images/neural_lm.png)

---

## 📒 Module 5 & 6: POS Tagging

### HMM for POS Tagging
![Hidden states (tags) with transition and emission probabilities](./images/hmm.png)

### Viterbi Algorithm Trellis
![Dynamic programming trellis with scores and backpointers](./images/viterbi.png)

---





## Module 1: Introduction to NLP (4 Marks)

### 1.1 What is NLP?
- **Natural Language Understanding (NLU)**: Converting language to machine-understandable representation
- **Natural Language Generation (NLG)**: Converting machine representation to human language

### 1.2 Applications of NLP
| Application | Description |
|------------|-------------|
| Speech Recognition | Converting spoken language to text |
| Machine Translation | Translating between languages |
| Information Extraction | Extracting structured data from text |
| Sentiment Analysis | Determining emotional tone |
| Question Answering | Answering questions from text |
| Text Summarization | Creating concise summaries |
| Chatbots/Virtual Assistants | Interactive conversational agents |
| Named Entity Recognition | Identifying entities (person, place, org) |
| POS Tagging | Labeling grammatical categories |
| Document Classification | Categorizing documents |

### 1.3 Why is NLP Difficult?
1. **Ambiguity at all levels**
2. **Multiple languages and cultural entities**
3. **Language keeps changing** (new words, rules, exceptions)
4. **Noisy data sources**

### 1.4 Types of Ambiguities
| Type | Example |
|------|---------|
| **Structural** | "Visiting relatives can be a nuisance" (two meanings) |
| **Grammatical** | "Can" - container (Noun), ability (Modal), to preserve (Verb) |
| **Lexical - Polysemy** | "Bank" - river bank vs. financial bank |
| **Lexical - Homonymy** | Different words that sound/look same |

### 1.5 Levels of Language Analysis
```
Pragmatic (Context-dependent meaning)
    ↓
Discourse (Sentence relationships)
    ↓
Semantic (Word meaning)
    ↓
Syntactic (Sentence structure)
    ↓
Lexical (Word categories)
    ↓
Morphological (Word formation)
```

### 1.6 Evaluation Metrics
| Metric | Formula |
|--------|---------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) |

### 1.7 NLP Tools
- **Open Source**: NLTK, spaCy, Hugging Face Transformers, Gensim, LangChain
- **Commercial**: Google Cloud AI, Amazon Comprehend, Microsoft Azure AI, IBM Watson, OpenAI API

---

## Module 2: N-gram Language Modeling (4 Marks)

### 2.1 Language Model Definition
A **Language Model** assigns probabilities to sequences of words:
- **P(W)** = P(w₁, w₂, ..., wₙ) - probability of a sentence
- **P(wₙ | w₁...wₙ₋₁)** - probability of next word given history

### 2.2 Chain Rule of Probability
```
P(w₁, w₂, ..., wₙ) = P(w₁) × P(w₂|w₁) × P(w₃|w₁,w₂) × ... × P(wₙ|w₁...wₙ₋₁)
```

### 2.3 Markov Assumption
> The probability of a word depends only on the previous (n-1) words.

| Model | Formula | Context Window |
|-------|---------|----------------|
| **Unigram** | P(wₙ) | 0 words |
| **Bigram** | P(wₙ \| wₙ₋₁) | 1 word |
| **Trigram** | P(wₙ \| wₙ₋₂, wₙ₋₁) | 2 words |

### 2.4 N-gram Probability Calculation (MLE)

**Bigram Formula:**
```
P(wₙ | wₙ₋₁) = Count(wₙ₋₁, wₙ) / Count(wₙ₋₁)
```

**Trigram Formula:**
```
P(wₙ | wₙ₋₂, wₙ₋₁) = Count(wₙ₋₂, wₙ₋₁, wₙ) / Count(wₙ₋₂, wₙ₋₁)
```

### 2.5 The Zero Problem & Smoothing

**Problem**: Unseen n-grams have P = 0, making entire sentence probability = 0

**Solutions:**

#### a) Laplace (Add-1) Smoothing
```
P_Laplace(wₙ | wₙ₋₁) = [Count(wₙ₋₁, wₙ) + 1] / [Count(wₙ₋₁) + V]
```
Where V = vocabulary size

#### b) Linear Interpolation
```
P(wₙ|wₙ₋₂,wₙ₋₁) = λ₁·P(wₙ|wₙ₋₂,wₙ₋₁) + λ₂·P(wₙ|wₙ₋₁) + λ₃·P(wₙ)
```
Where λ₁ + λ₂ + λ₃ = 1

#### c) Backoff
Use lower-order n-gram if higher-order has zero count

#### d) Stupid Backoff (Web-Scale)
```
S(wᵢ | wᵢ₋ₖ₊₁...wᵢ₋₁) = Count(wᵢ₋ₖ₊₁...wᵢ) / Count(wᵢ₋ₖ₊₁...wᵢ₋₁)  if count > 0
                        = 0.4 × S(wᵢ | wᵢ₋ₖ₊₂...wᵢ₋₁)              otherwise
```

### 2.6 Perplexity (Evaluation Metric)

**Formula:**
```
PP(W) = P(w₁, w₂, ..., wₙ)^(-1/N) = ⁿ√(1 / P(W))
```

**Interpretation:**
- Lower perplexity = Better model
- Perplexity of 10 = model choosing from ~10 equally likely words
- Like a "branching factor"

**Example Calculation:**
```
Sentence: "I love NLP models" (4 words)
P(I)=0.10, P(love|I)=0.20, P(NLP|love)=0.25, P(models|NLP)=0.50

P(sentence) = 0.10 × 0.20 × 0.25 × 0.50 = 0.0025
PP = (0.0025)^(-1/4) = (1/0.0025)^(1/4) = 400^0.25 ≈ 4.47
```

| N-gram | Perplexity (WSJ) |
|--------|------------------|
| Unigram | 962 |
| Bigram | 170 |
| Trigram | 109 |

---

## Module 3: Neural Language Models & LLMs (4 Marks)

### 3.1 Limitations of N-gram Models
1. No semantic similarity (can't generalize "cat" to "dog")
2. Limited context window
3. Sparse representations
4. No handling of long-range dependencies

### 3.2 Neural Language Model Architecture
```
Input: One-hot vectors → Word Embeddings → Hidden Layer → Softmax → Output
```

**Key Equations:**
```
e = [E·x₁; E·x₂; E·x₃]        # Concatenate embeddings
h = σ(W·e + b)                 # Hidden layer
y = softmax(U·h)               # Output probabilities
```

### 3.3 Why Neural LMs Work Better
- Use **embeddings** instead of sparse one-hot vectors
- Similar words have similar embeddings
- Can generalize: seen "cat gets fed" → can predict "dog gets fed"

### 3.4 N-gram vs LLM Comparison

| Feature | N-gram Models | LLMs |
|---------|--------------|------|
| Training Data | Small | Huge |
| Compute | Low | Very High |
| Interpretability | Excellent | Poor |
| Long Context | No | Yes |
| Language Quality | Low-Medium | Very High |
| Hallucination | Almost None | Possible |
| Deployment | Easy, Lightweight | Heavy, Expensive |

### 3.5 Large Language Models (LLMs)

**Key Concepts:**
- **Pre-training**: Training on massive datasets to learn general language patterns
- **Fine-tuning**: Specializing on smaller task-specific datasets
- **Transfer Learning**: Using pre-trained knowledge for new tasks

### 3.6 Prompt Engineering

| Type | Description | Example |
|------|-------------|---------|
| **Zero-shot** | No examples provided | "Translate: Hello → French" |
| **One-shot** | One example provided | "Hello→Bonjour. Goodbye→?" |
| **Few-shot** | Multiple examples provided | Several input-output pairs |
| **Chain-of-Thought** | Step-by-step reasoning | "Let's think step by step..." |

---

## Module 4: Vector Semantics & Word Embeddings (9 Marks Total)

### 4.1 Distributional Hypothesis
> **"You shall know a word by the company it keeps"** - J.R. Firth

Words appearing in similar contexts have similar meanings.

### 4.2 Types of Word Representations

| Type | Characteristics |
|------|-----------------|
| **Sparse Vectors** | High-dimensional (~50,000), mostly zeros, count-based |
| **Dense Vectors (Embeddings)** | Low-dimensional (50-300), learned, mostly non-zero |

### 4.3 TF-IDF (Term Frequency - Inverse Document Frequency)

**Term Frequency (TF):**
```
tf(t,d) = log₁₀(count(t,d) + 1)
```

**Inverse Document Frequency (IDF):**
```
idf(t) = log₁₀(N / df(t))
```
Where N = total documents, df(t) = documents containing term t

**TF-IDF Weight:**
```
w(t,d) = tf(t,d) × idf(t)
```

**Example Calculation:**
```
Corpus: N = 100 documents
Word "neural" appears in 10 documents
In Document D1, "neural" appears 5 times

TF = 1 + log₁₀(5) = 1 + 0.7 = 1.7
IDF = log₁₀(100/10) = log₁₀(10) = 1
TF-IDF = 1.7 × 1 = 1.7
```

### 4.4 Cosine Similarity

**Formula:**
```
cos(A, B) = (A · B) / (||A|| × ||B||) = Σ(aᵢ × bᵢ) / (√Σaᵢ² × √Σbᵢ²)
```

**Example:**
```
A = [1, 2, 0], B = [0, 3, 4]

Dot Product = (1×0) + (2×3) + (0×4) = 6
||A|| = √(1² + 2² + 0²) = √5 ≈ 2.236
||B|| = √(0² + 3² + 4²) = √25 = 5

Cosine Similarity = 6 / (2.236 × 5) = 6/11.18 ≈ 0.536
```

**Interpretation:**
- 1 = Identical direction (maximum similarity)
- 0 = Perpendicular (no similarity)
- -1 = Opposite directions

### 4.5 Euclidean Distance

**Formula:**
```
d(A, B) = √Σ(aᵢ - bᵢ)²
```

**Example:**
```
A = [1, 5], B = [4, 1]
d = √[(4-1)² + (1-5)²] = √[9 + 16] = √25 = 5
```

### 4.6 Document Embedding (Centroid Method)

**Formula:**
```
D_vector = (v₁ + v₂ + ... + vₙ) / n
```

**Example:**
```
"Apple" = [0.5, 0.5], "Red" = [0.1, 0.9]
Document = [(0.5+0.1)/2, (0.5+0.9)/2] = [0.3, 0.7]
```

### 4.7 Word2Vec: Skip-gram Model

**Objective:** Given a target word, predict context words

**Architecture:**
```
Input (One-hot) → Hidden Layer (Embedding) → Output (Softmax)
      V×1              V×D                      V×1
```

**Training Pairs (Window = 2):**
```
Sentence: "The quick brown fox jumps"
Target: "brown"
Context words: "The", "quick", "fox", "jumps"
Training pairs: (brown, The), (brown, quick), (brown, fox), (brown, jumps)
```

### 4.8 Negative Sampling

**Problem:** Softmax over entire vocabulary is expensive

**Solution:** Train binary classifier to distinguish real pairs from fake pairs

**Objective Function:**
```
J = log σ(v_target · u_context) + Σ log σ(-v_target · u_negative)
```

### 4.9 Skip-gram Backward Propagation

**Given:**
- Target: "cat", Context: "meow" (Positive, y = 1)
- v_cat = [0.5, 0.5], u_meow = [1.0, 0.0]
- P = σ(v · u) = 0.62, Learning rate η = 0.1

**Update Rule:**
```
EI = P - y = 0.62 - 1 = -0.38
Gradient = EI × u_meow = -0.38 × [1.0, 0.0] = [-0.38, 0.0]
v_new = v_old - η × Gradient
v_new = [0.5, 0.5] - 0.1 × [-0.38, 0.0] = [0.538, 0.5]
```

### 4.10 CBOW (Continuous Bag of Words)

**Objective:** Given context words, predict target word

**Comparison:**
| Feature | Skip-gram | CBOW |
|---------|-----------|------|
| Input | Target word | Context words |
| Output | Context words | Target word |
| Speed | Slower | Faster |
| Rare words | Better | Worse |
| Data needed | Less | More |

### 4.11 Word Analogies (Parallelogram Method)

**Formula:**
```
v_King ≈ v_Queen - v_Woman + v_Man
```

**Examples:**
- Paris : France :: Tokyo : Japan
- King : Queen :: Man : Woman

### 4.12 GloVe (Global Vectors)

**Key Difference from Word2Vec:**
- Uses global co-occurrence statistics
- Learns from ratios of co-occurrence probabilities

**Objective:**
```
J = Σ f(Xᵢⱼ) × (vᵢᵀuⱼ + bᵢ + b̃ⱼ - log(Xᵢⱼ))²
```

### 4.13 Count-based vs Prediction-based Comparison

| Feature | LSA (Count-based) | Word2Vec (Prediction-based) |
|---------|-------------------|---------------------------|
| Method | SVD on count matrix | Neural network prediction |
| Sparsity | Dense (after SVD) | Dense |
| Interpretability | Latent topics | Not interpretable |
| Training | Single pass | Iterative |

---

## Module 5: POS Tagging & HMM (4 Marks)

### 5.1 Part-of-Speech Tags (Penn Treebank)

| Tag | Meaning | Example |
|-----|---------|---------|
| NN | Noun, singular | dog, cat |
| NNS | Noun, plural | dogs, cats |
| VB | Verb, base form | eat, run |
| VBZ | Verb, 3rd person singular | eats, runs |
| VBD | Verb, past tense | ate, ran |
| JJ | Adjective | happy, big |
| RB | Adverb | quickly |
| DT | Determiner | the, a |
| IN | Preposition | in, on, at |
| PRP | Personal pronoun | I, you, he |

### 5.2 Hidden Markov Model (HMM)

**Components:**
- **Hidden States (Q)**: POS tags
- **Observations (O)**: Words
- **Start Probabilities (π)**: P(tag | Start)
- **Transition Probabilities (A)**: P(tagᵢ | tagᵢ₋₁)
- **Emission Probabilities (B)**: P(word | tag)

### 5.3 HMM Probability Formulas

**Transition Probability:**
```
P(tagᵢ | tagᵢ₋₁) = Count(tagᵢ₋₁, tagᵢ) / Count(tagᵢ₋₁)
```

**Emission Probability:**
```
P(word | tag) = Count(tag, word) / Count(tag)
```

### 5.4 HMM Disambiguation Example

**Problem:** Disambiguate "book" in "read book"
- Previous tag: VB
- Possible tags for "book": NN, VB

**Given:**
```
Transitions: P(NN|VB) = 0.4, P(VB|VB) = 0.1
Emissions: P("book"|NN) = 0.05, P("book"|VB) = 0.01
```

**Calculation:**
```
Score(NN) = P(NN|VB) × P("book"|NN) = 0.4 × 0.05 = 0.020
Score(VB) = P(VB|VB) × P("book"|VB) = 0.1 × 0.01 = 0.001
```

**Result:** NN wins (0.020 > 0.001), so "book" is tagged as **Noun**

---

## Module 6: Viterbi Algorithm & Advanced POS Models (5 Marks)

### 6.1 Viterbi Algorithm

**Purpose:** Find the most probable sequence of hidden states

**Complexity:** O(N² × T) where N = number of states, T = sequence length

**Three Steps:**
1. **Initialization:** Calculate scores for first observation
2. **Recursion:** For each subsequent observation, find best path
3. **Backtrace:** Follow backpointers to recover best sequence

### 6.2 Viterbi Formula

**Initialization (t=1):**
```
V₁(j) = π(j) × P(o₁ | j)
```

**Recursion (t>1):**
```
Vₜ(j) = max[Vₜ₋₁(i) × P(j|i)] × P(oₜ|j)
```

### 6.3 Viterbi Example: "Time flies"

**Given:**
- States: N (Noun), V (Verb)
- Start Probs: π(N) = 0.8, π(V) = 0.2
- Emissions: P("Time"|N) = 0.5, P("Time"|V) = 0.1

**Step 1 - Initialization:**
```
V₁(N) = π(N) × P("Time"|N) = 0.8 × 0.5 = 0.4
V₁(V) = π(V) × P("Time"|V) = 0.2 × 0.1 = 0.02
```

**Best tag for "Time": N (0.4 > 0.02)**

### 6.4 Full Viterbi Table Example

**Sentence:** "The doctor is in"

| Word | the | doctor | is | in | STOP |
|------|-----|--------|----|----|------|
| **VB** | 0 | 0.00021 | 0.027216 | 0 | - |
| **NN** | 0 | 0.0756 | 0.001512 | 0 | 0.005443 |
| **Det** | 0.21 | 0 | 0 | 0 | - |
| **Prep** | 0 | 0 | 0 | 0.005443 | - |

**Best sequence:** Det → NN → VB → Prep

### 6.5 Maximum Entropy Markov Model (MEMM)

**Key Difference from HMM:**
- HMM is **Generative**: P(word | tag)
- MEMM is **Discriminative**: P(tag | word, features)

**Advantages:**
- Can use arbitrary overlapping features
- Features: word shape, suffixes, prefixes, surrounding words

**Feature Examples for word "back":**
- Current word = "back"
- Previous tag = MD
- Previous word = "will"
- Next word = "the"
- Contains capital? No
- Suffix = "ack"

### 6.6 MEMM vs HMM Comparison

| Aspect | HMM | MEMM |
|--------|-----|------|
| Type | Generative | Discriminative |
| Features | Limited | Rich, overlapping |
| Independence | Strong assumptions | Flexible |
| Training | EM algorithm | Gradient descent |

### 6.7 Neural POS Tagging

**RNN-based Tagger:**
- Input: Sequence of word embeddings
- Output: Probability distribution over tags at each timestep
- Activation: Softmax

**Bi-LSTM-CRF:**
- Bidirectional context
- CRF layer ensures valid tag sequences
- Accuracy: ~97-98%

**Transformer-based (BERT/RoBERTa):**
- Pre-trained language models
- Fine-tuned for POS tagging
- Accuracy: ~98.5-99.5%

---

# 🎯 Practice Questions with Solutions

---

## Q1. Module 1: Introduction (4 Marks)

### Question 1.1
**List any four applications of NLP with brief descriptions.**

**Solution:**
1. **Machine Translation**: Automatically translating text between languages (e.g., Google Translate)
2. **Sentiment Analysis**: Determining the emotional tone of text (positive/negative/neutral)
3. **Named Entity Recognition**: Identifying and classifying entities like person names, organizations, locations
4. **Question Answering**: Systems that can answer questions posed in natural language

### Question 1.2
**Explain the levels of language analysis with examples.**

**Solution:**
1. **Morphological**: Word formation - "unhappiness" = un + happy + ness
2. **Lexical**: Word categorization - "bank" can be noun or verb
3. **Syntactic**: Sentence structure - "The cat sat on mat" is grammatical
4. **Semantic**: Meaning - "Colorless green ideas sleep furiously" is syntactically correct but semantically meaningless
5. **Pragmatic**: Context-dependent - "Can you pass the salt?" is a request, not a question
6. **Discourse**: Multi-sentence - "John went to the store. He bought milk." - "He" refers to John

### Question 1.3
**Identify the type of ambiguity in each:**
a) "I saw the man with telescope"
b) "The bank is flooded"
c) "Can you can a can?"

**Solution:**
a) **Structural ambiguity**: Two interpretations - I used telescope to see man, or man had telescope
b) **Lexical ambiguity (Homonymy)**: "bank" = river bank or financial institution
c) **Grammatical ambiguity**: "can" appears as modal verb, main verb, and noun

---

## Q2. Module 2: Language Models (4 Marks)

### Question 2.1
**Given the following corpus, calculate the bigram probability P("am"|"I"):**
```
Corpus: "I am happy. I am learning NLP. I like NLP."
```

**Solution:**
```
Count("I") = 3
Count("I", "am") = 2
Count("I", "like") = 1

P("am"|"I") = Count("I", "am") / Count("I") = 2/3 ≈ 0.667
```

### Question 2.2
**Calculate the perplexity of a bigram model on the test sentence "I like learning" given:**
- P(I) = 0.3
- P(like|I) = 0.2
- P(learning|like) = 0.15

**Solution:**
```
P(sentence) = P(I) × P(like|I) × P(learning|like)
P(sentence) = 0.3 × 0.2 × 0.15 = 0.009

N = 3 words
PP = P(sentence)^(-1/N) = (0.009)^(-1/3) = (1/0.009)^(1/3)
PP = (111.11)^(1/3) ≈ 4.81
```

### Question 2.3
**Apply Laplace smoothing to calculate P("NLP"|"learn") given:**
- Count("learn") = 50
- Count("learn", "NLP") = 0
- Vocabulary size V = 1000

**Solution:**
```
P_Laplace("NLP"|"learn") = [Count("learn","NLP") + 1] / [Count("learn") + V]
P_Laplace("NLP"|"learn") = (0 + 1) / (50 + 1000) = 1/1050 ≈ 0.00095
```

### Question 2.4
**Using linear interpolation with λ₁=0.6, λ₂=0.3, λ₃=0.1, calculate P(w₃|w₁,w₂) given:**
- P(w₃|w₁,w₂) = 0.02 (trigram)
- P(w₃|w₂) = 0.05 (bigram)
- P(w₃) = 0.001 (unigram)

**Solution:**
```
P_interpolated = λ₁×P(w₃|w₁,w₂) + λ₂×P(w₃|w₂) + λ₃×P(w₃)
P_interpolated = 0.6×0.02 + 0.3×0.05 + 0.1×0.001
P_interpolated = 0.012 + 0.015 + 0.0001 = 0.0271
```

---

## Q3. Module 3: Neural LM & LLM (4 Marks)

### Question 3.1
**Compare N-gram and Neural Language Models on three parameters.**

**Solution:**
| Parameter | N-gram LM | Neural LM |
|-----------|-----------|-----------|
| Context | Fixed, limited (n-1 words) | Longer, flexible context |
| Generalization | Cannot generalize (exact match) | Generalizes via embeddings |
| Data Sparsity | Suffers from zero counts | Handles unseen combinations |

### Question 3.2
**Distinguish between Pre-training and Fine-tuning in LLMs.**

**Solution:**
- **Pre-training**: Training the model on massive datasets to learn general language patterns through tasks like next word prediction. The model learns grammar, facts, and reasoning abilities.
- **Fine-tuning**: Training the pre-trained model on a smaller, task-specific dataset to specialize it for a particular task (e.g., sentiment analysis, medical diagnosis).

### Question 3.3
**What is the difference between Zero-shot and Few-shot inference? Give an example of Few-shot.**

**Solution:**
- **Zero-shot**: Model performs task with no examples, just instructions
- **Few-shot**: Model is given a few examples of input-output pairs before the actual query

**Few-shot Example:**
```
Translate English to French:
Hello → Bonjour
Dog → Chien
Cat → ?
```
Model predicts: "Chat"

### Question 3.4
**List three advantages of using Transfer Learning in NLP.**

**Solution:**
1. **Reduced training time**: Leverage pre-trained representations instead of training from scratch
2. **Less data required**: Can achieve good performance with smaller task-specific datasets
3. **Better generalization**: Pre-trained models capture rich linguistic knowledge that transfers across tasks

---

## Q4. Module 4: Vector Semantics (4 Marks)

### Question 4.1
**Calculate TF-IDF for the word "machine" in document D1:**
- Document D1 contains "machine" 8 times
- Total documents N = 500
- "machine" appears in 25 documents

**Solution:**
```
TF = 1 + log₁₀(8) = 1 + 0.903 = 1.903
IDF = log₁₀(500/25) = log₁₀(20) = 1.301
TF-IDF = 1.903 × 1.301 = 2.476
```

### Question 4.2
**Calculate Cosine Similarity between vectors A = [2, 1, 3] and B = [1, 2, 2].**

**Solution:**
```
Dot Product = (2×1) + (1×2) + (3×2) = 2 + 2 + 6 = 10
||A|| = √(4 + 1 + 9) = √14 ≈ 3.742
||B|| = √(1 + 4 + 4) = √9 = 3

Cosine Similarity = 10 / (3.742 × 3) = 10/11.226 ≈ 0.891
```

### Question 4.3
**Given word embeddings, calculate document vector using centroid method:**
- "Natural" = [0.4, 0.6, 0.2]
- "Language" = [0.5, 0.3, 0.7]
- "Processing" = [0.3, 0.6, 0.4]

**Solution:**
```
Document vector = (v_Natural + v_Language + v_Processing) / 3
= ([0.4+0.5+0.3]/3, [0.6+0.3+0.6]/3, [0.2+0.7+0.4]/3)
= (1.2/3, 1.5/3, 1.3/3)
= [0.4, 0.5, 0.433]
```

### Question 4.4
**Calculate Euclidean distance between "cat" = [0.2, 0.8, 0.5] and "dog" = [0.3, 0.7, 0.6].**

**Solution:**
```
d = √[(0.3-0.2)² + (0.7-0.8)² + (0.6-0.5)²]
d = √[0.01 + 0.01 + 0.01]
d = √0.03 ≈ 0.173
```

---

## Q5. Module 4: Word Embedding (5 Marks)

### Question 5.1
**For Skip-gram with window size 2, identify training pairs for target word "learning" in:**
```
"I love learning NLP models"
```

**Solution:**
- Target word: "learning"
- Window: ±2 words
- Context words: "love", "NLP" (within window)
- Training pairs:
  - (learning, love)
  - (learning, NLP)
  - Also: (learning, I), (learning, models) if full window considered

### Question 5.2
**Perform one iteration of Skip-gram with Negative Sampling:**
- Target: "code", Context: "python" (Positive, y=1)
- v_code = [0.3, 0.4], u_python = [0.5, 0.6]
- Predicted P = σ(v·u) = 0.58
- Learning rate η = 0.1

**Solution:**
```
Step 1: Calculate error
EI = P - y = 0.58 - 1 = -0.42

Step 2: Calculate gradient w.r.t v_code
Gradient = EI × u_python = -0.42 × [0.5, 0.6] = [-0.21, -0.252]

Step 3: Update v_code
v_new = v_old - η × Gradient
v_new = [0.3, 0.4] - 0.1 × [-0.21, -0.252]
v_new = [0.3, 0.4] + [0.021, 0.0252]
v_new = [0.321, 0.4252]
```

### Question 5.3
**Using the parallelogram method, calculate vector for "Queen" given:**
- v_King = [0.8, 0.6, 0.2]
- v_Man = [0.5, 0.3, 0.1]
- v_Woman = [0.4, 0.7, 0.3]

**Solution:**
```
Analogy: Man is to King as Woman is to Queen
v_Queen ≈ v_King - v_Man + v_Woman
v_Queen = [0.8, 0.6, 0.2] - [0.5, 0.3, 0.1] + [0.4, 0.7, 0.3]
v_Queen = [0.8-0.5+0.4, 0.6-0.3+0.7, 0.2-0.1+0.3]
v_Queen = [0.7, 1.0, 0.4]
```

### Question 5.4
**Compare Skip-gram and CBOW on:**
1. Training speed
2. Performance on rare words

**Solution:**
1. **Training Speed**: CBOW is faster because it predicts one word from multiple context words, while Skip-gram predicts multiple context words from one target word
2. **Rare Words**: Skip-gram performs better on rare words because each rare word gets updated multiple times (once for each context word), while CBOW averages context vectors which dilutes rare word signals

### Question 5.5
**Given embedding matrix M, extract embedding for "India" (index 3):**
```
M = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9],
     [3, 5, 7]]  ← India (row 3)
```

**Solution:**
```
One-hot vector for "India" (index 3): [0, 0, 0, 1]

Embedding = One-hot × M^T
Or simply: Select row 3 of matrix M

Embedding for "India" = [3, 5, 7]
```

---

## Q6. Module 5: POS Tagging - HMM (4 Marks)

### Question 6.1
**Calculate transition probabilities from the following tagged corpus:**
```
Tagged: "The/DT cat/NN sat/VB on/IN the/DT mat/NN"
DT appears: 2 times (followed by NN: 2 times)
NN appears: 2 times (followed by VB: 1, followed by END: 1)
```

**Solution:**
```
P(NN|DT) = Count(DT→NN) / Count(DT) = 2/2 = 1.0
P(VB|NN) = Count(NN→VB) / Count(NN) = 1/2 = 0.5
P(END|NN) = Count(NN→END) / Count(NN) = 1/2 = 0.5
```

### Question 6.2
**Disambiguate "watch" in "I watch TV":**
- Previous tag: PRP (pronoun)
- Candidates: NN, VB
- P(NN|PRP) = 0.1, P(VB|PRP) = 0.6
- P("watch"|NN) = 0.02, P("watch"|VB) = 0.05

**Solution:**
```
Score(NN) = P(NN|PRP) × P("watch"|NN) = 0.1 × 0.02 = 0.002
Score(VB) = P(VB|PRP) × P("watch"|VB) = 0.6 × 0.05 = 0.030

Decision: 0.030 > 0.002, so "watch" is tagged as VB (Verb)
```

### Question 6.3
**In HMM-based POS tagging, identify:**
a) What are the hidden states?
b) What are the observations?
c) What algorithm finds the best tag sequence?

**Solution:**
a) **Hidden States**: The POS tags (Noun, Verb, Adjective, etc.)
b) **Observations**: The actual words in the sentence
c) **Algorithm**: Viterbi Algorithm

### Question 6.4
**Why is emission probability P("the"|VB) typically zero or very low?**

**Solution:**
The word "the" is a determiner and almost never appears as a verb in any corpus. The emission probability is based on how often a word appears with a particular tag. Since "the" → VB is essentially never observed in training data, P("the"|VB) ≈ 0. This acts as a "veto" in HMM tagging, preventing "the" from ever being tagged as a verb.

---

## Q7. Module 6: Viterbi & Advanced Models (5 Marks)

### Question 7.1
**Fill the Viterbi table for "She dances" given:**

Transition Probabilities:
|        | VB  | MD  | PRP |
|--------|-----|-----|-----|
| START  | 0.1 | 0.2 | 0.7 |
| PRP    | 0.4 | 0.4 | 0.9 |

Emission Probabilities:
|     | She | dances |
|-----|-----|--------|
| VB  | 0   | 0.3    |
| PRP | 1   | 0      |

**Solution:**

**Step 1: Initialization (Word = "She")**
```
V₁(VB) = P(VB|START) × P("She"|VB) = 0.1 × 0 = 0
V₁(PRP) = P(PRP|START) × P("She"|PRP) = 0.7 × 1 = 0.7
```

**Step 2: Recursion (Word = "dances")**
```
V₂(VB) = max[V₁(VB)×P(VB|VB), V₁(PRP)×P(VB|PRP)] × P("dances"|VB)
       = max[0×0.1, 0.7×0.4] × 0.3
       = 0.28 × 0.3 = 0.084

V₂(PRP) = max[...] × P("dances"|PRP) = ... × 0 = 0
```

**Final Table:**
|     | She | dances |
|-----|-----|--------|
| VB  | 0   | 0.084  |
| PRP | 0.7 | 0      |

**Best sequence: PRP → VB**

### Question 7.2
**Backtracking: Given these Viterbi values at final step:**
- State N: Value=0.008, Backpointer=V
- State V: Value=0.002, Backpointer=N

**What is the best tag sequence for a 2-word sentence?**

**Solution:**
```
1. Final step (t=2): Max value is 0.008 at State N
2. Tag₂ = N
3. Backpointer from N points to V
4. Tag₁ = V

Best sequence: V → N
```

### Question 7.3
**Why is MEMM more flexible than HMM for POS tagging?**

**Solution:**
1. **Discriminative vs Generative**: MEMM directly models P(Tag|Word) while HMM models P(Word|Tag). This allows MEMM to use overlapping features without violating independence assumptions.

2. **Rich Features**: MEMM can incorporate arbitrary features like:
   - Word suffixes ("-ing", "-ed")
   - Capitalization
   - Previous words (not just previous tags)
   - Surrounding context on both sides
   
3. **No Independence Assumption**: HMM requires independence between observations given states, MEMM doesn't have this constraint.

### Question 7.4
**In an RNN-based POS tagger:**
a) What is the dimension of the output layer at each timestep?
b) What activation function produces tag probabilities?

**Solution:**
a) **Dimension**: Size of the Tagset (number of unique POS tags, e.g., 45 for Penn Treebank)
b) **Activation**: Softmax (produces probability distribution summing to 1)

### Question 7.5
**Calculate Viterbi score using log probabilities:**
- log V₁(A) = -1.5
- log P(B|A) = -0.3 (transition)
- log P(word|B) = -1.0 (emission)

**Solution:**
```
In log space, multiplication becomes addition:
log V₂(B) = log V₁(A) + log P(B|A) + log P(word|B)
log V₂(B) = -1.5 + (-0.3) + (-1.0)
log V₂(B) = -2.8
```

---

# 📝 Additional Practice Questions

---

## Set A: Conceptual Questions

### A1. Explain the distributional hypothesis and its importance in NLP.

### A2. What is the "curse of dimensionality" in NLP and how do word embeddings address it?

### A3. Compare TF-IDF weighted sparse vectors with dense word embeddings.

### A4. Explain why log probabilities are used instead of raw probabilities in language models.

### A5. What is the Label Bias Problem in MEMMs?

---

## Set B: Calculation Problems

### B1. TF-IDF Calculation
Given corpus of N=1000 documents:
- Word "algorithm" appears 15 times in Document D5
- "algorithm" appears in 50 documents total
- Word "the" appears 100 times in D5
- "the" appears in all 1000 documents

Calculate TF-IDF for both words in D5.

**Solution:**
```
For "algorithm":
TF = 1 + log₁₀(15) = 1 + 1.176 = 2.176
IDF = log₁₀(1000/50) = log₁₀(20) = 1.301
TF-IDF = 2.176 × 1.301 = 2.831

For "the":
TF = 1 + log₁₀(100) = 1 + 2 = 3
IDF = log₁₀(1000/1000) = log₁₀(1) = 0
TF-IDF = 3 × 0 = 0
```

### B2. Cosine Similarity with Normalized Vectors
If vectors are already unit normalized (||A|| = ||B|| = 1):
A = [0.6, 0.8] and B = [0.8, 0.6]

Calculate cosine similarity.

**Solution:**
```
For unit vectors: cos(A,B) = A · B
= (0.6×0.8) + (0.8×0.6)
= 0.48 + 0.48
= 0.96
```

### B3. Perplexity Comparison
Model A: P(test sentence) = 0.001
Model B: P(test sentence) = 0.0001
Sentence length: 5 words

Which model is better?

**Solution:**
```
PP(A) = (0.001)^(-1/5) = (1000)^(1/5) = 3.98
PP(B) = (0.0001)^(-1/5) = (10000)^(1/5) = 6.31

Model A is better (lower perplexity)
```

### B4. Word2Vec Update with Negative Sample
- Target: "apple", Context: "fruit" (Positive, y=1)
- Negative sample: "car" (y=0)
- v_apple = [0.5, 0.5]
- u_fruit = [0.4, 0.8], u_car = [0.9, 0.1]
- σ(v·u_fruit) = 0.65, σ(v·u_car) = 0.55
- Learning rate = 0.1

Calculate updated v_apple.

**Solution:**
```
Error for positive: E_pos = 0.65 - 1 = -0.35
Error for negative: E_neg = 0.55 - 0 = 0.55

Gradient from positive: -0.35 × [0.4, 0.8] = [-0.14, -0.28]
Gradient from negative: 0.55 × [0.9, 0.1] = [0.495, 0.055]

Total gradient = [-0.14 + 0.495, -0.28 + 0.055] = [0.355, -0.225]

v_new = [0.5, 0.5] - 0.1 × [0.355, -0.225]
v_new = [0.5 - 0.0355, 0.5 + 0.0225]
v_new = [0.4645, 0.5225]
```

### B5. HMM Path Probability
Calculate probability of tagging "Time flies" as "NN VB":
- P(NN|START) = 0.3, P(VB|NN) = 0.4
- P("Time"|NN) = 0.01, P("flies"|VB) = 0.02

**Solution:**
```
P(NN VB, "Time flies") = P(NN|START) × P("Time"|NN) × P(VB|NN) × P("flies"|VB)
= 0.3 × 0.01 × 0.4 × 0.02
= 0.000024
```

---

## Set C: Application Questions

### C1. Design a simple POS tagging system using rules. What are its limitations?

### C2. How would you handle out-of-vocabulary words in:
a) N-gram language models
b) Word2Vec embeddings
c) HMM-based POS taggers

### C3. Explain how you would evaluate a POS tagger. What metrics would you use?

### C4. What preprocessing steps are needed before applying TF-IDF?

### C5. How does window size in Word2Vec affect the type of similarity captured?

**Solutions:**

**C5 Solution:**
- **Small window (2-3)**: Captures syntactic similarity - words with similar grammatical roles
  - Example: "Hogwarts" neighbors include other fictional schools like "Sunnydale"
- **Large window (5+)**: Captures semantic/topical similarity - words from same topic
  - Example: "Hogwarts" neighbors include Harry Potter terms like "Dumbledore", "Malfoy"

---

# 🔑 Key Formulas Quick Reference

| Concept | Formula |
|---------|---------|
| **Bigram Probability** | P(wₙ\|wₙ₋₁) = Count(wₙ₋₁,wₙ) / Count(wₙ₋₁) |
| **Perplexity** | PP = P(W)^(-1/N) |
| **Laplace Smoothing** | P = (C + 1) / (N + V) |
| **TF** | tf = 1 + log₁₀(count + 1) |
| **IDF** | idf = log₁₀(N / df) |
| **TF-IDF** | w = tf × idf |
| **Cosine Similarity** | cos(A,B) = (A·B) / (\|\|A\|\| × \|\|B\|\|) |
| **Euclidean Distance** | d = √Σ(aᵢ - bᵢ)² |
| **Word Analogy** | v_target = v_a* - v_a + v_b |
| **SGNS Gradient** | ∇v = (σ(v·u) - y) × u |
| **Viterbi Init** | V₁(j) = π(j) × B(j,o₁) |
| **Viterbi Recursion** | Vₜ(j) = max[Vₜ₋₁(i) × A(i,j)] × B(j,oₜ) |
| **HMM Score** | Score = Transition × Emission |

---

# 📚 Study Tips

1. **Focus on problem-solving**: 6 out of 7 questions are problem-based
2. **Practice calculations**: TF-IDF, Cosine Similarity, Perplexity, Word2Vec updates, Viterbi
3. **Understand the intuition**: Know WHY each formula works, not just HOW
4. **Time management**: 30 marks in 2 hours = ~4 minutes per mark
5. **Show your work**: Partial credit available for correct approach
6. **Memorize key formulas**: Use the quick reference above
7. **Practice backtracking**: In Viterbi, follow pointers correctly

---

---

# 🎥 Recommended YouTube Videos (Short & Focused)

## Module 1: Introduction to NLP
| Topic | Video | Channel | Duration |
|-------|-------|---------|----------|
| NLP Overview | [NLP Zero to Hero](https://www.youtube.com/watch?v=fNxaJsNG3-s) | TensorFlow | ~10 min |
| Levels of Language | [What is NLP?](https://www.youtube.com/watch?v=CMrHM8a3hqw) | IBM Technology | ~6 min |

## Module 2: N-gram Language Models
| Topic | Video | Channel | Duration |
|-------|-------|---------|----------|
| N-grams Basics | [What is N-Gram in NLP? Beginner to Pro](https://www.youtube.com/watch?v=89A4jGvaaKk) | DataMites | ~8 min |
| N-gram Language Model | [N-gram Language Modeling - Theory, Math, Code](https://www.youtube.com/watch?v=Saq1QagC8zU) | Weights & Biases | ~15 min |
| **Perplexity** | [Perplexity Explained: Why Lower is Better](https://www.youtube.com/watch?v=qb0MO1EVf00) | Serrano.Academy | ~12 min |
| Perplexity | [What is Perplexity?](https://www.youtube.com/watch?v=NURcDHhYe98) | Hugging Face | ~5 min |
| Smoothing | [Laplace Smoothing Explained](https://www.youtube.com/watch?v=gCI-ZC7irbY) | StatQuest | ~10 min |

## Module 3: Neural LM & LLMs
| Topic | Video | Channel | Duration |
|-------|-------|---------|----------|
| Neural Nets Basics | [Neural Networks Explained](https://www.youtube.com/watch?v=aircAruvnKk) | 3Blue1Brown | ~19 min |
| **LLMs Intro** | [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) | Andrej Karpathy | ~60 min |
| Prompt Engineering | [ChatGPT Prompt Engineering](https://www.youtube.com/watch?v=_ZvnD96BtNQ) | freeCodeCamp | ~17 min |
| Transfer Learning | [Transfer Learning Explained](https://www.youtube.com/watch?v=yofjFQddwHE) | DeepLizard | ~8 min |

## Module 4: Vector Semantics (TF-IDF, Cosine)
| Topic | Video | Channel | Duration |
|-------|-------|---------|----------|
| **TF-IDF** | [Calculate TF-IDF in NLP (Simple Example)](https://www.youtube.com/watch?v=vZAXpvHhQow) | Normalized Nerd | ~10 min |
| TF-IDF | [TF-IDF Explained Simply](https://www.youtube.com/watch?v=D2V1okCEsiE) | Code Academy | ~8 min |
| **Cosine Similarity** | [Cosine Similarity Explained](https://www.youtube.com/watch?v=e9U0QAFbfLI) | Bhavesh Bhatt | ~7 min |
| Embeddings Intro | [Word Embeddings Explained](https://www.youtube.com/watch?v=viZrOnJclY0) | Luis Serrano | ~12 min |

## Module 4: Word Embeddings (Word2Vec, GloVe)
| Topic | Video | Channel | Duration |
|-------|-------|---------|----------|
| **Word2Vec** ⭐ | [Word2Vec - A Simple Explanation](https://www.youtube.com/watch?v=UqRCEmrv1gQ) | codebasics | ~16 min |
| Skip-gram & CBOW | [Word2Vec Skip-gram & CBOW](https://www.youtube.com/watch?v=UqRCEmrv1gQ) | StatQuest | ~20 min |
| **Illustrated Word2Vec** ⭐ | [The Illustrated Word2vec](https://www.youtube.com/watch?v=ISPId9Lhc1g) | Jay Alammar | ~25 min |
| Negative Sampling | [Negative Sampling Explained](https://www.youtube.com/watch?v=CNp9-pN4hpU) | Rahul Patwari | ~15 min |
| GloVe | [GloVe Explained](https://www.youtube.com/watch?v=QoUYlxl1RGI) | Xander Steenbrugge | ~10 min |
| Word Analogies | [King - Man + Woman = Queen](https://www.youtube.com/watch?v=gQddtTdmG_8) | Computerphile | ~12 min |

## Module 5: POS Tagging & HMM
| Topic | Video | Channel | Duration |
|-------|-------|---------|----------|
| POS Tagging Intro | [POS Tagging Explained](https://www.youtube.com/watch?v=6xWdo5-dVvg) | Krish Naik | ~12 min |
| **HMM Explained** ⭐ | [Hidden Markov Models](https://www.youtube.com/watch?v=kqSzLo9fenk) | Luis Serrano | ~15 min |
| HMM for POS | [HMM POS Tagging](https://www.youtube.com/watch?v=IqXdjdOgXPM) | Coursera NLP | ~10 min |
| Transition/Emission | [HMM Probabilities Explained](https://www.youtube.com/watch?v=fv6KBEg9k-U) | ritvikmath | ~12 min |

## Module 6: Viterbi Algorithm & Advanced Models
| Topic | Video | Channel | Duration |
|-------|-------|---------|----------|
| **Viterbi Algorithm** ⭐ | [Viterbi Algorithm Step by Step](https://www.youtube.com/watch?v=6JVqutwtzmo) | Neso Academy | ~18 min |
| Viterbi for POS | [Viterbi Decoding in HMM](https://www.youtube.com/watch?v=IqXdjdOgXPM) | Stanford NLP | ~12 min |
| MEMM vs HMM | [Maximum Entropy Models](https://www.youtube.com/watch?v=8LXjFeH2g6o) | Michael Collins | ~15 min |
| Neural POS Tagging | [RNN for Sequence Labeling](https://www.youtube.com/watch?v=WCUNPb-5EYI) | DeepMind | ~20 min |

---

## 🎯 Top 10 Must-Watch Videos (Exam Focused)

1. ⭐ **[TF-IDF Calculation Example](https://www.youtube.com/watch?v=vZAXpvHhQow)** - Normalized Nerd (~10 min)
2. ⭐ **[Word2Vec Explained Simply](https://www.youtube.com/watch?v=UqRCEmrv1gQ)** - codebasics (~16 min)
3. ⭐ **[The Illustrated Word2vec](https://www.youtube.com/watch?v=ISPId9Lhc1g)** - Jay Alammar (~25 min)
4. ⭐ **[Cosine Similarity](https://www.youtube.com/watch?v=e9U0QAFbfLI)** - Short & clear (~7 min)
5. ⭐ **[N-gram Language Models](https://www.youtube.com/watch?v=Saq1QagC8zU)** - With code (~15 min)
6. ⭐ **[Perplexity Explained](https://www.youtube.com/watch?v=qb0MO1EVf00)** - Visual explanation (~12 min)
7. ⭐ **[HMM Intuition](https://www.youtube.com/watch?v=kqSzLo9fenk)** - Luis Serrano (~15 min)
8. ⭐ **[Viterbi Algorithm](https://www.youtube.com/watch?v=6JVqutwtzmo)** - Step-by-step (~18 min)
9. ⭐ **[Skip-gram with Negative Sampling](https://www.youtube.com/watch?v=CNp9-pN4hpU)** - Math explained (~15 min)
10. ⭐ **[Neural Language Models](https://www.youtube.com/watch?v=OHyygU1cU0w)** - Stanford NLP (~20 min)

---

## 📺 Playlists for Deep Dive

| Playlist | Channel | Videos |
|----------|---------|--------|
| [Stanford CS224N: NLP with Deep Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z) | Stanford | 20+ videos |
| [NLP Tutorial for Beginners](https://www.youtube.com/playlist?list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm) | Krish Naik | 40+ videos |
| [Natural Language Processing](https://www.youtube.com/playlist?list=PLQiyVNMpDLKnZYBTUOlSI9mi9wAErFtFm) | Deeplearning.AI | 10+ videos |
| [NLP with Python](https://www.youtube.com/playlist?list=PLBZBJbE_rGRU33W-vT6LKhRWl7IzVG-C2) | codebasics | 15+ videos |

---

**Good luck with your exam! 🎓**

*Last updated: January 2026*
