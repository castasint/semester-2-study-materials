# Session 6: Part-of-Speech Tagging
## AIMLCZG530 - Natural Language Processing

---

# 1. Introduction to POS Tagging

## 1.1 What is POS Tagging?

**Definition**: Assigning grammatical categories (parts of speech) to each word in a sentence.

**Example**:
```
The    quick   brown   fox    jumps   over   the    lazy    dog
DT     JJ      JJ      NN     VBZ     IN     DT     JJ      NN
```

## 1.2 Why POS Tagging Matters

| Application | How POS Helps |
|-------------|---------------|
| **Parsing** | Constrains possible parse trees |
| **NER** | "Washington" - person vs. place |
| **Word Sense** | "book" as noun vs. verb |
| **Text-to-Speech** | Pronunciation (REcord vs. reCORD) |
| **Sentiment** | Identify adjectives for opinion |

---

# 2. English Word Classes

## 2.1 Open vs Closed Classes

### Open Classes (Content Words)
New words added frequently

| Class | Examples | Function |
|-------|----------|----------|
| **Nouns** | dog, computer, happiness | Things, concepts |
| **Verbs** | run, think, compute | Actions, states |
| **Adjectives** | big, happy, neural | Modify nouns |
| **Adverbs** | quickly, very, well | Modify verbs/adj |

### Closed Classes (Function Words)
Fixed set, rarely changes

| Class | Examples | Function |
|-------|----------|----------|
| **Determiners** | the, a, this, each | Specify nouns |
| **Pronouns** | I, you, he, it, they | Replace nouns |
| **Prepositions** | in, on, at, under | Relations |
| **Conjunctions** | and, but, or, because | Connect clauses |
| **Particles** | up, off, out (in "give up") | With verbs |
| **Modals** | can, will, should, must | Possibility, necessity |

---

# 3. The Penn Treebank POS Tagset

## 3.1 Overview
- **45 tags** for English
- Most widely used tagset
- Distinguishes subtle differences

## 3.2 Common Tags

### Nouns
| Tag | Description | Example |
|-----|-------------|---------|
| NN | Noun, singular | cat, book |
| NNS | Noun, plural | cats, books |
| NNP | Proper noun, singular | John, London |
| NNPS | Proper noun, plural | Americans |

### Verbs
| Tag | Description | Example |
|-----|-------------|---------|
| VB | Verb, base form | run, eat |
| VBD | Verb, past tense | ran, ate |
| VBG | Verb, gerund/present participle | running, eating |
| VBN | Verb, past participle | run, eaten |
| VBP | Verb, non-3rd person singular present | run, eat |
| VBZ | Verb, 3rd person singular present | runs, eats |

### Adjectives & Adverbs
| Tag | Description | Example |
|-----|-------------|---------|
| JJ | Adjective | big, happy |
| JJR | Adjective, comparative | bigger, happier |
| JJS | Adjective, superlative | biggest, happiest |
| RB | Adverb | quickly, very |
| RBR | Adverb, comparative | faster |
| RBS | Adverb, superlative | fastest |

### Function Words
| Tag | Description | Example |
|-----|-------------|---------|
| DT | Determiner | the, a, this |
| IN | Preposition/subordinating conj | in, of, because |
| CC | Coordinating conjunction | and, but, or |
| PRP | Personal pronoun | I, you, he |
| PRP$ | Possessive pronoun | my, your, his |
| MD | Modal | can, will, should |
| TO | "to" | to (infinitive) |

### Other
| Tag | Description | Example |
|-----|-------------|---------|
| CD | Cardinal number | one, 2, 100 |
| WH words | WDT, WP, WRB | which, who, where |
| . | Sentence-final punctuation | . ! ? |
| , | Comma | , |

---

# 4. Challenges in POS Tagging

## 4.1 Ambiguity

Many words have multiple possible tags:

| Word | Possible Tags | Context Example |
|------|---------------|-----------------|
| **book** | NN, VB | "read the book" vs. "book a flight" |
| **back** | RB, NN, VB, JJ | "go back", "hurt my back", "back the car" |
| **that** | DT, IN, WDT | "that book", "said that", "the book that" |
| **can** | MD, NN, VB | "can run", "a can", "can vegetables" |

## 4.2 Statistics on Ambiguity

- ~40% of word tokens are ambiguous
- ~14% of word types are ambiguous
- Most frequent words are most ambiguous

## 4.3 Context Dependence

**Same word, different tags**:
```
Time flies like an arrow.
   NN  VBZ  IN  DT  NN

Fruit flies like a banana.
   NN   NNS  VBP DT   NN
```

---

# 5. Approaches to POS Tagging

## 5.1 Rule-based

- Hand-written rules
- Example: "word ending in -ing before noun → JJ"
- Pros: Interpretable
- Cons: Hard to maintain, incomplete

## 5.2 Statistical

| Method | Description |
|--------|-------------|
| **HMM** | Hidden Markov Model |
| **CRF** | Conditional Random Fields |
| **MEMM** | Maximum Entropy Markov Model |

## 5.3 Neural

| Method | Description |
|--------|-------------|
| **RNN/LSTM** | Sequence models |
| **Bi-LSTM-CRF** | Bidirectional + CRF |
| **Transformers** | BERT-based taggers |

---

# 6. Markov Chains

## 6.1 Definition

A **Markov Chain** is a model for sequences where:
- Current state depends only on previous state
- Transition probabilities are fixed

## 6.2 Components

| Component | Description |
|-----------|-------------|
| **States (Q)** | Set of possible states |
| **Transitions (A)** | P(state_i → state_j) |
| **Initial (π)** | P(starting in state_i) |

## 6.3 First-order Markov Assumption

```
P(qᵢ | q₁, q₂, ..., qᵢ₋₁) = P(qᵢ | qᵢ₋₁)
```

## 6.4 Example: Weather

States: {Sunny, Rainy}

Transition matrix:
|  | Sunny | Rainy |
|--|-------|-------|
| **Sunny** | 0.8 | 0.2 |
| **Rainy** | 0.4 | 0.6 |

P(Sunny → Sunny) = 0.8
P(Sunny → Rainy) = 0.2

---

# 7. Hidden Markov Model (HMM)

## 7.1 Why "Hidden"?

- We observe **words** (outputs)
- Tags are **hidden** (we want to find them)

## 7.2 HMM Components

| Component | Symbol | Description |
|-----------|--------|-------------|
| **States** | Q | POS tags {NN, VB, DT, ...} |
| **Observations** | O | Words in vocabulary |
| **Transition probs** | A | P(tagᵢ \| tagᵢ₋₁) |
| **Emission probs** | B | P(word \| tag) |
| **Initial probs** | π | P(tag \| start) |

## 7.3 Visual Representation

```
Hidden States (Tags):   DT ----→ NN ----→ VB ----→ IN ----→ NN
                         ↓        ↓        ↓        ↓        ↓
Observations (Words):  The      cat      sat      on       mat
```

## 7.4 Calculating Probabilities

### Transition Probability
```
P(tagᵢ | tagᵢ₋₁) = Count(tagᵢ₋₁ → tagᵢ) / Count(tagᵢ₋₁)
```

**Example**:
```
Corpus: DT NN VB, DT NN VB NN, DT JJ NN

Count(DT) = 3
Count(DT → NN) = 2
Count(DT → JJ) = 1

P(NN | DT) = 2/3 = 0.667
P(JJ | DT) = 1/3 = 0.333
```

### Emission Probability
```
P(word | tag) = Count(tag, word) / Count(tag)
```

**Example**:
```
Count(NN) = 100
Count(NN, "cat") = 5
Count(NN, "dog") = 8

P("cat" | NN) = 5/100 = 0.05
P("dog" | NN) = 8/100 = 0.08
```

---

# 8. HMM for POS Tagging

## 8.1 The Tagging Problem

**Given**: Sequence of words W = w₁, w₂, ..., wₙ
**Find**: Best sequence of tags T = t₁, t₂, ..., tₙ

**Objective**:
```
T* = argmax P(T | W)
       T
```

## 8.2 Using Bayes' Rule

```
P(T | W) = P(W | T) × P(T) / P(W)
```

Since P(W) is constant for all T:
```
T* = argmax P(W | T) × P(T)
       T
```

## 8.3 HMM Decomposition

**Likelihood** (emission):
```
P(W | T) = ∏ᵢ P(wᵢ | tᵢ)
```

**Prior** (transition):
```
P(T) = ∏ᵢ P(tᵢ | tᵢ₋₁)
```

**Combined**:
```
T* = argmax ∏ᵢ P(wᵢ | tᵢ) × P(tᵢ | tᵢ₋₁)
       T
```

## 8.4 HMM Disambiguation Example

**Sentence**: "Time flies"
**Previous tag for "Time"**: NN

**Candidate tags for "flies"**: NN, VBZ

**Given**:
```
P(NN | NN) = 0.2
P(VBZ | NN) = 0.5
P("flies" | NN) = 0.02
P("flies" | VBZ) = 0.03
```

**Calculations**:
```
Score(NN) = P(NN|NN) × P("flies"|NN) = 0.2 × 0.02 = 0.004
Score(VBZ) = P(VBZ|NN) × P("flies"|VBZ) = 0.5 × 0.03 = 0.015
```

**Result**: VBZ wins (0.015 > 0.004)

---

# 9. Key Formulas

| Concept | Formula |
|---------|---------|
| Transition | P(tᵢ\|tᵢ₋₁) = C(tᵢ₋₁,tᵢ) / C(tᵢ₋₁) |
| Emission | P(w\|t) = C(t,w) / C(t) |
| HMM Score | P(w\|t) × P(t\|prev_t) |
| Best Tags | argmax ∏ᵢ P(wᵢ\|tᵢ) × P(tᵢ\|tᵢ₋₁) |

---

# 📝 Practice Questions

## Q1. Tag these sentences using Penn Treebank tags:
a) "The dog runs quickly."
b) "She can book a flight."

## Q2. Calculate transition probabilities from:
```
DT NN VB IN DT NN
DT JJ NN VBZ RB
```

## Q3. HMM Disambiguation
Previous tag: VB
Word: "book"
Candidates: NN, VB
P(NN|VB)=0.4, P(VB|VB)=0.1
P("book"|NN)=0.05, P("book"|VB)=0.02
Which tag wins?

## Q4. Why is "the" almost always tagged as DT?

## Q5. Explain the difference between open and closed word classes.

---

*Reference: Session 6 - Part-of-Speech Tagging*
