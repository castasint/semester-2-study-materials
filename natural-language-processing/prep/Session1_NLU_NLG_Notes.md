# Session 1: Natural Language Understanding and Generation
## AIMLCZG530 - Natural Language Processing

---

# 1. The Study of Language

## 1.1 What is Natural Language?
- **Natural Language** refers to languages that have evolved naturally among humans (e.g., English, Hindi, Spanish)
- Contrasts with **Formal Languages** (programming languages, mathematical notation)
- Characterized by ambiguity, irregularity, and context-dependence

## 1.2 Key Challenges in Language Processing
| Challenge | Description |
|-----------|-------------|
| **Ambiguity** | Words/sentences can have multiple meanings |
| **Variability** | Same meaning expressed in many ways |
| **Context Dependence** | Meaning depends on surrounding context |
| **Idioms & Metaphors** | Non-literal expressions |
| **World Knowledge** | Understanding requires background knowledge |

---

# 2. Applications of Natural Language Understanding

## 2.1 Core NLP Applications

### Information Extraction
- **Named Entity Recognition (NER)**: Identifying names, places, organizations
- **Relation Extraction**: Finding relationships between entities
- **Event Extraction**: Identifying events and their participants

### Text Classification
- **Sentiment Analysis**: Positive/Negative/Neutral classification
- **Spam Detection**: Email/message filtering
- **Topic Classification**: Categorizing documents

### Machine Translation
- Translating text between languages
- Statistical MT → Neural MT → Transformer-based MT
- Examples: Google Translate, DeepL

### Question Answering
- **Factoid QA**: Who, What, When, Where questions
- **Reading Comprehension**: Answer from given passage
- **Open-domain QA**: Answer from knowledge base

### Dialogue Systems
- **Task-oriented**: Booking, customer service
- **Chatbots**: Conversational agents
- **Virtual Assistants**: Siri, Alexa, Google Assistant

### Text Generation
- **Summarization**: Condensing long documents
- **Language Modeling**: Predicting next word
- **Creative Writing**: Story generation

## 2.2 Industry Applications

| Domain | Application |
|--------|-------------|
| **Healthcare** | Clinical note analysis, drug interaction |
| **Finance** | Fraud detection, sentiment from news |
| **Legal** | Contract analysis, case research |
| **E-commerce** | Review analysis, product recommendations |
| **Social Media** | Trend detection, content moderation |

---

# 3. Evaluating Language Understanding Systems

## 3.1 Classification Metrics

### Confusion Matrix
```
                    Predicted
                  Positive  Negative
Actual  Positive    TP        FN
        Negative    FP        TN
```

### Core Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall correctness |
| **Precision** | TP / (TP + FP) | Of predicted positives, how many correct? |
| **Recall** | TP / (TP + FN) | Of actual positives, how many found? |
| **F1 Score** | 2 × (P × R) / (P + R) | Harmonic mean of P and R |

### When to Use What?
- **High Precision needed**: Spam detection (avoid false positives)
- **High Recall needed**: Disease diagnosis (avoid missing cases)
- **Balanced**: Use F1 Score

## 3.2 Sequence Labeling Metrics

### For POS Tagging / NER
- **Token-level Accuracy**: % of tokens correctly labeled
- **Entity-level F1**: For NER, exact entity match

## 3.3 Generation Metrics

### BLEU Score (Machine Translation)
- Compares n-gram overlap with reference
- Range: 0 to 1 (higher is better)

### Perplexity (Language Models)
- Measures how well model predicts text
- Lower is better
- Formula: `PP = P(text)^(-1/N)`

## 3.4 Human Evaluation
- **Fluency**: Is the output grammatically correct?
- **Adequacy**: Does it convey the correct meaning?
- **Coherence**: Is it logically consistent?

---

# 4. The Different Levels of Language Analysis

## 4.1 Overview of Linguistic Levels

```
┌─────────────────────────────────────┐
│         PRAGMATICS                  │  ← Context, speaker intent
├─────────────────────────────────────┤
│         DISCOURSE                   │  ← Multi-sentence relationships
├─────────────────────────────────────┤
│         SEMANTICS                   │  ← Meaning of words/sentences
├─────────────────────────────────────┤
│         SYNTAX                      │  ← Sentence structure
├─────────────────────────────────────┤
│         LEXICAL                     │  ← Word categories (POS)
├─────────────────────────────────────┤
│         MORPHOLOGY                  │  ← Word internal structure
└─────────────────────────────────────┘
```

## 4.2 Detailed Analysis of Each Level

### Level 1: Morphological Analysis
**Definition**: Study of word formation and internal structure

| Concept | Example |
|---------|---------|
| **Root/Stem** | "run" in "running" |
| **Prefix** | "un" in "unhappy" |
| **Suffix** | "ing" in "running" |
| **Inflection** | run → runs, ran, running |
| **Derivation** | happy → unhappy → unhappiness |

**NLP Tasks**: Stemming, Lemmatization

### Level 2: Lexical Analysis
**Definition**: Identifying word categories and properties

| Category | Examples | Function |
|----------|----------|----------|
| **Noun (N)** | cat, happiness | Entity/thing |
| **Verb (V)** | run, think | Action/state |
| **Adjective (Adj)** | happy, big | Modifies noun |
| **Adverb (Adv)** | quickly, very | Modifies verb/adj |
| **Determiner (Det)** | the, a, this | Specifies noun |
| **Preposition (Prep)** | in, on, at | Relates elements |

**NLP Tasks**: POS Tagging, Word Sense Disambiguation

### Level 3: Syntactic Analysis
**Definition**: Study of sentence structure and grammar

**Key Concepts**:
- **Phrase Structure**: NP, VP, PP
- **Parse Trees**: Hierarchical sentence structure
- **Grammatical Relations**: Subject, Object, Modifier

**Example**:
```
Sentence: "The cat sat on the mat"

        S
       / \
      NP   VP
     /    /  \
   Det N  V   PP
   |   |  |  /  \
  The cat sat P  NP
              |  /  \
             on Det  N
                |    |
               the  mat
```

**NLP Tasks**: Parsing (Constituency, Dependency)

### Level 4: Semantic Analysis
**Definition**: Study of meaning

| Concept | Description | Example |
|---------|-------------|---------|
| **Word Meaning** | Dictionary definition | "bank" = financial institution |
| **Compositional** | Meaning from parts | "red car" = car that is red |
| **Polysemy** | Multiple related meanings | "run" = jog, operate, flow |
| **Synonymy** | Same meaning | happy ≈ joyful |
| **Antonymy** | Opposite meaning | hot ↔ cold |

**NLP Tasks**: Word Sense Disambiguation, Semantic Role Labeling

### Level 5: Discourse Analysis
**Definition**: Analysis beyond single sentences

| Concept | Description | Example |
|---------|-------------|---------|
| **Coreference** | Same entity references | "John... He..." |
| **Coherence** | Logical flow | Cause-effect relations |
| **Discourse Relations** | How sentences connect | Contrast, elaboration |

**NLP Tasks**: Coreference Resolution, Discourse Parsing

### Level 6: Pragmatic Analysis
**Definition**: Meaning in context, speaker intent

| Concept | Description | Example |
|---------|-------------|---------|
| **Speech Acts** | What is accomplished | "Can you pass the salt?" = Request |
| **Implicature** | Implied meaning | "It's cold" = Close the window |
| **Presupposition** | Assumed information | "John's wife" assumes John is married |

**NLP Tasks**: Intent Detection, Dialogue Act Classification

---

# 5. Types of Ambiguity

## 5.1 Lexical Ambiguity

### Homonymy
- **Same spelling/sound, different meanings**
- Examples:
  - "bank" (river bank vs. financial bank)
  - "bat" (animal vs. sports equipment)

### Polysemy
- **Related meanings of same word**
- Examples:
  - "head" (body part, leader, top)
  - "run" (jog, operate, flow)

## 5.2 Structural (Syntactic) Ambiguity

**Multiple parse trees possible**

Example 1: "I saw the man with the telescope"
- Interpretation A: I used telescope to see the man
- Interpretation B: The man had a telescope

Example 2: "Flying planes can be dangerous"
- Interpretation A: The act of flying planes is dangerous
- Interpretation B: Planes that are flying are dangerous

## 5.3 Semantic Ambiguity

**Scope ambiguity**

Example: "Every student read a book"
- Interpretation A: Each student read their own book (different books)
- Interpretation B: All students read the same book

## 5.4 Pragmatic Ambiguity

**Context-dependent interpretation**

Example: "Can you open the door?"
- Literal: Asking about ability
- Pragmatic: Request to open the door

---

# 6. The Organization of Natural Language Understanding Systems

## 6.1 Traditional NLP Pipeline

```
Raw Text
    ↓
┌──────────────────┐
│  Tokenization    │  → Split into words/tokens
└──────────────────┘
    ↓
┌──────────────────┐
│  Normalization   │  → Lowercase, remove punctuation
└──────────────────┘
    ↓
┌──────────────────┐
│  POS Tagging     │  → Assign word categories
└──────────────────┘
    ↓
┌──────────────────┐
│  Parsing         │  → Build syntax tree
└──────────────────┘
    ↓
┌──────────────────┐
│  NER             │  → Identify named entities
└──────────────────┘
    ↓
┌──────────────────┐
│  Semantic        │  → Extract meaning
│  Analysis        │
└──────────────────┘
    ↓
Application-Specific Processing
```

## 6.2 Modern End-to-End Systems

### Neural Approach
```
Raw Text → Embeddings → Neural Network → Output
```

- **Pre-trained Models**: BERT, GPT, RoBERTa
- **Fine-tuning**: Task-specific adaptation
- **End-to-end**: Single model for multiple tasks

## 6.3 Key Components

### Preprocessing
| Step | Purpose | Example |
|------|---------|---------|
| Tokenization | Split text | "I'm happy" → ["I", "'m", "happy"] |
| Lowercasing | Normalize case | "The" → "the" |
| Stopword Removal | Remove common words | Remove "the", "is", "a" |
| Stemming | Reduce to root | "running" → "run" |
| Lemmatization | Dictionary form | "better" → "good" |

### Feature Extraction
- **Bag of Words**: Word frequency vectors
- **TF-IDF**: Weighted word importance
- **Word Embeddings**: Dense vector representations
- **Contextualized Embeddings**: BERT, ELMo

---

# 7. NLP Tools and Libraries

## 7.1 Open Source Tools

| Tool | Language | Strengths |
|------|----------|-----------|
| **NLTK** | Python | Educational, comprehensive |
| **spaCy** | Python | Fast, production-ready |
| **Hugging Face** | Python | Pre-trained transformers |
| **Gensim** | Python | Topic modeling, Word2Vec |
| **Stanford CoreNLP** | Java | Full pipeline |
| **LangChain** | Python | LLM applications |

## 7.2 Commercial APIs

| Service | Provider | Features |
|---------|----------|----------|
| **Google Cloud NLP** | Google | Entity, sentiment, syntax |
| **Amazon Comprehend** | AWS | Entity, keyphrase, topics |
| **Azure Text Analytics** | Microsoft | Sentiment, language detection |
| **OpenAI API** | OpenAI | GPT-4, embeddings |

---

# 8. Key Takeaways

1. **NLP spans multiple linguistic levels** from morphology to pragmatics
2. **Ambiguity** is the central challenge in NLP
3. **Evaluation metrics** depend on the task type
4. **Modern systems** use end-to-end neural approaches
5. **Pre-trained models** have revolutionized NLP

---

# 📝 Practice Questions

## Q1. List and explain the six levels of language analysis.

## Q2. Identify the type of ambiguity:
a) "The chicken is ready to eat"
b) "I saw her duck"
c) "Time flies like an arrow"

## Q3. Calculate Precision, Recall, and F1 for:
- TP = 40, FP = 10, FN = 5, TN = 45

## Q4. Compare the traditional NLP pipeline with modern end-to-end approaches.

## Q5. List 5 applications of NLP and their key challenges.

---

*Reference: Session 1 - Natural Language Understanding and Generation*
