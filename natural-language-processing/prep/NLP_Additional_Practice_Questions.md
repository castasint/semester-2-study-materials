# NLP Additional Practice Questions Set
## AIMLCZG530 - Natural Language Processing

---

> 💡 **Visual References:** See the concept diagrams in `images/` folder for visual explanations of TF-IDF, Cosine Similarity, Word2Vec, HMM, and Viterbi algorithms.

# Module 1: Introduction to NLP (4 Marks Questions)

## Question 1.1: NLP Applications
**Question:**
List and briefly describe any FOUR applications of Natural Language Processing in real-world systems.

**Solution:**
1. **Machine Translation**: Automatically translating text from one language to another (e.g., Google Translate converting English to Hindi)
2. **Sentiment Analysis**: Determining the emotional tone or opinion expressed in text (e.g., analyzing product reviews as positive or negative)
3. **Question Answering Systems**: Systems that answer questions posed in natural language by extracting information from documents (e.g., IBM Watson)
4. **Chatbots/Virtual Assistants**: Interactive AI systems that understand and respond to user queries in natural language (e.g., Siri, Alexa)

---

## Question 1.2: Ambiguity Types
**Question:**
Identify the type of ambiguity in each of the following sentences:
1. "Flying planes can be dangerous"
2. "I saw her duck"
3. "The bank is on the river side"

**Solution:**
1. **Structural Ambiguity**: Two interpretations possible:
   - Flying planes (the act of flying planes) can be dangerous
   - Flying planes (planes that are flying) can be dangerous
   
2. **Lexical + Structural Ambiguity**: 
   - "duck" as noun (the bird) - I saw her pet duck
   - "duck" as verb - I saw her duck (lower her head)
   
3. **Lexical Ambiguity (Homonymy)**: "bank" can mean:
   - Financial institution near the river
   - The river bank itself

---

## Question 1.3: Levels of Language Analysis
**Question:**
Explain with examples how the following sentence fails at different levels of language analysis:
- "Colorless green ideas sleep furiously"

**Solution:**
- **Morphological Level**: ✓ Valid (all words properly formed)
- **Lexical Level**: ✓ Valid (all words exist in dictionary)
- **Syntactic Level**: ✓ Valid (follows grammar: Adj Adj Noun Verb Adverb)
- **Semantic Level**: ✗ Invalid 
  - "Colorless" and "green" contradict each other
  - "Ideas" cannot "sleep"
  - Abstract concepts cannot be "colorless" or "green"
- **Pragmatic Level**: ✗ Invalid (no meaningful use in any context)

This is Noam Chomsky's famous example demonstrating that grammatical correctness doesn't imply meaningfulness.

---

## Question 1.4: Evaluation Metrics
**Question:**
A POS tagger is evaluated on a test set with the following results:
- True Positives (correctly tagged "Noun"): 80
- False Positives (incorrectly tagged as "Noun"): 20
- False Negatives (missed "Noun" tags): 15
- True Negatives: 185

Calculate Precision, Recall, and F1-Score for the "Noun" class.

**Solution:**
```
Precision = TP / (TP + FP) = 80 / (80 + 20) = 80/100 = 0.80 or 80%

Recall = TP / (TP + FN) = 80 / (80 + 15) = 80/95 ≈ 0.842 or 84.2%

F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
         = 2 × (0.80 × 0.842) / (0.80 + 0.842)
         = 2 × 0.6736 / 1.642
         = 1.3472 / 1.642
         ≈ 0.821 or 82.1%
```

---

# Module 2: N-gram Language Models (4 Marks Questions)

## Question 2.1: Bigram Probability Calculation
**Question:**
Given the following corpus:
```
"I love NLP. I love machine learning. NLP is interesting."
```
Calculate:
1. P(love | I)
2. P(NLP | love)
3. P(is | NLP)

**Solution:**
First, tokenize and count:
- "I" appears: 2 times (followed by "love": 2 times)
- "love" appears: 2 times (followed by "NLP": 1, "machine": 1)
- "NLP" appears: 2 times (followed by ".": 1, "is": 1)

```
1. P(love | I) = Count(I, love) / Count(I) = 2/2 = 1.0

2. P(NLP | love) = Count(love, NLP) / Count(love) = 1/2 = 0.5

3. P(is | NLP) = Count(NLP, is) / Count(NLP) = 1/2 = 0.5
```

---

## Question 2.2: Perplexity Calculation
**Question:**
A bigram language model gives the following probabilities for the sentence "AI is powerful":
- P(AI | START) = 0.1
- P(is | AI) = 0.4
- P(powerful | is) = 0.05

Calculate the perplexity of this sentence.

**Solution:**
```
Step 1: Calculate sentence probability
P(sentence) = P(AI|START) × P(is|AI) × P(powerful|is)
P(sentence) = 0.1 × 0.4 × 0.05 = 0.002

Step 2: Count words
N = 3 words

Step 3: Calculate perplexity
PP = P(sentence)^(-1/N)
PP = (0.002)^(-1/3)
PP = (1/0.002)^(1/3)
PP = (500)^(1/3)
PP ≈ 7.94

Interpretation: Model is as confused as choosing from ~8 equally likely words.
```

---

## Question 2.3: Laplace Smoothing
**Question:**
Apply Laplace (Add-1) smoothing to calculate P(transformer | language) given:
- Count(language, transformer) = 0
- Count(language) = 200
- Vocabulary size V = 5000

Compare with the MLE estimate.

**Solution:**
```
MLE Estimate (No Smoothing):
P_MLE(transformer | language) = 0/200 = 0

Laplace Smoothed Estimate:
P_Laplace(transformer | language) = (Count + 1) / (Total + V)
                                  = (0 + 1) / (200 + 5000)
                                  = 1 / 5200
                                  ≈ 0.000192

Comparison:
- MLE gives 0, which causes zero probability for any sentence containing this bigram
- Laplace gives small but non-zero probability (0.019%), solving the zero problem
- Note: Laplace "steals" too much probability from frequent n-grams
```

---

## Question 2.4: Linear Interpolation
**Question:**
Using linear interpolation with weights λ₁ = 0.5, λ₂ = 0.3, λ₃ = 0.2, calculate the interpolated probability P(learning | machine, deep) given:
- P(learning | machine, deep) = 0.02 (trigram - unseen, use backoff value)
- P(learning | deep) = 0.08 (bigram)
- P(learning) = 0.001 (unigram)

**Solution:**
```
P_interpolated = λ₁ × P_trigram + λ₂ × P_bigram + λ₃ × P_unigram
              = 0.5 × 0.02 + 0.3 × 0.08 + 0.2 × 0.001
              = 0.01 + 0.024 + 0.0002
              = 0.0342

The interpolated probability (0.0342) is higher than the trigram alone (0.02)
because it incorporates evidence from lower-order n-grams.
```

---

## Question 2.5: Stupid Backoff
**Question:**
Calculate S(happy | am, I) using Stupid Backoff with α = 0.4 given:
- Count(I, am, happy) = 0
- Count(am, happy) = 5
- Count(am) = 100

**Solution:**
```
Since Count(I, am, happy) = 0, we back off to bigram:

S(happy | am, I) = 0.4 × S(happy | am)

S(happy | am) = Count(am, happy) / Count(am) = 5/100 = 0.05

S(happy | am, I) = 0.4 × 0.05 = 0.02

Note: Stupid Backoff doesn't produce true probabilities (doesn't sum to 1),
but is fast and effective for web-scale data.
```

---

# Module 3: Neural Language Models & LLMs (4 Marks Questions)

## Question 3.1: N-gram vs Neural LM Comparison
**Question:**
Compare N-gram Language Models and Neural Language Models on the following parameters:
1. Ability to handle similar words
2. Memory requirements
3. Training complexity
4. Handling of long-range dependencies

**Solution:**
| Parameter | N-gram LM | Neural LM |
|-----------|-----------|-----------|
| Similar Words | Cannot generalize (treats "dog" and "cat" as completely different) | Uses embeddings to recognize similar words share properties |
| Memory | Stores all n-gram counts (can be very large for higher-order n-grams) | Compact parameter matrix (embeddings + weights) |
| Training | Simple counting, fast | Iterative gradient descent, GPU-intensive |
| Long-range | Limited to n-1 words of context | Can capture longer dependencies (especially RNN/Transformer) |

---

## Question 3.2: Pre-training and Fine-tuning
**Question:**
Explain the concepts of Pre-training and Fine-tuning in Large Language Models with an example use case.

**Solution:**
**Pre-training:**
- Training on massive unlabeled text data (billions of words)
- Task: Predict next word (autoregressive) or masked word (BERT-style)
- Model learns: Grammar, syntax, world knowledge, reasoning patterns
- Example: GPT-3 pre-trained on internet text including books, Wikipedia, websites

**Fine-tuning:**
- Taking pre-trained model and training further on task-specific data
- Uses smaller, labeled dataset
- Adapts general knowledge to specific domain/task

**Example Use Case: Medical Diagnosis Chatbot**
1. Pre-training: Model learns general English from web text
2. Fine-tuning: Train on medical texts, doctor-patient conversations
3. Result: Model understands medical terminology and can assist with diagnosis

---

## Question 3.3: Prompting Techniques
**Question:**
Explain Zero-shot, One-shot, and Few-shot prompting with examples for a sentiment classification task.

**Solution:**
**Task:** Classify movie reviews as Positive or Negative

**Zero-shot Prompting:**
```
Classify the sentiment of this review as Positive or Negative:
Review: "The movie was absolutely brilliant!"
Sentiment:
```
(No examples provided, model uses pre-trained knowledge)

**One-shot Prompting:**
```
Classify the sentiment:
Review: "I hated this film" → Negative

Review: "The movie was absolutely brilliant!"
Sentiment:
```
(One example provided)

**Few-shot Prompting:**
```
Classify the sentiment:
Review: "I hated this film" → Negative
Review: "Best movie ever!" → Positive  
Review: "Waste of time" → Negative

Review: "The movie was absolutely brilliant!"
Sentiment:
```
(Multiple examples provided)

**Key Insight:** More examples generally improve accuracy, especially for complex tasks.

---

## Question 3.4: Transfer Learning Benefits
**Question:**
List and explain THREE advantages of using Transfer Learning in NLP applications.

**Solution:**
1. **Reduced Data Requirements:**
   - Pre-trained models already understand language structure
   - Fine-tuning requires only 100s-1000s of labeled examples instead of millions
   - Critical for low-resource languages or specialized domains

2. **Faster Training Time:**
   - Most learning already done during pre-training
   - Fine-tuning takes hours/days instead of weeks/months
   - Significant cost and energy savings

3. **Better Generalization:**
   - Pre-trained models capture broad linguistic patterns
   - Less prone to overfitting on small datasets
   - Knowledge transfers across related tasks (e.g., NER helps POS tagging)

---

# Module 4: Vector Semantics (4 Marks Questions)

## Question 4.1: TF-IDF Calculation
**Question:**
Calculate TF-IDF for the words "deep" and "and" in Document D3 given:
- Corpus size: N = 200 documents
- "deep" appears 6 times in D3, in 20 documents total
- "and" appears 50 times in D3, in all 200 documents

**Solution:**
```
For "deep":
TF = 1 + log₁₀(6) = 1 + 0.778 = 1.778
IDF = log₁₀(200/20) = log₁₀(10) = 1.0
TF-IDF = 1.778 × 1.0 = 1.778

For "and":
TF = 1 + log₁₀(50) = 1 + 1.699 = 2.699
IDF = log₁₀(200/200) = log₁₀(1) = 0
TF-IDF = 2.699 × 0 = 0

Interpretation: Despite "and" appearing more frequently in D3,
its TF-IDF is 0 because it provides no discriminative information
(appears in all documents). "deep" has meaningful TF-IDF weight
as it distinguishes D3 from other documents.
```

---

## Question 4.2: Cosine Similarity
**Question:**
Calculate the Cosine Similarity between:
- Document A vector: [3, 4, 0, 2]
- Document B vector: [1, 2, 3, 0]

Interpret the result.

**Solution:**
```
Step 1: Dot Product
A · B = (3×1) + (4×2) + (0×3) + (2×0) = 3 + 8 + 0 + 0 = 11

Step 2: Magnitudes
||A|| = √(9 + 16 + 0 + 4) = √29 ≈ 5.385
||B|| = √(1 + 4 + 9 + 0) = √14 ≈ 3.742

Step 3: Cosine Similarity
cos(A,B) = 11 / (5.385 × 3.742) = 11 / 20.15 ≈ 0.546

Interpretation:
- Similarity of 0.546 indicates moderate similarity
- Documents share some common terms but are not highly similar
- Range is [0,1] for non-negative vectors where 1 = identical
```

---

## Question 4.3: Document Embedding
**Question:**
Create a document embedding using the centroid method for the sentence "Machine learning is fun" given:
- "Machine" = [0.8, 0.2, 0.5]
- "learning" = [0.7, 0.4, 0.6]
- "is" = [0.1, 0.1, 0.1]
- "fun" = [0.3, 0.9, 0.4]

**Solution:**
```
Centroid Method: Average all word vectors

Document_vector = (v_Machine + v_learning + v_is + v_fun) / 4

Dimension 1: (0.8 + 0.7 + 0.1 + 0.3) / 4 = 1.9 / 4 = 0.475
Dimension 2: (0.2 + 0.4 + 0.1 + 0.9) / 4 = 1.6 / 4 = 0.400
Dimension 3: (0.5 + 0.6 + 0.1 + 0.4) / 4 = 1.6 / 4 = 0.400

Document Embedding = [0.475, 0.400, 0.400]

Alternative: Sum without averaging = [1.9, 1.6, 1.6]
(Both are valid approaches)
```

---

## Question 4.4: Euclidean vs Cosine Distance
**Question:**
Given two word vectors:
- A = [1, 0]
- B = [3, 0]
- C = [0.5, 0.5]

1. Calculate Euclidean distance between A and B
2. Calculate Cosine similarity between A and B
3. Which is more similar to A: B or C (using Cosine)?

**Solution:**
```
1. Euclidean Distance (A, B):
d = √[(3-1)² + (0-0)²] = √4 = 2

2. Cosine Similarity (A, B):
cos = (1×3 + 0×0) / (√1 × √9) = 3 / 3 = 1.0
(Perfect similarity - same direction!)

3. Cosine Similarity (A, C):
cos = (1×0.5 + 0×0.5) / (√1 × √0.5) = 0.5 / (1 × 0.707) = 0.707

Comparison:
- cos(A,B) = 1.0 → A and B point in exact same direction
- cos(A,C) = 0.707 → C is at 45° angle to A

B is more similar to A despite being further away in Euclidean distance!
This shows why Cosine is preferred for text similarity (ignores magnitude).
```

---

# Module 4: Word Embedding (5 Marks Questions)

## Question 5.1: Skip-gram Training Pairs
**Question:**
For the sentence "Natural language processing enables understanding" with window size = 2, list all training pairs when the target word is "processing".

**Solution:**
```
Target word: "processing" (position 3)
Window size: 2 (±2 words)

Context words within window:
- Position 1: "Natural" (distance 2, within window)
- Position 2: "language" (distance 1, within window)
- Position 4: "enables" (distance 1, within window)
- Position 5: "understanding" (distance 2, within window)

Training pairs for Skip-gram (target, context):
1. (processing, Natural)
2. (processing, language)
3. (processing, enables)
4. (processing, understanding)

Total: 4 training pairs for this target word.
```

---

## Question 5.2: Skip-gram Backward Propagation
**Question:**
Perform one gradient update for Skip-gram with Negative Sampling:
- Target: "neural", Context: "network" (Positive pair, y=1)
- v_neural = [0.4, 0.6]
- u_network = [0.8, 0.2]
- Predicted probability P = σ(v · u) = 0.68
- Learning rate η = 0.05

Calculate the updated target vector v_neural.

**Solution:**
```
Step 1: Calculate dot product (verify)
v · u = 0.4×0.8 + 0.6×0.2 = 0.32 + 0.12 = 0.44
σ(0.44) ≈ 0.608 (close to given 0.68)

Step 2: Calculate error term
Error = P - y = 0.68 - 1 = -0.32

Step 3: Calculate gradient w.r.t v_neural
∇v = Error × u_network
∇v = -0.32 × [0.8, 0.2]
∇v = [-0.256, -0.064]

Step 4: Update v_neural (gradient ascent for positive pairs)
v_new = v_old - η × ∇v
v_new = [0.4, 0.6] - 0.05 × [-0.256, -0.064]
v_new = [0.4, 0.6] - [-0.0128, -0.0032]
v_new = [0.4 + 0.0128, 0.6 + 0.0032]
v_new = [0.4128, 0.6032]

Interpretation: The target vector moved slightly toward the context vector,
as expected for a positive pair.
```

---

## Question 5.3: Word Analogy (Parallelogram Method)
**Question:**
Using the parallelogram method, find the vector for "Tokyo" given the analogy:
"France : Paris :: Japan : ?"

Given vectors:
- v_France = [0.6, 0.3, 0.8]
- v_Paris = [0.8, 0.5, 0.9]
- v_Japan = [0.5, 0.4, 0.7]

**Solution:**
```
Analogy Formula: v_target = v_Paris - v_France + v_Japan

Explanation:
- v_Paris - v_France captures the "capital of" relationship
- Adding v_Japan applies this relationship to Japan

Calculation:
v_Tokyo = v_Paris - v_France + v_Japan
v_Tokyo = [0.8, 0.5, 0.9] - [0.6, 0.3, 0.8] + [0.5, 0.4, 0.7]

Dimension 1: 0.8 - 0.6 + 0.5 = 0.7
Dimension 2: 0.5 - 0.3 + 0.4 = 0.6
Dimension 3: 0.9 - 0.8 + 0.7 = 0.8

v_Tokyo ≈ [0.7, 0.6, 0.8]

In practice, we find the word whose embedding is nearest to this computed vector.
```

---

## Question 5.4: CBOW vs Skip-gram
**Question:**
Compare CBOW and Skip-gram on:
1. Input and Output
2. Training efficiency
3. Performance on rare words
4. Computational cost per training example

**Solution:**
| Aspect | CBOW | Skip-gram |
|--------|------|-----------|
| **Input** | Multiple context words | Single target word |
| **Output** | Single target word | Multiple context words |
| **Training Efficiency** | Faster (one prediction per context) | Slower (multiple predictions per target) |
| **Rare Words** | Worse (context averages dilute signal) | Better (each occurrence updates embedding) |
| **Computation** | Lower (one softmax per window) | Higher (multiple softmax per window) |

**When to use:**
- CBOW: Large corpus, frequent words important, training speed critical
- Skip-gram: Smaller corpus, rare words important, quality over speed

---

## Question 5.5: Extracting Embedding from Matrix
**Question:**
Given the embedding matrix W (4 words × 3 dimensions):
```
        dim1  dim2  dim3
cat      0.2   0.8   0.3
dog      0.3   0.7   0.4
king     0.9   0.1   0.6
queen    0.8   0.2   0.7
```

1. What is the embedding for "king"?
2. Calculate the semantic similarity between "cat" and "dog"
3. What operation retrieves the embedding from the one-hot vector?

**Solution:**
```
1. Embedding for "king" = [0.9, 0.1, 0.6]
   (Simply read row 3 from the matrix)

2. Similarity between "cat" and "dog":
   v_cat = [0.2, 0.8, 0.3]
   v_dog = [0.3, 0.7, 0.4]
   
   Dot product = 0.2×0.3 + 0.8×0.7 + 0.3×0.4 = 0.06 + 0.56 + 0.12 = 0.74
   ||cat|| = √(0.04 + 0.64 + 0.09) = √0.77 ≈ 0.877
   ||dog|| = √(0.09 + 0.49 + 0.16) = √0.74 ≈ 0.860
   
   Cosine = 0.74 / (0.877 × 0.860) = 0.74 / 0.754 ≈ 0.981
   
   Very high similarity (0.981) - expected for semantically similar words!

3. Operation: Matrix multiplication
   one_hot_king = [0, 0, 1, 0]
   embedding = one_hot_king × W = selects row 3 = [0.9, 0.1, 0.6]
   
   This "lookup trick" is why embeddings are faster than sparse representations.
```

---

# Module 5: POS Tagging - HMM (4 Marks Questions)

## Question 6.1: Transition Probability Calculation
**Question:**
Given the following tagged corpus, calculate the transition probabilities:
```
The/DT cat/NN sat/VB on/IN the/DT mat/NN
The/DT dog/NN runs/VBZ fast/RB
```

Calculate: P(NN|DT), P(VB|NN), P(IN|VB)

**Solution:**
```
Count occurrences:
DT occurs: 3 times
  - Followed by NN: 3 times (cat, mat, dog)
NN occurs: 3 times
  - Followed by VB: 1 time (cat→sat)
  - Followed by ./END: 1 time (mat)
  - Followed by VBZ: 1 time (dog→runs)
VB occurs: 1 time
  - Followed by IN: 1 time (sat→on)

Transition Probabilities:
P(NN|DT) = Count(DT→NN) / Count(DT) = 3/3 = 1.0
P(VB|NN) = Count(NN→VB) / Count(NN) = 1/3 ≈ 0.333
P(IN|VB) = Count(VB→IN) / Count(VB) = 1/1 = 1.0
```

---

## Question 6.2: HMM Disambiguation
**Question:**
Disambiguate the word "flies" in the sentence "Time flies like an arrow".
- Previous tag for "Time": NN
- Candidate tags for "flies": NN, VBZ
- P(NN|NN) = 0.2, P(VBZ|NN) = 0.5
- P("flies"|NN) = 0.03, P("flies"|VBZ) = 0.04

**Solution:**
```
HMM Score = Transition × Emission

For "flies" as NN (Noun):
Score(NN) = P(NN|NN) × P("flies"|NN)
         = 0.2 × 0.03
         = 0.006

For "flies" as VBZ (Verb, 3rd person singular):
Score(VBZ) = P(VBZ|NN) × P("flies"|VBZ)
           = 0.5 × 0.04
           = 0.020

Comparison: 0.020 > 0.006

Decision: "flies" is tagged as VBZ (Verb)

This makes sense: "Time flies" where "Time" is the subject and "flies" is the verb.
```

---

## Question 6.3: HMM Components
**Question:**
In the context of HMM-based POS tagging:
1. What do the hidden states represent?
2. What do the observations represent?
3. What does the emission probability model?
4. What does the transition probability model?

**Solution:**
```
1. Hidden States: The POS tags (Noun, Verb, Adjective, Determiner, etc.)
   - Called "hidden" because we don't directly observe them
   - Goal is to infer them from the words

2. Observations: The actual words in the sentence
   - These are what we can see/observe
   - E.g., "The", "cat", "sat", "on", "mat"

3. Emission Probability P(word|tag):
   - Models how likely a word is given a particular tag
   - E.g., P("cat"|NN) might be 0.01, P("cat"|VB) might be 0.0001
   - Captures lexical information

4. Transition Probability P(tag_i|tag_{i-1}):
   - Models how likely a tag is given the previous tag
   - E.g., P(NN|DT) is high because nouns often follow determiners
   - Captures syntactic patterns of the language
```

---

## Question 6.4: Emission vs Transition Roles
**Question:**
Why is P("the"|VB) typically very low, and how does this affect HMM tagging?

**Solution:**
```
Why P("the"|VB) is very low:
- "the" is a determiner, almost never used as a verb
- In training corpus, "the" appears thousands of times with DT tag
- "the" with VB tag is essentially never observed
- Therefore P("the"|VB) ≈ 0 by Maximum Likelihood Estimation

Effect on HMM Tagging:
1. Acts as a "veto" - prevents "the" from being tagged as verb
2. Even if transition P(VB|previous_tag) is high, the product becomes ~0
3. HMM will always prefer DT for "the" regardless of context

Example calculation:
Previous tag: NN, possible tags for "the": DT, VB
- Score(DT) = P(DT|NN) × P("the"|DT) = 0.15 × 0.9 = 0.135
- Score(VB) = P(VB|NN) × P("the"|VB) = 0.4 × 0.0001 = 0.00004

DT wins decisively (0.135 >> 0.00004)
```

---

# Module 6: Viterbi & Advanced Models (5 Marks Questions)

## Question 7.1: Viterbi Table Construction
**Question:**
Fill the complete Viterbi table for the sentence "I run" given:

Start Probabilities: π(PRP)=0.6, π(VB)=0.3, π(NN)=0.1

Transition Matrix (from row to column):
|      | PRP | VB  | NN  |
|------|-----|-----|-----|
| PRP  | 0.1 | 0.7 | 0.2 |
| VB   | 0.2 | 0.3 | 0.5 |
| NN   | 0.3 | 0.4 | 0.3 |

Emission Matrix:
|     | "I"  | "run" |
|-----|------|-------|
| PRP | 0.9  | 0.0   |
| VB  | 0.0  | 0.6   |
| NN  | 0.0  | 0.3   |

**Solution:**
```
STEP 1: Initialization (Word = "I")

V₁(PRP) = π(PRP) × P("I"|PRP) = 0.6 × 0.9 = 0.54
V₁(VB)  = π(VB) × P("I"|VB)   = 0.3 × 0.0 = 0.00
V₁(NN)  = π(NN) × P("I"|NN)   = 0.1 × 0.0 = 0.00

STEP 2: Recursion (Word = "run")

For each state, find max over all previous states:

V₂(PRP):
  From PRP: 0.54 × 0.1 = 0.054
  From VB:  0.00 × 0.2 = 0.000
  From NN:  0.00 × 0.3 = 0.000
  Max = 0.054 (from PRP)
  V₂(PRP) = 0.054 × P("run"|PRP) = 0.054 × 0.0 = 0.000

V₂(VB):
  From PRP: 0.54 × 0.7 = 0.378
  From VB:  0.00 × 0.3 = 0.000
  From NN:  0.00 × 0.4 = 0.000
  Max = 0.378 (from PRP)
  V₂(VB) = 0.378 × P("run"|VB) = 0.378 × 0.6 = 0.2268

V₂(NN):
  From PRP: 0.54 × 0.2 = 0.108
  From VB:  0.00 × 0.5 = 0.000
  From NN:  0.00 × 0.3 = 0.000
  Max = 0.108 (from PRP)
  V₂(NN) = 0.108 × P("run"|NN) = 0.108 × 0.3 = 0.0324

FINAL VITERBI TABLE:
|     | "I"  | "run"  | Backpointer |
|-----|------|--------|-------------|
| PRP | 0.54 | 0.000  | PRP         |
| VB  | 0.00 | 0.2268 | PRP         |
| NN  | 0.00 | 0.0324 | PRP         |

STEP 3: Termination
Maximum at t=2: V₂(VB) = 0.2268 → Tag = VB
Backpointer: VB came from PRP

BEST PATH: PRP → VB
"I" = PRP (Pronoun), "run" = VB (Verb)
```

---

## Question 7.2: Viterbi Backtracking
**Question:**
Given the following Viterbi values and backpointers for a 3-word sentence, recover the best tag sequence:

At t=3:
- State DT: Value=0.001, Backpointer→NN
- State NN: Value=0.015, Backpointer→VB
- State VB: Value=0.008, Backpointer→NN

At t=2:
- State NN: Backpointer→DT
- State VB: Backpointer→NN

At t=1:
- Start state (no backpointer needed)

**Solution:**
```
Step 1: Find Best Final State
Compare values at t=3:
- DT: 0.001
- NN: 0.015 ← Maximum
- VB: 0.008

Best final state: NN
Tag₃ = NN

Step 2: Follow Backpointers

From NN at t=3:
  Backpointer → VB
  Tag₂ = VB

From VB at t=2:
  Backpointer → NN
  Tag₁ = NN

(Wait, this seems wrong - let me re-read)

Actually from the given data:
- At t=3, NN has backpointer to VB
- At t=2, VB has backpointer to NN

So:
Tag₃ = NN (highest value at t=3)
Tag₂ = VB (backpointer from NN at t=3)
Tag₁ = NN (backpointer from VB at t=2)

BEST SEQUENCE: NN → VB → NN

(In practice this represents a sentence like "Dogs/NN eat/VB food/NN")
```

---

## Question 7.3: MEMM vs HMM Comparison
**Question:**
Explain THREE advantages of Maximum Entropy Markov Model (MEMM) over HMM for POS tagging.

**Solution:**
```
1. RICH FEATURE ENGINEERING:
   HMM: Limited to word identity and previous tag
   MEMM: Can use arbitrary overlapping features:
   - Word suffixes (-ing, -ed, -tion)
   - Capitalization (proper nouns)
   - Surrounding words (not just previous)
   - Word shape (Xx-like for title case)
   - Prefixes (un-, pre-, anti-)
   
2. DISCRIMINATIVE VS GENERATIVE:
   HMM: Models P(word|tag) - generative
   MEMM: Models P(tag|word, features) - discriminative
   
   Advantage: Discriminative models directly optimize for the
   classification task rather than modeling the data generation process.
   This typically leads to better accuracy.

3. NO INDEPENDENCE ASSUMPTIONS:
   HMM: Assumes observations are independent given states
   MEMM: No such restriction
   
   Example: In HMM, P(word₁|tag) is independent of P(word₂|tag)
   In MEMM, features can capture dependencies between words.

Additional: MEMM can handle unknown words better through features
like suffixes and word shape, rather than relying solely on <UNK>.
```

---

## Question 7.4: Neural POS Tagger Architecture
**Question:**
Describe the architecture of an RNN-based POS tagger and answer:
1. What is the input at each timestep?
2. What is the output dimension?
3. What activation function produces tag probabilities?
4. How does Bidirectional RNN improve tagging?

**Solution:**
```
ARCHITECTURE:
[Word] → [Embedding] → [Bi-LSTM] → [Dense Layer] → [Softmax] → [Tag]

1. INPUT AT EACH TIMESTEP:
   - Word embedding vector (dense representation)
   - Dimension: typically 50-300 dimensions
   - May include character-level embeddings for handling unknown words
   
2. OUTPUT DIMENSION:
   - Size of the tagset
   - Penn Treebank: 45 tags
   - Universal Dependencies: 17 tags
   - Each output is a probability distribution over all tags

3. ACTIVATION FUNCTION:
   - Softmax: Converts raw scores to probabilities
   - Formula: softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)
   - Ensures outputs sum to 1, interpretable as probabilities

4. BIDIRECTIONAL RNN IMPROVEMENT:
   Forward RNN: Processes left-to-right (past context)
   Backward RNN: Processes right-to-left (future context)
   
   Example: "I saw her duck"
   - To tag "duck", we need both:
     - Past: "saw her" → likely noun (she owns a duck)
     - Future: (end of sentence) → confirms noun interpretation
   
   Hidden state = [forward_hidden; backward_hidden] → concatenated
   
   Result: ~2-3% accuracy improvement over unidirectional
```

---

## Question 7.5: Log-Probability Viterbi
**Question:**
Calculate the Viterbi score in log space for a single step:
- log V₁(N) = -2.3
- log P(V|N) = -0.5 (transition)
- log P("runs"|V) = -1.2 (emission)

Why do we use log probabilities?

**Solution:**
```
CALCULATION:
In log space, multiplication becomes addition:

log V₂(V) = log V₁(N) + log P(V|N) + log P("runs"|V)
          = -2.3 + (-0.5) + (-1.2)
          = -4.0

If we need the actual probability:
V₂(V) = 10^(-4.0) = 0.0001

WHY LOG PROBABILITIES?

1. NUMERICAL UNDERFLOW PREVENTION:
   - Probabilities are small (0.001, 0.0001, etc.)
   - Multiplying many small numbers → extremely tiny results
   - Eventually computer rounds to 0 (underflow)
   
   Example without log:
   0.02 × 0.01 × 0.05 × 0.03 × ... × 0.02 = 1.2×10⁻⁴⁵ → 0!
   
   With log:
   -1.7 + (-2) + (-1.3) + (-1.5) + ... = -45 (perfectly representable)

2. COMPUTATIONAL EFFICIENCY:
   - Addition is faster than multiplication
   - Especially important for long sequences

3. NUMERICAL STABILITY:
   - Comparisons remain accurate
   - max(log a, log b) = log(max(a, b))
```

---

# Quick Reference: Common Mistakes to Avoid

| Module | Common Mistake | Correct Approach |
|--------|----------------|------------------|
| TF-IDF | Forgetting to add 1 before log | Use tf = 1 + log₁₀(count) |
| Cosine Sim | Not taking square root for magnitude | ||v|| = √(Σvᵢ²) |
| Perplexity | Using wrong N (word count) | Count actual words in test sentence |
| Word2Vec | Confusing v (target) and u (context) vectors | v is input embedding, u is output embedding |
| HMM | Multiplying probabilities wrong order | Score = Transition × Emission |
| Viterbi | Forgetting to multiply by emission | V(t) = max[V(t-1)×Trans] × Emit |
| Backtrack | Going forward instead of backward | Start from max final state, follow backpointers |

---

**Good luck with your preparation! 📚✨**
