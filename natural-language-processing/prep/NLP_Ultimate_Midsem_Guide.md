# 🎯 NLP MIDSEM - SUPER SIMPLE PREP GUIDE
## Explained Like You're 5 Years Old | Score 20+ in 5 Hours

---

# 🗺️ EXAM MAP - WHERE MARKS ARE HIDING

| Q# | What They'll Ask | Type | Marks | Difficulty |
|----|------------------|------|-------|------------|
| 1️⃣ | Introduction - NLP Apps | Write sentences | **4** | 😊 Easy |
| 2️⃣ | Language Models | 🔢 Calculate | **4** | 😐 Medium |
| 3️⃣ | Neural LM & LLM | Write + Apply | **4** | 😊 Easy |
| 4️⃣ | Vector Semantics | 🔢 Calculate | **4** | 😐 Medium |
| 5️⃣ | Word Embeddings | 🔢 Calculate | **5** | 😐 Medium |
| 6️⃣ | POS Tagging | 🔢 Calculate | **4** | 😐 Medium |
| 7️⃣ | Viterbi Algorithm | 🔢 Calculate | **5** | 😓 Hard |

**🎯 Secret**: 26 out of 30 marks = JUST CALCULATIONS. Learn formulas = Win!

---

# ⏰ YOUR 5-HOUR BATTLE PLAN

| Time | What to Study | Expected Marks |
|------|---------------|----------------|
| 11:45 AM - 12:45 PM | Q4 + Q5 (TF-IDF, Cosine, Word2Vec) | +9 |
| 12:45 PM - 1:45 PM | Q6 + Q7 (HMM, Viterbi) | +9 |
| 1:45 PM - 2:45 PM | Q2 + Q3 (N-gram, Perplexity, LLM) | +8 |
| 2:45 PM - 3:45 PM | Q1 (Theory + All Formula Review) | +4 |
| 3:45 PM - 4:45 PM | Practice 5 problems, eat, relax | 🧘 |

---

# 📌 MASTER FORMULA CARD (Screenshot This!)

---

## 📊 FORMULA 1: TF-IDF

**📝 Formula:**
```
TF = 1 + log₁₀(count)        ← count = times word appears in doc
IDF = log₁₀(N ÷ df)          ← N = total docs, df = docs with word
TF-IDF = TF × IDF
```

**⏰ When to Use:** Q4 asks "Calculate TF-IDF for word X in document Y"

**🔢 Quick Examples:**
```
Example 1: count=5, N=500, df=100
   TF = 1 + log(5) = 1 + 0.7 = 1.7
   IDF = log(500÷100) = log(5) = 0.7
   TF-IDF = 1.7 × 0.7 = 1.19 ✓

Example 2: count=10, N=1000, df=10
   TF = 1 + log(10) = 1 + 1 = 2
   IDF = log(1000÷10) = log(100) = 2
   TF-IDF = 2 × 2 = 4.0 ✓

Example 3: Word appears in ALL docs (df = N)
   IDF = log(N÷N) = log(1) = 0
   TF-IDF = anything × 0 = 0 ← Common word = useless!
```

---

## 📐 FORMULA 2: COSINE SIMILARITY

**📝 Formula:**
```
            Dot Product           a₁×b₁ + a₂×b₂ + ...
Cosine = ─────────────────── = ─────────────────────────────
         Length A × Length B   √(a₁²+a₂²+...) × √(b₁²+b₂²+...)
```

**⏰ When to Use:** Q4 asks "Find similarity between two vectors"

**🔢 Quick Examples:**
```
Example 1: A=[3,4], B=[4,3]
   Dot = 3×4 + 4×3 = 24
   Length A = √(9+16) = √25 = 5
   Length B = √(16+9) = √25 = 5
   Cosine = 24 ÷ (5×5) = 24÷25 = 0.96 ✓ (very similar!)

Example 2: A=[1,0], B=[0,1]
   Dot = 1×0 + 0×1 = 0
   Cosine = 0 ÷ anything = 0 ✓ (perpendicular = nothing in common)

Example 3: A=[2,1,0,2], B=[1,1,2,1]
   Dot = 2×1 + 1×1 + 0×2 + 2×1 = 5
   Length A = √(4+1+0+4) = √9 = 3
   Length B = √(1+1+4+1) = √7 = 2.65
   Cosine = 5 ÷ (3×2.65) = 5÷7.95 = 0.63 ✓
```

---

## 🎲 FORMULA 3: PERPLEXITY

**📝 Formula:**
```
PP = (1 ÷ P)^(1/N)    where P = multiply all word probabilities
                            N = number of words
                      
💡 LOWER = BETTER!
```

**⏰ When to Use:** Q2 asks "Calculate perplexity for this sentence"

**🔢 Quick Examples:**
```
Example 1: P(I)=0.4, P(love|I)=0.5, P(NLP|love)=0.2, N=3 words
   P = 0.4 × 0.5 × 0.2 = 0.04
   PP = (1÷0.04)^(1/3) = 25^(1/3) = ³√25 = 2.92 ✓

Example 2: P=0.0002, N=3
   PP = (1÷0.0002)^(1/3) = 5000^(1/3) = ³√5000 = 17.1 ✓
   (Higher PP = model more confused)

Example 3: P=0.001, N=4
   PP = (1÷0.001)^(1/4) = 1000^(1/4) = ⁴√1000 = 5.62 ✓
```

---

## 📈 FORMULA 4: BIGRAM & LAPLACE

**📝 Formulas:**
```
BIGRAM:   P(word|prev) = Count(prev,word) ÷ Count(prev)

LAPLACE:  P(word|prev) = (Count + 1) ÷ (Count(prev) + VocabSize)
          Use when Count = 0!
```

**⏰ When to Use:** Q2 asks "Calculate P(word|previous)" or "Apply smoothing"

**🔢 Quick Examples:**
```
Example 1: Bigram - C(I,love)=2, C(I)=2
   P(love|I) = 2 ÷ 2 = 1.0 ✓

Example 2: Bigram - C(I,NLP)=3, C(I)=10
   P(NLP|I) = 3 ÷ 10 = 0.3 ✓

Example 3: Laplace - C(the,cat)=0, C(the)=50, V=10000
   P(cat|the) = (0+1) ÷ (50+10000) = 1÷10050 = 0.0001 ✓
   (Add 1 to numerator, add vocab to denominator)
```

---

## 🏷️ FORMULA 5: HMM SCORE

**📝 Formula:**
```
Score(tag) = P(tag | prev_tag) × P(word | tag)
             ─────────────────   ─────────────
                TRANSITION         EMISSION

🎯 Pick the tag with HIGHEST score!
```

**⏰ When to Use:** Q6 asks "Which tag should this word get?"

**🔢 Quick Examples:**
```
Example 1: Word "flies" after a Noun
   
   Try NN (noun):
   Score = P(NN|NN) × P("flies"|NN) = 0.3 × 0.02 = 0.006
   
   Try VBZ (verb):
   Score = P(VBZ|NN) × P("flies"|VBZ) = 0.4 × 0.05 = 0.020
   
   Winner: VBZ ✓ (0.020 > 0.006)

Example 2: Word "book" after a Verb
   
   Try NN:  Score = 0.5 × 0.04 = 0.020 ← Winner! ✓
   Try VB:  Score = 0.1 × 0.03 = 0.003
```

---

## 🔄 FORMULA 6: WORD2VEC UPDATE

**📝 Formula:**
```
Error = σ(v·u) - y      ← y=1 for real pair, y=0 for fake pair
v_new = v_old - η × Error × u

💡 Real pair → vectors move CLOSER
💡 Fake pair → vectors move APART
```

**⏰ When to Use:** Q5 asks "Update the vector for this word pair"

**🔢 Quick Examples:**
```
Example 1: REAL pair (cat, meow), y=1
   σ(v·u) = 0.55
   Error = 0.55 - 1 = -0.45 (negative = under-predicted)
   v moves TOWARD u ✓

Example 2: FAKE pair (cat, pizza), y=0
   σ(v·u) = 0.60
   Error = 0.60 - 0 = +0.60 (positive = over-predicted)
   v moves AWAY from u ✓

Example 3: Full calculation
   v=[0.2,0.6], u=[0.5,0.3], σ=0.55, y=1, η=0.1
   Error = -0.45
   Gradient = -0.45 × [0.5,0.3] = [-0.225,-0.135]
   v_new = [0.2,0.6] - 0.1×[-0.225,-0.135]
         = [0.2+0.0225, 0.6+0.0135] = [0.2225, 0.6135] ✓
```

---

## 🔁 FORMULA 7: WORD ANALOGY

**📝 Formula:**
```
v_? = v_known - v_old_context + v_new_context

Pattern: A is to B as C is to ?
Formula: ? = C - A + B
```

**⏰ When to Use:** Q5 asks "Find the vector using analogy"

**🔢 Quick Examples:**
```
Example 1: King:Man :: Queen:Woman
   Queen = King - Man + Woman ✓

Example 2: v_Man=[0.5,0.3], v_Woman=[0.4,0.6], v_King=[0.8,0.4]
   v_Queen = [0.8,0.4] - [0.5,0.3] + [0.4,0.6]
           = [0.8-0.5+0.4, 0.4-0.3+0.6]
           = [0.7, 0.7] ✓

Example 3: Paris:France :: Tokyo:Japan
   Japan = France - Paris + Tokyo ✓
```

---

## 🎯 FORMULA 8: VITERBI

**📝 Formula:**
```
INIT:      V₁(tag) = π(tag) × P(word₁ | tag)
RECURSE:   Vₜ(tag) = max[Vₜ₋₁(prev) × P(tag|prev)] × P(wordₜ|tag)
BACKTRACK: Start from max final, follow pointers back
```

**⏰ When to Use:** Q7 asks "Find best tag sequence" or "Complete Viterbi table"

**🔢 Quick Example:**
```
Sentence: "The dog runs" | Tags: DT, NN, VBZ

STEP 1 - INIT ("The"):
   V₁(DT) = π(DT) × P("The"|DT) = 0.6 × 0.8 = 0.48 ← Best!
   V₁(NN) = 0.3 × 0.01 = 0.003

STEP 2 - RECURSE ("dog"):
   V₂(NN) = V₁(DT) × P(NN|DT) × P("dog"|NN)
          = 0.48 × 0.7 × 0.3 = 0.1008 ← Best!

STEP 3 - RECURSE ("runs"):
   V₃(VBZ) = V₂(NN) × P(VBZ|NN) × P("runs"|VBZ)
           = 0.1008 × 0.7 × 0.4 = 0.028 ← Best!

BACKTRACK: VBZ ← NN ← DT

ANSWER: "The"=DT, "dog"=NN, "runs"=VBZ ✓
```

---

# 📗 HOUR 1: TF-IDF & COSINE (Q4 = 4 Marks)

---

## 🧒 What is TF-IDF? (Like Explaining to a Kid)

**Imagine you're looking for a book about DOGS in a library:**
- If "dog" appears 100 times in a book → That book is probably about dogs! (High TF)
- If "dog" appears in only 2 out of 1000 books → Word "dog" is SPECIAL! (High IDF)
- If "the" appears in ALL books → Word "the" is BORING, not helpful (Low IDF)

**TF-IDF tells you: How IMPORTANT is this word for THIS document?**

---

## 🔢 TF-IDF FORMULA (Every Part Explained!)

### 📌 FORMULA BOX:
```
┌─────────────────────────────────────────────────────────┐
│  TF = 1 + log₁₀(count)                                  │
│  ──   ─   ─────────────                                 │
│  │    │        │                                        │
│  │    │        └── How many times word appears in doc   │
│  │    └── We add 1 so TF is never zero                  │
│  └── Term Frequency = How often word appears            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  IDF = log₁₀(N ÷ df)                                    │
│  ───   ───────────────                                  │
│   │         │    │                                      │
│   │         │    └── df = Document Frequency            │
│   │         │        (how many docs have this word)     │
│   │         └── N = Total number of documents           │
│   └── Inverse Doc Freq = Is this word rare/special?     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  TF-IDF = TF × IDF                                      │
│  ──────   ──   ───                                      │
│     │      │    │                                       │
│     │      │    └── How rare is the word overall?       │
│     │      └── How often in THIS document?              │
│     └── Final importance score                          │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 TF-IDF SOLVED EXAMPLE (Step-by-Step)

**QUESTION**: Calculate TF-IDF for word "machine" in Document D1:
- Word "machine" appears **5 times** in D1
- Total documents = **500**
- "machine" appears in **100** documents

**SOLUTION** (Follow the recipe!):

```
🥣 STEP 1: Calculate TF (Term Frequency)
   
   TF = 1 + log₁₀(count)
   TF = 1 + log₁₀(5)           ← "machine" appears 5 times
   TF = 1 + 0.699              ← log₁₀(5) = 0.699 (use calculator)
   TF = 1.699 ✓
   
   
🥣 STEP 2: Calculate IDF (Inverse Document Frequency)

   IDF = log₁₀(N ÷ df)
   IDF = log₁₀(500 ÷ 100)      ← 500 total docs, 100 have "machine"
   IDF = log₁₀(5)              ← 500÷100 = 5
   IDF = 0.699 ✓
   
   
🥣 STEP 3: Multiply them!

   TF-IDF = TF × IDF
   TF-IDF = 1.699 × 0.699
   TF-IDF = 1.188 ✓
   
   
📦 FINAL ANSWER: TF-IDF = 1.188
```

---

## 🚨 SPECIAL CASE: Common Words Like "the"

**QUESTION**: What's TF-IDF for "the" that appears in ALL documents?
- Count in D1 = 20
- Total docs = 500
- "the" appears in = 500 documents (ALL of them!)

**SOLUTION**:
```
TF = 1 + log₁₀(20) = 1 + 1.301 = 2.301

IDF = log₁₀(500 ÷ 500) = log₁₀(1) = 0  ← ZERO!

TF-IDF = 2.301 × 0 = 0 ✓

💡 INSIGHT: Words in ALL documents are useless for finding 
   specific documents, so TF-IDF = 0!
```

---

## 🧒 What is Cosine Similarity? (Kid-Friendly)

**Imagine two arrows pointing in directions:**
- If both arrows point the SAME way → They're twins! (Similarity = 1)
- If arrows are perpendicular (90°) → Nothing in common (Similarity = 0)
- If arrows point OPPOSITE → Total opposites (Similarity = -1)

**We use this to find:** Are two documents similar? Are two words related?

---

## 🔢 COSINE SIMILARITY FORMULA (Fully Explained)

### 📌 FORMULA BOX:
```
┌─────────────────────────────────────────────────────────┐
│                    A · B                                │
│  Cosine = ─────────────────────                         │
│            ||A|| × ||B||                                │
└─────────────────────────────────────────────────────────┘

What does each part mean?

┌─────────────────────────────────────────────────────────┐
│  A · B = Dot Product                                    │
│        = (a₁ × b₁) + (a₂ × b₂) + (a₃ × b₃) + ...       │
│                                                         │
│  Think: Multiply matching pairs, then add everything    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  ||A|| = Magnitude (Length) of A                        │
│        = √(a₁² + a₂² + a₃² + ...)                      │
│                                                         │
│  Think: Square each number, add them, take square root  │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 COSINE SIMILARITY SOLVED EXAMPLE

**QUESTION**: Find similarity between:
- Document A = [2, 1, 0, 2]
- Document B = [1, 1, 2, 1]

**SOLUTION** (Follow the recipe!):

```
🥣 STEP 1: Calculate Dot Product (A · B)

   Multiply matching pairs, then add:
   
   Position 1: 2 × 1 = 2
   Position 2: 1 × 1 = 1  
   Position 3: 0 × 2 = 0
   Position 4: 2 × 1 = 2
                     ────
   A · B = 2 + 1 + 0 + 2 = 5 ✓


🥣 STEP 2: Calculate ||A|| (Length of A)

   Square each number in A, add them, take √:
   
   A = [2, 1, 0, 2]
   Squares: 4 + 1 + 0 + 4 = 9
   ||A|| = √9 = 3 ✓


🥣 STEP 3: Calculate ||B|| (Length of B)

   B = [1, 1, 2, 1]
   Squares: 1 + 1 + 4 + 1 = 7
   ||B|| = √7 = 2.646 ✓


🥣 STEP 4: Put it all together!

   Cosine = (A · B) ÷ (||A|| × ||B||)
   Cosine = 5 ÷ (3 × 2.646)
   Cosine = 5 ÷ 7.938
   Cosine = 0.63 ✓


📦 FINAL ANSWER: Cosine Similarity = 0.63

💡 MEANING: 0.63 is pretty similar (closer to 1 than to 0)
```

---

# 📘 HOUR 1 (Continued): WORD EMBEDDINGS (Q5 = 5 Marks)

---

## 🧒 What is Word2Vec? (Kid-Friendly)

**Imagine every word is a person with a personality:**
- "King" and "Queen" hang out together (royalty friends)
- "Dog" and "Cat" hang out together (pet friends)
- "King" and "Dog" don't hang out much (different groups)

**Word2Vec learns these "friendships" as numbers (vectors)!**

---

## 🔢 WORD ANALOGY FORMULA (Fully Explained)

### 📌 FORMULA BOX:
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  v_Queen = v_King - v_Man + v_Woman                    │
│  ───────   ──────   ─────   ───────                    │
│     │         │        │        │                       │
│     │         │        │        └── Add the new context │
│     │         │        └── Subtract the old context     │
│     │         └── Start with known word                 │
│     └── What we want to find                            │
│                                                         │
│  Think: King is to Man as Queen is to Woman            │
│         So: Queen = King - Man + Woman                 │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 WORD ANALOGY SOLVED EXAMPLE

**QUESTION**: Given these word vectors:
- v_Man = [0.5, 0.3, 0.2]
- v_Woman = [0.4, 0.6, 0.3]
- v_King = [0.8, 0.4, 0.5]

Find v_Queen (King:Man :: Queen:Woman)

**SOLUTION**:
```
🥣 STEP 1: Write the formula

   v_Queen = v_King - v_Man + v_Woman


🥣 STEP 2: Substitute the numbers

   v_Queen = [0.8, 0.4, 0.5] - [0.5, 0.3, 0.2] + [0.4, 0.6, 0.3]


🥣 STEP 3: Do math for each position

   Position 1: 0.8 - 0.5 + 0.4 = 0.7
   Position 2: 0.4 - 0.3 + 0.6 = 0.7
   Position 3: 0.5 - 0.2 + 0.3 = 0.6


📦 FINAL ANSWER: v_Queen = [0.7, 0.7, 0.6]
```

---

## 🧒 Skip-gram vs CBOW (Super Simple!)

### Skip-gram (Target → Context)
```
Given: "cat" (center word)
Predict: "the", "sat", "on", "mat" (surrounding words)

Think: I know the main character, guess who's around them!
```

### CBOW (Context → Target)
```
Given: "the", "sat", "on", "mat" (surrounding words)
Predict: "cat" (center word)

Think: I know the friends, guess the main character!
```

**Memory trick**: 
- **S**kip-gram: **S**ingle word predicts **S**urroundings
- **C**BOW: **C**ontext predicts **C**enter

---

## 🔢 WORD2VEC UPDATE FORMULA (Fully Explained)

### 📌 FORMULA BOX:
```
┌─────────────────────────────────────────────────────────┐
│  For a word pair, we update the vector like this:       │
│                                                         │
│  Error = σ(v · u) - y                                   │
│  ─────   ───────   ─                                    │
│    │        │      │                                    │
│    │        │      └── y = 1 if real pair (cat-meow)    │
│    │        │          y = 0 if fake pair (cat-pizza)   │
│    │        └── σ(v · u) = model's current guess        │
│    │            (probability between 0 and 1)           │
│    └── How wrong was the model?                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  v_new = v_old - η × Error × u                          │
│  ─────   ─────   ─   ─────   ─                          │
│    │       │     │     │     │                          │
│    │       │     │     │     └── The other word's vec   │
│    │       │     │     └── How wrong we were            │
│    │       │     └── η = Learning rate (how big steps)  │
│    │       │         Usually 0.01 to 0.5                │
│    │       └── Old vector (before update)               │
│    └── New vector (after update)                        │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 WORD2VEC UPDATE SOLVED EXAMPLE (Positive Pair)

**QUESTION**: Update v_code given:
- Target: "code", Context: "python" (REAL pair, y = 1)
- v_code = [0.2, 0.6]
- u_python = [0.5, 0.3]
- σ(v · u) = 0.55 (model's current guess)
- η = 0.1 (learning rate)

**SOLUTION**:
```
🥣 STEP 1: Calculate Error (how wrong is the model?)

   Error = σ(v · u) - y
   Error = 0.55 - 1        ← y=1 because it's a real pair
   Error = -0.45 ✓
   
   💡 Negative error means model under-predicted!


🥣 STEP 2: Calculate Gradient (direction to move)

   Gradient = Error × u_python
   Gradient = -0.45 × [0.5, 0.3]
   Gradient = [-0.225, -0.135] ✓


🥣 STEP 3: Update the vector

   v_new = v_old - η × Gradient
   v_new = [0.2, 0.6] - 0.1 × [-0.225, -0.135]
   v_new = [0.2, 0.6] - [-0.0225, -0.0135]
   v_new = [0.2 + 0.0225, 0.6 + 0.0135]    ← minus a negative = plus!
   v_new = [0.2225, 0.6135] ✓


📦 FINAL ANSWER: v_code_new = [0.2225, 0.6135]

💡 INSIGHT: For REAL pairs, vectors move CLOSER together!
```

---

## 📝 WORD2VEC UPDATE SOLVED EXAMPLE (Negative Pair)

**QUESTION**: Update v_dog given:
- Target: "dog", Context: "pizza" (FAKE pair, y = 0)
- v_dog = [0.4, 0.8]
- u_pizza = [0.6, 0.2]
- σ(v · u) = 0.60
- η = 0.2

**SOLUTION**:
```
🥣 STEP 1: Error = 0.60 - 0 = 0.60 ← y=0 for fake pair

🥣 STEP 2: Gradient = 0.60 × [0.6, 0.2] = [0.36, 0.12]

🥣 STEP 3: v_new = [0.4, 0.8] - 0.2 × [0.36, 0.12]
                 = [0.4 - 0.072, 0.8 - 0.024]
                 = [0.328, 0.776] ✓


📦 FINAL ANSWER: v_dog_new = [0.328, 0.776]

💡 INSIGHT: For FAKE pairs, vectors move APART from each other!
```

---

# 📙 HOUR 2: HMM & VITERBI (Q6 + Q7 = 9 Marks)

---

## 🧒 What is POS Tagging? (Kid-Friendly)

**Every word has a job in a sentence:**
- "dog" → Noun (NN) - a thing
- "runs" → Verb (VBZ) - an action
- "the" → Determiner (DT) - points to something
- "happy" → Adjective (JJ) - describes something

**The computer needs to figure out each word's job!**

---

## 🧒 What is HMM? (Super Simple!)

**Think of it like a guessing game:**

1. You can't see the TAGS (they're hidden) 🙈
2. You CAN see the WORDS (they're visible) 👀
3. You use CLUES to guess the tags:
   - **Clue 1**: What tag usually comes after the previous tag?
   - **Clue 2**: What tag usually makes this word?

---

## 🔢 HMM FORMULA (Fully Explained)

### 📌 FORMULA BOX:
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Score = P(tag | prev_tag) × P(word | tag)             │
│  ─────   ─────────────────   ──────────────            │
│    │            │                   │                   │
│    │            │                   └── EMISSION:       │
│    │            │                       If this IS a    │
│    │            │                       noun, how often │
│    │            │                       is it "book"?   │
│    │            │                                       │
│    │            └── TRANSITION:                         │
│    │                After a verb, how often             │
│    │                does a noun come next?              │
│    │                                                    │
│    └── Final score - higher is better!                  │
│                                                         │
│  🎯 CHOOSE THE TAG WITH HIGHEST SCORE!                  │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 HMM SOLVED EXAMPLE (Word Disambiguation)

**QUESTION**: The word "flies" comes after a NOUN. What tag should "flies" get?
- Candidate tags: NN (noun) or VBZ (verb)
- P(NN | NN) = 0.3 ← Probability noun follows noun
- P(VBZ | NN) = 0.4 ← Probability verb follows noun
- P("flies" | NN) = 0.02 ← If it's a noun, how often is it "flies"
- P("flies" | VBZ) = 0.05 ← If it's a verb, how often is it "flies"

**SOLUTION**:
```
🥣 STEP 1: Calculate score for NN (noun)

   Score(NN) = P(NN | NN) × P("flies" | NN)
             = 0.3 × 0.02
             = 0.006 ✓


🥣 STEP 2: Calculate score for VBZ (verb)

   Score(VBZ) = P(VBZ | NN) × P("flies" | VBZ)
              = 0.4 × 0.05
              = 0.020 ✓


🥣 STEP 3: Compare and pick the winner!

   Score(NN)  = 0.006
   Score(VBZ) = 0.020  ← BIGGER = WINNER! 🏆


📦 FINAL ANSWER: "flies" = VBZ (verb)

💡 MEANING: "Time flies" → flies is an action verb!
   (Not the insect noun)
```

---

## 🧒 What is Viterbi? (Super Simple!)

**Viterbi is like finding the best path through a maze:**
- At each step, you calculate ALL possible scores
- You remember which path gave the best score
- At the end, you trace back to find the winning path!

**Three simple steps:**
1. **START**: Calculate scores for first word
2. **CONTINUE**: For each next word, find best path to each tag
3. **TRACE BACK**: Follow the winning path backwards

---

## 🔢 VITERBI FORMULA (Fully Explained)

### 📌 FORMULA BOX:
```
┌─────────────────────────────────────────────────────────┐
│  STEP 1 - INITIALIZATION (First word only)              │
│                                                         │
│  V₁(tag) = π(tag) × P(word₁ | tag)                     │
│  ──────   ──────   ────────────────                     │
│     │        │            │                             │
│     │        │            └── Emission: how likely      │
│     │        │                this word for this tag?   │
│     │        └── π = Start probability                  │
│     │            (how often does sentence start         │
│     │             with this tag?)                       │
│     └── V = Viterbi score for first position            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  STEP 2 - RECURSION (All other words)                   │
│                                                         │
│  Vₜ(j) = max[Vₜ₋₁(i) × A(i→j)] × B(j, wordₜ)           │
│  ─────   ─────────────────────   ──────────────         │
│    │              │                    │                │
│    │              │                    └── Emission     │
│    │              │                        P(word|tag)  │
│    │              └── Find the BEST previous path       │
│    │                  Try all previous tags, pick max   │
│    └── Score at time t for tag j                        │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 VITERBI COMPLETE SOLVED EXAMPLE

**QUESTION**: Tag the sentence "The dog runs"
- Tags available: DT (determiner), NN (noun), VBZ (verb)

**GIVEN DATA**:

**Start Probabilities** (how often sentences start with each tag):
| Tag | π (start prob) |
|-----|----------------|
| DT | 0.6 |
| NN | 0.3 |
| VBZ | 0.1 |

**Transition Probabilities** (what tag follows what):
| From ↓ To → | DT | NN | VBZ |
|-------------|-----|-----|-----|
| DT | 0.1 | 0.7 | 0.2 |
| NN | 0.1 | 0.2 | 0.7 |
| VBZ | 0.3 | 0.6 | 0.1 |

**Emission Probabilities** (word given tag):
| Word | P(word\|DT) | P(word\|NN) | P(word\|VBZ) |
|------|------------|------------|-------------|
| The | 0.8 | 0.01 | 0.01 |
| dog | 0.01 | 0.3 | 0.02 |
| runs | 0.01 | 0.05 | 0.4 |

---

### 🥣 STEP 1: INITIALIZATION (Word = "The")

```
Calculate V₁ for each possible tag:

V₁(DT) = π(DT) × P("The"|DT) 
       = 0.6 × 0.8 
       = 0.48 ⭐ HIGHEST!

V₁(NN) = π(NN) × P("The"|NN) 
       = 0.3 × 0.01 
       = 0.003

V₁(VBZ) = π(VBZ) × P("The"|VBZ) 
        = 0.1 × 0.01 
        = 0.001

📊 After "The": DT is winning with 0.48
```

---

### 🥣 STEP 2A: RECURSION (Word = "dog")

**For NN, try coming from each previous tag:**
```
From DT:  V₁(DT) × P(NN|DT) = 0.48 × 0.7 = 0.336 ← BEST PATH!
From NN:  V₁(NN) × P(NN|NN) = 0.003 × 0.2 = 0.0006
From VBZ: V₁(VBZ) × P(NN|VBZ) = 0.001 × 0.6 = 0.0006

V₂(NN) = 0.336 × P("dog"|NN) = 0.336 × 0.3 = 0.1008 ⭐
Backpointer: DT (came from DT)
```

**For VBZ:**
```
Best path: From DT: 0.48 × 0.2 = 0.096
V₂(VBZ) = 0.096 × P("dog"|VBZ) = 0.096 × 0.02 = 0.00192
```

---

### 🥣 STEP 2B: RECURSION (Word = "runs")

**For VBZ, try coming from each previous tag:**
```
From DT:  V₂(DT) × P(VBZ|DT) = 0.00048 × 0.2 = 0.000096
From NN:  V₂(NN) × P(VBZ|NN) = 0.1008 × 0.7 = 0.07056 ← BEST!
From VBZ: V₂(VBZ) × P(VBZ|VBZ) = 0.00192 × 0.1 = 0.000192

V₃(VBZ) = 0.07056 × P("runs"|VBZ) = 0.07056 × 0.4 = 0.02822 ⭐
Backpointer: NN (came from NN)
```

---

### 🥣 STEP 3: BACKTRACKING

```
┌───────────────────────────────────────┐
│  Final scores for "runs":             │
│  VBZ = 0.02822 ← WINNER! 🏆           │
│                                       │
│  Trace back:                          │
│  "runs" → VBZ (best=0.02822)          │
│     ↑ came from                       │
│  "dog" → NN (backpointer said NN)     │
│     ↑ came from                       │
│  "The" → DT (backpointer said DT)     │
└───────────────────────────────────────┘
```

---

### 📊 FINAL VITERBI TABLE

| Tag | "The" | "dog" | "runs" |
|-----|-------|-------|--------|
| DT | **0.48** | 0.00048 | - |
| NN | 0.003 | **0.1008** ←DT | - |
| VBZ | 0.001 | 0.00192 | **0.02822** ←NN |

**📦 FINAL ANSWER**: **DT → NN → VBZ**
- "The" = Determiner
- "dog" = Noun
- "runs" = Verb

---

# 📕 HOUR 3: LANGUAGE MODELS (Q2 = 4 Marks)

---

## 🧒 What is a Language Model? (Kid-Friendly)

**It predicts what word comes next!**

Example: "I love ___"
- A good model might guess: "you" (80%), "pizza" (10%), "coding" (5%)...
- A bad model might guess: "the" (50%), "banana" (50%)

---

## 🔢 BIGRAM FORMULA (Fully Explained)

### 📌 FORMULA BOX:
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  P(word | prev) = Count(prev, word) ÷ Count(prev)      │
│  ─────────────   ─────────────────   ──────────        │
│        │                │                  │            │
│        │                │                  └── How many │
│        │                │                      times did│
│        │                │                      prev     │
│        │                │                      appear?  │
│        │                └── How many times did we see   │
│        │                    these two words together?   │
│        └── Probability of "word" following "prev"       │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 BIGRAM SOLVED EXAMPLE

**QUESTION**: Given this corpus, find P(love | I):
```
<s> I love NLP </s>
<s> I love coding </s>
<s> NLP is fun </s>
```

**SOLUTION**:
```
🥣 STEP 1: Count how many times "I" appears

   Sentence 1: "I" appears once
   Sentence 2: "I" appears once
   Sentence 3: "I" doesn't appear
   
   Count(I) = 2 ✓


🥣 STEP 2: Count how many times "I love" appears together

   Sentence 1: "I love" ✓
   Sentence 2: "I love" ✓
   
   Count(I, love) = 2 ✓


🥣 STEP 3: Apply the formula

   P(love | I) = Count(I, love) ÷ Count(I)
               = 2 ÷ 2
               = 1.0 ✓


📦 FINAL ANSWER: P(love | I) = 1.0 (or 100%)

💡 MEANING: Every time we saw "I", it was followed by "love"!
```

---

## 🔢 LAPLACE SMOOTHING FORMULA (Fully Explained)

**Problem**: What if we never saw "I cat" together? P = 0÷2 = 0!

**Solution**: Add 1 to everything (fake it till you make it!)

### 📌 FORMULA BOX:
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  P_smooth = (Count + 1) ÷ (N + V)                      │
│             ─────────     ───────                       │
│                 │            │─── V = Vocabulary size   │
│                 │            │    (total unique words)  │
│                 │            └── N = How many times     │
│                 │                prev word appeared     │
│                 └── Add 1 to the count (even if 0!)    │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 LAPLACE SMOOTHING SOLVED EXAMPLE

**QUESTION**: Calculate P(cat | the) with Laplace smoothing:
- Count("the", "cat") = 0 (never saw them together!)
- Count("the") = 50
- Vocabulary size V = 10,000

**SOLUTION**:
```
🥣 Apply Laplace formula:

   P_smooth(cat | the) = (Count + 1) ÷ (N + V)
                       = (0 + 1) ÷ (50 + 10,000)
                       = 1 ÷ 10,050
                       = 0.0000995 ✓


📦 FINAL ANSWER: P ≈ 0.0001

💡 MEANING: Now it's not zero! Small, but possible!
```

---

## 🔢 PERPLEXITY FORMULA (Fully Explained)

### 📌 FORMULA BOX:
```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  PP = (1 ÷ P(sentence))^(1÷N)                          │
│  ──   ─────────────────  ────                           │
│  │           │            │                             │
│  │           │            └── N = number of words       │
│  │           └── Total probability of sentence          │
│  │               (multiply all word probabilities)      │
│  └── Perplexity = "how confused is the model?"          │
│                                                         │
│  💡 LOWER PERPLEXITY = BETTER MODEL!                    │
│  💡 PP of 10 means choosing from ~10 equally likely     │
│     words at each position                              │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 PERPLEXITY SOLVED EXAMPLE

**QUESTION**: Calculate perplexity for "I love NLP" (N=3 words):
- P(I | start) = 0.4
- P(love | I) = 0.5
- P(NLP | love) = 0.2

**SOLUTION**:
```
🥣 STEP 1: Calculate P(sentence)

   P(sentence) = P(I) × P(love|I) × P(NLP|love)
               = 0.4 × 0.5 × 0.2
               = 0.04 ✓


🥣 STEP 2: Calculate (1 ÷ P)

   1 ÷ 0.04 = 25


🥣 STEP 3: Take the Nth root (N=3)

   PP = 25^(1/3)        ← Cube root of 25
   PP = ³√25
   PP ≈ 2.92 ✓


📦 FINAL ANSWER: Perplexity ≈ 2.92

💡 MEANING: Model is choosing from about 3 words at each step.
            That's pretty good! (lower = better)
```

---

# 📒 HOUR 4: THEORY (Q1 + Q3)

---

## Q1: NLP APPLICATIONS (Memorize 4!)

| Application | What it does | Example |
|-------------|--------------|---------|
| 🌐 Machine Translation | Language A → Language B | Google Translate |
| 😊 Sentiment Analysis | Is text happy/sad/angry? | Product reviews |
| 🏷️ NER | Find names, places, orgs | "Apple is in California" |
| ❓ Question Answering | Answer questions | Siri, Alexa |

---

## Q1: LEVELS OF LANGUAGE (Memorize Order!)

```
🎭 Pragmatic    = "Can you pass salt?" means PLEASE PASS IT
    ↓
📖 Discourse    = "John went. He bought" - He = John  
    ↓
💭 Semantic     = Word meanings
    ↓
📝 Syntactic    = Grammar rules
    ↓
🏷️ Lexical      = Word categories (noun, verb)
    ↓
🔤 Morphological = Word parts (un + happy + ness)
```

**Memory Trick**: **P**lease **D**on't **S**leep, **S**tudy **L**ate **M**orning

---

## Q1: TYPES OF AMBIGUITY

| Type | Example | Why ambiguous? |
|------|---------|----------------|
| **Structural** | "I saw man with telescope" | WHO has the telescope? |
| **Lexical** | "The bank is flooded" | River bank? Money bank? |
| **Grammatical** | "Can you can a can?" | can = ability/verb/noun |

---

## Q3: PROMPT TYPES

| Type | How many examples? | Example |
|------|-------------------|---------|
| **Zero-shot** | 0 | "Translate: Hello → French" |
| **One-shot** | 1 | "Hello→Bonjour. Goodbye→?" |
| **Few-shot** | 2-5 | Multiple examples first |
| **Chain-of-Thought** | Step by step | "Let's solve step by step..." |

---

# ⚠️ COMMON MISTAKES - DON'T DO THESE!

| ❌ MISTAKE | ✅ CORRECT |
|-----------|-----------|
| TF = log₁₀(count) | TF = **1 +** log₁₀(count) |
| PP = P^(1/N) | PP = P^(**-1/N**) or (1/P)^(1/N) |
| HMM = just transition | HMM = Transition **×** Emission |
| Viterbi: forgot emission | Must multiply by **P(word\|tag)** at end! |
| Cosine: forgot magnitude | Calculate **BOTH** ||A|| **AND** ||B|| |
| Skip-gram = context→target | Skip-gram = **target→context** |

---

# 🏆 EXAM STRATEGY

**Answer in this order:**
1. **Q4** (TF-IDF, Cosine) - 4 marks - Direct calculation
2. **Q5** (Word2Vec) - 5 marks - Formula-based
3. **Q6** (HMM) - 4 marks - Multiplication only
4. **Q7** (Viterbi) - 5 marks - Takes time, do carefully
5. **Q2** (N-gram, PP) - 4 marks - Easy formulas
6. **Q3** (LLM) - 4 marks - Theory, relax
7. **Q1** (Intro) - 4 marks - Just write points

**🍀 SHOW ALL STEPS = PARTIAL MARKS!**

---

# 🎓 GOOD LUCK! YOU'VE GOT THIS!
