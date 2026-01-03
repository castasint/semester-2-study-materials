# NLP Regular Exam Solutions - Expanded Guide

## Question 1: Text Preprocessing
**Question:**
You are preparing data from social media comments. Consider the text: *"I'd love 2 go, but I can't. The concert is 100% sold out :("*
Perform the following advanced steps one by one: Tokenization, Stop Word Removal, Lemmatization.

### **Answer**
1.  **Tokenization:** `["I", "'d", "love", "2", "go", ",", "but", "I", "ca", "n't", ".", "The", "concert", "is", "100", "%", "sold", "out", ":("]`
2.  **Stop Word Removal:** `["love", "2", "go", "concert", "100", "%", "sold", ":("]` *(Removing I, 'd, but, ca, n't, ., The, is, out)*
3.  **Lemmatization:** `["love", "2", "go", "concert", "100", "%", "sell", ":("]` *(sold -> sell)*

### **Theory**
Computers cannot understand raw sentences. We must break them down into standard units.
1.  **Tokenization:** Breaking a string into atomic units (words/punctuation).
2.  **Stop Words:** Removing high-frequency words that carry little unique meaning (is, the, a).
3.  **Lemmatization:** Using a dictionary/morphological analysis to return a word to its root (better than stemming which just chops off ends).

### **Formula**
*   **Lemma(w)** = MorphologicalRoot(w)
    *   e.g., Lemma(sold) = sell, Lemma(better) = good.

### **Explanation like a 1-year-old**
Imagine you have a toy box full of messy Lego blocks.
1.  **Tokenization:** First, we pull all the blocks apart so they are separate pieces.
2.  **Stop Word Removal:** We throw away the plain grey blocks because they are boring and we have too many of them.
3.  **Lemmatization:** If we have a "broken" red block and a "perfect" red block, we fix the broken one so they both look like just a "red block". Now everything is neat!

### **Key Concepts**
*   **Normalization:** Making text standard so "run" and "running" match.
*   **Noise Reduction:** Removing "the" helps models focus on keywords like "concert".
*   **Contraction expansion:** "Can't" -> "Can" + "Not" (or ca + n't).

### **Extra Practice Examples**
*   **Example A:**
    *   *Input:* "Dr. Smith won't drive."
    *   *Step 1 (Token):* `["Dr.", "Smith", "wo", "n't", "drive", "."]`
    *   *Step 2 (Stop):* `["Dr.", "Smith", "drive"]` (Removing 'wo', 'n't', '.')
    *   *Step 3 (Lemma):* `["Dr.", "Smith", "drive"]` (Already base forms)
*   **Example B:**
    *   *Input:* "It's rainin' cats & dogs!"
    *   *Step 1 (Token):* `["It", "'s", "rainin'", "cats", "&", "dogs", "!"]`
    *   *Step 2 (Stop):* `["rainin'", "cats", "dogs"]` (Removing It, 's, &, !)
    *   *Step 3 (Lemma):* `["rain", "cat", "dog"]` (Fixing slang "rainin'" -> rain, plurals -> singular)

---

## Question 2: N-gram Probability & Smoothing
**Question:**
Toy corpus: "read a book", "read a blog".
Calculate MLE $P(\text{map}|\text{a})$ for test bigram "read a map". Then apply Add-1 Smoothing ($V=5$).

### **Answer**
**a) MLE Probability:** $0$
**b) Laplace Smoothed Probability:** $1/7$ or $\approx 0.142$

### **Theory**
Language models predict the next word based on history. If a word never appeared in the training data (Oov/Sparsity), MLE gives it 0% probability. This crashes the whole sentence calculation (multiplication by zero). Smoothing fixes this by "stealing" probability mass from frequent words and giving it to unseen words.

### **Math**
**a) MLE:**
*   Count("a map") = 0
*   Count("a") = 2 ("a book", "a blog")
*   $P = 0 / 2 = 0$

**b) Add-1 Smoothing:**
*   Count("a map") = 0 (add 1 -> 1)
*   Count("a") = 2 (add $V$ -> $2 + 5 = 7$)
*   $P = 1 / 7$

### **Formula**
**MLE:**
$$P(w_i|w_{i-1}) = \frac{count(w_{i-1}, w_i)}{count(w_{i-1})}$$

**Laplace (Add-1):**
$$P(w_i|w_{i-1}) = \frac{count(w_{i-1}, w_i) + 1}{count(w_{i-1}) + V}$$

### **Explanation like a 1-year-old**
We are guessing what toy comes after "a".
We saw "a book" and "a blog". We never saw "a map".
**MLE:** The computer says "Impossible! Maps don’t exist!"
**Smoothing:** We say, "Wait, maybe a map exists, we just missed it." We pretend we saw it **one time**. Now the computer says, "Okay, maybe there is a small chance it's a map."

### **Key Concepts**
*   **Sparsity:** We never have enough data to see every possible sentence.
*   **Zero Probability Problem:** Zeros destroy calculations.
*   **Vocabulary (V):** The list of all known words.

### **Extra Practice Examples**
*   **Example A (One-word-corpus):**
    *   *Corpus:* "Eat apple. Eat banana." ($V=3$: Eat, apple, banana).
    *   *Task:* Prob of "Eat carrot" (Carrot is OOV/new).
    *   *MLE:* 0/2 = 0.
    *   *Add-1:* Count("Eat carrot")=0->1. Count("Eat")=2. V=3. $P = 1/(2+3) = 1/5 = 0.2$.
*   **Example B (Higher Counts):**
    *   *Corpus:* "The cat" (appears 10 times), "The dog" (appears 90 times). V=100.
    *   *Task:* Prob of "The bird" (unseen).
    *   *Add-1:* Count("The bird")=0->1. Count("The")=100. V=100. $P = 1/(100+100) = 1/200 = 0.005$.

---

## Question 3: Skip-Gram Negative Sampling (SGNS) Loss
**Question:**
Target: "doctor", Context: "hospital". Negatives: "car", "banana".
Vectors provided. Compute Total SGNS Loss.

### **Answer**
**Total Loss = 3.261**

### **Theory**
Word2Vec (Skip-Gram) tries to move context words closer to the target word and random "negative" words further away. The "Loss" is a score of how *bad* the model currently is. We want high similarity for (doctor, hospital) and low similarity for (doctor, car).

### **Math**
1.  **Pos (Hosp):** $2(1) + -1(2) + 1(-1) = -1$.
    *   Sigmoid $\sigma(-1) \approx 0.27$. Loss $= -\ln(0.27) \approx 1.31$.
2.  **Neg (Car):** $-1(1) + 1(2) + 0.5(-1) = 0.5$.
    *   Sigmoid $\sigma(0.5) \approx 0.62$. Loss $= -\ln(1 - 0.62)$ or $-\ln(\sigma(-0.5))$
    *   We use $\sigma(-z)$ for negatives. $\sigma(-0.5) \approx 0.38$. Loss $= -\ln(0.38) \approx 0.97$.
3.  **Neg (Banana):** $0.5(1) + -0.5(2) + -1(-1) = 0.5$.
    *   Same as Car. Loss $\approx 0.97$.
4.  **Total:** $1.31 + 0.97 + 0.97 = 3.25$ (approx).

### **Formula**
$$L = - \log(\sigma(u \cdot v)) - \sum \log(\sigma(-u_{neg} \cdot v))$$
*   Where $\sigma(x) = \frac{1}{1+e^{-x}}$

### **Explanation like a 1-year-old**
We want the word "Doctor" to hold hands with "Hospital".
We want "Doctor" to run away from "Banana".
Currently, "Doctor" is pushing "Hospital" away (bad!) and holding hands with "Banana" (bad!).
The "Loss" is the teacher giving the model zero stars because it got everything wrong.

### **Key Concepts**
*   **Dot Product:** Measures similarity.
*   **Sigmoid:** Squashes numbers between 0 and 1 (probability).
*   **Negative Sampling:** efficient training by only looking at a few "wrong" words instead of valid vocabulary.

### **Extra Practice Examples**
*   **Example A (Perfect Model):**
    *   *Vectors:* Target `[1,0]`, Positive `[1,0]`, Negative `[0,1]`.
    *   *Pos Dot:* $1*1=1$. $\sigma(1) \approx 0.73$. loss $= -\ln(0.73) \approx 0.31$.
    *   *Neg Dot:* $1*0=0$. $\sigma(0)=0.5$. we want $\sigma(-0) = 0.5$. loss $= -\ln(0.5) \approx 0.69$.
    *   *Total:* $1.0$ (Low loss).
*   **Example B (Worst Model):**
    *   *Vectors:* Target `[1,0]`, Positive `[-1,0]` (Opposite!), Negative `[1,0]` (Same!).
    *   *Pos Dot:* $-1$. $\sigma(-1) \approx 0.26$. loss $= -\ln(0.26) \approx 1.34$.
    *   *Neg Dot:* $1$. $\sigma(1) \approx 0.73$. we want $\sigma(-1) \approx 0.26$. loss $= -\ln(0.26) \approx 1.34$.
    *   *Total:* $2.68$ (High loss).

---

## Question 4: Word Embeddings & Operations
**Question:**
Extract 'desert'. Calculate distance between 'ocean' and 'forest'. compute 'forest + climate + mountain'.

### **Answer**
**a)** 'desert' vector: `[7, 6, 1]`
**b)** 'ocean': `[5, 8, 6]`, 'forest': `[2, 3, 5]`. Use Cosine Similarity.
**c)** Sum: `[14, 14, 13]`

### **Theory**
Words are represented as lists of numbers (vectors). Words with similar meanings have vectors pointing in similar directions. We can do math with these vectors!

### **Math**
**c) Addition:**
$$[2,3,5] + [6,7,6] + [6,4,2] = [2+6+6, 3+7+4, 5+6+2] = [14, 14, 13]$$

### **Formula**
**Vector Addition:** $\vec{v}_{sentence} = \sum \vec{v}_{word}$
**Cosine Similarity:** $\frac{A \cdot B}{||A|| ||B||}$

### **Explanation like a 1-year-old**
Imagine a map of a city.
"Forest" is at house number `[2, 3, 5]`.
"Climate" is a delivery truck moving `[6, 7, 6]` blocks.
If we start at the Forest and drive the Climate truck, where do we end up? We just add the numbers together to find the new spot.

### **Key Concepts**
*   **Distributed Representation:** Meaning is spread across dimensions, not just one ID.
*   **Semantic Composition:** Meaning of a phrase can often be found by adding meanings of words.

### **Extra Practice Examples**
*   **Example A (Analogy):**
    *   *Vectors:* King `[5]`, Man `[2]`, Woman `[3]`.
    *   *Task:* Compute "Queen". Formula: King - Man + Woman.
    *   *Math:* $5 - 2 + 3 = 6$. The Queen vector should be near `[6]`.
*   **Example B (Classification Centroid):**
    *   *Task:* Represent document "Apple Banana".
    *   *Vectors:* Apple `[1, 1]`, Banana `[2, 2]`.
    *   *Centroid (Average):* `[(1+2)/2, (1+2)/2] = [1.5, 1.5]`.

---

## Question 5: Training Error & Updates
**Question:**
"Happy" vs "Joyful" (Target 1) and "Sad" (Target 0). Compute products, sigmoids, and errors. Explain signs.

### **Answer**
1.  **Joyful:** Dot=0.09, Sig=0.52. **Error = -0.48**.
2.  **Sad:** Dot=0.05, Sig=0.51. **Error = +0.51**.

**Meaning:**
*   Negative error for Joyful -> "Score was too low, boost it!"
*   Positive error for Sad -> "Score was too high, lower it!"

### **Theory**
This is Gradient Descent. We compare what the model *thought* it saw (prediction) vs the *truth* (target). The difference is the error signal used to update the weights.

### **Math**
1.  **Joyful:** $0.52 - 1.0 = -0.48$
2.  **Sad:** $0.51 - 0.0 = +0.51$

### **Formula**
$$Error = \text{Prediction}(\sigma) - \text{Target}(y)$$
Update rule depends on this sign.

### **Explanation like a 1-year-old**
**Robot:** "I think Happy and Joyful are 50% friends."
**Teacher:** "Wrong! They are 100% friends!"
**Robot:** "Oops, my score was too low (Negative error). I will move them closer."

**Robot:** "I think Happy and Sad are 50% friends."
**Teacher:** "Wrong! They are enemies!"
**Robot:** "Oops, my score was too high (Positive error). I will push them apart."

### **Key Concepts**
*   **Backpropagation:** Sending error info backwards to fix the brain.
*   **Targets:** 1 for neighbors, 0 for random words.

### **Extra Practice Examples**
*   **Example A (Overconfident Wrong):**
    *   *Target:* 0 (should be enemies).
    *   *Prediction:* 0.99 (Robot thinks they are best friends).
    *   *Error:* $0.99 - 0 = +0.99$. Huge Positive error ("Stop being friends!").
*   **Example B (Overconfident Right):**
    *   *Target:* 1 (should be friends).
    *   *Prediction:* 0.99 (Robot thinks they are best friends).
    *   *Error:* $0.99 - 1 = -0.01$. Tiny Negative error ("You are doing great, just a tiny bit more").

---

## Question 6: POS Tagging
**Question:**
"Book the table" vs "Read the book". Why POS tagging?

### **Answer**
**a)** POS tagging tells us the first "book" is a **Verb** (action) and the second is a **Noun** (object). This ensures we don't try to "read a reservation" or "reserve a paperback".
**b)** Noun: `NN`, Verb: `VB`.

### **Theory**
Many words are **ambiguous** (polysemous). Knowing the Grammatical category (Part of Speech) helps solve this ambiguity for translation, speech synthesis, etc.

### **Formula**
$P(tag|word) \text{ vs } P(tag|context)$

### **Explanation like a 1-year-old**
The word "Book" is wearing a costume.
Sometimes it wears a "Work Uniform" (Verb - to book a ticket).
Sometimes it wears a "Party Hat" (Noun - a storybook).
We put a sticker on the word saying "Uniform" or "Hat" so we know what it is doing.

### **Key Concepts**
*   **Disambiguation:** Selecting the right meaning.
*   **Penn Treebank:** The standard list of stickers (tags) used in NLP.

### **Extra Practice Examples**
*   **Example A:**
    *   *Sentence:* "I can open the can."
    *   *Can 1:* Modal Verb (MD) - ability.
    *   *Can 2:* Noun (NN) - metal container.
*   **Example B:**
    *   *Sentence:* "Time flies like an arrow."
    *   *Time:* Noun (NN).
    *   *Flies:* Verb (VBZ).
    *   *Like:* Preposition (IN).
    *   *Contrast:* In "I like flies", Like is a Verb and Flies is a Noun!

---

## Question 7: Modern NLP (BERT vs LLM)
**Question:**
How do BERT (Pre-trained) and GPT-4 (LLM) do POS tagging deeply?

### **Answer**
**a) BERT (Fine-tuning):** It has already read the whole internet (pre-training) and understands grammar. We add a tiny "classifier" on top and train it specifically on POS examples to adjust the weights slightly.
**b) LLM (Prompting):** We don't train it. We just ask it nicely in English: "Here is a sentence, please list the POS tags." (Few-shot or Zero-shot prompting).

### **Theory**
*   **BERT (Discriminative):** Pre-train + Fine-tune. Requires labeled data and training.
*   **GPT (Generative):** In-context learning. Requires no weight updates, just good instructions.

### **Formula**
*   **BERT:** $W_{new} = W_{old} - \alpha \nabla L$
*   **GPT:** $P(\text{tags} | \text{Prompt} + \text{Sentence})$

### **Explanation like a 1-year-old**
**BERT:** Like sending a genius student to college for 4 years to learn "Language", then giving them a 1-week crash course on "Tagging".
**GPT-4:** Like asking a super-genius who already knows everything, "Hey, can you just do this for me?" and they just say the answer.

### **Key Concepts**
*   **Fine-tuning:** Adjusting a big model for a small task.
*   **Prompt Engineering:** Designing the right question for an AI.

### **Extra Practice Examples**
*   **Example A (Sentiment Analysis):**
    *   *BERT:* Feed thousands of movie reviews labeled "Positive/Negative". Update weights (Fine-tune).
    *   *GPT:* Prompt: "Classify this review: 'The movie was dull.' Answer:" GPT says "Negative".
*   **Example B (Summarization):**
    *   *BERT:* Harder for BERT (it's not a generator), needs special architecture (Encoder-Decoder like BART).
    *   *GPT:* Prompt: "Summarize this news article in 3 lines." GPT generates the summary instantly.
