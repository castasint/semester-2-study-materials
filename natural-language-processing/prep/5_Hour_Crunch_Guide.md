# 🚨 5-HOUR EXAM CRUNCH GUIDE
## NLP Mid-Semester - Emergency Study Plan

**Time Available**: 5 hours
**Exam Time**: ~11:23 PM IST

---

# ⏰ Hour-by-Hour Study Plan

## Hour 1 (18:25 - 19:25): Formulas & Core Concepts
**Priority: CRITICAL CALCULATIONS**

### Must Memorize (15 min):
```
TF-IDF:     TF = 1 + log₁₀(count),  IDF = log₁₀(N/df)
Cosine:     cos(A,B) = (A·B) / (||A|| × ||B||)
Perplexity: PP = P(W)^(-1/N)
Bigram:     P(wₙ|wₙ₋₁) = C(wₙ₋₁,wₙ) / C(wₙ₋₁)
Laplace:    P = (C+1) / (N+V)
HMM Score:  Transition × Emission
Viterbi:    Vₜ(j) = max[Vₜ₋₁(i) × A(i,j)] × B(j,oₜ)
Analogy:    v_target = v_a* - v_a + v_b
```

### Quick Videos (30 min):
- 📺 [What is Perplexity?](https://www.youtube.com/watch?v=NURcDHhYe98) - 4 min
- 📺 [TF-IDF Explained](https://www.youtube.com/watch?v=D2V1okCEsiE) - 7 min
- 📺 [Cosine Similarity](https://www.youtube.com/watch?v=e9U0QAFbfLI) - 8 min

### Practice (15 min):
- Do 2-3 TF-IDF + Cosine calculations from `NLP_Exam_Preparation_Guide.md`

---

## Hour 2 (19:25 - 20:25): Word Embeddings & N-grams
**Priority: WORD2VEC & PERPLEXITY**

### Review (20 min):
Read from `Session2_Vector_Semantics_Notes.md`:
- Skip-gram vs CBOW (which is which)
- Word Analogy parallelogram method
- Gradient update formula

### Quick Videos (25 min):
- 📺 [Word2Vec Explained](https://www.youtube.com/watch?v=viZrOnJclY0) - 15 min (watch at 1.5x = 10 min)
- 📺 [N-gram Models](https://www.youtube.com/watch?v=GiyMGBuu45w) - 11 min (watch at 1.5x = 7 min)

### Practice (15 min):
- Word2Vec update calculation
- Perplexity calculation
- Smoothing (Laplace)

---

## Hour 3 (20:25 - 21:25): HMM & Viterbi
**Priority: MOST LIKELY EXAM QUESTIONS**

### Review (15 min):
Read from `Session6_POS_Tagging_Notes.md` and `Session7_Viterbi_MEMM_Neural_Notes.md`:
- HMM components (States, Observations, Transition, Emission)
- Viterbi table construction

### Essential Video (20 min):
- 📺 [Viterbi Algorithm Solved Problem](https://www.youtube.com/watch?v=6JVqutwtzmo) - 12 min
- 📺 [HMM POS Tagging](https://www.youtube.com/watch?v=kqSzLo9fenk) - 15 min (1.5x = 10 min)

### Practice (25 min):
- Complete ONE full Viterbi table calculation
- HMM disambiguation problem
- Backtracking to get final sequence

---

## Hour 4 (21:25 - 22:25): LLM, Prompting & Theory
**Priority: CONCEPTUAL QUESTIONS**

### Quick Read (30 min):
From `Session5_LLM_Prompting_Notes.md`:
- Zero-shot vs One-shot vs Few-shot (with examples)
- Chain-of-Thought prompting
- Pre-training vs Fine-tuning

From `Session1_NLU_NLG_Notes.md`:
- 6 Levels of Language Analysis
- Types of Ambiguity (Structural, Lexical, Grammatical)

### Quick Video (15 min):
- 📺 [Zero-Shot, Few-Shot & CoT](https://www.youtube.com/watch?v=V1KZn8WT6Go) - 15 min (1.5x = 10 min)

### Review Diagrams (15 min):
Look at all images in `prep/images/` folder for visual memory

---

## Hour 5 (22:25 - 23:25): Final Revision & Mock
**Priority: EXAM SIMULATION**

### Quick Cheatsheet Review (20 min):
- Read `NLP_Quick_Reference_Cheatsheet.md` completely
- Focus on "Common Mistakes to Avoid" section

### Mock Problem Solving (30 min):
From `NLP_Additional_Practice_Questions.md`, solve:
- 1 TF-IDF + Cosine problem
- 1 Perplexity problem
- 1 Viterbi problem
- 1 Word Analogy problem

### Mental Checklist (10 min):
```
□ TF-IDF formula & calculation
□ Cosine similarity steps
□ Perplexity interpretation (lower = better)
□ Viterbi: Init → Recurse → Backtrack
□ Skip-gram: target → context
□ CBOW: context → target
□ Zero/Few-shot examples ready
□ 6 levels: Morphological → Pragmatic
```

---

# 📋 CRITICAL FORMULAS CARD

Print/Screenshot this:

```
╔══════════════════════════════════════════════════╗
║             NLP EXAM FORMULA CARD                ║
╠══════════════════════════════════════════════════╣
║ TF = 1 + log₁₀(count)     │ IDF = log₁₀(N/df)   ║
║ TF-IDF = TF × IDF                                ║
╠══════════════════════════════════════════════════╣
║ Cosine = (A·B) / (||A|| × ||B||)                ║
║ ||A|| = √(a₁² + a₂² + ...)                      ║
╠══════════════════════════════════════════════════╣
║ Perplexity = P(sentence)^(-1/N)                 ║
║ Lower PP = Better model                          ║
╠══════════════════════════════════════════════════╣
║ Bigram: P(w|prev) = C(prev,w) / C(prev)         ║
║ Laplace: P = (C+1) / (N+V)                       ║
╠══════════════════════════════════════════════════╣
║ HMM Score = P(tag|prev_tag) × P(word|tag)       ║
║ Viterbi: V(j) = max[V(i)×A(i,j)] × B(j,obs)    ║
╠══════════════════════════════════════════════════╣
║ Word Analogy: v_? = v_b - v_a + v_a*            ║
║ Word2Vec: v_new = v_old - η × Error × u         ║
╚══════════════════════════════════════════════════╝
```

---

# 🎬 ESSENTIAL VIDEOS ONLY (Total: ~45 min at 1.5x)

| Priority | Video | Time | Topic |
|----------|-------|------|-------|
| ⭐⭐⭐ | [Perplexity](https://www.youtube.com/watch?v=NURcDHhYe98) | 4 min | Must know |
| ⭐⭐⭐ | [TF-IDF](https://www.youtube.com/watch?v=D2V1okCEsiE) | 7 min | Calculation |
| ⭐⭐⭐ | [Viterbi Solved](https://www.youtube.com/watch?v=6JVqutwtzmo) | 12 min | Problem |
| ⭐⭐ | [Word2Vec](https://www.youtube.com/watch?v=viZrOnJclY0) | 15 min | Update formula |
| ⭐⭐ | [CoT Prompting](https://www.youtube.com/watch?v=V1KZn8WT6Go) | 15 min | Concepts |

---

# 📁 FILES TO FOCUS ON

1. **`NLP_Quick_Reference_Cheatsheet.md`** - All formulas in one place
2. **`NLP_Exam_Preparation_Guide.md`** - Practice questions with solutions
3. **`images/` folder** - Visual diagrams for quick recall

---

# ⚠️ COMMON MISTAKES TO AVOID

| Mistake | Correct |
|---------|---------|
| TF without +1 | TF = **1** + log₁₀(count) |
| Wrong perplexity power | PP = P^(**-1/N**) not P^(1/N) |
| Forgetting to backtrack Viterbi | Always trace back from max final state |
| Confusing Skip-gram/CBOW | Skip-gram: target→context, CBOW: context→target |
| Missing magnitude in cosine | Calculate BOTH ||A|| AND ||B|| |

---

**🍀 Good luck! You've got this! Remember: show all steps for partial credit!**

---

*Exam Time: ~11:23 PM | Current: 6:23 PM | Time Left: 5 hours*
