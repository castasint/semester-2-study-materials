# 🎯 DRL ACTUAL EXAM ANALYSIS - Dec 2025

> **This is the REAL exam from December 2025 with solutions!**
> **Pattern: 4 Questions × 7.5 marks = 30 marks**

---

## 📊 ACTUAL EXAM PATTERN BREAKDOWN

| Q# | Topic | Marks | Parts |
|----|-------|-------|-------|
| **Q1** | RL Basics + Value Iteration | 7.5M | a(1.5) + b(2) + c(4) |
| **Q2** | MDP Formulation | 7.5M | a(2.5) + b(2) + c(1.5) + d(1.5) |
| **Q3** | Multi-Armed Bandits | 7.5M | a(3) + b(1.5) + c(1) + d(2) |
| **Q4** | MC Control + Returns | 7.5M | a(2) + b(2) + c(3.5) |

---

## 📋 DETAILED QUESTION ANALYSIS

### Q1: RL Basics + Value Iteration (7.5 marks)

**Q1a (1.5 marks):** Why RL fits a scenario + Immediate Reward vs Long-term Value
- **Practice Set Coverage:** Q1, Q2, Q7 ✅

**Q1b (2 marks):** Classify as MAB vs Finite MDP (2 cases)
- **Practice Set Coverage:** Q11, Q20, Q21 ✅

**Q1c (4 marks):** **Synchronous Value Iteration - 1 iteration** ⭐⭐⭐
- Given: States, Actions, Transition probabilities, Rewards, γ
- Calculate: V₁(s) for all states
- **Practice Set Coverage:** Q42 ✅ (but this exam had specific format)

**Actual Calculation from Exam:**
```
Formula: V₁(s) = max_a Σ P(s'|s,a) × R(s,a,s')  [since V₀ = 0]

State: Resting
  Q(A) = 0.6(1) + 0.4(2) = 0.6 + 0.8 = 1.4
  Q(B) = 1.0(2) = 2.0
  V₁(Resting) = max(1.4, 2.0) = 2.0

State: Moderate
  Q(A) = 0.4(2) + 0.6(1) = 0.8 + 0.6 = 1.4
  Q(B) = 1.0(1) = 1.0
  V₁(Moderate) = max(1.4, 1.0) = 1.4

State: High
  Q(A) = 0.5(1) + 0.5(1) = 1.0
  Q(B) = 1.0(1) = 1.0
  V₁(High) = max(1.0, 1.0) = 1.0

Answer: V = [2.0, 1.4, 1.0]
```

---

### Q2: MDP Formulation (7.5 marks)

**Q2a (2.5 marks):** Formulate MDP - State, Actions, Dynamics, Diagram
- **Practice Set Coverage:** Q22, Q26, Q27 ✅

**Q2b (2 marks):** Reward Design Analysis - Compare two designs
- **Practice Set Coverage:** Conceptual understanding from Q6, Q7 ✅

**Q2c (1.5 marks):** Episodic vs Continuing task
- **Practice Set Coverage:** Q28 ✅

**Q2d (1.5 marks):** Impact of low γ on learning
- **Practice Set Coverage:** Q23 (concept) ✅

---

### Q3: Multi-Armed Bandits (7.5 marks) ⭐⭐⭐

**Q3a (3 marks):** **Exponential Recency-Weighted Average with α = 0.5** ⭐
- Given: History table of (step, action, reward)
- Calculate: Q-values for all actions after each step
- **Practice Set Coverage:** Q13, Q17 ✅

**Actual Calculation from Exam:**
```
Formula: Q_new = Q_old + α(R - Q_old)  where α = 0.5

Initial: Q(S)=0, Q(T)=0, Q(R)=0, Q(C)=0

Step 1: Action S, Reward 7
  Q(S) = 0 + 0.5(7-0) = 3.5

Step 2: Action T, Reward 5
  Q(T) = 0 + 0.5(5-0) = 2.5

Step 3: Action R, Reward 6
  Q(R) = 0 + 0.5(6-0) = 3.0

Step 4: Action C, Reward 4
  Q(C) = 0 + 0.5(4-0) = 2.0

Step 5: Action S, Reward 8
  Q(S) = 3.5 + 0.5(8-3.5) = 3.5 + 2.25 = 5.75

... continue for all steps

Best Intervention: S (Value: 5.75)
```

**Q3b (1.5 marks):** Significance of α + What if α=1?
- α controls forgetting. α=1 means memoryless (only last reward matters)
- **Practice Set Coverage:** Q17 ✅

**Q3c (1 mark):** UCB confidence level meaning
- Controls exploration bonus
- **Practice Set Coverage:** Q16 ✅

**Q3d (2 marks):** ε-greedy analysis - identify exploration vs exploitation
- **Practice Set Coverage:** Q14, Q5 ✅

---

### Q4: MC Control + Returns (7.5 marks) ⭐⭐⭐

**Q4a (2 marks):** Why V(s) insufficient, need Q(s,a) in model-free
- **Practice Set Coverage:** Q75, Q71 ✅

**Q4b (2 marks):** Deterministic policy fails in First-Visit MC + Fix
- **Practice Set Coverage:** Q54, Q55 ✅

**Q4c (3.5 marks):** **Return Calculation + Q-table Update** ⭐
- Given: Episode trajectory with (state, action, reward, next_state)
- Calculate: Returns G_t for each (s,a) pair
- Update: Q-table with First-Visit MC
- **Practice Set Coverage:** Q53, Q57 ✅

**Actual Calculation from Exam:**
```
Episode: (s₀,a₁,2,s₀) → (s₀,a₃,0,s₁) → (s₁,a₂,3,s₁) → (s₁,a₂,-1,Term)
γ = 0.8

Returns (working backwards):
G₃ = -1 (terminal reward)
G₂ = 3 + 0.8(-1) = 3 - 0.8 = 2.2
G₁ = 0 + 0.8(3) + 0.8²(-1) = 0 + 2.4 - 0.64 = 1.76
G₀ = 2 + 0.8(0) + 0.8²(3) + 0.8³(-1) = 2 + 0 + 1.92 - 0.512 = 3.408

First-Visit Q-values:
Q(s₀, a₁) = 3.408  ← First visit at t=0
Q(s₀, a₃) = 1.76   ← First visit at t=1
Q(s₁, a₂) = 2.2    ← First visit at t=2 (ignore t=3, not first)

Greedy Policy:
π(s₀) = a₁ (highest Q)
π(s₁) = a₂ (highest Q)
```

---

## ✅ PRACTICE SET COVERAGE ANALYSIS

| Exam Question | Practice Set Coverage | Status |
|---------------|----------------------|--------|
| Q1a: RL vs SL + Reward vs Value | Q1, Q2, Q6 | ✅ Covered |
| Q1b: MAB vs MDP Classification | Q11, Q20, Q21 | ✅ Covered |
| Q1c: Value Iteration | Q42 | ✅ Covered (but practice!) |
| Q2a: MDP Formulation | Q22, Q26 | ✅ Covered |
| Q2b: Reward Design | Q6, Q7 | ✅ Conceptually covered |
| Q2c: Episodic vs Continuing | Q28 | ✅ Covered |
| Q2d: Impact of γ | Q23 | ✅ Covered |
| Q3a: α-update Q-values | Q13, Q17 | ✅ Covered |
| Q3b: α significance | Q17 | ✅ Covered |
| Q3c: UCB confidence | Q16 | ✅ Covered |
| Q3d: ε-greedy analysis | Q5, Q14 | ✅ Covered |
| Q4a: V(s) vs Q(s,a) | Q71, Q75 | ✅ Covered |
| Q4b: MC + Deterministic | Q54 | ✅ Covered |
| Q4c: Return + Q-table | Q53 | ✅ Covered |

### Verdict: **100% of exam topics are covered in the practice set!** ✅

---

## 🎯 SCORING BREAKDOWN FOR 80+

| Question | Easy Parts | Medium Parts | Hard Parts |
|----------|------------|--------------|------------|
| Q1 (7.5) | 1a (1.5) | 1b (2) | 1c (4) ⭐ |
| Q2 (7.5) | 2c (1.5), 2d (1.5) | 2a (2.5), 2b (2) | |
| Q3 (7.5) | 3b (1.5), 3c (1) | 3d (2) | 3a (3) ⭐ |
| Q4 (7.5) | 4a (2), 4b (2) | | 4c (3.5) ⭐ |

### High-Value Targets (⭐):
1. **Q1c: Value Iteration** - 4 marks
2. **Q3a: α-update Table** - 3 marks
3. **Q4c: Return + Q-table** - 3.5 marks

**Total from ⭐ parts: 10.5 marks = 35% of exam!**

---

## 📚 PRIORITIZED CONCEPTS FOR 4-HOUR STUDY

### 🔴 CRITICAL (Must Master - 60% of time)

#### 1. Value Iteration Calculation (1.5 hours)
```
V₁(s) = max_a Σ P(s'|s,a) × R(s,a,s')  [when V₀ = 0]
V_{k+1}(s) = max_a Σ P(s'|s,a) × [R + γV_k(s')]

Practice: Q42 from practice set
```

#### 2. Incremental Q-Update with α (1 hour)
```
Q_new = Q_old + α(R - Q_old)

For α = 0.5:
Q_new = Q_old + 0.5(R - Q_old)
      = 0.5 × Q_old + 0.5 × R

Practice: Q13, Q17 from practice set
```

#### 3. Return Calculation (Discounted) (1 hour)
```
Work BACKWARDS from terminal:
G_T = R_T (terminal reward)
G_{T-1} = R_{T-1} + γG_T
G_{T-2} = R_{T-2} + γG_{T-1}
...

Practice: Q23, Q53 from practice set
```

### 🟡 IMPORTANT (Know Well - 30% of time)

#### 4. ε-greedy Analysis (20 min)
- Greedy action: Highest Q-value
- If chose non-greedy → Exploration (random)
- If chose greedy → Could be exploitation OR random

#### 5. MAB vs MDP Classification (15 min)
- **MAB**: Stateless, actions don't affect future
- **MDP**: State exists, actions change state

#### 6. Why Q(s,a) not V(s) in Model-Free (10 min)
- Model-free: Don't know P(s'|s,a)
- V(s) tells value of state, not which action is best
- Q(s,a) directly compares actions

#### 7. Episodic vs Continuing (10 min)
- **Episodic**: Terminal state exists (games)
- **Continuing**: No end (temperature control)

### 🟢 GOOD TO KNOW (5-10 min each)

8. UCB confidence parameter controls exploration
9. α=1 means memoryless (only last reward)
10. Low γ makes agent myopic (short-sighted)
11. Exploring Starts or ε-soft to fix MC

---

## ⏰ 4-HOUR STUDY PLAN (Starting 2:00 PM)

```
2:00 PM - 3:30 PM: VALUE ITERATION (1.5 hours)
════════════════════════════════════════════════
├── 2:00-2:30: Formula understanding
│   V₁(s) = max_a Σ P(s'|s,a) × R(s,a,s')
│   
├── 2:30-3:00: Practice Q42 from practice set
│   
└── 3:00-3:30: Do the actual exam Q1c problem
    (States: Resting, Moderate, High)

3:30 PM - 4:30 PM: Q-UPDATE WITH α (1 hour)
════════════════════════════════════════════════
├── 3:30-3:50: Formula + Example
│   Q_new = Q_old + α(R - Q_old)
│   
├── 3:50-4:10: Practice Q13, Q17
│   
└── 4:10-4:30: Do actual exam Q3a
    (Build the 8-step table)

4:30 PM - 5:30 PM: RETURNS + MC (1 hour)
════════════════════════════════════════════════
├── 4:30-4:50: Return formula (work backwards!)
│   G_t = R_{t+1} + γG_{t+1}
│   
├── 4:50-5:10: First-Visit MC concept (Q53, Q54)
│   
└── 5:10-5:30: Do actual exam Q4c
    (Episode return calculation)

5:30 PM - 6:00 PM: QUICK CONCEPTS (30 min)
════════════════════════════════════════════════
├── MAB vs MDP classification
├── ε-greedy analysis
├── V(s) vs Q(s,a) explanation
├── Episodic vs Continuing
└── UCB confidence, α=1 meaning, low γ impact
```

---

## 📝 FORMULAS TO MEMORIZE

```
1. VALUE ITERATION:
   V_{k+1}(s) = max_a Σ P(s'|s,a) × [R + γV_k(s')]
   
   When V_k = 0:
   V_1(s) = max_a Σ P(s'|s,a) × R(s,a,s')

2. INCREMENTAL Q-UPDATE:
   Q_{n+1} = Q_n + α(R_n - Q_n)
   
   For α = 0.5:
   Q_new = 0.5 × Q_old + 0.5 × R

3. RETURN (DISCOUNTED):
   G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
   G_t = R_{t+1} + γG_{t+1}

4. ε-GREEDY PROBABILITIES:
   P(best) = 1 - ε + ε/|A|
   P(other) = ε/|A|

5. UCB:
   A = argmax_a [Q(a) + c√(ln(t)/N(a))]
```

---

## 🎯 80+ STRATEGY

| Part | Marks | Target | How |
|------|-------|--------|-----|
| Q1c (VI) | 4 | 4 | Master VI formula + calculation |
| Q3a (α-update) | 3 | 3 | Practice table building |
| Q4c (Returns) | 3.5 | 3 | Work backwards, First-Visit MC |
| Q1a, Q1b | 3.5 | 3 | Concepts |
| Q2 (MDP) | 7.5 | 6 | Formulation + concepts |
| Q3b,c,d | 4.5 | 4 | α, UCB, ε-greedy reasoning |
| Q4a, Q4b | 4 | 3.5 | V vs Q, MC with deterministic |
| **Total** | **30** | **26.5** | **88%** ✅ |

---

## ✅ KEY TAKEAWAYS

1. **Practice Set covers 100%** of the actual exam topics
2. **Biggest marks**: Value Iteration (4), Q-update table (3), Returns (3.5)
3. **Focus on calculations** - they're worth 10.5 marks (35%)
4. **Concepts are short** - 1-2 sentence answers expected
5. **Time management**: 18 min per question

**You CAN get 80+ with focused 4-hour prep on these specific topics!**
