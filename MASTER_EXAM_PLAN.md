# 🎯 MASTER EXAM STUDY PLAN - JAN 4TH, 2025
## DRL + DNN + ML SysOps | Starting 2:00 PM

---

## 📅 EXAM SCHEDULE

| Time | Subject | Marks | Target |
|------|---------|-------|--------|
| **9:00 AM** | DRL (Deep Reinforcement Learning) | 30 | **20+** |
| **1:00 PM** | DNN (Deep Neural Networks) | 100→30 | **80+ / 24 scaled** |
| **4:30 PM** | ML System Optimization | 30 | **20+** |

---

## ⏰ STUDY SCHEDULE (Starting 2:00 PM, Jan 3rd)

```
📗 2:00 PM - 4:00 PM: DRL Part 1 (2 hours)
   └─ Value Iteration, Q-update with α

☕ 4:00 PM - 4:15 PM: BREAK (Snack + Walk)

📗 4:15 PM - 5:30 PM: DRL Part 2 (1.25 hours)
   └─ Returns, MC, Quick concepts

📘 5:30 PM - 7:00 PM: DNN Part 1 (1.5 hours)
   └─ Perceptron, Gradient Descent, Softmax

🍽️ 7:00 PM - 7:30 PM: DINNER BREAK

📘 7:30 PM - 8:30 PM: DNN Part 2 (1 hour)
   └─ DFNN Forward, Metrics, Code patterns

📙 8:30 PM - 9:30 PM: ML SysOps Part 1 (1 hour)
   └─ Amdahl's Law (CRITICAL - practice 5 problems)

☕ 9:30 PM - 9:45 PM: BREAK

📙 9:45 PM - 10:30 PM: ML SysOps Part 2 (45 min)
   └─ MapReduce, k-Means, Parameter Server

📋 10:30 PM - 11:00 PM: Final Formula Review

😴 11:00 PM - 5:30 AM: SLEEP (6.5 hours)

🌅 5:30 AM - 8:30 AM: Final revisions before DRL

📝 9:00 AM: DRL EXAM
📝 1:00 PM: DNN EXAM  
📝 4:30 PM: ML SysOps EXAM
```

---

# 🔴 DRL EXAM GUIDE (Target: 20+/30)

## Pattern (from Dec 2025 actual exam)
**4 Questions × 7.5 marks = 30 marks**

| Q# | Topic | Marks | Key Skill |
|----|-------|-------|-----------|
| Q1 | RL Basics + Value Iteration | 7.5 | Calculation |
| Q2 | MDP Formulation | 7.5 | Design + Concept |
| Q3 | MAB + Q-Update | 7.5 | Table calculation |
| Q4 | MC + Returns | 7.5 | Return calculation |

## HIGH-YIELD FORMULAS (Memorize!)

### 1. Value Iteration (4 marks likely) ⭐⭐⭐
```
V₁(s) = max_a Σ P(s'|s,a) × R(s,a,s')  [when V₀ = 0]

Example:
Q(Mode_A) = 0.6(1) + 0.4(2) = 1.4
Q(Mode_B) = 1.0(2) = 2.0
V(state) = max(1.4, 2.0) = 2.0
```

### 2. Incremental Q-Update (3 marks likely) ⭐⭐⭐
```
Q_new = Q_old + α(R - Q_old)

For α = 0.5:
Q = 3.5 + 0.5(8 - 3.5) = 3.5 + 2.25 = 5.75
```

### 3. Return Calculation (3.5 marks likely) ⭐⭐⭐
```
Work BACKWARDS from terminal:
G_t = R_{t+1} + γ × G_{t+1}

Example (γ=0.8):
G₂ = 3 + 0.8(-1) = 2.2
G₁ = 0 + 0.8(2.2) = 1.76
G₀ = 2 + 0.8(1.76) = 3.41
```

### 4. ε-Greedy Probabilities
```
P(best action) = 1 - ε + ε/|A|
P(other action) = ε/|A|
```

## Quick Concepts
- **MAB vs MDP**: MAB is stateless; MDP has state transitions
- **V(s) vs Q(s,a)**: Model-free needs Q(s,a) to compare actions
- **Episodic**: Has terminal state; Continuing: No end
- **α = 1**: Memoryless (only last reward matters)

---

# 🔵 DNN EXAM GUIDE (Target: 80+/100 = 24+/30 scaled)

## Pattern (from actual pattern document)
**5 Questions × 20 marks = 100 marks**

| Q# | Topic | Parts | Key Skill |
|----|-------|-------|-----------|
| Q1 | Perceptron | Calc(6) + Linear Sep(4) + Overfitting(5) + Code(5) |
| Q2 | Linear Regression | GD(6) + Code(5) + RMSE(6) + Compare(3) |
| Q3 | Binary Classification | Sigmoid(6) + Code(5) + Metrics(6) + Compare(3) |
| Q4 | Multi-class | Softmax(6) + Code(5) + Metrics(6) + Compare(3) |
| Q5 | DFNN | Forward(6) + Code(5) + Design(6) + Params(3) |

## HIGH-YIELD FORMULAS

### 1. Perceptron Update (6 marks) ⭐⭐⭐
```
z = w₀x₀ + w₁x₁ + w₂x₂
ŷ = sign(z)  →  +1 if z≥0, else -1
Δwᵢ = η(target - ŷ) × xᵢ
```

### 2. Gradient Descent (6 marks) ⭐⭐⭐
```
ŷ = Xw
e = ŷ - y
∇J = (1/N) × Xᵀ × e
w_new = w - η × ∇J
```

### 3. Sigmoid + BCE (6 marks) ⭐⭐⭐
```
ŷ = σ(z) = 1/(1 + e^(-z))
BCE = -[y log(ŷ) + (1-y) log(1-ŷ)]
Gradient = (ŷ - y) × x

Key values: σ(0)=0.5, σ(1)=0.73, σ(-1)=0.27
```

### 4. Softmax + CCE (6 marks) ⭐⭐⭐
```
ŷₖ = e^zₖ / Σe^zⱼ
CCE = -log(ŷ_true_class)

Key: e⁰=1, e¹=2.72, e²=7.39
```

### 5. DFNN Forward Pass (6 marks) ⭐⭐⭐
```
Layer 1: z⁽¹⁾ = xW⁽¹⁾ + b⁽¹⁾, h⁽¹⁾ = ReLU(z⁽¹⁾)
Layer 2: z⁽²⁾ = h⁽¹⁾W⁽²⁾ + b⁽²⁾, ŷ = σ(z⁽²⁾)
```

### 6. Parameter Count (3 marks)
```
Total = Σ nₗ(nₗ₋₁ + 1)
Example 100→64→32→3: 6464 + 2080 + 99 = 8643
```

### 7. Confusion Matrix Metrics (6 marks)
```
Accuracy = (TP + TN) / Total
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
```

## Code Blanks Pattern
```python
np.ones(...)     # Bias column
np.zeros(...)    # Initialize weights
np.dot(w, x)     # or w @ x - Weighted sum
1/(1+np.exp(-z)) # Sigmoid
np.maximum(0,z)  # ReLU
(y_pred - y)     # Error
```

---

# 🟢 ML SYSOPS EXAM GUIDE (Target: 20+/30)

## HIGH-YIELD FORMULAS

### 1. Amdahl's Law (8-10 marks - GUARANTEED!) ⭐⭐⭐
```
Speedup(p) = 1 / (f + (1-f)/p)
Max Speedup = 1/f

Where:
  f = serial fraction
  p = processors

Example: f=0.2, p=4
Speedup = 1/(0.2 + 0.8/4) = 1/(0.2+0.2) = 1/0.4 = 2.5
```

### Quick Reference
| f | Max Speedup | p=4 | p=8 |
|---|-------------|-----|-----|
| 0.1 | 10 | 3.08 | 4.71 |
| 0.2 | 5 | 2.50 | 3.33 |
| 0.25 | 4 | 2.29 | 2.91 |

### 2. MapReduce (5-6 marks) ⭐⭐
```python
# Word Count
MAP:    emit(word, 1)
REDUCE: return sum(values)

# Average
MAP:    emit(key, (value, 1))
REDUCE: return sum_values / sum_counts
```

### 3. k-Means Parallelization
- **ASSIGN phase**: Embarrassingly parallel (each processor handles subset)
- **UPDATE phase**: Local sums → Global reduce → New centers

### 4. Parameter Server
- Workers PULL parameters
- Workers compute LOCAL gradients
- Workers PUSH gradients
- Server AGGREGATES and UPDATES

---

## 📋 STUDY PRIORITY BY TIME

### 2:00-5:30 PM: DRL (3.5 hours)
| Concept | Time | Practice |
|---------|------|----------|
| Value Iteration | 1.5 hr | Practice Set Q42 |
| Q-update with α | 1 hr | Practice Set Q13, Q17 |
| Returns + MC | 1 hr | Practice Set Q53 |

### 5:30-8:30 PM: DNN (3 hours)
| Concept | Time | Practice |
|---------|------|----------|
| Perceptron table | 40 min | dnn_practice A1 |
| Softmax + CCE | 40 min | dnn_practice D1 |
| GD iteration | 30 min | dnn_practice B1 |
| DFNN forward | 30 min | dnn_practice E1 |
| Metrics | 20 min | dnn_practice C2 |
| Code patterns | 20 min | All B parts |

### 8:30-10:30 PM: ML SysOps (2 hours)
| Concept | Time | Practice |
|---------|------|----------|
| Amdahl's Law | 1 hr | 5 different calculations |
| MapReduce | 30 min | Word count, average |
| k-Means + PS | 30 min | Conceptual review |

---

## 🎯 SCORE TARGETS

| Exam | Target | Strategy |
|------|--------|----------|
| **DRL** | 20+/30 | Master 3 calculations (10.5 marks) |
| **DNN** | 80+/100 (24+/30) | All Part A + B + Metrics |
| **ML SysOps** | 20+/30 | Amdahl's Law alone = 10 marks |

---

## 📁 DOCUMENTS TO READ (In Order)

### 🔴 DRL Documents (Read during 2:00-5:30 PM)

| Priority | File | What to Focus On |
|----------|------|------------------|
| 1️⃣ | **`/sourcecode/DRL_DEC2025_EXAM_ANALYSIS.md`** | Actual exam pattern + solutions |
| 2️⃣ | `/deep-reinforcement-learning/study-materials/drl_practice_problems.md` | Q13, Q17, Q42, Q53 |
| 3️⃣ | `/deep-reinforcement-learning/study-materials/drl_formula_sheet.md` | Quick reference |
| 4️⃣ | `/deep-reinforcement-learning/DRL_regular_dec25_solved.pdf` | Actual exam solutions |

### 🔵 DNN Documents (Read during 5:30-8:30 PM)

| Priority | File | What to Focus On |
|----------|------|------------------|
| 1️⃣ | **`/sourcecode/DNN_80PLUS_COMPLETE_GUIDE.md`** | Complete guide with all solutions |
| 2️⃣ | `/deep-neural-networks/study-materials/dnn_practice_problems.md` | Problems A1, B1, C1, D1, E1 |
| 3️⃣ | `/deep-neural-networks/study-materials/dnn_formula_sheet.md` | Quick reference |

### 🟢 ML SysOps Documents (Read during 8:30-10:30 PM)

| Priority | File | What to Focus On |
|----------|------|------------------|
| 1️⃣ | **`/sourcecode/MLSO_EXAM_GUIDE_JAN4.md`** | Amdahl's Law examples |
| 2️⃣ | `/ml-sys-ops/study-materials/mlso_formula_sheet.md` | All formulas |
| 3️⃣ | `/ml-sys-ops/study-materials/mlso_practice_problems.md` | MapReduce examples |

---

## 🏃 QUICK START GUIDE

**Open these 3 files NOW:**
```
1. /sourcecode/MASTER_EXAM_PLAN.md      ← This file (overview)
2. /sourcecode/DRL_DEC2025_EXAM_ANALYSIS.md  ← Start DRL here
3. /deep-reinforcement-learning/study-materials/drl_practice_problems.md
```

---

**START NOW! Focus on calculations - they give guaranteed marks! 💪**
