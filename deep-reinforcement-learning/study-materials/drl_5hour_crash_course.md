# 🚀 DRL 5-Hour Crash Course

> **AIMLCZG512 | Exam: 4 Questions × 7.5% | Closed Book | 2 Hours**

---

## ⏱️ TIME ALLOCATION

| Hour | Topic | Weight |
|------|-------|--------|
| **1** | Multi-Armed Bandits | ~25% |
| **2** | MDP & Bellman Equations | ~30% |
| **3** | Dynamic Programming | ~25% |
| **4** | Monte Carlo Methods | ~20% |
| **5** | Practice Problems | Review |

---

# HOUR 1: Multi-Armed Bandits

## 🎯 Key Formula #1: Incremental Update

```
Q_{n+1} = Q_n + α [R_n - Q_n]
        = Q_n + α × Error
```

**Practice right now:**
```
Q = 3.0, R = 5.0, α = 0.1
Q_new = 3.0 + 0.1(5.0 - 3.0) = 3.0 + 0.2 = 3.2
```

## 🎯 Key Formula #2: ε-Greedy

```
|A| = number of actions
Best action:  P = 1 - ε + ε/|A|
Other action: P = ε/|A|
```

**Practice:**
```
ε = 0.2, |A| = 4
P(best) = 1 - 0.2 + 0.2/4 = 0.8 + 0.05 = 0.85
P(other) = 0.2/4 = 0.05
```

## 🎯 Key Formula #3: UCB

```
A = argmax [ Q(a) + c√(ln t / N(a)) ]
```

## ✅ Hour 1 Checkpoint
- [ ] Can do incremental Q update
- [ ] Can calculate ε-greedy probabilities
- [ ] Know α = 1/n vs constant α difference

---

# HOUR 2: MDP & Bellman Equations

## 🎯 MDP = (S, A, P, R, γ)

```
S = States
A = Actions
P(s'|s,a) = Transition probability
R = Reward
γ = Discount factor
```

## 🎯 Return Calculation (Work Backwards!)

```
Given rewards [r₁, r₂, r₃], γ = 0.9:

G₃ = 0 (terminal)
G₂ = r₃ + γ×G₃ = r₃
G₁ = r₂ + γ×G₂
G₀ = r₁ + γ×G₁
```

**Practice:**
```
Rewards: [1, 2, 3], γ = 0.9

G₃ = 0
G₂ = 3 + 0.9(0) = 3
G₁ = 2 + 0.9(3) = 2 + 2.7 = 4.7
G₀ = 1 + 0.9(4.7) = 1 + 4.23 = 5.23
```

## 🎯 BELLMAN EQUATIONS (MEMORIZE!)

### Bellman Optimality for V*:
```
V*(s) = max_a [ R(s,a) + γ Σ_s' P(s'|s,a) V*(s') ]
```

### Bellman Optimality for Q*:
```
Q*(s,a) = R(s,a) + γ Σ_s' P(s'|s,a) max_a' Q*(s',a')
```

## ✅ Hour 2 Checkpoint
- [ ] Can calculate returns from rewards
- [ ] Can write Bellman equation
- [ ] Know V*(s) = max Q*(s,a)

---

# HOUR 3: Dynamic Programming

## 🎯 Value Iteration (Most Important!)

```
Repeat:
  For each state s:
    V(s) ← max_a [ R(s,a) + γ Σ_s' P(s'|s,a) V(s') ]
```

**Example - One Step:**
```
2×2 Grid, goal at (2,2), R = -1 per step, γ = 1.0

Initial V = [0, 0, 0, 0]

V(1,1) = max of:
  right: -1 + V(1,2) = -1
  down:  -1 + V(2,1) = -1
  = -1

V(1,2) = max of:
  down → goal: 0 + V(goal) = 0  ← Best!
  = 0
```

## 🎯 Policy Iteration

```
1. Evaluate policy: compute V^π
2. Improve policy: π(s) = argmax_a Q(s,a)
3. Repeat until stable
```

## ✅ Hour 3 Checkpoint
- [ ] Can do one value iteration step
- [ ] Understand evaluate → improve cycle
- [ ] Know difference: value iter vs policy iter

---

# HOUR 4: Monte Carlo Methods

## 🎯 Key Concept: Learn from Episodes

```
Episode: s₀ → s₁ → s₂ → ... → Terminal
Returns: G_t = sum of discounted future rewards
```

## 🎯 First-Visit vs Every-Visit

```
First-Visit MC: Use return from FIRST visit to state
Every-Visit MC: Use returns from ALL visits
```

**Example:**
```
Episode: A → B → A → C
Returns: G(A first)=10, G(A second)=7

First-Visit:  V(A) = 10
Every-Visit:  V(A) = (10+7)/2 = 8.5
```

## 🎯 ε-Soft Policy Update

```
After computing Q(s,a):

a* = argmax Q(s,a)

π(a*|s) = 1 - ε + ε/|A|
π(other|s) = ε/|A|
```

## ✅ Hour 4 Checkpoint
- [ ] Know first-visit vs every-visit difference
- [ ] Can update ε-soft policy
- [ ] Understand: MC needs complete episodes

---

# HOUR 5: Practice & Review

## 📝 Must-Do Problems

### Problem 1: Incremental Update
```
Q = 4.0, α = 0.2, R = 6.0
Q_new = ?

Answer: 4.0 + 0.2(6.0 - 4.0) = 4.0 + 0.4 = 4.4
```

### Problem 2: Return Calculation
```
Rewards: [2, 3, 5], γ = 0.9
G₀ = ?

G₂ = 5
G₁ = 3 + 0.9(5) = 7.5
G₀ = 2 + 0.9(7.5) = 8.75
```

### Problem 3: ε-Greedy
```
ε = 0.3, |A| = 5, best action = a₂
P(a₂) = ?

P(a₂) = 1 - 0.3 + 0.3/5 = 0.7 + 0.06 = 0.76
```

### Problem 4: Bellman Equation
```
V(s₂) = 10, γ = 0.9
s₁ → s₂ with R = 5
V(s₁) = ?

V(s₁) = 5 + 0.9(10) = 14
```

### Problem 5: Value Iteration
```
From state A:
  action a₁ → B with R=2, V(B)=5, γ=0.9
  action a₂ → C with R=1, V(C)=8, γ=0.9

V(A) = max(2 + 0.9×5, 1 + 0.9×8)
     = max(6.5, 8.2)
     = 8.2
```

---

## 📋 FINAL CHECKLIST

### Formulas to Memorize:
```
1. Q_{n+1} = Q_n + α(R - Q_n)

2. V*(s) = max_a [R(s,a) + γΣP(s'|s,a)V*(s')]

3. G_t = R_{t+1} + γG_{t+1}

4. P(best) = 1 - ε + ε/|A|
```

### Concepts to Know:
- [ ] Exploration vs Exploitation
- [ ] Markov Property
- [ ] Why γ < 1 for infinite horizons
- [ ] On-policy vs Off-policy (MC only)
- [ ] Model-based (DP) vs Model-free (MC)

---

## 🎯 EXAM STRATEGY

1. **Read all questions first** - 4 questions
2. **Do numerical problems first** - guaranteed marks
3. **Show all steps** - partial credit
4. **Write formulas** even if stuck on answer
5. **Time: 30min per question**

---

## 📊 Quick Reference Values

```
Common γ values: 0.9, 0.95, 0.99, 1.0
Common ε values: 0.1, 0.2, 0.3
Common α values: 0.1, 0.2, 1/n

ln(10) ≈ 2.3
ln(100) ≈ 4.6
√2 ≈ 1.41
√3 ≈ 1.73
```

---

**You've got this! 💪**
