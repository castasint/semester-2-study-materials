# 📝 DRL Practice Problems - Midterm Prep

> **AIMLCZG512 | 4 Questions × 7.5% = 30% | Closed Book**

---

## SECTION 1: Multi-Armed Bandits (Session 2-3)

### Problem 1: Incremental Update ⭐

A 3-armed bandit has the following Q-values after 10 steps:
- Q(a₁) = 2.5 (selected 4 times)
- Q(a₂) = 3.0 (selected 5 times)  
- Q(a₃) = 1.8 (selected 1 time)

You select action a₂ and receive reward R = 4.5.

**Questions:**
a) Calculate the new Q(a₂) using sample average method. (2M)
b) Calculate the new Q(a₂) using constant step-size α = 0.1. (2M)
c) Which method would you prefer for a non-stationary problem? Why? (1.5M)

---

**Solution:**

```
a) Sample average method:
   n = 5 (times a₂ was selected)
   New n = 6
   α = 1/n = 1/6 ≈ 0.167
   
   Q_new(a₂) = Q(a₂) + α[R - Q(a₂)]
             = 3.0 + 0.167 × [4.5 - 3.0]
             = 3.0 + 0.167 × 1.5
             = 3.0 + 0.25
             = 3.25

b) Constant step-size α = 0.1:
   Q_new(a₂) = Q(a₂) + α[R - Q(a₂)]
             = 3.0 + 0.1 × [4.5 - 3.0]
             = 3.0 + 0.1 × 1.5
             = 3.0 + 0.15
             = 3.15

c) Constant step-size (α = 0.1) is preferred because:
   - It gives more weight to recent rewards
   - Decaying α (1/n) treats all rewards equally
   - Non-stationary means reward distributions change
   - Need to "forget" old experience and adapt
```

---

### Problem 2: ε-Greedy Action Selection ⭐

Given Q-values for a 4-armed bandit:
- Q(a₁) = 2.0, Q(a₂) = 5.0, Q(a₃) = 3.5, Q(a₄) = 4.0

With ε = 0.2:

**Questions:**
a) What is the probability of selecting each action? (3M)
b) If you use UCB with c = 2, t = 100, and N(a) = [20, 30, 25, 25], which action is selected? (4.5M)

---

**Solution:**

```
a) ε-Greedy probabilities:
   Greedy action = a₂ (has max Q = 5.0)
   |A| = 4 actions
   
   P(a₂) = 1 - ε + ε/|A| = 1 - 0.2 + 0.2/4 = 0.8 + 0.05 = 0.85
   P(a₁) = ε/|A| = 0.2/4 = 0.05
   P(a₃) = ε/|A| = 0.2/4 = 0.05
   P(a₄) = ε/|A| = 0.2/4 = 0.05
   
   Verify: 0.85 + 0.05 + 0.05 + 0.05 = 1.0 ✓

b) UCB calculation:
   UCB(a) = Q(a) + c × √(ln t / N(a))
   
   ln(100) = 4.605
   
   UCB(a₁) = 2.0 + 2 × √(4.605/20) = 2.0 + 2 × √0.230 = 2.0 + 2 × 0.480 = 2.96
   UCB(a₂) = 5.0 + 2 × √(4.605/30) = 5.0 + 2 × √0.154 = 5.0 + 2 × 0.392 = 5.78
   UCB(a₃) = 3.5 + 2 × √(4.605/25) = 3.5 + 2 × √0.184 = 3.5 + 2 × 0.429 = 4.36
   UCB(a₄) = 4.0 + 2 × √(4.605/25) = 4.0 + 2 × √0.184 = 4.0 + 2 × 0.429 = 4.86
   
   Selected action = argmax UCB = a₂ (5.78)
```

---

### Problem 3: Non-Stationary Update

For a bandit with α = 0.2, the sequence of rewards for action a₁ is: [2, 4, 3, 5].
Initial Q₁(a₁) = 0.

**Calculate Q after each reward.** (6M)

---

**Solution:**

```
Q₁ = 0 (initial)

After R₁ = 2:
  Q₂ = Q₁ + α(R₁ - Q₁) = 0 + 0.2(2 - 0) = 0.4

After R₂ = 4:
  Q₃ = Q₂ + α(R₂ - Q₂) = 0.4 + 0.2(4 - 0.4) = 0.4 + 0.72 = 1.12

After R₃ = 3:
  Q₄ = Q₃ + α(R₃ - Q₃) = 1.12 + 0.2(3 - 1.12) = 1.12 + 0.376 = 1.496

After R₄ = 5:
  Q₅ = Q₄ + α(R₄ - Q₄) = 1.496 + 0.2(5 - 1.496) = 1.496 + 0.701 = 2.197

Final Q(a₁) = 2.197
```

---

## SECTION 2: Markov Decision Processes (Session 3-5)

### Problem 4: Bellman Equation Calculation ⭐

Consider a simple MDP:
- States: {s₁, s₂, s₃}, where s₃ is terminal
- Actions: {a}
- Transitions: s₁ --a--> s₂ (prob=1, reward=2), s₂ --a--> s₃ (prob=1, reward=5)
- γ = 0.9

**Questions:**
a) Write the Bellman equation for V*(s). (2M)
b) Calculate V*(s₁), V*(s₂), V*(s₃). (4M)
c) What is Q*(s₁, a)? (1.5M)

---

**Solution:**

```
a) Bellman Optimality Equation:
   V*(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V*(s')]

b) Calculate values (work backwards from terminal):
   
   V*(s₃) = 0 (terminal state)
   
   V*(s₂) = max_a [R(s₂,a) + γ × P(s₃|s₂,a) × V*(s₃)]
          = 5 + 0.9 × 1 × 0
          = 5
   
   V*(s₁) = max_a [R(s₁,a) + γ × P(s₂|s₁,a) × V*(s₂)]
          = 2 + 0.9 × 1 × 5
          = 2 + 4.5
          = 6.5

c) Q*(s₁, a) = R(s₁,a) + γ × P(s₂|s₁,a) × V*(s₂)
             = 2 + 0.9 × 1 × 5
             = 6.5
   
   Note: Q*(s,a) = V*(s) when there's only one action
```

---

### Problem 5: Stochastic MDP ⭐

MDP with states {A, B, C} where C is terminal.
From state A with action a:
- P(B|A,a) = 0.7, R = 3
- P(C|A,a) = 0.3, R = 10

γ = 0.95, V(B) = 8, V(C) = 0

**Calculate V(A).** (5M)

---

**Solution:**

```
Using Bellman equation:
V(A) = max_a [Σ_s' P(s'|A,a) × (R(A,a,s') + γ V(s'))]

With only one action a:
V(A) = P(B|A,a) × (R_AB + γ V(B)) + P(C|A,a) × (R_AC + γ V(C))
     = 0.7 × (3 + 0.95 × 8) + 0.3 × (10 + 0.95 × 0)
     = 0.7 × (3 + 7.6) + 0.3 × (10 + 0)
     = 0.7 × 10.6 + 0.3 × 10
     = 7.42 + 3.0
     = 10.42
```

---

### Problem 6: Return Calculation ⭐

An episode generates the following sequence:
- s₀ → s₁ (r₁ = 1) → s₂ (r₂ = 3) → s₃ (r₃ = 2) → Terminal (r₄ = 10)

With γ = 0.9:

**Questions:**
a) Calculate G₀ (return from s₀). (3M)
b) Calculate G₂ (return from s₂). (2M)
c) If this is the only episode and we use first-visit MC, what is V(s₁)? (2.5M)

---

**Solution:**

```
a) Calculate returns from end:
   G₄ = 0 (terminal)
   G₃ = r₄ + γG₄ = 10 + 0.9×0 = 10
   G₂ = r₃ + γG₃ = 2 + 0.9×10 = 2 + 9 = 11
   G₁ = r₂ + γG₂ = 3 + 0.9×11 = 3 + 9.9 = 12.9
   G₀ = r₁ + γG₁ = 1 + 0.9×12.9 = 1 + 11.61 = 12.61

b) G₂ = 11 (calculated above)

c) First-visit MC:
   V(s₁) = average of returns from first visits to s₁
   
   s₁ is visited once at step 1
   Return from s₁ = G₁ = 12.9
   
   With only one episode:
   V(s₁) = 12.9
```

---

## SECTION 3: Dynamic Programming (Session 4-5)

### Problem 7: Value Iteration ⭐

2×2 Gridworld:
```
┌───┬───┐
│ A │ B │
├───┼───┤
│ C │ G │
└───┴───┘
```
- G is goal (terminal, reward = 0)
- All other transitions: reward = -1
- Actions: up, down, left, right
- If action leads outside grid, stay in place
- γ = 1.0

**Perform one iteration of Value Iteration starting from V(s) = 0 for all s.** (7.5M)

---

**Solution:**

```
Initial: V(A) = V(B) = V(C) = V(G) = 0

For state A:
  up:    → A, R = -1, V' = 0 → -1 + 1.0×0 = -1
  down:  → C, R = -1, V' = 0 → -1 + 1.0×0 = -1
  left:  → A, R = -1, V' = 0 → -1 + 1.0×0 = -1
  right: → B, R = -1, V' = 0 → -1 + 1.0×0 = -1
  V(A) = max(-1, -1, -1, -1) = -1

For state B:
  up:    → B, R = -1, V' = 0 → -1
  down:  → G, R = 0, V' = 0 → 0 + 0 = 0  ← Can reach goal!
  left:  → A, R = -1, V' = 0 → -1
  right: → B, R = -1, V' = 0 → -1
  V(B) = max(-1, 0, -1, -1) = 0

For state C:
  up:    → A, R = -1, V' = 0 → -1
  down:  → C, R = -1, V' = 0 → -1
  left:  → C, R = -1, V' = 0 → -1
  right: → G, R = 0, V' = 0 → 0  ← Can reach goal!
  V(C) = max(-1, -1, -1, 0) = 0

V(G) = 0 (terminal)

After 1 iteration:
V(A) = -1, V(B) = 0, V(C) = 0, V(G) = 0
```

---

### Problem 8: Policy Evaluation

Given policy π that always moves right, for the same gridworld above.
Calculate V^π(A) after one iteration, starting from V(s) = 0. (5M)

---

**Solution:**

```
Policy π: always move right

State transitions under π:
  A → B (R = -1)
  B → B (hits wall, stays, R = -1)
  C → G (R = 0)
  G is terminal

Policy Evaluation update:
V^π(s) = R(s, π(s)) + γ × V^π(s')

After iteration 1 (starting from all 0):
  V^π(A) = -1 + 1.0 × V(B) = -1 + 0 = -1
  V^π(B) = -1 + 1.0 × V(B) = -1 + 0 = -1
  V^π(C) = 0 + 1.0 × V(G) = 0 + 0 = 0
  
Note: B doesn't reach goal under this policy, so V(B) will stay negative.
```

---

## SECTION 4: Monte Carlo Methods (Session 6-8)

### Problem 9: First-Visit vs Every-Visit MC ⭐

Episode: A → B → A → B → C (terminal)
Rewards: r₁=1, r₂=2, r₃=3, r₄=4
γ = 1.0

**Questions:**
a) Calculate returns for each visit to each state. (3M)
b) What values do first-visit MC assign? (2M)
c) What values do every-visit MC assign? (2.5M)

---

**Solution:**

```
Sequence: A(t=0) → B(t=1) → A(t=2) → B(t=3) → C(terminal)
Rewards after each transition: r₁=1, r₂=2, r₃=3, r₄=4

a) Returns (γ = 1.0):
   G from C: 0
   G from B(t=3): 4 + 0 = 4
   G from A(t=2): 3 + 4 = 7
   G from B(t=1): 2 + 7 = 9
   G from A(t=0): 1 + 9 = 10

b) First-visit MC (only first occurrence):
   V(A) = G from first visit = 10
   V(B) = G from first visit = 9
   V(C) = 0

c) Every-visit MC (average all visits):
   V(A) = average(10, 7) = 8.5
   V(B) = average(9, 4) = 6.5
   V(C) = 0
```

---

### Problem 10: MC Control with ε-soft Policy

After one episode, you have:
- Q(s₁, a₁) = 3.0
- Q(s₁, a₂) = 5.0
- Q(s₁, a₃) = 2.0

Using ε = 0.3, update the ε-soft policy for state s₁. (5M)

---

**Solution:**

```
|A| = 3 actions
ε = 0.3

Greedy action = a₂ (max Q = 5.0)

Updated policy π(a|s₁):
  π(a₂|s₁) = 1 - ε + ε/|A| = 1 - 0.3 + 0.3/3 = 0.7 + 0.1 = 0.8
  π(a₁|s₁) = ε/|A| = 0.3/3 = 0.1
  π(a₃|s₁) = ε/|A| = 0.3/3 = 0.1

Verify: 0.8 + 0.1 + 0.1 = 1.0 ✓
```

---

### Problem 11: Incremental MC Update

Using incremental MC update with α = 0.1:
- Current Q(s,a) = 4.5
- New return G = 7.0

**Calculate new Q(s,a).** (3M)

---

**Solution:**

```
Incremental update formula:
Q(s,a) ← Q(s,a) + α[G - Q(s,a)]

Q_new(s,a) = 4.5 + 0.1 × (7.0 - 4.5)
           = 4.5 + 0.1 × 2.5
           = 4.5 + 0.25
           = 4.75
```

---

## SECTION 5: Mixed Problems

### Problem 12: Conceptual Questions

a) What is the Markov property? Why is it important? (2M)
b) Difference between on-policy and off-policy learning? (2M)
c) Why do we need exploring starts in MC control? (1.5M)
d) What happens if γ = 0? γ = 1? (2M)

---

**Solution:**

```
a) Markov Property:
   The future depends only on current state, not history.
   P(s_{t+1}|s_t, a_t, s_{t-1},...) = P(s_{t+1}|s_t, a_t)
   
   Importance:
   - Allows recursive value function definitions
   - Enables Bellman equations
   - Makes DP and MC methods tractable

b) On-policy vs Off-policy:
   On-policy: Learn about policy currently being used
              (e.g., SARSA, on-policy MC)
   Off-policy: Learn about different policy than one being used
              (e.g., Q-learning, importance sampling MC)

c) Exploring Starts:
   Needed to guarantee all (s,a) pairs are visited.
   Without it, deterministic policy may never explore some actions.
   Ensures convergence to optimal Q-values.

d) Discount factor effects:
   γ = 0: Only immediate reward matters (myopic)
          G_t = R_{t+1}
   γ = 1: All future rewards equally important
          May diverge for non-episodic tasks
          Treats $1 today = $1 in 100 years
```

---

### Problem 13: Compare Algorithms

Fill in the table: (6M)

| Aspect | DP | MC | TD |
|--------|----|----|-----|
| Requires model? | | | |
| Bootstraps? | | | |
| Works with episodes only? | | | |

---

**Solution:**

```
| Aspect | DP | MC | TD |
|--------|----|----|-----|
| Requires model? | Yes | No | No |
| Bootstraps? | Yes | No | Yes |
| Works with episodes only? | No | Yes | No |

Explanation:
- DP: Needs P(s'|s,a) and R - full model
- MC: Learns from complete episode returns
- TD: Learns from partial episodes, bootstraps from estimates
```

---

## 📊 Answer Key Summary

| Problem | Topic | Key Answer |
|---------|-------|------------|
| 1 | Incremental update | Q_new = 3.25 (sample avg), 3.15 (α=0.1) |
| 2 | ε-greedy, UCB | P(a₂) = 0.85; UCB selects a₂ |
| 3 | Non-stationary | Final Q = 2.197 |
| 4 | Bellman | V*(s₁) = 6.5 |
| 5 | Stochastic MDP | V(A) = 10.42 |
| 6 | Returns | G₀ = 12.61, G₂ = 11 |
| 7 | Value iteration | V(A)=-1, V(B)=0, V(C)=0 |
| 8 | Policy evaluation | V^π(A) = -1 |
| 9 | First/Every visit MC | FV: V(A)=10; EV: V(A)=8.5 |
| 10 | ε-soft policy | π(a₂)=0.8, π(a₁)=π(a₃)=0.1 |
| 11 | Incremental MC | Q_new = 4.75 |
| 12 | Concepts | Markov, on/off policy, γ effects |
| 13 | Algorithm comparison | See table |

---

**Good luck! 🍀**
