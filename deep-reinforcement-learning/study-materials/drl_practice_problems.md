# 📝 DRL Practice Problems - Midterm Prep (Updated)

> **AIMLCZG512 | Pattern: Bandits, Gridworld, Modeling | Closed Book**

---

## SECTION 1: Multi-Armed Bandits (Session 2-3)

### Problem 1: UCB from History Table (Exam Pattern) ⭐

A news recommender system has 4 categories. The following table shows the history of the last 8 steps (1=Click, 0=No Click):

| Step | Category | Click? |
|------|----------|--------|
| 1 | Tech | 1 |
| 2 | Sports | 0 |
| 3 | Politics | 1 |
| 4 | Entertainment | 1 |
| 5 | Tech | 0 |
| 6 | Sports | 1 |
| 7 | Politics | 1 |
| 8 | Sports | 1 |

**Questions:**
a) Calculate Q-values for all categories based on this history. (2M)
b) Using UCB with $c=1.5$, identify the next action to select. Show calculations. (3M)
c) If using $\varepsilon$-greedy with $\varepsilon=0.2$, what is the probability of selecting 'Tech' at step 9? (2M)

---

**Solution:**

```
a) Calculate Q(a) = Total Rewards / Count
   
   Tech:    Returns=[1, 0] → Q = 1/2 = 0.5
   Sports:  Returns=[0, 1, 1] → Q = 2/3 ≈ 0.67
   Politics: Returns=[1, 1] → Q = 2/2 = 1.0
   Entertainment: Returns=[1] → Q = 1/1 = 1.0

b) UCB Calculation for Step 9 (t=9)
   Formula: UCB(a) = Q(a) + c × √(ln t / N(a))
   ln(9) ≈ 2.2
   
   Tech (N=2):    0.5 + 1.5 × √(2.2/2) = 0.5 + 1.5 × 1.05 = 2.07
   Sports (N=3):  0.67 + 1.5 × √(2.2/3) = 0.67 + 1.5 × 0.86 = 1.96
   Politics (N=2): 1.0 + 1.5 × √(2.2/2) = 1.0 + 1.5 × 1.05 = 2.57
   Entertainment (N=1): 1.0 + 1.5 × √(2.2/1) = 1.0 + 1.5 × 1.48 = 3.22
   
   Next action: Entertainment (Highest UCB = 3.22)

c) ε-Greedy Probability (ε=0.2)
   Best actions (tie): Politics, Entertainment (Q=1.0)
   Wait! Tie-breaking is random.
   
   P(exploit) = 1 - ε = 0.8
   P(explore) = ε = 0.2
   
   Is 'Tech' a greedy action? No (Q=0.5 < 1.0).
   So 'Tech' can only be chosen during exploration.
   
   P(Tech) = ε / |A| = 0.2 / 4 = 0.05
   
   (Note: If Tech was one of the best actions, logic would be different)
```

---

### Problem 2: Incremental Update

A 3-armed bandit has the following Q-values after 10 steps:
- Q(a₁) = 2.5 (selected 4 times)
- Q(a₂) = 3.0 (selected 5 times)  
- Q(a₃) = 1.8 (selected 1 time)

You select action a₂ and receive reward R = 4.5.

**Questions:**
a) Calculate the new Q(a₂) using sample average method. (2M)
b) Calculate the new Q(a₂) using constant step-size α = 0.1. (2M)

---

**Solution:**

```
a) Sample average method:
   n = 5 (times a₂ was selected)
   New n = 6
   α = 1/n = 1/6 ≈ 0.167
   
   Q_new(a₂) = Q(a₂) + α[R - Q(a₂)]
             = 3.0 + 0.167 × [4.5 - 3.0]
             = 3.25

b) Constant step-size α = 0.1:
   Q_new(a₂) = 3.0 + 0.1(1.5) = 3.15
```

---

## SECTION 2: MDP Modeling & Gridworld (Session 3-5)

### Problem 3: MDP Design (Exam Pattern) ⭐

An intelligent traffic controller manages a 4-way intersection. It observes:
- Traffic density on North, South, East, West roads (Low, Med, High)
- Pedestrian button status (Pressed, Not Pressed)

It can choose actions:
- A1: Allow N-S traffic
- A2: Allow E-W traffic
- A3: Allow Pedestrians

**Questions:**
a) Define the State Space and Action Space formally. (2M)
b) Design a Reward Function to minimize waiting time and ensure safety. (2M)
c) Is this suitable for Model-Based or Model-Free RL? Why? (1M)

---

**Solution:**

```
a) State & Action Space:
   State S = < D_N, D_S, D_E, D_W, Ped >
   Where D_i ∈ {Low, Med, High} and Ped ∈ {0, 1}
   Total states = 3 × 3 × 3 × 3 × 2 = 162 states
   
   Action Space A = {A1, A2, A3}

b) Reward Function:
   Goal: Minimize wait time, maximize safety.
   
   R = - (Total Vehicles Waiting) + (Safety Bonus)
   
   Example logic:
   - If A1 chosen: R = - (D_E + D_W)  [Penalty for waiting cars]
   - If A3 chosen when Ped=1: R = +50 [Safety bonus]
   - If A3 chosen when Ped=0: R = -10 [Unnecessary stop]

c) Algorithm Choice:
   Model-Free (like Q-Learning) is better because:
   - The environment dynamics (exact probability of traffic arrival) are complex and hard to model perfectly.
   - Easier to learn from experience (interactions).
```

---

### Problem 4: Gridworld Value Update (Numerical) ⭐

Consider a 3x3 grid world.
- Goal at (3,3) with Reward +10.
- All other moves give Reward -1.
- Discount γ = 0.9.
- Agent at (2,2). Deterministic moves.

**Questions:**
a) Write the specific Bellman Optimality Equation for V*(2,2). (2M)
b) Assume currently V(s)=0 for all states except V(3,3)=0. Perform one Value Iteration update for V(2,2). (3M)

---

**Solution:**

```
a) Bellman Equation for (2,2):
   V*(2,2) = max_a [ R(s,a) + γ V*(s') ]
   
   Specific expansion:
   V*(2,2) = max {
      R(N) + 0.9 V*(1,2),
      R(S) + 0.9 V*(3,2),
      R(E) + 0.9 V*(2,3),
      R(W) + 0.9 V*(2,1)
   }

b) Value Update (Iteration 1):
   R = -1 for all moves.
   Current V(s') are all 0.
   
   Move North → (1,2): -1 + 0.9(0) = -1
   Move South → (3,2): -1 + 0.9(0) = -1
   Move East  → (2,3): -1 + 0.9(0) = -1
   Move West  → (2,1): -1 + 0.9(0) = -1
   
   V_new(2,2) = max(-1, -1, -1, -1) = -1
   
   (Note: If V(3,3) was 10, then being next to it would give higher value. 
   Here we assumed V=0 init, so first update just propagates immediate reward.)
```

---

### Problem 5: Return Calculation

Episode: A → B → Terminal.
Rewards: R₁=2, R₂=5.
γ = 0.5.

**Calculate G₀ (return from A).**

---

**Solution:**

```
G_T = 0
G_B = R₂ + γ(0) = 5
G_A = R₁ + γ(G_B) = 2 + 0.5(5) = 2 + 2.5 = 4.5
```

---

## SECTION 3: Monte Carlo & Concepts

### Problem 6: ε-Soft Policy Update

After one episode, you have Q-values for state S:
- Q(a₁) = 5.0
- Q(a₂) = 2.0

Using ε = 0.4.

**Questions:**
a) Which action is greedy?
b) Calculate the new probabilities π(a₁|S) and π(a₂|S).

---

**Solution:**

```
a) Greedy action = a₁ (Max Q)

b) ε-Soft Probabilities:
   |A| = 2
   
   P(a₁) = 1 - ε + ε/|A|
         = 1 - 0.4 + 0.4/2
         = 0.6 + 0.2
         = 0.8
   
   P(a₂) = ε/|A|
         = 0.4/2
         = 0.2
   
   Check: 0.8 + 0.2 = 1.0 ✓
```

---

### Problem 7: Algorithms Conceptual

**Match the scenarios to the algorithm (DP, MC, TD):**

1.  "I have a perfect map of the maze."  -> (____)
2.  "I need to learn by walking through the maze, but I update my plan after every step." -> (____)
3.  "I learn only after finishing the whole maze run." -> (____)

---

**Solution:**

```
1. DP (Dynamic Programming) - Model-based, uses perfect map.
2. TD (Temporal Difference) - Model-free, bootstraps (updates every step).
3. MC (Monte Carlo) - Model-free, updates after full episode.
```

---

## 📊 Answer Key Summary

| Problem | Type | Key Answer |
|---------|------|------------|
| 1 | Bandit History | Q: 0.5, 0.67, 1.0, 1.0 → Next: Entertainment |
| 2 | Q Update | 3.25 (sample avg) |
| 3 | Modeling | States=162, R depends on wait time |
| 4 | Gridworld | V(2,2) = -1 (first iter) |
| 5 | Return | G = 4.5 |
| 6 | ε-Soft | P(Best) = 0.8 |
| 7 | Concepts | 1-DP, 2-TD, 3-MC |

---

**Good luck! 🍀**
