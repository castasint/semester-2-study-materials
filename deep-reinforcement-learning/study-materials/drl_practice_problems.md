# 📝 DRL Practice Problems - Midterm Prep (Updated V2)

> **AIMLCZG512 | Patterns: History Table, Modeling, Diagram VI | Closed Book**

---

## SECTION 1: Multi-Armed Bandits (Session 2-3)

### Problem 1: UCB from History Table (QP-1 Pattern) ⭐

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

### Problem 2: Modeling Non-Stationary Bandits (QP-2 Pattern) ⭐

An eCommerce platform shows one of 3 brands (Adidas, Nike, Sketchers) to users. User preferences change over time due to market trends.

**Questions:**
a) Why is this better modeled as a Non-Stationary Bandit rather than a standard MAB? (2M)
b) Which algorithm is better suited: Sample Average or Constant Step-Size ($\alpha$)? Why? (1.5M)
c) List two hyperparameters involved and their impact. (1.5M)

**Solution:**

```
a) Non-Stationary because reward probabilities change over time. 
   Standard MAB assumes fixed distributions. A brand popular today might not be popular next month.
   
b) Constant Step-Size (α) is better.
   Sample average (1/n) gives equal weight to all past rewards.
   Constant α gives more weight to recent rewards (exponential decay), allowing the agent to adapt to trends.

c) Hyperparameters:
   1. Epsilon (ε): Controls exploration. Higher ε adapts faster but exploits less.
   2. Step-size (α): Controls learning rate. Higher α forgets old history faster.
```

---

## SECTION 2: MDP Modeling & Gridworld (Session 3-5)

### Problem 3: MDP Design (QP-1 & QP-2 Pattern) ⭐

A robot delivers parcels in a grid town. It receives +10 for delivery, -10 for wrong delivery, and -1 per step.

**Questions:**
a) Define the State Space formally. (Includes what?)
b) Define the structure of an Episode for this task.
c) Calculate return for: 3 deliveries in 5 steps (total). γ=0.9.

**Solution:**

```
a) State S = (Robot_Position, Parcel_Status)
   Need to track where robot is AND which parcels are picked/delivered.

b) Episode: Starts when robot leaves depot, ends when ALL parcels delivered.

c) Return Calculation:
   Rewards: -1, -1, +10(del), -1, +10(del), +10(del) ... (example sequence)
   Total Undiscounted: ΣR
   Total Discounted: Σ γ^t R_t
```

---

### Problem 4: Diagram-based Value Iteration (QP-2 Pattern) ⭐

Consider this system with 3 states (S1, S2, S3) and 2 actions (Roll, Wait).
Transitions shown in diagram (simplified text version):
- S1 --Roll (0.5, R=1)--> S2
- S1 --Roll (0.5, R=3)--> S3
- S1 --Wait (1.0, R=0)--> S1
- (Assume similar transitions for others)

**Task:**
Perform 2 iterations of Value Iteration starting from V=0. γ=0.5.

**Solution Approach:**

```
Iteration 1 (V=0 everywhere):
V_new(S1) = max_a [ R(s,a) + γ Σ P V_old ]
Since V_old=0, this reduces to max expected immediate reward.

Q(S1, Wait) = 0 + 0 = 0
Q(S1, Roll) = (0.5×1 + 0.5×3) + 0 = 0.5 + 1.5 = 2.0
V(S1) = 2.0

Iteration 2:
Use V=[2.0, ... ] to compute next values.
```

---

### Problem 5: Gridworld Value Update with Notation (QP-1 Pattern) ⭐

Consider a 3x3 grid world.
- Goal at (3,3) with Reward +10.
- All other moves give Reward -1.
- Discount γ = 0.9.
- Agent at (2,2). Deterministic moves.

**Questions:**
a) Write the specific Bellman Optimality Equation for V*(2,2). (2M)
b) Assume currently V(s)=0 for all states except V(3,3)=0. Perform one Value Iteration update for V(2,2). (3M)

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
   V_new(2,2) = max(-1, -1, -1, -1) = -1
```

---

## SECTION 3: Monte Carlo & Return Calculation

### Problem 6: Return Calculation (Discounted vs Undiscounted)

Detailed in Problem 3c.
Key Concept: Discouting reduces value of future rewards. If γ is small, agent prefers immediate rewards (mypoic). If γ is large, agent plans for long term.

---

### Problem 7: ε-Soft Policy Update

After one episode, you have Q-values for state S:
- Q(a₁) = 5.0
- Q(a₂) = 2.0

Using ε = 0.4.

**Questions:**
a) Which action is greedy?
b) Calculate the new probabilities π(a₁|S) and π(a₂|S).

**Solution:**

```
a) Greedy action = a₁ (Max Q)

b) ε-Soft Probabilities:
   |A| = 2
   
   P(a₁) = 1 - ε + ε/|A|
         = 1 - 0.4 + 0.4/2
         = 0.8
   
   P(a₂) = ε/|A|
         = 0.2
```

---

**Good luck! 🍀**
