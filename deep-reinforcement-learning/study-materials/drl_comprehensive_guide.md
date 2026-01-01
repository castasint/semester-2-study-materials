# 📚 Deep Reinforcement Learning - Comprehensive Study Guide

> **AIMLCZG512 | Midterm Prep | Sessions 1-8**
> **Exam: 4 Questions × 7.5% = 30% | Closed Book | 2 Hours**

---

# Table of Contents

1. [Session 1: Introduction to Reinforcement Learning](#session-1-introduction-to-reinforcement-learning)
2. [Session 2-3: Multi-Armed Bandits](#session-2-3-multi-armed-bandits)
3. [Session 3-5: Markov Decision Processes (MDP)](#session-3-5-markov-decision-processes)
4. [Session 4-5: Dynamic Programming](#session-4-5-dynamic-programming)
5. [Session 5-8: Monte Carlo Methods](#session-5-8-monte-carlo-methods)

---

# Session 1: Introduction to Reinforcement Learning

## 1.1 What is Reinforcement Learning?

```
┌─────────────────────────────────────────────────────────────┐
│           AGENT-ENVIRONMENT INTERACTION                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│         ┌──────────┐                                        │
│         │  AGENT   │                                        │
│         │(Learner) │                                        │
│         └────┬─────┘                                        │
│              │                                               │
│     Action aₜ│  ▲ State sₜ, Reward rₜ                       │
│              │  │                                            │
│              ▼  │                                            │
│      ┌───────────────┐                                      │
│      │  ENVIRONMENT  │                                      │
│      └───────────────┘                                      │
│                                                              │
│  At each time step t:                                       │
│    1. Agent observes state sₜ                               │
│    2. Agent takes action aₜ                                 │
│    3. Environment returns reward rₜ₊₁ and new state sₜ₊₁    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 1.2 Elements of Reinforcement Learning

| Element | Symbol | Description |
|---------|--------|-------------|
| **Policy** | π | Strategy the agent uses to select actions |
| **Reward Signal** | r | Immediate feedback from environment |
| **Value Function** | V(s), Q(s,a) | Long-term expected reward |
| **Model** | P, R | Agent's representation of environment (optional) |

## 1.3 RL vs Supervised vs Unsupervised Learning

| Aspect | Supervised | Unsupervised | RL |
|--------|------------|--------------|-----|
| **Feedback** | Labels | None | Rewards |
| **Goal** | Predict | Discover patterns | Maximize reward |
| **Data** | i.i.d samples | i.i.d samples | Sequential |
| **Exploration** | No | No | Yes |

## 1.4 Key Characteristics of RL

1. **Trial-and-error learning** - Learn from experience
2. **Delayed reward** - Actions affect future rewards
3. **Exploration vs Exploitation** - Balance trying new vs using known good
4. **No supervisor** - Only reward signal

---

# Session 2-3: Multi-Armed Bandits

## 2.1 The k-Armed Bandit Problem ⭐

```
Problem Setup:
─────────────
• You have k slot machines (arms)
• Each arm has unknown reward distribution
• At each step, you choose one arm
• Goal: Maximize total reward over time

Challenge: Exploration vs Exploitation
• Exploitation: Pull best known arm
• Exploration: Try other arms to learn more
```

## 2.2 Action-Value Methods

### True Action Value
$$q_*(a) = \mathbb{E}[R_t | A_t = a]$$

### Estimated Action Value (Sample Average)
$$Q_t(a) = \frac{\text{sum of rewards when } a \text{ taken}}{\text{number of times } a \text{ taken}} = \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{1}_{A_i=a}}{\sum_{i=1}^{t-1} \mathbb{1}_{A_i=a}}$$

## 2.3 Incremental Update Formula ⭐ VERY IMPORTANT

$$Q_{n+1} = Q_n + \frac{1}{n}[R_n - Q_n]$$

**General form:**
$$Q_{n+1} = Q_n + \alpha[R_n - Q_n]$$

Where:
- $\alpha = \frac{1}{n}$ for stationary problems
- $\alpha = $ constant (e.g., 0.1) for non-stationary problems

**Intuition:**
```
NewEstimate = OldEstimate + StepSize × [Target - OldEstimate]
                                        └─────── Error ──────┘
```

## 2.4 Non-Stationary Problems

For non-stationary environments (reward distributions change over time):

$$Q_{n+1} = Q_n + \alpha[R_n - Q_n]$$
$$= (1-\alpha)Q_n + \alpha R_n$$
$$= (1-\alpha)^n Q_1 + \sum_{i=1}^{n} \alpha(1-\alpha)^{n-i} R_i$$

**Effect:** Recent rewards have more weight (exponential recency-weighted average)

## 2.5 Action Selection Methods

### Greedy Selection
$$A_t = \arg\max_a Q_t(a)$$

### ε-Greedy Selection ⭐
```
With probability (1-ε): Select greedy action argmax_a Q(a)
With probability ε:     Select random action uniformly
```

**Trade-off:**
- Small ε → More exploitation
- Large ε → More exploration

### Optimistic Initial Values
Set $Q_1(a)$ high (e.g., +5) to encourage exploration initially.

### Upper Confidence Bound (UCB) ⭐
$$A_t = \arg\max_a \left[ Q_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}} \right]$$

Where:
- $N_t(a)$ = number of times action $a$ selected
- $c$ = exploration parameter
- $\sqrt{\frac{\ln t}{N_t(a)}}$ = uncertainty bonus

**Intuition:** Actions with fewer selections get higher bonus.

## 2.6 Worked Example: Incremental Q-Update

```
Given:
  Q(a) = 3.5 (current estimate)
  n = 5 (action taken 5 times)
  R = 4.0 (new reward)

Calculate Q(a) after update:
  α = 1/n = 1/5 = 0.2
  Error = R - Q(a) = 4.0 - 3.5 = 0.5
  
  Q_new(a) = Q(a) + α × Error
           = 3.5 + 0.2 × 0.5
           = 3.5 + 0.1
           = 3.6
```

---

# Session 3-5: Markov Decision Processes

## 3.1 MDP Definition ⭐ FUNDAMENTAL

An MDP is defined by the tuple $(S, A, P, R, \gamma)$:

| Component | Symbol | Description |
|-----------|--------|-------------|
| **States** | $S$ | Set of all possible states |
| **Actions** | $A$ | Set of all possible actions |
| **Transition Probability** | $P(s'|s,a)$ | Prob. of reaching $s'$ from $s$ via $a$ |
| **Reward Function** | $R(s,a,s')$ | Immediate reward |
| **Discount Factor** | $\gamma \in [0,1]$ | Weight for future rewards |

## 3.2 The Markov Property

$$P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)$$

**Meaning:** The future depends only on the present state, not on history.

## 3.3 Returns

### Episodic Tasks (with terminal state)
$$G_t = R_{t+1} + R_{t+2} + R_{t+3} + ... + R_T$$

### Continuing Tasks (infinite horizon)
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

### Recursive Form ⭐
$$G_t = R_{t+1} + \gamma G_{t+1}$$

## 3.4 Value Functions ⭐ VERY IMPORTANT

### State Value Function $V^\pi(s)$
$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s\right]$$

### Action Value Function $Q^\pi(s,a)$
$$Q^\pi(s,a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

### Relationship
$$V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s,a)$$
$$Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')$$

## 3.5 Bellman Expectation Equations ⭐ MUST MEMORIZE

### For State Value
$$V^\pi(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]$$

### For Action Value
$$Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \sum_{a'} \pi(a'|s') Q^\pi(s',a')$$

## 3.6 Optimal Value Functions

### Optimal State Value
$$V^*(s) = \max_\pi V^\pi(s) = \max_a Q^*(s,a)$$

### Optimal Action Value
$$Q^*(s,a) = \max_\pi Q^\pi(s,a)$$

### Bellman Optimality Equations ⭐ MUST MEMORIZE
$$V^*(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s') \right]$$

$$Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q^*(s',a')$$

## 3.7 Worked Example: Calculate V(s)

```
Simple 3-state MDP:
  States: s1, s2, s3 (terminal)
  γ = 0.9
  
  From s1: action a → s2 with reward 2
  From s2: action a → s3 with reward 5
  
Calculate V*(s1):
  V*(s3) = 0 (terminal state)
  V*(s2) = max_a [R(s2,a) + γV*(s3)]
         = 5 + 0.9 × 0 = 5
  V*(s1) = max_a [R(s1,a) + γV*(s2)]
         = 2 + 0.9 × 5 = 2 + 4.5 = 6.5
```

---

# Session 4-5: Dynamic Programming

## 4.1 Requirements for DP

- **Perfect model** of environment (P, R known)
- **Finite MDP** (finite states and actions)

## 4.2 Policy Evaluation (Prediction) ⭐

Compute $V^\pi$ for a given policy $\pi$.

**Iterative Algorithm:**
```
Initialize V(s) = 0 for all s
Repeat:
  Δ = 0
  For each s ∈ S:
    v = V(s)
    V(s) = Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
    Δ = max(Δ, |v - V(s)|)
Until Δ < θ (small threshold)
```

## 4.3 Policy Improvement

Given $V^\pi$, improve policy by acting greedily:

$$\pi'(s) = \arg\max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]$$

**Policy Improvement Theorem:**
If $\pi'(s) = \arg\max_a Q^\pi(s,a)$ for all $s$, then $V^{\pi'} \geq V^\pi$.

## 4.4 Policy Iteration ⭐

```
Algorithm: Policy Iteration
───────────────────────────
1. Initialize π arbitrarily

2. Policy Evaluation:
   Compute V^π using iterative policy evaluation

3. Policy Improvement:
   For each s:
     old_action = π(s)
     π(s) = argmax_a [R(s,a) + γ Σ_s' P(s'|s,a) V^π(s')]
   
4. If policy changed, go to step 2
   Else return π (optimal policy)
```

## 4.5 Value Iteration ⭐

Combines evaluation and improvement in single update:

```
Algorithm: Value Iteration
──────────────────────────
1. Initialize V(s) = 0 for all s

2. Repeat:
   Δ = 0
   For each s ∈ S:
     v = V(s)
     V(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
     Δ = max(Δ, |v - V(s)|)
   Until Δ < θ

3. Output optimal policy:
   π*(s) = argmax_a [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
```

## 4.6 Comparison: Policy Iteration vs Value Iteration

| Aspect | Policy Iteration | Value Iteration |
|--------|------------------|-----------------|
| Per iteration | Full policy evaluation | Single sweep |
| Convergence | Fewer iterations | More iterations |
| Per-step cost | Higher | Lower |
| Memory | Store V and π | Store V only |

## 4.7 Generalized Policy Iteration (GPI)

```
         Evaluation
      ┌──────────────┐
      │              │
      ▼              │
   Policy ────────► Value
      │              ▲
      │              │
      └──────────────┘
         Improvement
```

Key idea: Evaluation and improvement can be interleaved.

## 4.8 Worked Example: Value Iteration Step

```
Gridworld: 2×2 grid, goal at (2,2)
States: (1,1), (1,2), (2,1), (2,2)
Actions: up, down, left, right
γ = 0.9, R = -1 per step, R = 0 at goal

Current V:
  V(1,1) = 0, V(1,2) = 0, V(2,1) = 0, V(2,2) = 0

Update V(1,1):
  Action right → goes to (1,2): -1 + 0.9 × V(1,2) = -1 + 0 = -1
  Action down  → goes to (2,1): -1 + 0.9 × V(2,1) = -1 + 0 = -1
  Action up    → stays at (1,1): -1 + 0.9 × V(1,1) = -1 + 0 = -1
  Action left  → stays at (1,1): -1 + 0.9 × V(1,1) = -1 + 0 = -1
  
  V(1,1) = max(-1, -1, -1, -1) = -1

After many iterations, V converges to optimal values.
```

---

# Session 5-8: Monte Carlo Methods

## 5.1 Key Differences from DP

| Aspect | Dynamic Programming | Monte Carlo |
|--------|---------------------|-------------|
| Model | Required (P, R) | Not required |
| Learning | From model | From experience |
| Updates | Bootstrapping | Full episode returns |
| Applicability | Model-based | Model-free |

## 5.2 Monte Carlo Prediction ⭐

**Goal:** Estimate $V^\pi$ or $Q^\pi$ from sample episodes.

### First-Visit MC
```
For each episode:
  For each state s visited for FIRST time in episode:
    Append return G following first visit to Returns(s)
  V(s) = average(Returns(s))
```

### Every-Visit MC
```
For each episode:
  For each visit to state s:
    Append return G following visit to Returns(s)
  V(s) = average(Returns(s))
```

## 5.3 MC Estimation of Action Values

To estimate $Q(s,a)$:
- Need to visit all state-action pairs
- Problem: Some pairs may never be visited if policy is deterministic

**Solution: Exploring Starts**
- Start each episode at random (s,a) pair
- Ensures all pairs visited eventually

## 5.4 Monte Carlo Control ⭐

### MC with Exploring Starts
```
Algorithm: Monte Carlo ES
─────────────────────────
1. Initialize Q(s,a) arbitrarily, π(s) arbitrarily

2. Repeat:
   a. Generate episode starting from random (s₀,a₀)
   
   b. For each (s,a) in episode:
      G = return following (s,a)
      Append G to Returns(s,a)
      Q(s,a) = average(Returns(s,a))
   
   c. For each s in episode:
      π(s) = argmax_a Q(s,a)
```

### On-Policy MC Control (ε-soft) ⭐
```
Algorithm: On-Policy First-Visit MC Control
───────────────────────────────────────────
1. Initialize Q(s,a) arbitrarily
   Initialize π as ε-soft (ε-greedy)

2. Repeat:
   a. Generate episode using π
   
   b. For each (s,a) in episode:
      G = return following first visit to (s,a)
      Append G to Returns(s,a)
      Q(s,a) = average(Returns(s,a))
   
   c. For each s in episode:
      a* = argmax_a Q(s,a)
      For all a:
        if a = a*: π(a|s) = 1 - ε + ε/|A|
        else:      π(a|s) = ε/|A|
```

## 5.5 ε-Soft Policies

Policy where $\pi(a|s) \geq \frac{\varepsilon}{|A|}$ for all actions.

**ε-Greedy Policy:**
$$\pi(a|s) = \begin{cases} 1 - \varepsilon + \frac{\varepsilon}{|A|} & \text{if } a = a^* \\ \frac{\varepsilon}{|A|} & \text{otherwise} \end{cases}$$

Where $a^* = \arg\max_a Q(s,a)$

## 5.6 Calculating Returns ⭐

Given episode rewards: $R_1, R_2, ..., R_T$

**From end of episode:**
```
G_T = 0
G_{t} = R_{t+1} + γ G_{t+1}

Example: γ = 0.9, rewards = [1, 2, 3] (R₁=1, R₂=2, R₃=3)

G_3 = 0 (terminal)
G_2 = R_3 + γG_3 = 3 + 0.9×0 = 3
G_1 = R_2 + γG_2 = 2 + 0.9×3 = 2 + 2.7 = 4.7
G_0 = R_1 + γG_1 = 1 + 0.9×4.7 = 1 + 4.23 = 5.23
```

## 5.7 Worked Example: First-Visit MC

```
Episode: s1 → s2 → s1 → s3 (terminal)
Rewards: r1=2, r2=1, r3=4, γ = 1.0

Returns calculation:
  G from s3: 0 (terminal)
  G from s1 (second visit): 4 + 0 = 4
  G from s2: 1 + 4 = 5
  G from s1 (first visit): 2 + 5 = 7

First-Visit MC updates:
  V(s1) ← average including 7 (uses first visit only)
  V(s2) ← average including 5
  V(s3) = 0
```

---

# 📋 Formula Quick Reference

## Incremental Update
```
Q_{n+1} = Q_n + α[R_n - Q_n]
```

## Bellman Equations
```
V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V^π(s')]

V*(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V*(s')]

Q*(s,a) = R(s,a) + γ Σ_s' P(s'|s,a) max_a' Q*(s',a')
```

## UCB Action Selection
```
A_t = argmax_a [Q_t(a) + c√(ln t / N_t(a))]
```

## Return Calculation
```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
G_t = R_{t+1} + γG_{t+1}
```

## ε-Greedy Policy
```
π(a|s) = 1 - ε + ε/|A|  if a = argmax Q(s,a)
       = ε/|A|          otherwise
```

---

# 🎯 Exam Tips

1. **Bellman equations** - Will definitely appear
2. **Incremental Q update** - Know the formula cold
3. **Value iteration step** - Practice working through one iteration
4. **MC return calculation** - Compute G from episode end
5. **ε-greedy probabilities** - Calculate for given ε and |A|

---

**Focus on numerical problems - they give guaranteed marks!**
