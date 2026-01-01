# 🎮 Deep Reinforcement Learning - Crash Course

> **5-Hour Crash Course | For Midterm Exam | Core Concepts**

---

## 📚 KEY TOPICS FOR EXAM

1. MDP Fundamentals
2. Bellman Equations
3. Value Iteration & Policy Iteration
4. Q-Learning & SARSA
5. Deep Q-Networks (DQN)
6. Policy Gradient Methods

---

# HOUR 1: MDP Fundamentals

## 1.1 Markov Decision Process (MDP)

An MDP is defined by the tuple: **(S, A, P, R, γ)**

| Component | Symbol | Description |
|-----------|--------|-------------|
| **States** | S | Set of all possible states |
| **Actions** | A | Set of all possible actions |
| **Transition** | P(s'|s,a) | Probability of reaching s' from s via action a |
| **Reward** | R(s,a,s') | Immediate reward for transition |
| **Discount** | γ ∈ [0,1] | Future reward discount factor |

## 1.2 Key Definitions

### Policy (π)
```
π(a|s) = Probability of taking action a in state s
```

### State Value Function V^π(s)
```
V^π(s) = Expected total discounted reward starting from s, following π

V^π(s) = E[R₀ + γR₁ + γ²R₂ + ... | s₀ = s, π]
       = E[Σ γᵗRₜ | s₀ = s, π]
```

### Action Value Function Q^π(s,a)
```
Q^π(s,a) = Expected total reward starting from s, taking a, then following π

Q^π(s,a) = E[Σ γᵗRₜ | s₀ = s, a₀ = a, π]
```

### Relationship
```
V^π(s) = Σ_a π(a|s) Q^π(s,a)

Q^π(s,a) = R(s,a) + γ Σ_s' P(s'|s,a) V^π(s')
```

## 1.3 Discount Factor (γ)

| γ Value | Meaning |
|---------|---------|
| γ = 0 | Only care about immediate reward |
| γ = 1 | Future = Present (may diverge for infinite horizon) |
| γ = 0.9 | Standard value, balance short/long term |

---

# HOUR 2: Bellman Equations ⭐ MOST IMPORTANT

## 2.1 Bellman Expectation Equation

### For V^π(s):
```
V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V^π(s')]
```

### For Q^π(s,a):
```
Q^π(s,a) = R(s,a) + γ Σ_s' P(s'|s,a) Σ_a' π(a'|s') Q^π(s',a')
```

## 2.2 Bellman Optimality Equation

### For V*(s):
```
V*(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V*(s')]
```

### For Q*(s,a):
```
Q*(s,a) = R(s,a) + γ Σ_s' P(s'|s,a) max_a' Q*(s',a')
```

## 2.3 Worked Example: Gridworld

```
Simple 3-state MDP:
  S = {s1, s2, s3}
  A = {left, right}
  γ = 0.9

Transitions (deterministic):
  s1 --right--> s2   (reward = 0)
  s2 --right--> s3   (reward = +10)
  s3 is terminal

Calculate V*(s1):
  V*(s3) = 0 (terminal)
  V*(s2) = max[R(s2,right) + γV*(s3)] = 10 + 0.9(0) = 10
  V*(s1) = max[R(s1,right) + γV*(s2)] = 0 + 0.9(10) = 9
```

---

# HOUR 3: Value Iteration & Policy Iteration

## 3.1 Value Iteration Algorithm

```
Algorithm: Value Iteration
─────────────────────────
1. Initialize V(s) = 0 for all s
2. Repeat until convergence:
   For each state s:
     V(s) ← max_a [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
3. Extract policy:
   π*(s) = argmax_a [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
```

### Convergence
- Δ = max_s |V_new(s) - V_old(s)|
- Stop when Δ < threshold (e.g., 0.001)

## 3.2 Policy Iteration Algorithm

```
Algorithm: Policy Iteration
───────────────────────────
1. Initialize π arbitrarily
2. Repeat until π stable:
   
   Policy Evaluation:
     Solve V^π(s) = Σ_a π(a|s)[R(s,a) + γ Σ_s' P(s'|s,a) V^π(s')]
     (iteratively or by solving linear system)
   
   Policy Improvement:
     For each state s:
       π(s) ← argmax_a [R(s,a) + γ Σ_s' P(s'|s,a) V^π(s')]
```

## 3.3 Comparison

| Aspect | Value Iteration | Policy Iteration |
|--------|-----------------|------------------|
| Per iteration | Simple max operation | Full policy evaluation |
| Convergence | Slower iterations | Fewer iterations |
| Memory | Store V only | Store V and π |

---

# HOUR 4: Q-Learning & SARSA (Model-Free)

## 4.1 Q-Learning Algorithm ⭐ VERY IMPORTANT

```
Algorithm: Q-Learning (Off-Policy)
──────────────────────────────────
Initialize Q(s,a) arbitrarily
For each episode:
  Initialize s
  For each step:
    Choose a from s using policy (e.g., ε-greedy)
    Take action a, observe r, s'
    
    Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    
    s ← s'
  Until s is terminal
```

### Key Update:
```
Q(s,a) ← Q(s,a) + α × TD_error

Where: TD_error = r + γ max_a' Q(s',a') - Q(s,a)
                = (target) - (current estimate)
```

## 4.2 SARSA Algorithm (On-Policy)

```
Algorithm: SARSA
────────────────
Initialize Q(s,a) arbitrarily
For each episode:
  Initialize s
  Choose a from s using policy (e.g., ε-greedy)
  For each step:
    Take action a, observe r, s'
    Choose a' from s' using policy (ε-greedy)
    
    Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
    
    s ← s', a ← a'
  Until s is terminal
```

### Key Difference:
```
Q-Learning: Q(s,a) ← ... + α[r + γ max_a' Q(s',a') - Q(s,a)]  (uses max)
SARSA:      Q(s,a) ← ... + α[r + γ Q(s',a') - Q(s,a)]        (uses actual a')
```

## 4.3 ε-Greedy Policy

```
With probability ε: take random action (exploration)
With probability 1-ε: take greedy action argmax_a Q(s,a)

Typical: ε starts at 1.0, decays to 0.1 over time
```

## 4.4 Worked Example: Q-Learning Update

```
Given:
  Current Q(s,a) = 5.0
  Learning rate α = 0.1
  Discount γ = 0.9
  Reward r = 2
  max_a' Q(s',a') = 8.0

Calculate new Q(s,a):
  Target = r + γ × max_a' Q(s',a')
         = 2 + 0.9 × 8.0
         = 2 + 7.2
         = 9.2
  
  TD_error = Target - Q(s,a) = 9.2 - 5.0 = 4.2
  
  Q_new(s,a) = Q(s,a) + α × TD_error
             = 5.0 + 0.1 × 4.2
             = 5.0 + 0.42
             = 5.42
```

---

# HOUR 5: Deep Q-Networks (DQN) & Policy Gradients

## 5.1 Why Deep RL?

| Problem | Solution |
|---------|----------|
| Large state space | Use neural net to approximate Q(s,a) |
| Continuous states | Can't store table, need function |
| Generalization | Learn from similar states |

## 5.2 DQN Architecture

```
Input: State s (e.g., game pixels)
  ↓
Neural Network (CNN/MLP)
  ↓
Output: Q(s,a) for all actions

Loss = (r + γ max_a' Q_target(s',a') - Q(s,a))²
```

## 5.3 DQN Key Tricks

### Experience Replay
```
- Store transitions (s, a, r, s') in replay buffer
- Sample random mini-batches for training
- Breaks correlation between consecutive samples
```

### Target Network
```
- Separate network Q_target for computing targets
- Copy weights periodically: Q_target ← Q
- Stabilizes training
```

## 5.4 Policy Gradient Basics

Instead of learning Q-values, directly learn the policy π_θ(a|s).

### REINFORCE Algorithm
```
For each episode:
  Generate trajectory τ = (s₀, a₀, r₀, s₁, a₁, r₁, ...)
  
  For each step t:
    G_t = Σ_{k=0}^{T-t} γᵏ r_{t+k}  (return from step t)
    
    θ ← θ + α × G_t × ∇_θ log π_θ(a_t|s_t)
```

### Policy Gradient Theorem
```
∇_θ J(θ) = E[∇_θ log π_θ(a|s) × Q^π(s,a)]
```

## 5.5 Actor-Critic

Combines value-based and policy-based:
```
Actor: Policy network π_θ(a|s)
Critic: Value network V_φ(s) or Q_φ(s,a)

Update:
- Critic: Minimize TD error
- Actor: Use critic's value estimate to update policy
```

---

# 📋 FORMULA QUICK REFERENCE

## Bellman Equations
```
V*(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V*(s')]
Q*(s,a) = R(s,a) + γ Σ P(s'|s,a) max_a' Q*(s',a')
```

## Q-Learning Update
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

## SARSA Update
```
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
```

## TD Error
```
δ = r + γV(s') - V(s)    (for state values)
δ = r + γQ(s',a') - Q(s,a)  (for action values)
```

## Policy Gradient
```
θ ← θ + α × G_t × ∇log π_θ(a|s)
```

## Return (Discounted Sum)
```
G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... = Σ γᵏ r_{t+k}
```

---

# 📝 PRACTICE PROBLEMS

## Problem 1: Value Calculation
Given γ = 0.9, calculate V(s1) for:
```
s1 --a1--> s2 (r=2)
s2 --a1--> s3 (r=5)
s3 terminal

V(s3) = 0
V(s2) = 5 + 0.9(0) = 5
V(s1) = 2 + 0.9(5) = 2 + 4.5 = 6.5
```

## Problem 2: Q-Learning Update
```
Q(s,a) = 10, α = 0.2, γ = 0.95, r = 3, max Q(s',a') = 15

Target = 3 + 0.95(15) = 3 + 14.25 = 17.25
TD_error = 17.25 - 10 = 7.25
Q_new = 10 + 0.2(7.25) = 10 + 1.45 = 11.45
```

## Problem 3: ε-Greedy
With ε = 0.1 and Q-values Q(s,left) = 5, Q(s,right) = 8:
```
P(left) = ε/2 = 0.05
P(right) = (1-ε) + ε/2 = 0.9 + 0.05 = 0.95
```

---

# 🎯 EXAM TIPS

1. **Memorize Bellman equation** - guaranteed to appear
2. **Know Q-learning update step-by-step**
3. **Understand ε-greedy exploration**
4. **Practice small gridworld value calculations**
5. **Know difference between on-policy (SARSA) vs off-policy (Q-learning)**

---

**Focus on numerical problems - they give guaranteed marks!**
