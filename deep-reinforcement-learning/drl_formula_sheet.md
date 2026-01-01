# 📋 DRL Formula Sheet & Quick Reference

> **Deep Reinforcement Learning | Exam Quick Reference**

---

## 🎯 MDP Components

```
MDP = (S, A, P, R, γ)

S = State space
A = Action space  
P(s'|s,a) = Transition probability
R(s,a) = Reward function
γ = Discount factor (0 ≤ γ ≤ 1)
```

---

## 📊 Value Functions

### State Value
```
V^π(s) = E[Σ γᵗrₜ | s₀=s, π]
```

### Action Value (Q-value)
```
Q^π(s,a) = E[Σ γᵗrₜ | s₀=s, a₀=a, π]
```

### Relationship
```
V^π(s) = Σ_a π(a|s) Q^π(s,a)
```

---

## ⭐ BELLMAN EQUATIONS (Memorize!)

### Bellman Expectation (for policy π)
```
V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V^π(s')]
```

### Bellman Optimality (for optimal V*)
```
V*(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V*(s')]

Q*(s,a) = R(s,a) + γ Σ_s' P(s'|s,a) max_a' Q*(s',a')
```

---

## 🔄 ALGORITHMS

### Value Iteration
```
V(s) ← max_a [R(s,a) + γ Σ P(s'|s,a) V(s')]
Repeat until convergence
```

### Policy Iteration
```
1. Policy Evaluation: Compute V^π
2. Policy Improvement: π(s) ← argmax_a Q(s,a)
Repeat until policy stable
```

---

## ⭐ Q-LEARNING (Memorize!)

```
Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
                    \_________target__________/ 
```

### Components
```
α = learning rate (e.g., 0.1)
γ = discount factor (e.g., 0.9)
r = immediate reward
max_a' Q(s',a') = best Q-value in next state

TD_error = r + γ max_a' Q(s',a') - Q(s,a)
```

---

## 🔄 SARSA

```
Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') - Q(s,a)]
```

### Q-Learning vs SARSA
```
Q-Learning: uses max_a' Q(s',a')  → Off-policy
SARSA:      uses Q(s',a')         → On-policy
```

---

## 🎲 EXPLORATION

### ε-Greedy
```
P(random action) = ε
P(greedy action) = 1 - ε

Greedy action = argmax_a Q(s,a)
```

### Decay
```
ε_t = max(ε_min, ε₀ × decay^t)
```

---

## 🧠 DQN (Deep Q-Network)

### Loss Function
```
L = (r + γ max_a' Q_target(s',a') - Q(s,a))²
```

### Key Techniques
```
1. Experience Replay: Store (s,a,r,s') in buffer
2. Target Network: Separate network for targets
3. Gradient clipping: Prevent exploding gradients
```

---

## 📈 POLICY GRADIENT

### REINFORCE Update
```
θ ← θ + α × Gₜ × ∇log π_θ(a|s)

Gₜ = rₜ + γrₜ₊₁ + γ²rₜ₊₂ + ...
```

### Policy Gradient Theorem
```
∇J(θ) = E[∇log π_θ(a|s) × Q^π(s,a)]
```

---

## 🎭 ACTOR-CRITIC

```
Actor:  Updates policy π_θ
Critic: Estimates value V_φ or Q_φ

Advantage: A(s,a) = Q(s,a) - V(s)
```

---

## 🔢 QUICK CALCULATIONS

### Return Calculation
```
G = r₀ + γr₁ + γ²r₂ + ...
Example: r = [1, 2, 3], γ = 0.9
G = 1 + 0.9(2) + 0.81(3) = 1 + 1.8 + 2.43 = 5.23
```

### Q-Learning Step
```
Given: Q = 5, α = 0.1, γ = 0.9, r = 2, max Q' = 8

Target = 2 + 0.9(8) = 9.2
TD_error = 9.2 - 5 = 4.2
Q_new = 5 + 0.1(4.2) = 5.42
```

---

## 📝 COMMON EXAM QUESTIONS

1. **Calculate V(s)** given transitions and rewards
2. **One step of Q-learning** update
3. **Compare Q-learning vs SARSA**
4. **ε-greedy action selection**
5. **Bellman equation application**

---

**Focus on: Bellman equations + Q-learning update!**
