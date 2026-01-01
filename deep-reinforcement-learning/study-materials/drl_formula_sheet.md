# 📋 DRL Formula Sheet - Quick Reference

> **AIMLCZG512 | Midterm | Sessions 1-8 | Closed Book**

---

## 🎰 MULTI-ARMED BANDITS

### Action Value Estimate (Sample Average)
```
Q_t(a) = (Sum of rewards when a taken) / (Number of times a taken)
```

### Incremental Update ⭐ MEMORIZE
```
Q_{n+1} = Q_n + α [R_n - Q_n]

Where:
  α = 1/n     → Stationary problems
  α = constant → Non-stationary problems
```

### ε-Greedy Action Selection
```
With prob (1-ε): a = argmax_a Q(a)  [exploit]
With prob ε:     a = random action  [explore]
```

### ε-Greedy Probabilities
```
P(greedy action) = 1 - ε + ε/|A|
P(other action)  = ε/|A|
```

### UCB Action Selection
```
A_t = argmax_a [ Q_t(a) + c √(ln t / N_t(a)) ]
```

---

## 📊 MDP FUNDAMENTALS

### MDP Tuple
```
(S, A, P, R, γ)

S = State space
A = Action space
P(s'|s,a) = Transition probability
R(s,a) or R(s,a,s') = Reward
γ ∈ [0,1] = Discount factor
```

### Return (Discounted Sum)
```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...

Recursive: G_t = R_{t+1} + γG_{t+1}
```

### State Value Function
```
V^π(s) = E_π [ Σ γᵏ R_{t+k+1} | S_t = s ]
```

### Action Value Function
```
Q^π(s,a) = E_π [ Σ γᵏ R_{t+k+1} | S_t = s, A_t = a ]
```

---

## ⭐ BELLMAN EQUATIONS (MUST MEMORIZE!)

### Bellman Expectation
```
V^π(s) = Σ_a π(a|s) [ R(s,a) + γ Σ_s' P(s'|s,a) V^π(s') ]
```

### Bellman Optimality
```
V*(s) = max_a [ R(s,a) + γ Σ_s' P(s'|s,a) V*(s') ]

Q*(s,a) = R(s,a) + γ Σ_s' P(s'|s,a) max_a' Q*(s',a')
```

---

## 🔄 DYNAMIC PROGRAMMING

### Policy Evaluation (Prediction)
```
V(s) ← Σ_a π(a|s) [ R(s,a) + γ Σ_s' P(s'|s,a) V(s') ]
```

### Value Iteration
```
V(s) ← max_a [ R(s,a) + γ Σ_s' P(s'|s,a) V(s') ]
```

### Policy Improvement
```
π(s) ← argmax_a [ R(s,a) + γ Σ_s' P(s'|s,a) V(s') ]
```

---

## 🎲 MONTE CARLO

### First-Visit MC
```
V(s) = average of returns from FIRST visit to s in each episode
```

### Every-Visit MC
```
V(s) = average of returns from ALL visits to s
```

### MC Update (Incremental)
```
V(s) ← V(s) + α [ G - V(s) ]

Where G = return from that visit
```

### ε-Soft Policy Update
```
a* = argmax_a Q(s,a)

π(a*|s) = 1 - ε + ε/|A|
π(a≠a*|s) = ε/|A|
```

---

## 📐 QUICK CALCULATIONS

### Return from Rewards
```
Given rewards [r₁, r₂, r₃, ...], γ:

Work backwards:
G_T = 0
G_{t} = r_{t+1} + γ × G_{t+1}
```

### Example:
```
Rewards: [1, 2, 3], γ = 0.9

G₃ = 0
G₂ = 3 + 0.9(0) = 3
G₁ = 2 + 0.9(3) = 4.7
G₀ = 1 + 0.9(4.7) = 5.23
```

### ε-Greedy Example
```
ε = 0.2, |A| = 4

P(best action) = 1 - 0.2 + 0.2/4 = 0.85
P(other action) = 0.2/4 = 0.05
```

### Incremental Update Example
```
Q = 3.0, α = 0.1, R = 5.0

Q_new = 3.0 + 0.1(5.0 - 3.0)
      = 3.0 + 0.2
      = 3.2
```

---

## 🔑 KEY RELATIONSHIPS

```
V^π(s) = Σ_a π(a|s) Q^π(s,a)

Q^π(s,a) = R(s,a) + γ Σ_s' P(s'|s,a) V^π(s')

V*(s) = max_a Q*(s,a)

Q*(s,a) = R(s,a) + γ Σ_s' P(s'|s,a) V*(s')
```

---

## 📝 ALGORITHM COMPARISON

| Method | Model? | Bootstrap? | Episodes? |
|--------|--------|------------|-----------|
| DP | Yes | Yes | No |
| MC | No | No | Yes |
| TD | No | Yes | No |

---

## ⚠️ COMMON MISTAKES

1. **Forgetting γ** in return calculations
2. **Wrong α** for stationary vs non-stationary
3. **ε-greedy probability** - remember it's 1-ε+ε/|A|, not just 1-ε
4. **Returns calculated forward** - should be backward from end
5. **Bellman equation missing summation** over states

---

## 🎯 EXAM CHECKLIST

- [ ] Can write Bellman optimality equation
- [ ] Can calculate return from reward sequence
- [ ] Can do one step of value iteration
- [ ] Can compute ε-greedy probabilities
- [ ] Can do incremental Q update
- [ ] Know first-visit vs every-visit MC difference

---

**Good luck! 🍀**
