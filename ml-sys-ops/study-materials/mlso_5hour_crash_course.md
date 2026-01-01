# 📺 ML System Optimization - 5-Hour Crash Course

> **AIMLCZG516 | Midterm Focus | Sessions 1-9**

---

## ⏱️ TIME ALLOCATION (5 Hours)

| Hour | Topic | Weight | Focus |
|------|-------|--------|-------|
| **1** | Speedup & Amdahl's Law | 30% | 🧮 Numerical calculations |
| **2** | Parallelism Types & Memory Models | 20% | Concepts + Diagrams |
| **3** | k-Means & MapReduce | 25% | Algorithm + Code |
| **4** | Parameter Server & SGD | 15% | Architecture + Trade-offs |
| **5** | Cache Locality & Practice | 10% | Matrix multiplication + Review |

---

# HOUR 1: Speedup & Amdahl's Law (Most Tested!)

## 🎯 Master This Formula

```
                    1
Speedup(p) = ─────────────
               f + (1-f)/p

f = serial fraction
p = processors
```

## 📝 Step-by-Step Calculation Method

**Given:** T_seq = 100s, f = 0.2, p = 4

```
Step 1: Identify values
  f = 0.2 (serial fraction)
  1-f = 0.8 (parallel fraction)
  p = 4 processors

Step 2: Apply formula
  Speedup(4) = 1 / (0.2 + 0.8/4)
             = 1 / (0.2 + 0.2)
             = 1 / 0.4
             = 2.5×

Step 3: Calculate parallel time
  T_par = T_seq / Speedup = 100 / 2.5 = 40 seconds

Step 4: Max speedup (p → ∞)
  Speedup_max = 1 / f = 1 / 0.2 = 5×
```

## 🔢 Quick Reference Table

| f (serial) | Max Speedup | p=4 | p=8 | p=16 |
|------------|-------------|-----|-----|------|
| 0.0 | ∞ | 4.0 | 8.0 | 16.0 |
| 0.1 | 10 | 3.1 | 4.7 | 6.4 |
| 0.2 | 5 | 2.5 | 3.3 | 4.0 |
| 0.25 | 4 | 2.3 | 2.9 | 3.4 |
| 0.5 | 2 | 1.6 | 1.8 | 1.9 |

## ✅ Practice (15 min)

**Q1:** f = 0.15, p = 10. Find speedup and max speedup.
```
Speedup(10) = 1/(0.15 + 0.85/10) = 1/0.235 = 4.26×
Max = 1/0.15 = 6.67×
```

**Q2:** Speedup = 3, p = 5. Find serial fraction.
```
3 = 1/(f + (1-f)/5)
f + 0.2 - 0.2f = 0.333
0.8f = 0.133
f = 0.167 (16.7%)
```

---

# HOUR 2: Parallelism Types & Memory Models

## 🔄 Three Types of Parallelism

```
1. DATA PARALLELISM (SPMD)
   ✓ Same operation, different data
   ✓ Best for ML (most common)
   Example: Vector addition A[i] + B[i]

2. TASK PARALLELISM
   ✓ Different operations, concurrent
   ✓ Pipeline processing
   Example: Stage1 → Stage2 → Stage3

3. REQUEST PARALLELISM
   ✓ Independent requests
   ✓ No communication needed
   Example: Web server
```

## 💾 Memory Models (Draw This!)

```
SHARED MEMORY:                    DISTRIBUTED MEMORY:
                                  
   P1  P2  P3  Pp                  P1    P2    P3    Pp
    │   │   │   │                  │     │     │     │
    └───┴───┴───┘                 Mem   Mem   Mem   Mem
         │                         │     │     │     │
   ┌─────┴─────┐                  └─────┴─────┴─────┘
   │  SHARED   │                  │    NETWORK    │
   │  MEMORY   │                  └───────────────┘
   └───────────┘                  
                                  
✓ Fast (ns)                       ✓ Scalable
✗ Limited scale                   ✗ Slow (ms)
Thread programming                Message passing
```

## 📊 Memory Hierarchy (Know the Order!)

```
Fastest → Slowest:
  Registers → L1 → L2 → L3 → RAM → SSD → HDD → Network

Access time ratios (approximate):
  L1 : RAM : Network = 1 : 50 : 10,000
```

## ✅ Practice (10 min)

**Q:** Which parallelism type for:
- Training CNN on images? → **Data Parallel (SPMD)**
- Image → Preprocess → Train pipeline? → **Task Parallel**
- Inference API serving requests? → **Request Parallel**

---

# HOUR 3: k-Means & MapReduce

## 📊 k-Means Algorithm

```
1. Initialize k centers randomly
2. REPEAT:
   ASSIGN: point → nearest center
   UPDATE: center = mean(cluster points)
3. UNTIL convergence
```

## 🔁 k-Means MapReduce

```
ASSIGN Phase:
  MAP: For each point → (cluster_id, point)
  REDUCE: min distance → assign to cluster

UPDATE Phase:
  MAP: For each cluster → (cluster_id, (point, 1))
  REDUCE: sum points/count → new centroid
```

## 📝 MapReduce Template

```python
# Word Count Example
def map(doc):
    for word in doc.split():
        emit(word, 1)

def reduce(word, counts):
    emit(word, sum(counts))
```

## 🧮 k-Means Speedup

```
Sequential: O(n × k × d)
Parallel:   O(n/p × k × d)
Speedup ≈ p (nearly linear!)
```

## ✅ Practice (15 min)

**Q:** Write MAP output for: "cat dog cat bird dog dog"
```
("cat", 1), ("dog", 1), ("cat", 1), ("bird", 1), ("dog", 1), ("dog", 1)

After shuffle: 
  "cat" → [1, 1]
  "dog" → [1, 1, 1]
  "bird" → [1]

After reduce:
  ("cat", 2), ("dog", 3), ("bird", 1)
```

---

# HOUR 4: Parameter Server & Distributed SGD

## 🏗️ Parameter Server Architecture

```
         ┌─────────────────┐
         │ PARAMETER SERVER│ ← Stores model w
         └────────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───┴───┐    ┌───┴───┐    ┌───┴───┐
│Worker1│    │Worker2│    │WorkerP│
│ Data1 │    │ Data2 │    │ DataP │
└───────┘    └───────┘    └───────┘

Iteration:
1. Workers PULL w
2. Workers compute gradients
3. Workers PUSH gradients
4. Server updates w
```

## ⚡ Sync vs Async SGD

| Aspect | Synchronous | Asynchronous |
|--------|-------------|--------------|
| Barrier | Yes (wait all) | No |
| Speed | Slow | Fast |
| Convergence | Stable | May oscillate |
| Stragglers | Problem | No problem |

## 🧮 SGD Update Rule

```
w ← w - η × ∇L(w)

η = learning rate
∇L = gradient of loss
```

## ⚠️ Key Issue

**Communication is the bottleneck!**
```
If model = 10M params × 4 bytes = 40 MB
Network = 1 Gbps = 125 MB/s
Time per sync = 40 MB / 125 MB/s = 320 ms

This can dominate computation time!
```

## ✅ Practice (10 min)

**Q:** 5 workers, 1M parameters, 1 Gbps network. Computation = 50ms.
What fraction of time is communication?

```
Communication = 2 × (1M × 4 bytes) / 125 MB/s = 64 ms
Total = 50 + 64 = 114 ms
Fraction = 64/114 = 56% (more than half!)
```

---

# HOUR 5: Cache Locality & Final Review

## 💾 Cache Locality Types

```
TEMPORAL: Same data accessed repeatedly
  → Solution: Keep in cache (LRU)

SPATIAL: Nearby data accessed together
  → Solution: Prefetch, access row-wise
```

## 🧮 Matrix Multiplication (Key Example!)

```
IJK (Bad):
  b[k][j] accessed column-wise → Cache misses!

IKJ (Good):
  b[k][j] accessed row-wise → Cache hits!

Speedup: up to 7× for large matrices
```

## 📋 Final Checklist

- [ ] Amdahl's Law formula memorized
- [ ] Can calculate speedup given f and p
- [ ] Know 3 parallelism types
- [ ] Shared vs Distributed memory differences
- [ ] k-Means algorithm steps
- [ ] Can write MapReduce for word count
- [ ] Parameter server architecture
- [ ] Sync vs Async SGD trade-offs
- [ ] Why IKJ is better than IJK

## 🔥 Most Important Formulas

```
1. Speedup = 1 / (f + (1-f)/p)
2. Max Speedup = 1/f
3. Efficiency = Speedup/p
4. k-Means complexity = O(n × k × d)
5. SGD update: w ← w - η∇L
```

## 📝 3 Types of Questions to Expect

1. **Numerical:** Given f, p, calculate speedup
2. **Conceptual:** Compare sync/async, shared/distributed
3. **Algorithm:** Write MapReduce, explain k-means

---

## 📺 NPTEL Videos (Quick Reference)

| Topic | Search Term |
|-------|-------------|
| Speedup | "NPTEL parallel computing speedup amdahl" |
| MapReduce | "NPTEL big data mapreduce" |
| Parameter Server | "NPTEL distributed machine learning" |

---

**You're ready! Good luck! 🍀**
