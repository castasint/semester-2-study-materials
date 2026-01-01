# 📋 ML System Optimization - Formula Sheet & Quick Reference

> **AIMLCZG516 | Midterm | Sessions 1-9**

---

## 🎯 SPEEDUP & PERFORMANCE

### Speedup Definition
```
Speedup(p) = T_sequential / T_parallel(p)
```

### Amdahl's Law ⭐ MOST IMPORTANT
```
                    1
Speedup(p) = ─────────────
               f + (1-f)/p

Where:
  f = serial fraction (NOT parallelizable)
  p = number of processors
  (1-f) = parallel fraction
```

### Maximum Speedup
```
Speedup_max = 1/f   (when p → ∞)
```

### Efficiency
```
Efficiency(p) = Speedup(p) / p

Ideal efficiency = 1.0 (100%)
Linear speedup: Speedup = p
```

---

## 🔄 PARALLELISM TYPES

| Type | Description | Example |
|------|-------------|---------|
| **Data Parallelism** | Same op on different data | Vector addition |
| **Task Parallelism** | Different ops concurrently | Pipeline stages |
| **Request Parallelism** | Independent requests | Web server |
| **SPMD** | Single Program Multiple Data | Most ML |

---

## 🧮 k-MEANS CLUSTERING

### Algorithm Steps
```
1. Initialize k centers randomly
2. ASSIGN: Each point → nearest center
3. UPDATE: New center = mean of cluster
4. REPEAT until convergence
```

### Complexity
```
Sequential: O(n × k × d × I)
  n = data points
  k = clusters
  d = dimensions
  I = iterations

Parallel (p processors): O(n/p × k × d × I)
Speedup ≈ p (nearly linear)
```

---

## 📊 MAPREDUCE

### Functions
```python
MAP:    (key, value) → list of (key', value')
REDUCE: (key', list of values) → (key', aggregated_value)
```

### Common Patterns
| Pattern | Map Output | Reduce |
|---------|------------|--------|
| Count | (item, 1) | sum |
| Sum | (key, value) | sum |
| Average | (key, (value, 1)) | sum/count |
| Max/Min | (key, value) | max/min |

---

## 🖥️ MEMORY MODELS

### Shared Memory
```
✓ Fast communication (nanoseconds)
✓ Easy programming (threads)
✗ Limited scalability
Example: Multi-core CPU, GPU
```

### Distributed Memory
```
✓ High scalability
✗ Slow communication (milliseconds)
✗ Complex programming (messages)
Example: Clusters, Cloud
```

---

## 🌐 PARAMETER SERVER

### Architecture
```
Server: Stores global model parameters w
Workers: Store local data, compute gradients

Each iteration:
  1. Workers PULL current w
  2. Workers compute gradients ∇L
  3. Workers PUSH gradients to server
  4. Server updates: w ← w - η × avg(∇L)
```

### SGD Update Rule
```
w ← w - η × (1/|B|) × Σ ∇L(xᵢ, yᵢ, w)

η = learning rate
B = mini-batch
```

---

## 🌳 DECISION TREES

### Information Gain
```
IG(S, F) = H(S) - Σ (|Sᵥ|/|S|) × H(Sᵥ)
```

### Entropy
```
H(S) = -Σ pᵢ × log₂(pᵢ)
```

### Parallelization
```
Each branch → separate task
#parallel_tasks = #values of feature
```

---

## 💾 CACHE LOCALITY

### Types
```
TEMPORAL: Same location accessed repeatedly
SPATIAL: Nearby locations accessed together
```

### Matrix Multiplication
```
IJK order: Poor cache use for B (column access)
IKJ order: Good cache use (row access)
Speedup: up to 7x for large matrices!
```

### Access Time (Approximate)
```
L1 Cache:    ~4 cycles
L2 Cache:    ~12 cycles
L3 Cache:    ~40 cycles
RAM:         ~200 cycles
SSD:         ~50,000 cycles
HDD:         ~10,000,000 cycles
```

---

## 📐 QUICK CALCULATIONS

### Amdahl's Law Examples
```
f = 0:    Speedup(p) = p (ideal)
f = 0.1:  Speedup(∞) = 10
f = 0.2:  Speedup(∞) = 5
f = 0.5:  Speedup(∞) = 2
```

### Speedup Table
```
f = 0.2 (20% serial):
  p=2:  Speedup = 1.67
  p=4:  Speedup = 2.50
  p=8:  Speedup = 3.33
  p=16: Speedup = 4.00
  p=∞:  Speedup = 5.00
```

---

## 🔑 KEY TERMS

| Term | Meaning |
|------|---------|
| **SPMD** | Single Program Multiple Data |
| **MPI** | Message Passing Interface |
| **SIMD** | Single Instruction Multiple Data |
| **GPGPU** | General Purpose GPU computing |
| **Throughput** | Operations per second |
| **Latency** | Time for single operation |
| **Scalability** | Performance scales with resources |

---

## ⚠️ COMMON EXAM TRAPS

1. **Speedup > p is impossible** (for most cases)
2. **f = 0 means ideal parallelization** (not realistic)
3. **Communication overhead** reduces speedup in distributed systems
4. **Synchronization** can serialize parallel code
5. **Load imbalance** means some processors wait
6. **Row-major vs Column-major** matters for cache performance

---

**Good luck! 🍀**
