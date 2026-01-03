# 🎯 ML SYSTEM OPTIMIZATION EXAM GUIDE - Jan 4th, 4:30 PM

> **AIMLCZG516 ML System Optimization | 30 marks | 2 Hours | Closed Book**

---

## 🔥 HIGH-YIELD TOPICS (Focus 80% of time here)

### 1. AMDAHL'S LAW ⭐⭐⭐ (8-10 marks - GUARANTEED!)

#### The Formula (MEMORIZE!)
```
                    1
Speedup(p) = ─────────────
              f + (1-f)/p

Where:
  f = serial fraction (cannot be parallelized)
  (1-f) = parallel fraction
  p = number of processors
```

#### Maximum Speedup
```
As p → ∞:  Speedup_max = 1/f
```

#### 📝 Example 1: Basic Calculation
```
A program takes 100 seconds. 20% of it is sequential.
Calculate speedup with 4 processors.

Given: f = 0.2, p = 4

Speedup(4) = 1 / (0.2 + 0.8/4)
           = 1 / (0.2 + 0.2)
           = 1 / 0.4
           = 2.5

Parallel time = 100 / 2.5 = 40 seconds
```

#### 📝 Example 2: Maximum Speedup
```
f = 0.1 (10% sequential)

Maximum Speedup = 1/f = 1/0.1 = 10

Even with infinite processors, speedup cannot exceed 10x!
```

#### 📝 Example 3: Find Required Processors
```
Need 2.5x speedup. f = 0.25. How many processors?

2.5 = 1 / (0.25 + 0.75/p)
0.25 + 0.75/p = 1/2.5 = 0.4
0.75/p = 0.15
p = 0.75/0.15 = 5 processors
```

#### Quick Reference Table
| f (serial) | Max Speedup | Speedup with p=4 | Speedup with p=8 |
|------------|-------------|------------------|------------------|
| 0.05 | 20 | 3.48 | 5.93 |
| 0.10 | 10 | 3.08 | 4.71 |
| 0.20 | 5 | 2.50 | 3.33 |
| 0.25 | 4 | 2.29 | 2.91 |
| 0.50 | 2 | 1.60 | 1.78 |

---

### 2. EFFICIENCY ⭐⭐ (2-3 marks likely)

#### Formula
```
Efficiency(p) = Speedup(p) / p

Ideal efficiency = 1.0 (100%)
Linear speedup means efficiency = 1
```

#### 📝 Example
```
Speedup with 8 processors = 6

Efficiency = 6/8 = 0.75 = 75%

Meaning: We're using 75% of the parallel capacity effectively
```

---

### 3. MAPREDUCE ⭐⭐⭐ (5-6 marks likely)

#### Core Concept
```
MAP Function:
  Input: (key, value)
  Output: list of (new_key, new_value)
  
REDUCE Function:
  Input: (key, [list of values])
  Output: (key, aggregated_result)
```

#### 📝 Example: Word Count (CLASSIC!)
```python
# MAP: For each document
def map(doc_id, doc_text):
    for word in doc_text.split():
        emit(word, 1)

# Example: doc = "hello world hello"
# Emits: ("hello", 1), ("world", 1), ("hello", 1)

# SHUFFLE: Groups by key
# ("hello", [1, 1]), ("world", [1])

# REDUCE: For each word
def reduce(word, counts):
    emit(word, sum(counts))

# Result: ("hello", 2), ("world", 1)
```

#### Common MapReduce Patterns
| Task | MAP Output | REDUCE |
|------|------------|--------|
| Word Count | (word, 1) | sum |
| Sum by Category | (category, value) | sum |
| Average | (key, (value, 1)) | sum/count |
| Maximum | (key, value) | max |
| Filter | (match, record) | identity |

#### 📝 Example: Average Temperature by City
```python
# Data: [("NYC", 75), ("LA", 82), ("NYC", 68), ("LA", 79)]

# MAP
def map(record):
    city, temp = record
    emit(city, (temp, 1))  # (value, count)

# Emits: ("NYC", (75, 1)), ("LA", (82, 1)), ("NYC", (68, 1)), ("LA", (79, 1))

# After shuffle:
# ("NYC", [(75,1), (68,1)]), ("LA", [(82,1), (79,1)])

# REDUCE
def reduce(city, values):
    total_temp = sum(v[0] for v in values)
    total_count = sum(v[1] for v in values)
    emit(city, total_temp / total_count)

# Result: ("NYC", 71.5), ("LA", 80.5)
```

---

### 4. k-MEANS PARALLELIZATION ⭐⭐ (4-5 marks likely)

#### Sequential Algorithm
```
1. Initialize k centers randomly
2. REPEAT:
   a. ASSIGN: Each point → nearest center
   b. UPDATE: New center = mean of cluster points
3. UNTIL: Centers don't change
```

#### Parallel Version
```
ASSIGN Phase (Embarrassingly Parallel):
┌─────────────────────────────────────────────┐
│ Partition data among p processors           │
│                                             │
│ Each processor:                             │
│   - For each local point                    │
│   - Compute distance to ALL k centers       │
│   - Assign to nearest cluster               │
│                                             │
│ Time: O(n/p × k × d) for n points           │
└─────────────────────────────────────────────┘

UPDATE Phase (Parallel per Cluster):
┌─────────────────────────────────────────────┐
│ Each processor:                             │
│   - Compute LOCAL sum and count per cluster │
│                                             │
│ REDUCE:                                     │
│   - Global sum = Σ local sums               │
│   - Global count = Σ local counts           │
│   - New center = Global sum / Global count  │
└─────────────────────────────────────────────┘
```

#### Complexity
```
Sequential: O(n × k × d × I)
Parallel:   O(n/p × k × d × I) + communication

Where: n=points, k=clusters, d=dimensions, I=iterations
Speedup ≈ p (near linear for large n)
```

---

### 5. MEMORY MODELS ⭐⭐ (3-4 marks likely)

#### Shared Memory
```
┌────┐ ┌────┐ ┌────┐ ┌────┐
│ P1 │ │ P2 │ │ P3 │ │ P4 │
└──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘
   │      │      │      │
═══╧══════╧══════╧══════╧═══
│      SHARED MEMORY       │
═══════════════════════════

Pros:
✓ Fast communication (nanoseconds)
✓ Easy programming (threads, OpenMP)

Cons:
✗ Limited scalability
✗ Memory contention

Examples: Multi-core CPU, GPU (within device)
```

#### Distributed Memory
```
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Memory1 │  │ Memory2 │  │ Memory3 │
├─────────┤  ├─────────┤  ├─────────┤
│   P1    │  │   P2    │  │   P3    │
└────┬────┘  └────┬────┘  └────┬────┘
     │           │            │
═════╧═══════════╧════════════╧══════
│         NETWORK                   │
═════════════════════════════════════

Pros:
✓ High scalability (1000s of nodes)
✓ No memory contention

Cons:
✗ Slow communication (milliseconds)
✗ Complex programming (MPI)

Examples: Clusters, Cloud, HPC systems
```

---

### 6. PARALLELISM TYPES ⭐⭐ (2-3 marks likely)

#### Data Parallelism (SPMD)
```
Same operation, different data

Example: Vector Addition
  Processor 1: C[0:n/4] = A[0:n/4] + B[0:n/4]
  Processor 2: C[n/4:n/2] = A[n/4:n/2] + B[n/4:n/2]
  ...

Most common in ML! (batch processing)
```

#### Task Parallelism
```
Different operations, same time

Example: Pipeline
  Stage 1 → Stage 2 → Stage 3
  While Stage 1 processes item 2,
  Stage 2 can process item 1

Used in: Decision tree branches, pipeline stages
```

#### Request Parallelism
```
Independent requests processed concurrently

Example: Web server handling multiple requests
No communication between requests
```

---

### 7. PARAMETER SERVER ⭐⭐ (4-5 marks likely)

#### Architecture
```
        ┌───────────────────┐
        │  PARAMETER SERVER │
        │  (stores model w) │
        └─────────┬─────────┘
                  │
     ┌────────────┼────────────┐
     │            │            │
┌────┴────┐ ┌────┴────┐ ┌────┴────┐
│ Worker1 │ │ Worker2 │ │ Worker3 │
│ Data D1 │ │ Data D2 │ │ Data D3 │
└─────────┘ └─────────┘ └─────────┘
```

#### Training Loop
```
Each iteration:
  1. Workers PULL current parameters w from server
  2. Workers compute LOCAL gradients ∇L on their data
  3. Workers PUSH gradients to server
  4. Server AGGREGATES gradients (average)
  5. Server UPDATES: w ← w - η × avg(∇L)
```

#### Synchronous vs Asynchronous SGD
| Aspect | Synchronous | Asynchronous |
|--------|-------------|--------------|
| Barrier | Yes (waits for all) | No |
| Consistency | High | Stale gradients |
| Speed | Slower (stragglers) | Faster |
| Convergence | Better | May diverge |

---

### 8. CACHE LOCALITY ⭐ (2-3 marks likely)

#### Types
```
TEMPORAL Locality:
  Same location accessed repeatedly
  Example: Loop counter, frequently used variable

SPATIAL Locality:
  Nearby locations accessed together
  Example: Array elements, sequential instructions
```

#### Matrix Multiplication Optimization
```
Naive (IJK): Poor cache use - B accessed column-wise
for i: for j: for k:
    C[i][j] += A[i][k] * B[k][j]  ← B[k][j] is BAD!

Optimized (IKJ): Good cache use - B accessed row-wise
for i: for k: for j:
    C[i][j] += A[i][k] * B[k][j]  ← B[k][*] row reused

Result: Up to 7x speedup for large matrices!
```

---

## 🧮 QUICK CALCULATIONS

### Amdahl's Law Examples (MUST PRACTICE!)
```
f = 0.1:
  p=2:  1/(0.1 + 0.9/2) = 1/0.55 = 1.82
  p=4:  1/(0.1 + 0.9/4) = 1/0.325 = 3.08
  p=8:  1/(0.1 + 0.9/8) = 1/0.213 = 4.71
  p=∞:  1/0.1 = 10 (max)

f = 0.25:
  p=2:  1/(0.25 + 0.75/2) = 1/0.625 = 1.6
  p=4:  1/(0.25 + 0.75/4) = 1/0.438 = 2.29
  p=8:  1/(0.25 + 0.75/8) = 1/0.344 = 2.91
  p=∞:  1/0.25 = 4 (max)
```

---

## 📋 KEY TERMS

| Term | Meaning |
|------|---------|
| **SPMD** | Single Program Multiple Data |
| **MPI** | Message Passing Interface (distributed) |
| **OpenMP** | Shared memory parallelism API |
| **SIMD** | Single Instruction Multiple Data |
| **Throughput** | Operations per second |
| **Latency** | Time for single operation |
| **Straggler** | Slow worker in synchronous system |

---

## ✅ EXAM STRATEGY

1. **Amdahl's Law FIRST** - Guaranteed 8-10 marks
2. **Show ALL calculations** - Partial credit
3. **MapReduce diagrams** - Draw the flow
4. **k-Means concept** - Know parallel vs sequential

### Time Allocation (2 hours)
- Reading: 5 min
- Q1 (Amdahl's Law): 25 min
- Q2 (MapReduce): 25 min
- Q3 (Parallelism/Memory): 20 min
- Q4 (k-Means/Parameter Server): 20 min
- Review: 30 min

---

## 🎯 SELF-TEST

Can you:
- [ ] Calculate speedup using Amdahl's Law?
- [ ] Find maximum speedup from serial fraction?
- [ ] Find required processors for target speedup?
- [ ] Write MAP and REDUCE for word count?
- [ ] Explain k-Means parallelization?
- [ ] Draw parameter server architecture?
- [ ] Compare sync vs async SGD?

---

## 📝 COMMON EXAM TRAPS

1. **Speedup > p is impossible** (in standard cases)
2. **f = 0 is unrealistic** - No program is 100% parallel
3. **Max speedup = 1/f** - Don't forget!
4. **Communication overhead** - Reduces actual speedup
5. **Row-major vs Column-major** - Affects cache performance

---

**STUDY FOCUS: Amdahl's Law is your guaranteed 8-10 marks. Master it!**

**Good luck! 💪**
