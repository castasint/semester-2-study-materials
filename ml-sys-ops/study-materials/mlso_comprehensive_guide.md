# 📚 ML System Optimization - Comprehensive Study Guide

> **AIMLCZG516 | Midterm Prep | Sessions 1-9**

---

# Table of Contents

1. [Module 1: Introduction & Fundamentals](#module-1-introduction--fundamentals)
   - Session 1: ML Performance Metrics
   - Session 2: Parallel Programming Models & Speedup
   - Session 3: Modern Systems (Multicore, GPU, Clusters)
2. [Module 2: Parallel/Distributed ML Algorithms](#module-2-paralleldistributed-ml-algorithms)
   - Session 4: Task Parallelism
   - Session 5: k-Means Parallelization
   - Session 6: Review & MapReduce
   - Session 7: kNN & Decision Trees
3. [Module 3: Scale-out ML Systems](#module-3-scale-out-ml-systems)
   - Session 8: Parameter Server & Distributed SGD
   - Session 9: Neural Network Optimization & Locality

---

# Module 1: Introduction & Fundamentals

## Session 1: ML and Performance Metrics

### 1.1 Performance Metrics

| Metric | Definition | Formula |
|--------|------------|---------|
| **Time Complexity** | Algorithmic complexity | O(n), O(n²), O(n log n) |
| **Running Time** | Actual wall-clock time | Measured in seconds |
| **Throughput** | Operations per unit time | ops/second |
| **Response Time** | Time to complete one request | Latency in ms |
| **Memory Usage** | RAM/Storage required | Bytes (MB, GB) |

### 1.2 Training vs Deployment Environments

```
┌─────────────────────────────────────────────────────────────┐
│                    ML SYSTEM LIFECYCLE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   TRAINING PHASE                    DEPLOYMENT PHASE         │
│   ─────────────                     ────────────────         │
│   • Large datasets (TB-PB)          • Small input (per req)  │
│   • High compute (GPU clusters)     • Low latency required   │
│   • Batch processing                • Real-time inference    │
│   • Accuracy-focused                • Throughput-focused     │
│   • Cloud/Data center               • Edge/Mobile/Cloud      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Range of Systems

| System Type | Characteristics | Use Case |
|-------------|-----------------|----------|
| **Cloud/Data Center** | Massive scale, distributed | Training large models |
| **GPU Clusters** | High parallelism, SIMD | Deep learning training |
| **Multi-core CPU** | Shared memory, threads | General ML workloads |
| **Embedded/Mobile** | Resource constrained | Edge inference |
| **TinyML Devices** | Ultra-low power | IoT, sensors |

---

## Session 2: Parallel Programming Models & Speedup

### 2.1 Types of Parallelism

```
┌─────────────────────────────────────────────────────────────┐
│                    PARALLELISM TYPES                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. DATA PARALLELISM (SPMD)                                  │
│     ├── Same operation on different data chunks              │
│     ├── E.g., Vector addition: A[i] + B[i] on each core     │
│     └── Preferred when feasible (easy, efficient)            │
│                                                              │
│  2. TASK PARALLELISM (Pipeline)                              │
│     ├── Different operations run concurrently                │
│     ├── E.g., Stage1 → Stage2 → Stage3                      │
│     └── Dependencies between stages                          │
│                                                              │
│  3. REQUEST PARALLELISM                                      │
│     ├── Independent requests processed in parallel           │
│     ├── E.g., Web server handling multiple requests          │
│     └── No communication between requests                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 SPMD (Single Program Multiple Data)

- **Same program** executes on all processors
- Each processor works on **different data subset**
- Most common paradigm in ML parallelization

```python
# SPMD Example: Parallel Vector Addition
# Each processor p executes:
for i in range(start_p, end_p):
    C[i] = A[i] + B[i]
```

### 2.3 Speedup & Amdahl's Law ⭐ KEY CONCEPT

**Speedup Definition:**
$$\text{Speedup}(p) = \frac{T_{\text{sequential}}}{T_{\text{parallel}}(p)}$$

**Amdahl's Law:**
```
If fraction f of a program is NOT parallelizable:

                    1
    Speedup(p) = ─────────────
                  f + (1-f)/p

Where:
  f = serial fraction (cannot be parallelized)
  p = number of processors
  (1-f) = parallel fraction
```

**Key Insights:**
- If f = 0 (perfectly parallel): Speedup = p (ideal/linear)
- If f = 0.1 (10% serial): Max speedup ≈ 10 (even with ∞ processors)
- If f = 0.5 (50% serial): Max speedup = 2

### 2.4 Speedup Example Calculation

**Problem:** A program takes 100 seconds. 20% is inherently sequential.

```
Given: f = 0.2, T_seq = 100s

For p = 4 processors:
  Speedup(4) = 1 / (0.2 + 0.8/4) = 1 / (0.2 + 0.2) = 1/0.4 = 2.5
  T_parallel = 100 / 2.5 = 40 seconds

For p = 10 processors:
  Speedup(10) = 1 / (0.2 + 0.8/10) = 1 / (0.2 + 0.08) = 1/0.28 = 3.57
  T_parallel = 100 / 3.57 = 28 seconds

Maximum Speedup (p → ∞):
  Speedup_max = 1 / f = 1 / 0.2 = 5
```

### 2.5 Factors Limiting Parallelism

| Factor | Description | Impact |
|--------|-------------|--------|
| **Memory Contention** | Multiple processors accessing same memory | Serialization |
| **Data Dependencies** | Output of one task needed as input to another | Forced ordering |
| **Synchronization** | Waiting for other processors | Idle time |
| **Communication Overhead** | Data transfer between processors | Extra time |
| **Load Imbalance** | Uneven work distribution | Some processors idle |

---

## Session 3: Modern Systems Architecture

### 3.1 Shared Memory vs Distributed Memory

```
┌─────────────────────────────────────────────────────────────┐
│              SHARED MEMORY MODEL                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│        ┌────┐  ┌────┐  ┌────┐  ┌────┐                       │
│        │ P1 │  │ P2 │  │ P3 │  │ Pp │                       │
│        └──┬─┘  └──┬─┘  └──┬─┘  └──┬─┘                       │
│           │      │      │      │                            │
│        ═══╧══════╧══════╧══════╧═══                         │
│        │     INTERCONNECTION BUS    │                        │
│        ═════════════════════════════                         │
│                     │                                        │
│              ┌──────┴──────┐                                │
│              │   GLOBAL    │                                │
│              │   MEMORY    │                                │
│              └─────────────┘                                │
│                                                              │
│  • Multi-core CPUs, GPUs                                    │
│  • Fast communication (ns)                                  │
│  • Limited scalability                                      │
│  • Programming: Threads (OpenMP, pthreads)                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│            DISTRIBUTED MEMORY MODEL                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │ Memory  │   │ Memory  │   │ Memory  │   │ Memory  │     │
│  ├─────────┤   ├─────────┤   ├─────────┤   ├─────────┤     │
│  │   P1    │   │   P2    │   │   P3    │   │   Pp    │     │
│  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘     │
│       │            │            │            │              │
│  ═════╧════════════╧════════════╧════════════╧════════     │
│  │           COMMUNICATION NETWORK              │           │
│  ═══════════════════════════════════════════════            │
│                                                              │
│  • Clusters, Cloud                                          │
│  • Slower communication (ms)                                │
│  • High scalability                                         │
│  • Programming: MPI, Spark, MapReduce                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Memory Hierarchy

```
                    ┌──────────────┐
                    │  REGISTERS   │  ← Fastest (1 cycle)
                    │   (~1 KB)    │
                    └──────┬───────┘
                           │
                    ┌──────┴───────┐
                    │   L1 CACHE   │  ← ~4 cycles
                    │  (~32 KB)    │
                    └──────┬───────┘
                           │
                    ┌──────┴───────┐
                    │   L2 CACHE   │  ← ~12 cycles
                    │  (~256 KB)   │
                    └──────┬───────┘
                           │
                    ┌──────┴───────┐
                    │   L3 CACHE   │  ← ~40 cycles
                    │   (~8 MB)    │
                    └──────┬───────┘
                           │
                    ┌──────┴───────┐
                    │     RAM      │  ← ~200 cycles
                    │  (~16 GB)    │
                    └──────┬───────┘
                           │
                    ┌──────┴───────┐
                    │     SSD      │  ← ~50,000 cycles
                    │  (~512 GB)   │
                    └──────┬───────┘
                           │
                    ┌──────┴───────┐
                    │     HDD      │  ← ~10,000,000 cycles
                    │  (~2 TB)     │
                    └──────────────┘
```

---

# Module 2: Parallel/Distributed ML Algorithms

## Session 4: Task Parallelism & Problem Decomposition

### 4.1 Task Parallelism

Unlike data parallelism where same operation runs on different data, **task parallelism** runs different operations concurrently.

```
Example: Pipeline Parallelism

  Input → [Task A] → [Task B] → [Task C] → Output
              ↓          ↓          ↓
           Stage 1    Stage 2    Stage 3
           
  While Task A processes item 2,
  Task B can process item 1 (already done by A)
```

### 4.2 Problem Decomposition Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Domain Decomposition** | Split data among processors | Large datasets |
| **Functional Decomposition** | Split computation into stages | Pipeline workloads |
| **Recursive Decomposition** | Divide-and-conquer | Tree structures |

---

## Session 5: k-Means Parallelization ⭐ KEY ALGORITHM

### 5.1 k-Means Algorithm (Sequential)

```
Algorithm: k-Means Clustering
─────────────────────────────
Input: Dataset D, number of clusters k
Output: k cluster centers c₁, c₂, ..., cₖ

1. Initialize: Choose k random points as initial centers
2. REPEAT:
   a. ASSIGN: For each point x in D:
      - Compute distance to all k centers
      - Assign x to nearest cluster
   b. UPDATE: For each cluster j:
      - cⱼ = mean of all points assigned to cluster j
3. UNTIL: Centers converge (don't change)
```

### 5.2 k-Means Using MapReduce

```
Step 1: Initialize k centers (random selection)

Step 2 (Assign): MAP operation
  ┌─────────────────────────────────────────────┐
  │ MAP: For each point xᵢ                      │
  │   - Compute distances to all k centers      │
  │   - Output: (cluster_id, xᵢ)               │
  └─────────────────────────────────────────────┘
  
  REDUCE: min over distances
  ┌─────────────────────────────────────────────┐
  │ REDUCE: For each point                      │
  │   - Find cluster with minimum distance      │
  │   - Assign point to that cluster            │
  └─────────────────────────────────────────────┘

Step 3 (Update): MAP + REDUCE
  ┌─────────────────────────────────────────────┐
  │ MAP: For each cluster j                     │
  │   - Output: (j, point)                      │
  │                                             │
  │ REDUCE: For cluster j                       │
  │   - cⱼ = (sum of all points) / count       │
  │     i.e., compute centroid                  │
  └─────────────────────────────────────────────┘

Step 4: Check convergence, repeat if needed
```

### 5.3 k-Means Speedup Analysis

```
Sequential Time Complexity (per iteration):
  T_seq = |D| × (k + k) + k × |C|
        = 2k|D| + k|C|
        
  Where:
    |D| = number of data points
    k = number of clusters
    |C| = average cluster size

Parallel Time (p processors):
  T_par = |D|/p × 2k + |C|
  
Speedup:
  S(p) = T_seq / T_par ≈ p (nearly linear!)
```

### 5.4 k-Means Code (Spark-style)

```python
# Pseudocode for parallel k-Means

def parallel_kmeans(data, k, max_iters):
    # Initialize centers randomly
    centers = data.takeSample(k)
    
    for iteration in range(max_iters):
        # ASSIGN: Map each point to nearest center
        assignments = data.map(
            lambda x: (find_nearest_center(x, centers), x)
        )
        
        # UPDATE: Compute new centers
        new_centers = assignments \
            .mapValues(lambda x: (x, 1)) \
            .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
            .mapValues(lambda v: v[0] / v[1]) \
            .collect()
        
        # Check convergence
        if converged(centers, new_centers):
            break
        centers = new_centers
    
    return centers
```

---

## Session 6: Review & MapReduce

### 6.1 MapReduce Programming Model

```
┌─────────────────────────────────────────────────────────────┐
│                   MAPREDUCE FLOW                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT     MAP PHASE      SHUFFLE/SORT     REDUCE PHASE     │
│  DATA                                                        │
│                                                              │
│ ┌─────┐   ┌────────┐                      ┌────────┐        │
│ │Doc1 │→ │Mapper 1│→ (k1,v1)              │        │        │
│ └─────┘   └────────┘   ↘                  │        │        │
│                          ┌──────┐  (k1,*) │Reducer1│→Out1   │
│ ┌─────┐   ┌────────┐     │      │  ────→  │        │        │
│ │Doc2 │→ │Mapper 2│→ (k2,v2)────│ SORT │         │        │
│ └─────┘   └────────┘   ↗  │  BY  │         └────────┘        │
│                          │ KEY  │                            │
│ ┌─────┐   ┌────────┐     │      │  (k2,*) ┌────────┐        │
│ │Doc3 │→ │Mapper 3│→ (k1,v3)────│      │  ────→  │Reducer2│→Out2   │
│ └─────┘   └────────┘     └──────┘         │        │        │
│                                           └────────┘        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 MapReduce Functions

```python
# MAP: Transform input to key-value pairs
def map(key, value):
    # Process one input record
    # Emit zero or more (key, value) pairs
    emit(new_key, new_value)

# REDUCE: Aggregate values for each key
def reduce(key, values[]):
    # Process all values for one key
    result = aggregate(values)
    emit(key, result)
```

### 6.3 Common MapReduce Patterns

| Pattern | Example | Map Output | Reduce Op |
|---------|---------|------------|-----------|
| **Word Count** | Count words in docs | (word, 1) | sum |
| **Summation** | Total sales | (category, amount) | sum |
| **Average** | Avg temperature | (city, (temp, 1)) | sum/count |
| **Max/Min** | Highest score | (user, score) | max |
| **Filter** | Find matches | (match, record) | identity |

---

## Session 7: kNN & Decision Trees Parallelization

### 7.1 kNN (k-Nearest Neighbors Parallelization

```
Sequential kNN:
  For each query point q:
    1. Compute distance to ALL training points
    2. Find k nearest neighbors
    3. Vote on class label

Parallel kNN (Data Parallel):
  ┌───────────────────────────────────────────────┐
  │ Partition training data among p processors    │
  │                                               │
  │ Each processor:                               │
  │   - Compute distances to local data          │
  │   - Find LOCAL k nearest neighbors           │
  │                                               │
  │ REDUCE:                                       │
  │   - Merge all local k-nearest lists          │
  │   - Select GLOBAL k nearest                  │
  │   - Vote for final prediction                │
  └───────────────────────────────────────────────┘
```

### 7.2 Decision Trees Parallelization

```
Decision Tree Construction (ID3 Algorithm):
──────────────────────────────────────────
1. If all examples have same label → Return leaf
2. If no features left → Return leaf with majority label
3. Choose feature F with max Information Gain
4. For each value v of F:
   - Create branch
   - Recursively build subtree for subset with F=v

Parallelization Strategy (Task Parallelism):
  ┌───────────────────────────────────────────────┐
  │ At each level:                                │
  │   - Each branch can be built INDEPENDENTLY    │
  │   - Assign different branches to processors   │
  │                                               │
  │ Number of parallel tasks = number of values   │
  │ of the chosen feature                         │
  └───────────────────────────────────────────────┘
```

### 7.3 Information Gain

$$\text{IG}(S, F) = H(S) - \sum_{v \in Values(F)} \frac{|S_v|}{|S|} H(S_v)$$

Where:
- H(S) = Entropy of set S
- F = Feature being tested
- Sᵥ = Subset of S where F = v

$$H(S) = -\sum_{c} p_c \log_2(p_c)$$

---

# Module 3: Scale-out ML Systems

## Session 8: Parameter Server & Distributed SGD ⭐ KEY CONCEPT

### 8.1 Distributed ML Challenges

```
Training Data Size: 1TB to 1PB
Model Parameters: 10⁹ to 10¹² parameters

Examples:
  - Online Recommender: Millions of user profiles
  - Ad Click Predictor: High-dimensional feature vectors
```

### 8.2 Parameter Server Architecture

```
┌─────────────────────────────────────────────────────────────┐
│               PARAMETER SERVER MODEL                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│                 ┌───────────────────┐                       │
│                 │  PARAMETER SERVER │                       │
│                 │  (stores w)       │                       │
│                 └─────────┬─────────┘                       │
│                           │                                  │
│            ┌──────────────┼──────────────┐                  │
│            │              │              │                  │
│      ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐           │
│      │  WORKER   │  │  WORKER   │  │  WORKER   │           │
│      │    W0     │  │    W1     │  │    Wp     │           │
│      │  Data D0  │  │  Data D1  │  │  Data Dp  │           │
│      └───────────┘  └───────────┘  └───────────┘           │
│                                                              │
│  Each iteration:                                            │
│    1. Workers PULL current w from server                    │
│    2. Workers compute local gradients on their data         │
│    3. Workers PUSH gradients to server                      │
│    4. Server aggregates and updates w                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 8.3 ML as Regularized Error Minimization

```
Training objective function:

  F(w) = Σᵢ L(xᵢ, yᵢ, w) + λR(w)

Where:
  w = model parameters (weights)
  L = loss function (prediction error)
  R = regularizer (penalizes complexity)
  λ = regularization strength
```

### 8.4 Stochastic Gradient Descent (SGD)

```
Batch Gradient Descent:
  w ← w - η · ∇F(w)
  
  Problem: Computing gradient over ALL data is expensive

Stochastic Gradient Descent:
  For each mini-batch B:
    w ← w - η · (1/|B|) Σᵢ∈B ∇L(xᵢ, yᵢ, w)
    
  Advantage: Updates after each mini-batch
             Better for large datasets
```

### 8.5 Distributed SGD

```
Synchronous SGD:
  ┌────────────────────────────────────────────────┐
  │ 1. All workers compute gradients in parallel   │
  │ 2. BARRIER: Wait for all workers               │
  │ 3. Aggregate gradients (average)               │
  │ 4. Update model                                │
  │ 5. Repeat                                      │
  │                                                │
  │ ✓ Consistent                                   │
  │ ✗ Slow (waits for stragglers)                 │
  └────────────────────────────────────────────────┘

Asynchronous SGD:
  ┌────────────────────────────────────────────────┐
  │ 1. Worker computes gradient                    │
  │ 2. Worker sends update immediately             │
  │ 3. Server applies update                       │
  │ 4. Worker pulls new model, continues           │
  │                                                │
  │ ✓ Fast (no waiting)                           │
  │ ✗ Stale gradients (may hurt convergence)      │
  └────────────────────────────────────────────────┘
```

---

## Session 9: Neural Network Optimization & Locality

### 9.1 Locality of Reference

```
Two types of locality observed in programs:

TEMPORAL LOCALITY:
  - Recently accessed locations will be accessed again
  - E.g., loop variables, frequently used data
  
SPATIAL LOCALITY:
  - Locations near recently accessed will be accessed
  - E.g., array elements, sequential instructions
```

### 9.2 Memory Hierarchy Optimization

```
System designers exploit locality:

CACHING (temporal locality):
  - Keep recently accessed data close to processor
  
PRE-FETCHING (spatial locality):
  - Load nearby data before it's needed
  
BLOCKING/BUFFERING:
  - Access data in large chunks
  - Amortizes setup costs (disk seek, network latency)
```

### 9.3 Matrix Multiplication Optimization ⭐ KEY EXAMPLE

**Naive Algorithm (IJK order):**
```c
// Poor cache performance
for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
        for (k = 0; k < n; k++)
            c[i][j] += a[i][k] * b[k][j];

// Access pattern:
// a[i][*] accessed row-wise (good)
// b[*][j] accessed column-wise (BAD - cache misses!)
```

**Cache-Aware Algorithm (IKJ order):**
```c
// Better cache performance
for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) c[i][j] = 0;
    for (k = 0; k < n; k++)
        for (j = 0; j < n; j++)
            c[i][j] += a[i][k] * b[k][j];
}

// Access pattern:
// a[i][k] accessed once per outer iteration (good)
// b[k][*] accessed row-wise (GOOD!)
// c[i][*] cached and reused (BEST!)
```

### 9.4 Performance Comparison (from Lecture 9)

| Method | n=256 | n=512 | n=1024 | n=2048 | n=4096 |
|--------|-------|-------|--------|--------|--------|
| **IJK** | 0.11s | 0.93s | 10.41s | ~450s | ~4026s |
| **IKJ** | 0.14s | 1.12s | 8.98s | ~73s | ~581s |
| **Speedup** | 0.80 | 0.83 | 1.16 | **6.19** | **6.93** |

**Key insight:** Same algorithm, different loop order → **7x speedup** for large matrices!

---

# 📋 Quick Reference Formulas

## Speedup & Amdahl's Law
```
Speedup(p) = T_seq / T_par(p)

Amdahl's Law: Speedup(p) = 1 / (f + (1-f)/p)
  where f = serial fraction, p = processors

Max Speedup = 1/f  (when p → ∞)
```

## Efficiency
```
Efficiency(p) = Speedup(p) / p

Ideal efficiency = 1 (100%)
```

## k-Means Complexity
```
Per iteration: O(n × k × d)
  where n = data points, k = clusters, d = dimensions
```

## Information Gain
```
IG(S, F) = H(S) - Σᵥ (|Sᵥ|/|S|) × H(Sᵥ)

Entropy: H(S) = -Σ pᵢ log₂(pᵢ)
```

## SGD Update
```
w ← w - η × ∇L(w)
  where η = learning rate
```

---

# 📝 Practice Problems

## Problem 1: Amdahl's Law

A program runs in 200 seconds. 30% of it is inherently sequential.

a) What is the maximum possible speedup?
b) What speedup can be achieved with 8 processors?
c) How many processors are needed to achieve 2.5x speedup?

**Solution:**
```
Given: T_seq = 200s, f = 0.3

a) Max Speedup = 1/f = 1/0.3 = 3.33

b) Speedup(8) = 1/(0.3 + 0.7/8) = 1/(0.3 + 0.0875) = 1/0.3875 = 2.58

c) 2.5 = 1/(0.3 + 0.7/p)
   0.3 + 0.7/p = 0.4
   0.7/p = 0.1
   p = 7 processors
```

## Problem 2: k-Means Speedup

You have 10,000 data points, 5 clusters. Sequential time = 50ms per iteration.

a) If you use 10 processors, estimate the parallel time (assume ideal speedup).
b) What communication overhead would reduce speedup to only 5x?

**Solution:**
```
a) Ideal speedup = 10
   T_par = 50ms / 10 = 5ms per iteration

b) If actual speedup = 5:
   T_par = 50/5 = 10ms
   Computation time = 5ms
   Communication overhead = 10 - 5 = 5ms
```

## Problem 3: MapReduce Word Count

Write map and reduce functions for counting word frequency.

**Solution:**
```python
def map(doc_id, doc_text):
    for word in doc_text.split():
        emit(word, 1)

def reduce(word, counts):
    emit(word, sum(counts))
```

## Problem 4: Matrix Multiplication Cache Analysis

For n×n matrix multiplication, the naive IJK algorithm has O(n³) memory accesses.

a) How many cache misses for B matrix in IJK order (assuming cache holds one row)?
b) How many cache misses for B matrix in IKJ order?

**Solution:**
```
a) IJK order: B[k][j] accessed column-wise
   - Each B[k][j] access is a cache miss (different rows)
   - Total misses for B: O(n³)

b) IKJ order: B[k][j] accessed row-wise
   - Row B[k][*] loaded once per (i,k) iteration
   - Total misses for B: O(n²)
   
   Improvement: O(n) reduction in cache misses!
```

---

# 📺 Recommended Resources

## NPTEL Lectures
- Parallel Computing (IIT Kanpur)
- Distributed Systems (IIT Kharagpur)
- High Performance Computing (IIT Madras)

## Topics to Review
1. Amdahl's Law and speedup calculations
2. MapReduce programming model
3. k-Means parallelization
4. Parameter Server architecture
5. Cache locality and matrix multiplication
6. Shared vs Distributed memory models

---

**Good luck with your exam! 🍀**
