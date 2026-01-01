# 📝 ML System Optimization - Practice Problems

> **AIMLCZG516 | Midterm Prep | Sessions 1-9**

---

## SECTION A: Speedup & Amdahl's Law (6 marks each)

### Problem 1: Basic Speedup Calculation

A machine learning training algorithm takes 500 seconds on a single processor. Profiling shows that 25% of the code is inherently sequential (file I/O, synchronization).

**Questions:**
a) Calculate the maximum theoretical speedup achievable. (2M)
b) Calculate the speedup with 16 processors. (2M)
c) How many processors are needed to achieve 3× speedup? (2M)

---

**Solution:**

```
Given: T_seq = 500s, f = 0.25 (serial fraction)

a) Maximum Speedup (p → ∞):
   Speedup_max = 1/f = 1/0.25 = 4×
   
   Answer: Maximum speedup = 4×

b) Speedup with 16 processors:
   Speedup(16) = 1 / (f + (1-f)/p)
                = 1 / (0.25 + 0.75/16)
                = 1 / (0.25 + 0.047)
                = 1 / 0.297
                = 3.37×
   
   T_parallel = 500 / 3.37 = 148.4 seconds
   
   Answer: Speedup = 3.37×, T_parallel ≈ 148 seconds

c) For 3× speedup:
   3 = 1 / (0.25 + 0.75/p)
   0.25 + 0.75/p = 1/3 = 0.333
   0.75/p = 0.333 - 0.25 = 0.083
   p = 0.75 / 0.083 = 9.04
   
   Answer: Need at least 10 processors
```

---

### Problem 2: Efficiency Analysis

A distributed k-means algorithm achieves the following speedups:

| Processors (p) | Speedup |
|----------------|---------|
| 2 | 1.8 |
| 4 | 3.2 |
| 8 | 5.5 |
| 16 | 8.0 |

**Questions:**
a) Calculate the efficiency for each case. (2M)
b) What fraction of the code appears to be sequential? (2M)
c) Will adding more processors significantly help? Justify. (2M)

---

**Solution:**

```
a) Efficiency = Speedup / p

   p=2:  Eff = 1.8/2 = 0.90 (90%)
   p=4:  Eff = 3.2/4 = 0.80 (80%)
   p=8:  Eff = 5.5/8 = 0.69 (69%)
   p=16: Eff = 8.0/16 = 0.50 (50%)

b) Using Amdahl's Law with p=16:
   8 = 1 / (f + (1-f)/16)
   f + (1-f)/16 = 0.125
   f + 0.0625 - 0.0625f = 0.125
   0.9375f = 0.0625
   f ≈ 0.067 (6.7%)
   
   Answer: ~6.7% serial fraction

c) Max speedup = 1/f = 1/0.067 ≈ 15×
   
   With 16 processors already at 8×, we've reached about 53% of max.
   Adding more processors has diminishing returns.
   At p=32: Speedup ≈ 10.7× (only 33% improvement for 2× processors)
   
   Answer: Diminishing returns; not significantly helpful
```

---

### Problem 3: Comparing Parallel Approaches

Two approaches are proposed for parallelizing an ML algorithm:

- **Approach A**: 10% serial fraction, requires 5ms communication per iteration
- **Approach B**: 5% serial fraction, requires 20ms communication per iteration

Sequential running time = 1000ms per iteration.

**Questions:**
a) Calculate parallel time for both with 10 processors (ignore communication). (2M)
b) Include communication overhead. Which approach is better? (2M)
c) At what number of processors do both approaches have equal performance? (2M)

---

**Solution:**

```
a) Ignoring communication:
   
   Approach A: Speedup = 1/(0.1 + 0.9/10) = 1/0.19 = 5.26
               T_par = 1000/5.26 = 190ms
   
   Approach B: Speedup = 1/(0.05 + 0.95/10) = 1/0.145 = 6.90
               T_par = 1000/6.90 = 145ms

b) Including communication:
   
   Approach A: T_total = 190 + 5 = 195ms
   Approach B: T_total = 145 + 20 = 165ms
   
   Answer: Approach B is better (165ms < 195ms)

c) Setting them equal:
   1000/(1/(0.1 + 0.9/p)) + 5 = 1000/(1/(0.05 + 0.95/p)) + 20
   
   Solving numerically: p ≈ 25 processors
   
   At p > 25, Approach A becomes better (lower communication cost)
```

---

## SECTION B: k-Means & MapReduce (6 marks each)

### Problem 4: k-Means Parallelization

You have a dataset of 100,000 points in 50 dimensions, clustering into k=10 clusters.

**Questions:**
a) Write the time complexity for sequential k-means (per iteration). (2M)
b) If you use 20 processors with data parallelism, what's the parallel complexity? (2M)
c) Describe the communication needed between iterations. (2M)

---

**Solution:**

```
a) Sequential complexity per iteration:
   
   ASSIGN phase: O(n × k × d) = O(100000 × 10 × 50) = O(50,000,000)
   UPDATE phase: O(n × d) = O(100000 × 50) = O(5,000,000)
   
   Total: O(n × k × d) = O(50 million operations)

b) Parallel complexity (20 processors):
   
   ASSIGN phase: O(n/p × k × d) = O(5000 × 10 × 50) = O(2,500,000)
   UPDATE phase: O(n/p × d) + O(k × d) communication = O(250,000) + O(500)
   
   Speedup ≈ 20× (nearly linear)

c) Communication per iteration:
   
   - Each processor sends: local cluster sums and counts
     = k × d floating point numbers + k counts
     = 10 × 50 + 10 = 510 values
   - All-reduce to compute global centroids
   - Broadcast new centroids to all processors
   - Total: O(p × k × d) = O(20 × 10 × 50) = 10,000 values
```

---

### Problem 5: MapReduce Word Count

Given the following document:
```
"the cat sat on the mat the cat saw a rat"
```

**Questions:**
a) Show the output of the MAP phase. (2M)
b) Show the intermediate state after SHUFFLE (group by key). (2M)
c) Show the final output after REDUCE. (2M)

---

**Solution:**

```
a) MAP output (one (word, 1) per word):
   
   ("the", 1), ("cat", 1), ("sat", 1), ("on", 1), ("the", 1),
   ("mat", 1), ("the", 1), ("cat", 1), ("saw", 1), ("a", 1), ("rat", 1)

b) After SHUFFLE (grouped by key):
   
   "a"   → [1]
   "cat" → [1, 1]
   "mat" → [1]
   "on"  → [1]
   "rat" → [1]
   "sat" → [1]
   "saw" → [1]
   "the" → [1, 1, 1]

c) REDUCE output (sum the lists):
   
   ("a", 1)
   ("cat", 2)
   ("mat", 1)
   ("on", 1)
   ("rat", 1)
   ("sat", 1)
   ("saw", 1)
   ("the", 3)
```

---

### Problem 6: k-Means MapReduce Implementation

**Question:** 
Write pseudocode for k-means using MapReduce. Clearly show MAP and REDUCE functions for both ASSIGN and UPDATE phases. (6M)

---

**Solution:**

```python
# Global: K cluster centers are available to all mappers

# PHASE 1: ASSIGN (Map + Reduce)
def map_assign(point_id, point):
    """Assign each point to nearest cluster."""
    min_dist = infinity
    nearest_cluster = -1
    
    for j in range(K):
        dist = euclidean_distance(point, centers[j])
        if dist < min_dist:
            min_dist = dist
            nearest_cluster = j
    
    emit(nearest_cluster, point)

def reduce_assign(cluster_id, points_list):
    """Just pass through - points are now grouped by cluster."""
    for point in points_list:
        emit(cluster_id, point)


# PHASE 2: UPDATE (Map + Reduce)
def map_update(cluster_id, point):
    """Emit point with count for averaging."""
    emit(cluster_id, (point, 1))

def reduce_update(cluster_id, point_count_list):
    """Compute new centroid."""
    total_point = zero_vector()
    count = 0
    
    for (point, c) in point_count_list:
        total_point += point
        count += c
    
    new_center = total_point / count
    emit(cluster_id, new_center)


# MAIN LOOP
def kmeans_mapreduce(data, k, max_iter):
    centers = random_sample(data, k)
    
    for iteration in range(max_iter):
        # Run ASSIGN phase
        assignments = mapreduce(data, map_assign, reduce_assign)
        
        # Run UPDATE phase  
        new_centers = mapreduce(assignments, map_update, reduce_update)
        
        if converged(centers, new_centers):
            break
        centers = new_centers
    
    return centers
```

---

## SECTION C: Distributed Systems & Parameter Server (6 marks each)

### Problem 7: Parameter Server Architecture

A distributed ML system has:
- 10 worker nodes
- 1 parameter server
- Model with 10 million parameters
- Each parameter = 4 bytes (float)

**Questions:**
a) How much data is transferred per iteration (all workers to server)? (2M)
b) If network bandwidth is 1 Gbps, what's the minimum communication time? (2M)
c) If each worker takes 100ms to compute gradients, what's the iteration time? (2M)

---

**Solution:**

```
a) Data per iteration:
   - Each worker sends: 10M parameters × 4 bytes = 40 MB
   - 10 workers total: 10 × 40 MB = 400 MB per iteration
   
   Plus pulling new parameters: another 10 × 40 MB = 400 MB
   Total: 800 MB per iteration

b) Communication time:
   - Bandwidth: 1 Gbps = 125 MB/s
   - Time = 800 MB / 125 MB/s = 6.4 seconds
   
   Note: This is minimum; actual time may be higher due to
   serialization and protocol overhead

c) Total iteration time:
   - Gradient computation: 100ms (parallel)
   - Communication: 6400ms (sequential bottleneck)
   - Total: ~6.5 seconds per iteration
   
   Note: Communication DOMINATES! This is a common issue.
```

---

### Problem 8: Synchronous vs Asynchronous SGD

**Questions:**
a) Compare synchronous and asynchronous SGD in a table. (3M)
b) Why might asynchronous SGD converge to a different solution? (3M)

---

**Solution:**

```
a) Comparison Table:

| Aspect | Synchronous SGD | Asynchronous SGD |
|--------|-----------------|------------------|
| Barrier | Wait for all workers | No waiting |
| Consistency | All use same model version | Stale gradients possible |
| Speed | Limited by slowest worker | Faster iterations |
| Convergence | More stable | May oscillate |
| Scaling | Poor with stragglers | Better scaling |
| Implementation | Simpler | More complex |

b) Why different convergence:

   Asynchronous SGD may converge differently because:
   
   1. STALE GRADIENTS: Workers compute gradients on old model
      - Worker pulls w_t, computes gradient
      - By time it pushes, server has w_(t+k)
      - Gradient is for wrong model version!
   
   2. INCONSISTENT UPDATES: Updates can conflict
      - Two workers update same parameters simultaneously
      - Final value depends on timing
   
   3. MINI-BATCH VARIANCE: Effective batch size varies
      - Some updates use more current information
      - Introduces additional noise
   
   Result: May find different local minimum or take longer to converge.
```

---

## SECTION D: Cache Locality & Optimization (6 marks each)

### Problem 9: Matrix Multiplication Analysis

Consider matrix multiplication C = A × B for n×n matrices.

**Questions:**
a) How many memory accesses for the naive IJK implementation? (2M)
b) Explain why IKJ order is more cache-efficient. (2M)
c) For n=1024, cache line = 64 bytes, float = 4 bytes, calculate cache misses. (2M)

---

**Solution:**

```
a) Memory accesses for IJK:
   
   for i: for j: for k:
       c[i][j] += a[i][k] * b[k][j]
   
   Per inner loop: 3 accesses (a, b, c)
   Total iterations: n³
   Total accesses: 3n³
   
   For n=1024: 3 × 1024³ = 3.2 billion accesses

b) Why IKJ is better:
   
   IJK order:
   - a[i][k]: Row-wise access (GOOD) ✓
   - b[k][j]: Column-wise access (BAD) ✗
     → Each b[k][j] in different cache line!
   
   IKJ order:
   - a[i][k]: Accessed once per k iteration (GOOD) ✓
   - b[k][j]: Row-wise access (GOOD) ✓
   - c[i][j]: Same row reused (GOOD) ✓
   
   IKJ exploits spatial locality for B matrix!

c) Cache miss analysis for n=1024:
   
   Cache line = 64 bytes = 16 floats
   
   IJK for B (column access):
   - Each b[k][j] is in different row
   - Row width = 1024 × 4 = 4KB
   - Every access to B is a cache miss
   - B misses: n³ / 1 = n³ = 1 billion misses
   
   IKJ for B (row access):
   - Row loaded once, 16 elements used
   - B misses: n³ / 16 ≈ 64 million misses
   
   Improvement: 16× fewer cache misses
```

---

### Problem 10: Memory Hierarchy

A program accesses an array of 1 million integers (4 bytes each).

| Level | Size | Access Time |
|-------|------|-------------|
| L1 Cache | 32 KB | 1 ns |
| L2 Cache | 256 KB | 10 ns |
| L3 Cache | 8 MB | 50 ns |
| RAM | 16 GB | 200 ns |

**Questions:**
a) What's the average access time if 90% hit in L1, 5% in L2, 4% in L3, 1% in RAM? (2M)
b) If sequential access improves L1 hit rate to 98%, what's the new average? (2M)
c) Calculate the speedup from optimizing access pattern. (2M)

---

**Solution:**

```
a) Original average access time:
   
   T_avg = 0.90 × 1 + 0.05 × 10 + 0.04 × 50 + 0.01 × 200
        = 0.90 + 0.50 + 2.00 + 2.00
        = 5.40 ns

b) Optimized (98% L1, 1.5% L2, 0.4% L3, 0.1% RAM):
   
   T_avg = 0.98 × 1 + 0.015 × 10 + 0.004 × 50 + 0.001 × 200
        = 0.98 + 0.15 + 0.20 + 0.20
        = 1.53 ns

c) Speedup:
   
   Speedup = 5.40 / 1.53 = 3.53×
   
   Just by improving access pattern, we get 3.5× speedup!
```

---

## SECTION E: Conceptual Questions (3-5 marks each)

### Problem 11: Short Answers

a) Why is SPMD preferred over task parallelism for ML? (3M)

```
Answer:
1. SIMPLICITY: Same code on all processors, easier to write/debug
2. LOAD BALANCE: Data can be evenly distributed
3. SCALABILITY: Adding processors = adding data capacity
4. ML NATURE: Most ML operations are data-parallel
   (same operation on many data points)
```

b) What is the bottleneck in distributed SGD? (2M)

```
Answer:
COMMUNICATION is the main bottleneck because:
1. Parameters must be synchronized across workers
2. Network bandwidth << memory bandwidth
3. As model size grows, communication time dominates
```

c) How does data parallelism differ in shared memory vs distributed memory? (3M)

```
Answer:
Shared Memory:
- Data accessible to all processors
- No explicit data transfer needed
- Synchronization via locks/barriers
- E.g., OpenMP, pthreads

Distributed Memory:
- Data partitioned across nodes
- Explicit send/receive for data sharing
- Higher latency but more scalable
- E.g., MPI, Spark
```

---

### Problem 12: Design Question

Design a parallel algorithm for kNN classification with k=5 on a 4-node cluster. The training set has 1 million points, query set has 1000 points.

**Solution:**

```
DESIGN:
1. DATA DISTRIBUTION:
   - Split training data: 250,000 points per node
   - Replicate query points to all nodes (1000 × 4 bytes × dims)

2. LOCAL COMPUTATION (parallel):
   Each node:
   - For each query point q:
     - Find LOCAL top-5 nearest neighbors from local 250K points
     - Store (distance, class) for each

3. GLOBAL MERGE (reduce):
   - Gather all local top-5 lists (4 nodes × 5 = 20 candidates per query)
   - Select GLOBAL top-5 from 20 candidates
   - Vote: majority class among top-5

4. COMPLEXITY:
   Sequential: O(Q × N × d) = O(1000 × 1M × d)
   Parallel:   O(Q × N/p × d) + O(Q × p × k) communication
             = O(1000 × 250K × d) + O(1000 × 20)
   
   Speedup ≈ 4× (close to linear)

5. OPTIMIZATION:
   - Use KD-trees for faster neighbor search
   - Pre-filter distant points
   - Batch queries for better cache use
```

---

## 📊 Answer Key Summary

| Problem | Key Concepts | Marks |
|---------|--------------|-------|
| 1 | Amdahl's Law basic | 6 |
| 2 | Efficiency calculation | 6 |
| 3 | Communication overhead | 6 |
| 4 | k-Means complexity | 6 |
| 5 | MapReduce phases | 6 |
| 6 | k-Means MapReduce code | 6 |
| 7 | Parameter server analysis | 6 |
| 8 | Sync vs Async SGD | 6 |
| 9 | Cache analysis | 6 |
| 10 | Memory hierarchy | 6 |
| 11 | Short answers | 8 |
| 12 | System design | 6 |

---

**Good luck with your exam! 🍀**
