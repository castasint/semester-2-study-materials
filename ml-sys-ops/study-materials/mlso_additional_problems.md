# 📝 ML System Optimization - Additional Practice Problems

> **AIMLCZG516 | Midterm Prep | More Practice for Sessions 1-9**

---

## SECTION A: More Amdahl's Law Problems

### Problem 13: Serial Fraction Identification

A parallel program running on 4 processors achieves a speedup of 3.2. On 8 processors, it achieves a speedup of 5.0.

**Questions:**
a) Calculate the serial fraction using the 4-processor data. (2M)
b) Verify your answer using the 8-processor data. (2M)
c) Predict the speedup with 32 processors. (2M)

---

**Solution:**

```
a) Using 4 processors, Speedup = 3.2:
   3.2 = 1 / (f + (1-f)/4)
   3.2 = 1 / (f + 0.25 - 0.25f)
   3.2 = 1 / (0.75f + 0.25)
   0.75f + 0.25 = 1/3.2 = 0.3125
   0.75f = 0.0625
   f = 0.0833 (8.33% serial)

b) Verify with 8 processors:
   Speedup(8) = 1 / (0.0833 + 0.9167/8)
              = 1 / (0.0833 + 0.1146)
              = 1 / 0.1979
              = 5.05 ≈ 5.0 ✓ Verified!

c) Speedup with 32 processors:
   Speedup(32) = 1 / (0.0833 + 0.9167/32)
               = 1 / (0.0833 + 0.0286)
               = 1 / 0.1119
               = 8.94×
```

---

### Problem 14: Efficiency Threshold

A company requires at least 75% efficiency for their parallel system to be cost-effective.

**Questions:**
a) If the serial fraction is 5%, what's the maximum number of processors that maintains 75% efficiency? (3M)
b) If they need 16 processors, what's the maximum acceptable serial fraction? (3M)

---

**Solution:**

```
a) Efficiency = Speedup / p ≥ 0.75
   Speedup ≥ 0.75p
   
   1 / (f + (1-f)/p) ≥ 0.75p
   1 / (0.05 + 0.95/p) ≥ 0.75p
   
   Solving: 0.05 + 0.95/p ≤ 1/(0.75p)
   
   For p = 4: Speedup = 3.48, Eff = 0.87 ✓
   For p = 8: Speedup = 5.93, Eff = 0.74 ✗
   For p = 6: Speedup = 4.80, Eff = 0.80 ✓
   For p = 7: Speedup = 5.41, Eff = 0.77 ✓
   
   Answer: Maximum 7 processors

b) For 16 processors with 75% efficiency:
   Speedup = 0.75 × 16 = 12
   12 = 1 / (f + (1-f)/16)
   f + (1-f)/16 = 1/12 = 0.0833
   f + 0.0625 - 0.0625f = 0.0833
   0.9375f = 0.0208
   f = 0.022 = 2.2%
   
   Answer: Maximum serial fraction = 2.2%
```

---

### Problem 15: Communication Overhead

A distributed algorithm has:
- 10% serial fraction
- Communication time = 2ms per iteration
- Computation time = 20ms on single processor

**Questions:**
a) Calculate speedup with 8 processors ignoring communication. (2M)
b) Calculate actual speedup including communication. (2M)
c) At what number of processors does communication equal computation? (2M)

---

**Solution:**

```
a) Ignoring communication:
   f = 0.10
   Speedup(8) = 1 / (0.10 + 0.90/8)
              = 1 / (0.10 + 0.1125)
              = 1 / 0.2125
              = 4.71×
   T_compute = 20 / 4.71 = 4.25ms

b) Including communication:
   T_total = T_compute + T_comm = 4.25 + 2 = 6.25ms
   Actual Speedup = 20 / 6.25 = 3.2×
   
   Note: Communication reduces speedup from 4.71 to 3.2!

c) When communication = computation:
   T_par = 20 / Speedup(p) = T_comm
   20 × (0.10 + 0.90/p) = 2
   0.10 + 0.90/p = 0.1
   0.90/p = 0
   
   This never happens! Communication always adds, so:
   At p → ∞: T_compute → 20 × 0.10 = 2ms = T_comm
   
   Answer: At very large p, they become equal (both 2ms)
```

---

## SECTION B: More MapReduce Problems

### Problem 16: Average Calculation

Given sales data:
```
Store_A: 100, 200, 150
Store_B: 300, 250
Store_C: 175, 225, 200, 100
```

**Questions:**
a) Write MAP output for calculating average sales per store. (2M)
b) Write REDUCE logic. (2M)
c) Show final output. (2M)

---

**Solution:**

```
a) MAP output (emit (store, (sale, 1)) for each sale):
   
   ("Store_A", (100, 1))
   ("Store_A", (200, 1))
   ("Store_A", (150, 1))
   ("Store_B", (300, 1))
   ("Store_B", (250, 1))
   ("Store_C", (175, 1))
   ("Store_C", (225, 1))
   ("Store_C", (200, 1))
   ("Store_C", (100, 1))

b) REDUCE logic:
   
   def reduce(store, values):
       total_sum = 0
       total_count = 0
       for (sale, count) in values:
           total_sum += sale
           total_count += count
       average = total_sum / total_count
       emit(store, average)

c) Final output:
   
   After shuffle:
   "Store_A" → [(100,1), (200,1), (150,1)]
   "Store_B" → [(300,1), (250,1)]
   "Store_C" → [(175,1), (225,1), (200,1), (100,1)]
   
   After reduce:
   ("Store_A", 450/3) = ("Store_A", 150)
   ("Store_B", 550/2) = ("Store_B", 275)
   ("Store_C", 700/4) = ("Store_C", 175)
```

---

### Problem 17: Top-K with MapReduce

Find the top 3 most frequent words in a document collection.

**Questions:**
a) Describe the MapReduce approach (may need multiple phases). (3M)
b) Why can't this be done in a single MapReduce job efficiently? (3M)

---

**Solution:**

```
a) Two-Phase Approach:

PHASE 1: Word Count
  MAP:    (doc) → [(word, 1), ...]
  REDUCE: (word, [1,1,1,...]) → (word, count)
  
PHASE 2: Top-K Selection
  MAP:    (word, count) → ("top", (word, count))
          # All emit same key to force single reducer
  REDUCE: ("top", [(w1,c1), (w2,c2), ...]) →
          Sort by count, take top 3

b) Single job is inefficient because:

   1. Finding global top-K requires comparing ALL words
   2. In standard MapReduce, reducers work independently
   3. Each reducer only sees subset of words
   4. No way to compare across reducers in same job
   
   Alternatives:
   - Use combiner for local top-K, then single reducer
   - Use sampling approximation
   - Use specialized operators (Spark: takeOrdered)
```

---

### Problem 18: Join Operation

**Users table:**
```
user_id, name
1, Alice
2, Bob
3, Carol
```

**Orders table:**
```
order_id, user_id, amount
101, 1, 50
102, 2, 30
103, 1, 70
104, 3, 40
```

Write MapReduce to join these tables.

---

**Solution:**

```python
# MAP phase (run on both tables)
def map_users(line):
    user_id, name = parse(line)
    emit(user_id, ("user", name))

def map_orders(line):
    order_id, user_id, amount = parse(line)
    emit(user_id, ("order", order_id, amount))

# After shuffle, grouped by user_id:
# 1 → [("user", "Alice"), ("order", 101, 50), ("order", 103, 70)]
# 2 → [("user", "Bob"), ("order", 102, 30)]
# 3 → [("user", "Carol"), ("order", 104, 40)]

# REDUCE phase
def reduce(user_id, values):
    # Separate users and orders
    user_name = None
    orders = []
    
    for value in values:
        if value[0] == "user":
            user_name = value[1]
        else:
            orders.append((value[1], value[2]))
    
    # Emit joined records
    for order_id, amount in orders:
        emit((user_id, user_name, order_id, amount))

# Final output:
# (1, "Alice", 101, 50)
# (1, "Alice", 103, 70)
# (2, "Bob", 102, 30)
# (3, "Carol", 104, 40)
```

---

## SECTION C: More k-Means Problems

### Problem 19: k-Means Iteration

Given 2D points: {(1,1), (2,2), (8,8), (9,9), (10,8)}
Initial centers: C1 = (1,1), C2 = (10,10)

**Questions:**
a) Assign each point to nearest center. (2M)
b) Compute new centers. (2M)
c) Did the algorithm converge? (2M)

---

**Solution:**

```
a) Distance calculations:
   
   Point (1,1):
     d(C1) = √((1-1)² + (1-1)²) = 0
     d(C2) = √((1-10)² + (1-10)²) = √162 = 12.73
     → Assign to C1
   
   Point (2,2):
     d(C1) = √((2-1)² + (2-1)²) = √2 = 1.41
     d(C2) = √((2-10)² + (2-10)²) = √128 = 11.31
     → Assign to C1
   
   Point (8,8):
     d(C1) = √((8-1)² + (8-1)²) = √98 = 9.90
     d(C2) = √((8-10)² + (8-10)²) = √8 = 2.83
     → Assign to C2
   
   Point (9,9):
     d(C1) = √((9-1)² + (9-1)²) = √128 = 11.31
     d(C2) = √((9-10)² + (9-10)²) = √2 = 1.41
     → Assign to C2
   
   Point (10,8):
     d(C1) = √((10-1)² + (8-1)²) = √130 = 11.40
     d(C2) = √((10-10)² + (8-10)²) = √4 = 2.00
     → Assign to C2
   
   Cluster 1: {(1,1), (2,2)}
   Cluster 2: {(8,8), (9,9), (10,8)}

b) New centers:
   
   C1_new = ((1+2)/2, (1+2)/2) = (1.5, 1.5)
   C2_new = ((8+9+10)/3, (8+9+8)/3) = (9, 8.33)

c) Convergence check:
   
   C1 changed: (1,1) → (1.5, 1.5)
   C2 changed: (10,10) → (9, 8.33)
   
   Centers moved, so NOT converged. Need more iterations.
```

---

### Problem 20: k-Means Complexity Analysis

A k-means implementation processes 1 million points in 3D space with k=100 clusters.

**Questions:**
a) Calculate operations per iteration. (2M)
b) If each operation takes 1 nanosecond, what's the iteration time? (2M)
c) With 100 nodes, what's the parallel time (assume ideal)? (2M)

---

**Solution:**

```
a) Operations per iteration:
   
   ASSIGN phase:
   - Each point: compute distance to k centers
   - Distance = 3 subtractions + 3 squares + 2 additions + 1 sqrt
   - Approximate: 10 ops per distance
   - Per point: k × 10 = 100 × 10 = 1000 ops
   - All points: n × k × 10 = 10^6 × 10^3 = 10^9 ops
   
   UPDATE phase:
   - Sum all points in each cluster: n × d = 3 × 10^6 ops
   - Divide by count: k × d = 300 ops
   
   Total: ~10^9 operations per iteration

b) Sequential time:
   
   Time = 10^9 ops × 1 ns/op = 10^9 ns = 1 second per iteration

c) Parallel time (100 nodes, ideal):
   
   T_parallel = T_sequential / p = 1s / 100 = 10 ms per iteration
   
   (Plus communication overhead in practice)
```

---

## SECTION D: More Distributed Systems Problems

### Problem 21: Sync vs Async Trade-offs

Consider training a large neural network on 10 workers.

| Scenario | Computation Time | Network Latency |
|----------|-----------------|-----------------|
| A | 100ms | 10ms |
| B | 100ms | 50ms |
| C | 20ms | 50ms |

**Questions:**
a) Calculate iteration time for synchronous SGD in each scenario. (2M)
b) When would asynchronous SGD be preferred? (2M)
c) What is the "stale gradient" problem? (2M)

---

**Solution:**

```
a) Synchronous SGD iteration time:
   T_sync = T_compute + 2 × T_network (push + pull)
   
   Scenario A: 100 + 2(10) = 120ms
   Scenario B: 100 + 2(50) = 200ms
   Scenario C: 20 + 2(50) = 120ms

b) Async preferred when:
   
   - Network latency >> computation (Scenario C)
   - Workers have unequal speeds (stragglers)
   - Communication overhead dominates
   
   In Scenario B & C, communication is significant (50-250% of compute)
   Async would help by overlapping communication with computation.

c) Stale gradient problem:
   
   In async SGD:
   1. Worker pulls model version w_t
   2. Worker computes gradient ∇L(w_t)
   3. Before pushing, server already at w_{t+k}
   4. Gradient was computed on OLD model!
   
   Impact:
   - Updates based on outdated information
   - May push model in wrong direction
   - Can slow convergence or cause divergence
   
   Solutions:
   - Bounded staleness (limit k)
   - Learning rate adjustment
   - Gradient correction
```

---

### Problem 22: Data Distribution Strategies

You have 10TB of training data and 100 worker nodes.

**Questions:**
a) Describe random sharding vs hash-based sharding. (2M)
b) Which is better for k-means? Why? (2M)
c) What's the "data locality" principle? (2M)

---

**Solution:**

```
a) Sharding strategies:

   RANDOM SHARDING:
   - Randomly assign data to nodes
   - Each node gets ~100GB
   - Data distribution may be uneven
   - Simple to implement
   
   HASH-BASED SHARDING:
   - hash(data_id) % num_nodes determines location
   - Deterministic: same data always on same node
   - Even distribution guaranteed
   - Useful for lookups

b) For k-means:
   
   Random sharding is BETTER because:
   - k-means processes all points equally
   - No need to lookup specific points
   - Random ensures each node has representative sample
   - Cluster assignments don't depend on data location
   
   Hash-based offers no advantage for k-means since we
   never need to find a specific point.

c) Data locality principle:
   
   "Move computation to data, not data to computation"
   
   - Data transfer is expensive (network is slow)
   - Computation should happen where data resides
   - Minimize data movement across network
   
   Example: MapReduce schedules map tasks on nodes 
   that already have the input data blocks.
```

---

## SECTION E: More Cache & Optimization Problems

### Problem 23: Row vs Column Access

Consider a 1000×1000 matrix stored in row-major order.
Cache line size = 64 bytes, each element = 8 bytes.

**Questions:**
a) How many elements per cache line? (2M)
b) For row-wise traversal, how many cache misses? (2M)
c) For column-wise traversal, how many cache misses? (2M)

---

**Solution:**

```
a) Elements per cache line:
   64 bytes / 8 bytes = 8 elements per cache line

b) Row-wise traversal (good locality):
   - Access: a[0][0], a[0][1], ..., a[0][999], a[1][0], ...
   - Every 8 elements, one cache miss
   - Total accesses: 1000 × 1000 = 10^6
   - Cache misses: 10^6 / 8 = 125,000 misses

c) Column-wise traversal (bad locality):
   - Access: a[0][0], a[1][0], a[2][0], ..., a[999][0], a[0][1], ...
   - Each element in different cache line (rows are 8000 bytes apart)
   - Every access is a cache miss!
   - Cache misses: 10^6 misses
   
   Ratio: 1,000,000 / 125,000 = 8× more misses!
```

---

### Problem 24: Blocked Matrix Multiplication

For n×n matrix multiplication, blocking uses b×b blocks.

**Questions:**
a) Why does blocking improve cache performance? (2M)
b) What is the optimal block size? (2M)
c) Write pseudocode for blocked matrix multiply. (2M)

---

**Solution:**

```
a) Why blocking helps:

   Without blocking:
   - B matrix accessed column-wise (bad)
   - Each B element accessed n times, but evicted between uses
   
   With blocking:
   - Work on small b×b blocks that fit in cache
   - Each block loaded once, used completely
   - Temporal locality: reuse data while in cache
   
   Key insight: If block fits in cache, no evictions during
   processing of that block.

b) Optimal block size:
   
   Block should fit in L1 cache:
   - Three b×b blocks needed (A, B, C subblocks)
   - 3 × b² × 8 bytes ≤ L1 cache size
   
   For L1 = 32KB:
   3 × b² × 8 ≤ 32768
   b² ≤ 1365
   b ≤ 36
   
   Typical choice: b = 32 (power of 2)

c) Blocked matrix multiply:

   for i_block in range(0, n, b):
       for j_block in range(0, n, b):
           for k_block in range(0, n, b):
               # Multiply blocks
               for i in range(i_block, min(i_block+b, n)):
                   for j in range(j_block, min(j_block+b, n)):
                       for k in range(k_block, min(k_block+b, n)):
                           C[i][j] += A[i][k] * B[k][j]
```

---

### Problem 25: Memory Access Pattern Analysis

A neural network layer performs: output = activation(W × input + bias)
- W: 1024 × 1024 matrix
- input: 1024 × 1 vector
- Operations are performed element-by-element

**Questions:**
a) What's the memory access pattern for W × input? (2M)
b) How can this be optimized for cache? (2M)
c) Calculate speedup from optimization (assume 8 elements/cache line). (2M)

---

**Solution:**

```
a) Access pattern:

   Matrix-vector multiply: out[i] = Σ_j W[i][j] × input[j]
   
   For each output element i:
   - Access row i of W: W[i][0], W[i][1], ..., W[i][1023]
   - Access all of input: input[0], input[1], ..., input[1023]
   
   W is accessed row-wise (GOOD)
   Input is accessed 1024 times (needs to stay in cache)

b) Optimization strategies:

   1. ENSURE INPUT IS CACHED:
      - Input is only 1024 × 8 = 8KB (fits in L1!)
      - Should be loaded once and stay in cache
   
   2. PROCESS MULTIPLE OUTPUTS:
      - Load a block of rows of W
      - Compute multiple outputs before moving on
   
   3. USE SIMD:
      - Process 8 elements at once (AVX)
      - Vectorize the inner loop

c) Speedup calculation:

   Unoptimized (assume input reloaded each time):
   - W accesses: 1024 × 1024 = 1M, misses = 1M/8 = 128K
   - Input accesses: 1024 × 1024 = 1M, misses = 1M (worst case)
   - Total misses: 1.128M
   
   Optimized (input cached):
   - W misses: 128K (same)
   - Input misses: 1024/8 = 128 (loaded once!)
   - Total misses: 128K + 128 ≈ 128K
   
   Speedup in cache misses: 1.128M / 128K ≈ 8.8×
```

---

## 📊 Quick Answer Key

| Problem | Key Answer |
|---------|------------|
| 13 | f = 8.33% |
| 14 | Max 7 processors for 75% efficiency |
| 15 | Actual speedup = 3.2× (not 4.71×) |
| 16 | Average: A=150, B=275, C=175 |
| 17 | Need two MapReduce phases |
| 18 | Join on user_id key |
| 19 | Centers converge toward cluster means |
| 20 | 10^9 ops, 1s sequential, 10ms parallel |
| 21 | Async better when network >> compute |
| 22 | Random sharding for k-means |
| 23 | Column-wise: 8× more cache misses |
| 24 | Block size ~32 for 32KB L1 cache |
| 25 | 8.8× speedup from caching input |

---

**Keep practicing! 💪**
