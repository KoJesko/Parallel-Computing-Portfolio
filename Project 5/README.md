# Project 5: Efficient Parallel Jacobi Relaxation on a Multiprocessor

Parallel Jacobi relaxation with a convergence test, a hand-written barrier and a hand-written max-aggregation.

---

## 1. Contents

The same algorithm is implemented twice: in C\* on the simulated shared-memory multiprocessor used in the course, and in OpenMP/C11 on real hardware. Both are checked against a sequential reference. Both use the same exchange pattern for the barrier and the aggregation. The C\* version uses the butterfly form; the OpenMP version uses the dissemination form, which also works when the thread count isn't a power of two (the 24-thread runs in section 7.2).

| File | What it is |
|---|---|
| `JacobiParallel.c` | Main deliverable. C\* parallel Jacobi: butterfly barrier, butterfly max-aggregation, ping-pong grids, convergence test. |
| `JacobiSequential.c` | C\* sequential reference with the same arithmetic on one process. Gives the T(1) baseline and the correctness reference. |
| `JacobiNaiveBarrier.c` | C\* control: linear counter barrier, lock-based aggregation, copy-back sweep. Measures what the efficient version saves. |
| `JacobiOpenMP.c` | Real-hardware version: dissemination barrier and all-reduce on cache-line-padded atomics, adaptive load balancing, barrier microbenchmark. |
| `runcstar.sh` | Drives the interactive C\* simulator non-interactively so runs can be scripted and captured. |
| `benchmark.sh` | C\* performance study: P = 1 to 32, efficient vs naive, speedup/efficiency table. |
| `benchmark-openmp.sh` | Real-hardware study: correctness, strong scaling, large grids, barrier cost. |
| `corelist.sh` | Builds an `OMP_PLACES` list with one thread per physical core, fastest cores first. |
| `capture.sh` | Renders the captured run logs as terminal-style PNG screen captures. |
| `Makefile` | `make`, `make run`, `make bench`, `make cstar`. |
| `results/` | Captured output of every sample run (`*.txt`) and the screen captures (`capture-*.png`). |

---

## 2. Problem and algorithm

Steady-state temperature of a square plate: Laplace's equation with Dirichlet boundaries. The top edge is held at 100°, the other three at 0°. Each Jacobi sweep replaces every interior point with the average of its four neighbours:

```
new[i][j] = ( old[i-1][j] + old[i+1][j] + old[i][j-1] + old[i][j+1] ) / 4
```

Sweeps repeat until the largest change at any point falls below a tolerance:

```
max over all interior points of |new[i][j] - old[i][j]|  <  TOL
```

The test is a property of the whole grid. On every sweep the P per-process maxima have to be combined into one global maximum, and that value has to reach every process. That step is where the synchronisation design matters.

---

## 3. What makes it efficient

### 3.1 Butterfly barrier: O(log P), lock-free

`Barrier()` synchronises in `log2 P` rounds. In round *k*, each process exchanges flags with the partner whose process number differs from its own in bit *k*:

```
round 0:  0<->1  2<->3  4<->5  6<->7     (pairs)
round 1:  0<->2  1<->3  4<->6  5<->7     (pairs of pairs)
round 2:  0<->4  1<->5  2<->6  3<->7     (groups of four)
```

After `log2 P` rounds every process has synchronised with all P processes, directly or through intermediaries.

* No locks. A lock-based counter barrier serialises all P arrivals on one lock, which is O(P). This is O(log P).
* No hot spot. Each process spins on its own memory word. In a centralised barrier, all P processes spin on one release variable.
* No reset. The flag holds the sweep number as a monotonically increasing ticket, so flags never need clearing. The naive barrier needs an extra barrier just to publish the reset of its counter.

C\* has no bitwise operators, so the partner `me XOR 2^k` is computed arithmetically:

```c
bit = (me / step) % 2;
if (bit == 0) partner = me + step; else partner = me - step;
```

### 3.2 Aggregation inside the barrier

`AggregateMax()` runs the same butterfly pattern and also combines the partner's value each round. After round *k* a process holds the maximum over its block of 2^(k+1) processes, so after `log2 P` rounds every process holds the global maximum.

One call gives the sweep both things it needs: the barrier that ends it, and the global maximum for the convergence test. It takes O(log P) rounds with no master process and no broadcast, so the convergence test adds no synchronisation step of its own. The naive version needs a lock-protected shared maximum (P serialised critical sections), then a barrier, then a broadcast.

### 3.3 Ping-pong grids: one barrier per sweep

New values go into a second grid, and the two grids swap roles every sweep. A sweep reads only the old grid and writes only the new one, so no process can overwrite a value another process still needs. One barrier per sweep is enough, and there's no copy-back.

The naive version keeps a single grid. Each sweep then needs a copy-back pass (N² extra assignments) and three barriers.

### 3.4 Row-block partition

Each process owns a contiguous band of rows. Only two halo rows per process are shared, and they're read-only during a sweep.

### 3.5 Adaptive load balancing (OpenMP, `-L`)

An equal static partition only works well when every core runs at the same speed. The test machine has 8 fast P-cores and 16 slower E-cores. A barrier makes each sweep as slow as its slowest thread, so with an equal partition the P-cores sit idle at every barrier waiting for the E-cores.

With `-L`, each thread times its own band and publishes the rate it achieved. At a few geometrically spaced sweeps the partition is recomputed so each thread's row count is proportional to its measured speed. This changes which thread updates which row and nothing else. The arithmetic is untouched, and the sweep count and checksum stay the same (section 6).

---

## 4. Two barrier bugs

The first came from `Barrier` and `AggregateMax` sharing one slot array. Both use monotonic tickets. A barrier carrying a higher ticket leaves flags that let the next sweep's aggregation pass its spin immediately and read a stale value, so both the synchronisation and the maximum are lost. The result still looks plausible but drifts, and drifts differently on every run. The two functions now keep independent slot arrays (`bflag`/`bval` and `bsync`).

The second is handled with parity double buffering. Slots are indexed by sweep parity, so a thread that runs ahead into the next sweep writes the other buffer and can't overwrite a value a slower partner hasn't read yet. Two buffers are enough because a thread can't get two sweeps ahead: to finish sweep *s+1* it must have heard, directly or transitively, from every other thread at sweep *s+1*.

On real hardware the value is published before the flag with a release store and read with an acquire load. That guarantees a reader that sees the flag also sees the value on a weakly ordered machine. Each `(thread, round)` slot has its own 64-byte cache line, so spinning threads never share a line.

---

## 5. Build and run

### C\* simulator version

The simulator binary is not included. It's the course's tool, not my work. `runcstar.sh` looks for `cstar.exe` next to itself, then in the parent directory (where the course files live), or you can point it anywhere with `CSTAR=/path/to/cstar.exe`. It's a Win32 program, so on Linux and macOS it runs under Wine. If `CSTAR` names one of the native `cstar`, `cstar-mac` or `cstar-m1` builds, that build runs directly.

```bash
./runcstar.sh JacobiParallel.c 'UTILIZATION 0:32'   # one run
./benchmark.sh                                      # full P = 1..32 study
```

Interactively, the simulator takes `OPEN`, `RUN`, `TIME`, `UTILIZATION p:q` and `EXIT` at its `*` prompt.

### Real-hardware version

```bash
make                       # gcc -O3 -march=native -fopenmp
make run                   # sample run on all cores
./benchmark-openmp.sh      # full study
./JacobiOpenMP -n 2048 -p 24 -t 0.01 -L     # 2048x2048, 24 threads, balanced
./JacobiOpenMP -b -p 32 -s tree             # barrier cost, ns/op
```

Options: `-n` grid, `-p` threads, `-t` tolerance, `-i` max sweeps, `-s tree|linear|omp` synchronisation, `-c` test every K sweeps, `-L` adaptive balancing, `-b` barrier microbenchmark, `-r` repeats.

---

## 6. Validation

Jacobi relaxation is deterministic. Each new value depends only on the old grid, so the partition can't change the answer. That is the correctness check used throughout:

* The C\* sequential, efficient-parallel and naive-parallel programs converge in the same number of sweeps to the same checksum at every P tested (1, 2, 4, 8, 16, 32).
* The OpenMP version returns a bit-identical checksum for every thread count from 1 to 32, every synchronisation mode (`tree`, `linear`, `omp`), and with and without adaptive balancing.
* On a small grid the result is checked against physics. With one edge at 100° and three at 0°, the exact steady-state centre value is 25: the four rotations of the problem sum to a plate held at 100° everywhere.

The stopping rule caps the change per sweep, which is a local criterion. On a large grid it halts while heat is still spreading, so the centre of a 1024×1024 plate has barely warmed up when the test fires. That comes from the stopping rule, not from the solver.

---

## 7. Results

Full captured output is in `results/`, along with rendered screen captures of every sample run (`results/capture-*.png`, produced by `capture.sh`).

### 7.1 C\* simulated multiprocessor

32×32 grid, tolerance 0.01. All 13 configurations converge in 642 sweeps to checksum 24632.0, matching the sequential reference.

Efficient version (butterfly barrier and butterfly aggregation):

| P | T(P) | Speedup | Efficiency | Utilisation |
|---|---|---|---|---|
| 1 | 75,101,538 | 1.00 | 0.999 | n/a |
| 2 | 37,806,988 | 1.99 | 0.993 | 50% |
| 4 | 19,158,622 | 3.92 | 0.979 | 74% |
| 8 | 9,829,431 | 7.64 | 0.954 | 87% |
| 16 | 5,183,528 | 14.48 | 0.905 | 92% |
| 32 | 2,900,738 | 25.87 | 0.809 | 93% |

Naive control (linear counter barrier, locked maximum, copy-back sweep):

| P | T(P) | Speedup | Efficiency | vs. efficient |
|---|---|---|---|---|
| 8 | 10,656,240 | 7.04 | 0.880 | 1.08× slower |
| 16 | 5,675,208 | 13.22 | 0.827 | 1.10× slower |
| 32 | 3,454,692 | 21.72 | 0.679 | 1.19× slower |

At 32 processors the efficient barrier and aggregation cut total execution time by 16% and raise parallel efficiency from 0.68 to 0.81. The gap widens as P grows, which is what the O(P) versus O(log P) analysis predicts.

> The simulator's own `SPEEDUP` line reports 30.99 for the 32-processor run. That compares the parallel program against its own one-processor time, which already includes the barrier work. The 25.87 above is measured against the separate sequential program, the stricter baseline.

### 7.2 Real hardware: Intel Core i9-14900HX (8 P-cores + 16 E-cores, 32 threads)

Cost of one barrier plus global maximum, in ns per operation (best of 3):

| P | efficient (log P) | naive (O(P), locked) | OpenMP barrier | naive/eff | omp/eff |
|---|---|---|---|---|---|
| 2 | 61 | 136 | 2,817 | 2.2× | 46.2× |
| 4 | 142 | 1,087 | 3,645 | 7.7× | 25.8× |
| 8 | 211 | 2,703 | 4,597 | 12.8× | 21.8× |
| 16 | 287 | 6,753 | 9,722 | 23.5× | 33.9× |
| 24 | 430 | 13,187 | 13,901 | 30.6× | 32.3× |
| 32 | 480 | 17,666 | 19,651 | 36.8× | 40.9× |

The hand-written barrier grows logarithmically: 61 ns at 2 threads, 480 ns at 32, so 8× the cost for 16× the threads. The lock-based barrier grows linearly and ends up 36.8× slower. At 32 threads the fused barrier is also 40.9× faster than OpenMP's barrier. This benchmark doesn't separate how much of that gap comes from fusing the reduction into the barrier and how much from the OpenMP runtime's barrier itself.

Strong scaling, 1024×1024 grid, 400 sweeps, convergence tested every sweep:

| P | static Gpt/s | speedup | adaptive Gpt/s | speedup | gain from balancing |
|---|---|---|---|---|---|
| 1 | 3.18 | 1.00 | 3.25 | 1.02 | +2% |
| 2 | 5.70 | 1.79 | 5.46 | 1.72 | −4% |
| 4 | 10.55 | 3.32 | 9.41 | 2.96 | −12% |
| 8 | 21.53 | 6.78 | 19.97 | 6.29 | −8% |
| 16 | 13.95 | 4.39 | 29.23 | 9.20 | +52% |
| 24 | 21.47 | 6.76 | 30.92 | 9.74 | +31% |
| 32 | 12.72 | 4.01 | 21.14 | 6.66 | +40% |

Peak throughput is 30.9 billion point updates per second at 24 threads, 9.7× the single-thread rate. At that rate one sweep takes about 34 µs; the fused barrier costs 0.43 µs of it, where the lock-based barrier would cost about 13 µs. Two effects in the table come from this CPU rather than from the algorithm.

Threads 9 to 16 land on E-cores. Each sweep waits for the slowest thread, so with an equal partition throughput drops from 21.5 to 14.0 Gpt/s when the E-cores join. Adaptive balancing gives the E-cores fewer rows and brings it back to 29.2 Gpt/s.

32 threads is slower than 24. The last 8 threads are SMT siblings of P-cores that are already busy, so they add contention without adding capacity. If the assignment calls for using the whole machine, the right setting on this CPU is 24 threads, one per physical core, with adaptive balancing so the E-cores pull their weight.

On 2 to 8 threads, all on P-cores, balancing costs 4 to 12%. There's nothing to rebalance there, so that's the overhead of timing and repartitioning, though most of it is within run-to-run noise.

Timings on a laptop vary by roughly ±10% between runs with temperature and clock behaviour. The tables above are a single captured run (best of 5 repetitions per point), reproducible with `./benchmark-openmp.sh`.

Large grids (adaptive, all cores, 50 sweeps):

| N | points | time | Gpt/s | GFLOP/s | GB/s touched |
|---|---|---|---|---|---|
| 2048 | 4.2 M | 0.094 s | 2.23 | 17.88 | 35.8 |
| 4096 | 16.8 M | 0.580 s | 1.45 | 11.58 | 23.2 |
| 8192 | 67.1 M | 2.259 s | 1.49 | 11.88 | 23.8 |

Once the grids no longer fit in L3, the solver is memory-bandwidth bound and the rate flattens at about 23 GB/s. Barrier design barely matters at that point. It matters while the problem fits in cache and a sweep takes tens of microseconds, where a slow barrier is a large fraction of each one.

### 7.3 Cross-validation

The two implementations agree exactly. On the same 32×32 problem, both the C\* simulator and the OpenMP program converge in 642 sweeps to checksum 24632.0, with a centre value of 24.07 against the exact steady state of 25. The OpenMP version also returns a bit-identical checksum (`623412.6429` on the 256×256 case) for every thread count from 1 to 32, all three synchronisation modes, and with and without adaptive balancing.

---

## 8. Notes on the C\* dialect

Several design choices come from working within this compiler:

* No bitwise operators, so the butterfly partner `me XOR 2^k` is computed with division and modulo.
* Identifiers are case-insensitive. A parameter named `g` collides with a global array `G`.
* `int` is 16 bits. `(int) clock` overflows once simulated time passes 32,767, so times are printed as floats with `cout.precision(9)`.
* Console lines are 128 characters, so the grid is printed every 4th row and column.
* No `?:`, no unions, no enums. Library functions such as `fabs` need their header, so the absolute value is computed inline to keep the programs self-contained.
