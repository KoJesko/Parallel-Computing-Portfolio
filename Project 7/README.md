# Project 7: Matrix Multiplication on a Torus Multicomputer

An efficient parallel matrix multiplication program, C = A × B, for a multicomputer whose processors are connected as a 2-D torus. It uses Cannon's algorithm, whose block rotations each travel a single torus link.

The main submission runs on Bruce Lester's C\* simulator with `ARCHITECTURE TORUS`. The same algorithm is also implemented on real hardware: POSIX threads on all 32 logical CPUs of an i9-14900HX, and CUDA on an RTX 4060 Laptop GPU.

---

## 1. Contents

| File | What it is |
|---|---|
| `TorusMatMul.c` | Main deliverable: C\* program for a Q×Q torus, using Cannon's algorithm with whole-block messages and communication overlapped with computation. |
| `MatMulSequential.c` | C\* sequential reference with the same matrices, kernel and checks. Gives T(1). |
| `NaiveMatMul.c` | C\* control: processor 0 holds A and B, sends every processor its strips and collects the C blocks. This is the direct multicomputer translation of the course's `MatrixMultiplyParallel.c`. |
| `cpu_torus.c` | pthreads version: Q×Q torus of nodes with private blocks, rotations copied into neighbours' inboxes, AVX2 kernel, dynamic or static work split. |
| `cuda_torus.cu` | CUDA version: G×G torus of thread blocks visiting the K tiles in Cannon's order, with cuBLAS timed as a reference. |
| `runcstar.sh` | Drives the interactive C\* simulator non-interactively (Wine + `cstar.exe`). Settings that must come before `RUN` go in `CSTAR_PRE`. |
| `benchmark.sh` | C\* study: P = 1 to 256, comparing Cannon vs naive, torus vs mesh, and overlap vs no overlap. Runs pinned to CPUs 0–15 (the P-cores). |
| `benchmark-hw.sh` | Real-hardware study: CPU thread scaling (dynamic and static), GPU at N = 2048/4096/8192. |
| `utilization.sh` | Runs the 32-thread CPU job and the GPU job together and samples `/proc/stat`, `top` and `nvidia-smi` at the same moments. |
| `samples.sh`, `capture.sh` | Capture every sample run into `results/*.txt` and render them as PNG screen captures. |
| `Makefile` | `make`, `make cstar`, `make bench`, `make util`, `make capture`. |
| `math.h` | The course's C\* header. |
| `results/` | Captured output of every run (`*.txt`) and the rendered screen captures (`capture-1` … `capture-7`). |

---

## 2. Algorithm

### 2.1 Data layout

The N×N matrices are cut into Q×Q blocks of size NB×NB (NB = N/Q). Processor i sits at torus row i / Q, column i % Q and owns blocks A(r,c), B(r,c) and C(r,c). No processor ever holds more than three blocks, and the matrices are never collected in one place.

### 2.2 Cannon's algorithm

1. **Alignment.** Block row r of A rotates left by r places, and block column c of B rotates up by c places. Processor (r,c) then holds A(r, r+c) and B(r+c, c), with indices taken mod Q. Each block is sent straight to its destination. Row 0 of A and column 0 of B don't move.
2. **Q multiply steps.** Each processor multiplies the two blocks it holds into its C block, then every A block moves one place left and every B block moves one place up. After Q steps, processor (r,c) has accumulated A(r,k) × B(k,c) for every k, which is C(r,c).

```
 alignment (Q = 4): A row r shifts left r     each step: A one left, B one up
   A00 A01 A02 A03                              <-- A --<-- A --<-- A --<-- A --<  (wraps)
   A11 A12 A13 A10                                  ^       ^       ^       ^
   A22 A23 A20 A21                                  B       B       B       B
   A33 A30 A31 A32                                  ^       ^       ^       ^  (wraps)
```

### 2.3 Why a torus, and the efficiency measures

* **One link per rotation.** The rotations are cyclic: a block leaving column 0 re-enters at column Q−1. On a torus that is a wraparound link, so every rotation message travels exactly one hop. Each processor sends and receives one A block and one B block per step, and no link carries two messages in the same direction.
* **Whole-block messages.** A block is a `struct`, and one `send` carries all NB² values. In the simulator a block message costs about 2 time units per float plus about 10 per hop. Sending element by element would add NB² message start-ups per rotation.
* **Overlap.** The next step's blocks are sent before the local multiply starts and received after it finishes (double buffering), so the transfers run while the processor computes.
* **No host distribution.** Each processor generates its own blocks of A and B, so there is no O(P) distribution phase through one node.
* **Exact checking.** The matrix entries are small integers, so every product and sum is exact in `float`. Every row sum and column sum of C is checked against Σₖ A(i,k)·rowsum(B)ₖ, which costs O(N²) instead of another N³. The checks run after T(P) is taken.

T(P) is the time the last processor finishes its C block. The check results then travel to processor 0 over torus links in ⌊Q/2⌋ + ⌊Q/2⌋ one-hop steps: columns 1..Q/2 pass west, and the rest pass east across the wraparound link.

---

## 3. Build and run

### C\* simulator (main submission)

`runcstar.sh` looks for `cstar.exe` next to itself, in the parent directory or in `../parallel/`. You can also set `CSTAR=/path/to/cstar.exe`. It runs under Wine.

```bash
./runcstar.sh MatMulSequential.c                    # T(1) reference
./runcstar.sh TorusMatMul.c 'UTILIZATION 0:15'      # 4 x 4 torus
./runcstar.sh NaiveMatMul.c                         # naive control
./benchmark.sh                                      # full P = 1 .. 256 study
```

To change the torus size by hand, edit `ARCHITECTURE TORUS(4)`, `#define Q`, `#define P` and `#define NB` together. `benchmark.sh` does this for each size, and also builds the MESH2 and `OVERLAP 0` variants.

Each `cstar.exe` simulates the whole multicomputer by itself; separate runs are only separate configurations. `benchmark.sh` pins all simulator runs to CPUs 0–15, the eight P-cores and their hyperthreads (`CPUS=0-15`, `JOBS=16`).

### Real hardware

```bash
make                         # gcc -pthread, nvcc (+ cuBLAS)
./cpu_torus                  # 4 x 4 torus, 2 threads per node = 32 threads, N = 4096
./cpu_torus 2 6 4096         # Q W N: 2 x 2 torus, 6 threads per node
./cpu_torus -s 4 2           # static split instead of dynamic
./cuda_torus                 # N = 8192, 64 x 64 torus of thread blocks
./cuda_torus -n 8192         # same without the Cannon skew
./benchmark-hw.sh            # full study
./utilization.sh             # CPU + GPU utilization during one run
```

---

## 4. Results: C\* simulated torus multicomputer

N = 96 (884,736 multiply-adds), default simulator settings, T(1) = 36,499,233 from `MatMulSequential.c`. All 33 configurations pass all 192 row/column checks.

**Cannon's algorithm (`TorusMatMul.c`):**

| Torus | P | Block | T(P) | Speedup | Efficiency | Utilization |
|---|---|---|---|---|---|---|
| 1×1 | 1 | 96×96 | 38,508,467 | 0.95 | 0.948 | 99% |
| 2×2 | 4 | 48×48 | 9,712,453 | 3.76 | 0.939 | 99% |
| 3×3 | 9 | 32×32 | 4,352,883 | 8.39 | 0.932 | 99% |
| 4×4 | 16 | 24×24 | 2,469,335 | 14.78 | 0.924 | 99% |
| 6×6 | 36 | 16×16 | 1,116,756 | 32.68 | 0.908 | 99% |
| 8×8 | 64 | 12×12 | 639,755 | 57.05 | 0.891 | 99% |
| 12×12 | 144 | 8×8 | 296,223 | 123.22 | 0.856 | 99% |
| 16×16 | 256 | 6×6 | 175,220 | 208.31 | 0.814 | 98% |

**Naive control (`NaiveMatMul.c`, same torus, processor 0 distributes and collects):**

| Torus | P | T(P) | Speedup | Efficiency | Cannon is faster by |
|---|---|---|---|---|---|
| 2×2 | 4 | 13,541,815 | 2.70 | 0.674 | 1.39× |
| 4×4 | 16 | 6,721,223 | 5.43 | 0.339 | 2.72× |
| 8×8 | 64 | 7,500,613 | 4.87 | 0.076 | 11.7× |
| 16×16 | 256 | 12,875,549 | 2.83 | 0.011 | 73.5× |

Cannon's algorithm keeps 81% efficiency at 256 processors. Its per-processor work falls as N³/P, while its communication per processor is only 2Q block messages of NB² floats, each crossing one link. The naive version peaks at a speedup of about 5.5 around P = 16–36, then gets slower as processors are added. Processor 0 must push (P−1)·2·N·NB floats out through its own links, one strip after another, while the other processors sit idle (utilization falls to 1% at P = 256).

The 1×1 Cannon run is 5% slower than the sequential program because accessing blocks through struct fields costs more in the simulator than accessing plain arrays. Speedups are measured against the faster sequential program.

**Torus vs mesh, and overlap.** These are the same program with the wraparound links removed (`MESH2`), and the same program with the rotations sent after the multiply instead of before it:

| Torus | P | Mesh slower by | No overlap slower by |
|---|---|---|---|
| 4×4 | 16 | 0.00% | 0.10% |
| 8×8 | 64 | 0.03% | 0.23% |
| 12×12 | 144 | 0.11% | 0.34% |
| 16×16 | 256 | 0.27% | 0.41% |

Both effects are real but small at N = 96, and both grow with P. The simulator charges only about 10 time units per hop. A block message costs about 2 per float, and a multiply-add about 40, so even at P = 256 one 6×6 block transfer is tiny next to the 6³ multiply-adds it feeds. On a mesh, the wraparound rotation crosses Q−1 links instead of one: 150 extra time units per step at Q = 16, against roughly 8,600 for the local multiply. The torus advantage would grow with a slower network, smaller blocks or larger Q. With the default network, the main win in this study is choosing an algorithm that only ever talks to torus neighbours (Cannon vs naive: up to 73×).

Full output: `results/results-cstar.txt` and `results/capture-4-cstar-summary.png`.

---

## 5. Results: real hardware

Intel Core i9-14900HX (8 P-cores with 2 hyperthreads each, plus 16 E-cores, 32 logical CPUs) and an RTX 4060 Laptop GPU (24 SMs). Single precision, N = 4096 on the CPU. CPU times are the best of 3. GPU times are kernel only, best of 10. Every run passes all 2N exact row/column checks. Full output is in `results/results-hw.txt`.

### 5.1 CPU (`cpu_torus`, dynamic work split)

| Threads | Torus × threads/node | Time | GFLOP/s | Speedup |
|---|---|---|---|---|
| 1 | 1×1 × 1 | 1.772 s | 77.6 | 1.00 |
| 4 | 2×2 × 1 | 0.462 s | 297.2 | 3.87 |
| 8 | 2×2 × 2 | 0.312 s | 440.7 | 5.74 |
| 16 | 4×4 × 1 | 0.268 s | 513.2 | 6.69 |
| 24 | 2×2 × 6 | 0.217 s | **632.6** | **8.24** |
| 32 | 4×4 × 2 | 0.244 s | 563.7 | 7.34 |

* **Up to 4 threads**, all threads run on P-cores and scaling is nearly linear (3.87× on 4).
* **Beyond 8 threads**, the gains shrink for three reasons:
  * On a laptop, the all-core clock drops well below the 5.8 GHz single-core turbo under the power limit.
  * Threads 9–24 run on 4.1 GHz E-cores.
  * Threads 25–32 are second hyperthreads on P-cores that are already busy, so the 24-thread run (one thread per physical core) is the fastest.
* **Dynamic vs static.** The dynamic split lets every thread take 48-row pieces of any node's work at each step, so the P-cores do more of it. That is 1.2–1.7× faster than the static split, where each thread works only on its own node's rows: 563.7 vs 476.6 GFLOP/s at 32 threads, and 513.2 vs 308.1 at 16.
* **Rotations.** A rotation copies one block (4 MB at 32 threads) into each neighbour's inbox. That is small next to a step's 2·NB³ multiply-adds, and it overlaps with them.

The single-thread kernel reaches 96 GFLOP/s on a 1024-wide block but 77 GFLOP/s on the 4096-wide block that the 1-thread run uses (TLB pressure from rows 16 KB apart). The 1-thread time is therefore a slightly pessimistic baseline. Two kernel fixes were needed to get here: padding the row stride (power-of-two strides mapped whole panels onto the same cache sets) and packing each B slice into a contiguous buffer.

### 5.2 GPU (`cuda_torus`)

| N | Grid (torus) | K order | Time | GFLOP/s | cuBLAS GFLOP/s |
|---|---|---|---|---|---|
| 2048 | 16×16 | Cannon | 6.4 ms | 2,676 | 4,080 |
| 2048 | 16×16 | unskewed | 3.5 ms | 4,940 | 5,513 |
| 4096 | 32×32 | Cannon | 17.0 ms | 8,069 | 9,840 |
| 4096 | 32×32 | unskewed | 16.4 ms | 8,404 | 9,840 |
| 8192 | 64×64 | Cannon | 141 ms | 7,776 | 7,772 |
| 8192 | 64×64 | unskewed | 142 ms | 7,747 | 7,897 |

At N = 4096–8192 the kernel runs at 7.7–8.4 TFLOP/s, 82–100% of cuBLAS on the same matrices. In the standalone sample run it reached 8.2 TFLOP/s, 104% of cuBLAS (`capture-5`).

The Cannon skew matters on a network but not much on a GPU. Unlike torus processors, thread blocks share an L2 cache. When every block in a row asks for the same A tile at the same time (unskewed order), that tile is served from L2 once, so skewing the order buys nothing and costs L2 reuse when the matrices are small. The skew was neutral to slightly positive at N = 8192: +0.4% here, +7.5% in an earlier run (thermal noise is large at 86 °C). It was slower at N = 2048. It is kept as the default because it is the faithful mapping of the torus algorithm, and `-n` turns it off.

### 5.3 CPU and GPU utilization in the same run

`utilization.sh` runs `cpu_torus 4 2 10240` (32 threads) and `cuda_torus 8192 150` at the same time and samples both:

| t (s) | Whole-machine CPU busy | GPU utilization | GPU power |
|---|---|---|---|
| 1–8 | 96.5 – 100.0% | 99 – 100% | 75 – 82 W |

At the same moment:
* Every one of the 32 logical CPUs is 97–100% busy (per-CPU `/proc/stat`).
* `top` shows `cpu_torus` at 2753% CPU.
* `nvidia-smi` shows 100% GPU-Util with `cuda_torus` as the compute process.

Both programs still pass all of their exact checks while running together. See `results/utilization.txt` and `results/capture-7-utilization.png`.

---

## 6. Notes on the C\* dialect

* Multicomputer syntax: `ARCHITECTURE TORUS(Q);` declares a Q×Q torus in row-major order with wraparound links: on TORUS(4), 0→3 and 0→12 are each one hop. `block stream s[P];` declares message channels, `send`/`recv` pass messages, and `forall i = 0 to P-1 @ i do` places process `i` on processor `i`.
* Streams can carry a `struct`, so a whole block is one message.
* **Identifiers are case-insensitive.** `c` clashes with `C`, `s` with `S`, `p` with `#define P`, and `Cout` is the keyword `cout`.
* `int` is 16 bits and integer literals above 32,767 are rejected, so T(1) is stored as `#define SEQTIME4 3649.9233` and multiplied by 10,000 when used.
* The simulator's total memory is limited. The naive program keeps processor 0's full matrices inside a host-only function, so the other 255 processes don't each allocate them.
* `CONGESTION ON` and `DELAY d` must be given before `RUN` (`CSTAR_PRE` in `runcstar.sh`). The study uses the default network.
