# Project 6: Numerical Integration on a 2-D Mesh

Conversion of the numerical integration program of Figure 5.8 to a 2-D mesh multicomputer, with a summation method designed for the mesh instead of a lock-protected shared sum.

The integral is

```
integral from 0 to 2 of sqrt(4 - t*t) dt  =  pi
```

with `f` exactly as given in the assignment:

```c
float f(float t)
{
  return( sqrt(4 - t*t) );
}
```

---

## 1. Contents

The main submission runs on the course's C\* simulator with `ARCHITECTURE MESH2`. The same algorithm is also implemented on real hardware: MPI, pthreads on all 32 logical CPUs, and CUDA.

| File | What it is |
|---|---|
| `MeshIntegrate.c` | Main deliverable: C\* program for an M×M mesh: local trapezoid sums, then neighbour-only mesh summation toward the centre processor. |
| `IntegrateSequential.c` | C\* sequential reference with the same arithmetic. Gives T(1) and the reference answer. |
| `MeshIntegrateNaive.c` | C\* control: every process sends its sum straight to processor 0 (the direct translation of Figure 5.8's shared sum). |
| `mesh_integrate.c` | MPI version on an `MPI_Cart_create` 2-D mesh, same summation. |
| `cpu_mesh.c` | pthreads version for the 32-thread i9-14900HX: threads pinned one per CPU, mailbox mesh summation, dynamic or static work split. |
| `cuda_mesh.cu` | CUDA version: 32×8 thread-block meshes inside a GX×GY grid mesh, row-then-column summation at both levels. |
| `runcstar.sh` | Drives the interactive C\* simulator non-interactively (Wine + `cstar.exe`). |
| `benchmark.sh` | C\* study: P = 1 to 144, mesh summation vs naive gather, speedup/efficiency table. |
| `benchmark-hw.sh` | Real-hardware study: MPI, pthreads (dynamic and static), CUDA. |
| `utilization.sh` | Runs the 32-thread CPU job and the GPU job together and samples `top` and `nvidia-smi` at the same moments. |
| `samples.sh`, `capture.sh` | Capture every sample run into `results/*.txt` and render them as PNG screen captures. |
| `Makefile` | `make`, `make cstar`, `make bench`, `make capture`. |
| `math.h` | The course's C\* header (needed for `sqrt`). |
| `results/` | Captured output of every run (`*.txt`), the rendered screen captures (`capture-*.png`) and a desktop screenshot of CPU and GPU utilization (`screenshot-utilization.png`). |

---

## 2. Algorithm

### 2.1 Local computation (unchanged from Figure 5.8)

N = 28,800 trapezoids of width w = (b − a)/N are split into P equal contiguous blocks. Each process sums f at the left end of each of its trapezoids:

```
localsum = w * (f(t) + f(t+w) + ... )      over its N/P points
```

and the final answer adds the endpoint correction once:

```
integral = (sum of all localsums) + w/2 * (f(b) - f(a))
```

which is the trapezoidal rule. N = 28,800 divides evenly by every P tested (1, 4, 9, 16, 25, 36, 64, 100, 144), so the problem size is the same at every mesh size.

### 2.2 Why the summation has to change

Figure 5.8 adds each local sum into a shared `globalsum` inside a lock. A multicomputer has no shared memory. The direct translation is to send all P local sums to processor 0, but that is a poor fit for a mesh:

* Processor 0 performs P − 1 receives one after another, so the summation is O(P).
* Messages from the far corner cross 2(M − 1) links, and every one of them uses the links next to processor 0, which become a hot spot.

### 2.3 Mesh summation toward the centre (`MeshIntegrate.c`)

Process `i` runs on processor `i`, at row `i / M`, column `i % M`. The summation uses only messages to a direct mesh neighbour:

1. Row phase: in every row, the left half passes a running sum rightward and the right half passes one leftward. Both halves run at the same time and meet at the middle column after ⌊M/2⌋ steps.
2. Column phase: the middle column does the same vertically. The grand total arrives at the centre processor after ⌊M/2⌋ more steps.

```
 row phase (M = 5)                     column phase
  o -> o -> O <- o <- o                     o
  o -> o -> O <- o <- o                     v
  o -> o -> O <- o <- o                     o
  o -> o -> O <- o <- o                     v
  o -> o -> O <- o <- o                     *  <- total at centre
                                            ^
                                            o
                                            ^
                                            o
```

Properties:

* About M = √P steps instead of P − 1.
* Every message travels exactly one link. No link carries more than one message.
* No processor receives more than three messages (left, right, and one vertical, or two vertical at the centre).
* Ending at the centre instead of a corner halves the number of steps: (M − 1) + (M − 1) to a corner, ⌊M/2⌋ + ⌊M/2⌋ to the centre.

The centre processor adds the endpoint correction and prints the answer and the performance summary.

---

## 3. Build and run

### C\* simulator (main submission)

`runcstar.sh` looks for `cstar.exe` next to itself, in the parent directory, or in `../parallel/`, or you can set `CSTAR=/path/to/cstar.exe`. It runs under Wine on Linux.

```bash
./runcstar.sh IntegrateSequential.c                   # T(1) reference
./runcstar.sh MeshIntegrate.c 'UTILIZATION 0:15'      # 4 x 4 mesh
./runcstar.sh MeshIntegrateNaive.c                    # naive control
./benchmark.sh                                        # full P = 1 .. 144 study
```

To change the mesh size by hand, edit `ARCHITECTURE MESH2(4)`, `#define M` and `#define P` together. `benchmark.sh` does this automatically for each size.

### Real hardware

```bash
make                                    # mpicc, gcc -pthread, nvcc
mpirun -np 16 ./mesh_integrate 1000000000
./cpu_mesh 32 1000000000                # add -s for the static split
./cuda_mesh 1000000000
./benchmark-hw.sh                       # full study
./utilization.sh                        # CPU + GPU utilization during one run
```

---

## 4. Results: C\* simulated mesh multicomputer

N = 28,800. Every configuration gives the same result, 3.14159241298 (error 2.41 × 10⁻⁷, the trapezoidal-rule error for this N), which matches the sequential reference. T(1) = 1,238,584 from `IntegrateSequential.c`.

Mesh summation toward the centre:

| Mesh | P | T(P) | Speedup | Efficiency | Utilisation |
|---|---|---|---|---|---|
| 1×1 | 1 | 1,267,642 | 0.98 | 0.977 | 99% |
| 2×2 | 4 | 317,345 | 3.90 | 0.976 | 99% |
| 3×3 | 9 | 141,470 | 8.76 | 0.973 | 99% |
| 4×4 | 16 | 80,013 | 15.48 | 0.967 | 99% |
| 5×5 | 25 | 51,640 | 23.98 | 0.959 | 98% |
| 6×6 | 36 | 36,369 | 34.06 | 0.946 | 97% |
| 8×8 | 64 | 21,413 | 57.84 | 0.904 | 93% |
| 10×10 | 100 | 14,817 | 83.59 | 0.836 | 86% |
| 12×12 | 144 | 11,565 | 107.10 | 0.744 | 77% |

Naive control (every process sends to processor 0):

| Mesh | P | T(P) | Speedup | Efficiency | Mesh summation is |
|---|---|---|---|---|---|
| 4×4 | 16 | 80,163 | 15.45 | 0.966 | 0.2% faster |
| 6×6 | 36 | 36,983 | 33.49 | 0.930 | 1.7% faster |
| 8×8 | 64 | 22,733 | 54.48 | 0.851 | 5.8% faster |
| 10×10 | 100 | 17,075 | 72.54 | 0.725 | 13.2% faster |
| 12×12 | 144 | 15,013 | 82.50 | 0.573 | 23.0% faster |

On small meshes the computation dominates and the two summations cost about the same. The naive gather's cost grows with P, while the mesh summation's grows only with √P, so the gap widens quickly. At 144 processors the mesh summation is 23% faster overall, and parallel efficiency is 0.74 instead of 0.57.

All runs are in `results/results-cstar.txt` and `results/capture-4-cstar-summary.png`.

---

## 5. Results: real hardware

Intel Core i9-14900HX (8 P-cores + 16 E-cores, 32 logical CPUs) and an RTX 4060 Laptop GPU (24 SMs), n = 10⁹ trapezoids. The CPU programs report the best of 5 repetitions; CUDA times are kernel time only, best of 10. Full output is in `results/results-hw.txt`.

| P | MPI time | MPI speedup | pthreads (dynamic) time | speedup | pthreads (static) time | speedup |
|---|---|---|---|---|---|---|
| 1 | 0.954 s | 1.00 | 0.934 s | 1.00 | 0.915 s | 1.00 |
| 2 | 0.506 s | 1.89 | 0.491 s | 1.90 | 0.487 s | 1.88 |
| 4 | 0.279 s | 3.43 | 0.258 s | 3.62 | 0.256 s | 3.58 |
| 8 | 0.145 s | 6.57 | 0.154 s | 6.06 | 0.155 s | 5.92 |
| 16 | 0.115 s | 8.27 | 0.099 s | 9.43 | 0.118 s | 7.77 |
| 24 | 0.079 s | 12.13 | 0.073 s | 12.81 | 0.079 s | 11.57 |
| 32 | 0.087 s | 10.95 | 0.076 s | 12.30 | 0.082 s | 11.14 |

| CUDA grid | Threads | Time | G evaluations/s | vs 1 CPU thread |
|---|---|---|---|---|
| 4×4 | 4,096 | 13.5 ms | 74 | 69× |
| 16×16 | 65,536 | 8.3 ms | 120 | 112× |
| 32×32 | 262,144 | 8.1 ms | 123 | 115× |

Up to 8 workers, everything runs on the P-cores and scaling is close to linear. Workers 9 to 24 run on E-cores, which are about 30% slower. With an equal static split, the P-cores finish first and wait for the E-cores. With the dynamic split (`cpu_mesh` default), workers take 262,144-interval chunks from an atomic counter as they finish, so the P-cores do more of the work: 9.43× instead of 7.77× at 16 threads. Workers 25 to 32 are the second hyperthread of a P-core that is already busy, so they add almost nothing. On this CPU, 24 threads (one per physical core) is the fastest setting.

The mesh summation itself is negligible on the CPU: about 0.4 to 1.7 µs for the 32-thread mesh, measured from the moment the last thread finishes computing. Threads are pinned one per physical core, fastest cores first, before any second hyperthread is used.

### 5.1 CPU and GPU utilization in the same run

`utilization.sh` runs `cpu_mesh 32` (n = 4 × 10¹⁰) and `cuda_mesh` (n = 10¹⁰, 200 repeats) at the same time and samples both:

| t (s) | Whole-machine CPU busy | GPU utilization |
|---|---|---|
| 1–8 | 99.9 – 100.0% | 99 – 100% |

At the same moment, `top` reports `cpu_mesh` at 3128% CPU (100% = one logical CPU, so about 31.3 of the 32 CPUs are busy), and `nvidia-smi` reports 99% GPU-Util with `cuda_mesh` as the compute process. See `results/utilization.txt` and `results/capture-7-utilization.png`. `results/screenshot-utilization.png` is a desktop screenshot of a second run: `cpu_mesh` at 3108% CPU in `top` and 100% GPU-Util in `nvidia-smi`.

### 5.2 Accuracy

With n = 10⁹ the CPU versions give 3.141592652738 (error 8.5 × 10⁻¹⁰) from 4 workers up. The 1- and 2-worker runs differ in the last few digits (…864 and …756) because the partial sums are added in a different order. The error stops improving beyond about n = 10⁸ because `f` takes a `float` argument as specified: near t = 2 a float can only resolve steps of about 2.4 × 10⁻⁷, so more points stop adding information. Partial sums are accumulated in `double` on the CPU and with Kahan-compensated `float` per thread on the GPU, so summation round-off is not what limits the result.

---

## 6. Notes on the C\* dialect

* Multicomputer syntax: `ARCHITECTURE MESH2(M);` declares an M×M mesh, `float stream s[P];` declares message channels, `send(s[k], v)` / `recv(s[k], v)` pass messages, and `forall i = 0 to P-1 @ i do` places process `i` on processor `i`.
* `int` is 16 bits and integer literals above 32,767 are rejected ("Number too big"), so T(1) is stored as `#define SEQTIMEK 1238.584` and multiplied by 1000 when used. `#define` only accepts a single constant, not an expression.
* `sqrt` needs the course's `math.h`. `fabs` is avoided; the absolute error is computed inline.
* Simulated time is printed with `cout << clock` at `precision(9)`, because `(int) clock` overflows.
