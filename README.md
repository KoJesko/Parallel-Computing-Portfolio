# Parallel Computing Portfolio

Projects from a parallel computing course. Each one is written first for **Bruce P. Lester's C\* parallel computer simulator** (shared memory, 2-D mesh, torus and hypercube architectures). The later projects then port the same algorithm to real hardware: **OpenMP, pthreads, MPI and CUDA** on a 32-thread Intel i9-14900HX laptop with an RTX 4060, and an 8-core Ryzen 7 9700X desktop with an RTX 5070 Ti.

Every parallel program is checked against a sequential reference, and most have a deliberately **naive control version**, so the speedups show what the better algorithm actually buys.

## Highlights

| | Project | Architecture | Result |
|:-:|---|---|---|
| 7 | [Cannon's matrix multiply](Project%207/) | 2-D torus · pthreads + AVX2 · CUDA | **208× on 256 processors** (81% efficiency), and up to 73× faster than the naive gather. 633 GFLOP/s on the CPU, **8.4 TFLOP/s on the GPU** (82–100% of cuBLAS) |
| 8 | [Image filtering on a hypercube](Project%208/Project8/) | 7-D hypercube (128 nodes) · MS-MPI · CUDA | Filter phase scales **128× on 128 nodes** thanks to a Gray-code strip layout where every halo exchange is one hop. CUDA is ~21× faster than 8 MPI cores |
| 6 | [Numerical integration on a mesh](Project%206/) | 2-D mesh · MPI · pthreads · CUDA | **107× on 144 processors**. Summing toward the centre of the mesh beats the naive gather by 23% |
| 5 | [Jacobi relaxation](Project%205/) | Shared memory · OpenMP | A hand-written O(log P) butterfly barrier costs 480 ns at 32 threads, **36.8× cheaper than a locked barrier** and 41× cheaper than OpenMP's. Peaks at 30.9 billion point updates/s |

## All projects

| # | Topic | What's in it |
|:-:|---|---|
| [1](Project%201/) | First C\* programs | Sum and factorial, to get started with the simulator |
| [2](Project%202/) | Parallel rank sort | The same sort on a 64-processor shared-memory machine and on an 8×8 mesh, plus timing only the parallel phase (`seqoff`/`seqon`) |
| [3](Project%203/) | Prime sieve pipeline | n/p portions linked by a stream pipeline. **93.7× speedup** on 128 processes for n = 32,000, and a look at why efficiency drops as the pipeline gets longer |
| [4](Project%204/) | Bucket sort | One spinlock per bucket with a minimal critical section. 7.1× with uniform data, **1.0× with skewed data**, which shows load imbalance wiping out all the parallelism |
| [5](Project%205/) | Jacobi relaxation | Butterfly barrier with the max-reduction fused in, ping-pong grids, and an OpenMP port with dissemination barriers and adaptive load balancing for P-cores vs E-cores |
| [6](Project%206/) | Trapezoidal integration of π | Neighbour-only summation on the mesh (O(√P) steps instead of O(P)), plus MPI `Cart_create`, pinned pthreads and CUDA versions |
| [7](Project%207/) | Matrix multiply | Cannon's algorithm with whole-block messages and computation overlapped with communication, compared against mesh and no-overlap variants. AVX2 CPU kernel and a CUDA torus of thread blocks |
| [8](Project%208/Project8/) | Spatial image filter | Binomial-tree broadcast, Gray-code layout and a two-phase halo exchange on a simulated 7-D hypercube. Real MPI from 1 to 128 ranks, and CUDA |

Projects 5–8 each have their own detailed write-up covering the algorithm, efficiency analysis, full results tables and screen captures. Projects 3 and 4 have a `PerformanceSummary.txt`.

`parallel/` holds the example programs and headers that came with the course.

## Running things

**C\* simulator programs** need Lester's C\* Compiler and Parallel Computer Simulation System (v2.2). Its license doesn't allow redistribution, so it isn't in this repo; use the copy that comes with the textbook. Put `cstar.exe` in `parallel/` (it's gitignored there) or set `CSTAR=/path/to/cstar.exe`. On Linux it runs under Wine. Projects 5–7 include a `runcstar.sh` that drives the interactive simulator non-interactively, so runs can be scripted and benchmarked:

```sh
cd "Project 7"
make cstar          # or ./runcstar.sh TorusMatMul.c
```

**Real-hardware versions** in projects 5–7 build with the included `Makefile` (needs gcc with OpenMP, an MPI implementation, and the CUDA toolkit for the `.cu` files):

```sh
cd "Project 6"
make && make bench
```

Project 8 targets Windows (MSVC, MS-MPI, CUDA). See `Project 8/Project8/build.bat` and the `run_*.bat` scripts; `run_simulator.bat` expects `cstar.exe` in the same folder.

## Things these projects cover

- Synchronisation cost: O(P) locked barriers vs O(log P) butterfly and dissemination barriers
- Matching the communication pattern to the network topology: neighbour-only mesh sums, one-hop torus rotations, Gray-code hypercube layouts
- Amdahl's law in practice: the 128-node filter is limited by sequential file I/O through node 0
- Hybrid CPUs: the static work split loses to dynamic chunking once threads run on E-cores, and second hyperthreads add almost nothing
- GPU vs network intuition: Cannon's skew helps on a torus but barely matters on a GPU, because thread blocks share L2

## License

BSD 3-Clause for my own code (see [LICENSE](LICENSE)). The C\* simulator and the course-supplied example files belong to their respective authors.
