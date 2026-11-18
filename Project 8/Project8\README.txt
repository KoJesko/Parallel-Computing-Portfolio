PROJECT 8 - MPI-based parallel spatial filtering of an image on a
            7-dimensional hypercube
=====================================================================

NOTE: All runs and screen captures were made on my personal gaming
desktop: AMD Ryzen 7 9700X (8 cores), NVIDIA GeForce RTX 5070 Ti,
Windows 11 Home.  The desktop's normal background programs were running,
so the real-hardware timings vary a little from run to run.

The required program runs in Bruce Lester's C* simulator on a simulated
7-D hypercube (128 processors).  The same algorithm was also run on real
hardware: with MS-MPI on all 8 CPU cores, and with CUDA on the GPU.


1. FILES
---------------------------------------------------------------------
SpatialFilterHypercube.c   MAIN PROGRAM - C* simulator, MPI mode,
                           ARCHITECTURE HYPERCUBE(7) = 128 processors
SpatialFilterSequential.c  sequential C* version (speedup baseline)
native\SpatialFilterMPI.c  real MPI version (MS-MPI), any power-of-two
                           number of ranks: 8 = one per core, 128 = 7-D
native\SpatialFilterCUDA.cu  CUDA GPU version
native\image_io.h          shared image I/O / reference filter (native)
native\imagetool.c         makes sample inputs, converts results to BMP,
                           verifies results pixel by pixel
build.bat                  builds the native programs (MSVC, MS-MPI, CUDA)
run_simulator.bat [smooth|edge|sharpen]   C* simulator demo
run_mpi.bat [sample|scaling|sustained]    8-core MPI demo
run_cuda.bat                              GPU demo

image256_smooth.txt   sample inputs, 256 x 256 grayscale test image
image256_edge.txt       with a 3x3 mask: Gaussian smoothing,
image256_sharpen.txt    Laplacian edge detection, sharpening
image128_emboss.txt     (128 x 128, emboss mask)
out_*.pgm             filtered results from the simulator (PGM images)
images\*.bmp          inputs and results as viewable images
screenshots\*.png     screen captures of the sample runs
logs\*.txt            full text output of every run

Input file format (whitespace-separated integers):
    rows cols
    k00 k01 k02 k10 k11 k12 k20 k21 k22      3x3 mask
    divisor offset
    rows*cols pixel values 0..255
Each output pixel = clamp( sum(mask * 3x3 neighbourhood) / divisor
                           + offset, 0, 255 ).  Image borders are
handled by replicating the edge pixels.


2. PARALLEL ALGORITHM (7-D hypercube, 128 nodes)
---------------------------------------------------------------------
* The image is cut into 128 horizontal strips (2 rows each for 256x256).
  Strip s is placed on node Gray(s) (reflected binary Gray code).
  Consecutive Gray codes differ in exactly one bit, so neighbouring
  strips always sit on directly connected hypercube nodes, and every
  halo exchange is a single hop.
* The header (size + mask) is broadcast with a binomial tree along the
  hypercube dimensions 6,5,...,0: 7 steps, each one over a hypercube
  link.
* Node 0 streams the pixels from the input file and sends each strip to
  its owner.  No node ever holds the whole image, so each node needs
  only a few KB (the simulator allows less than 32K words per node).
* Halo exchange runs in two phases: pairs (even strip, next strip),
  then pairs (odd strip, next strip).  The whole exchange takes 2
  message steps no matter how many nodes there are.
* Each node filters its own strip.  The mask is fully unrolled and
  edges are clamped once per column.
* Node 0 collects the strips in strip order and writes the PGM image.
* MPI_Reduce (sum and max) of the per-node filter time gives the
  filter-phase speedup.
The simulator has no bitwise operators, so the Gray code XOR is done
with integer arithmetic (XorInt / Gray / InvGray).

The real MPI version uses the same scheme, generalised to P = 2^d ranks:
a binomial-tree scatter and gather over the hypercube dimensions,
MPI_Sendrecv halo exchange with the Gray-code neighbours, and optional
repeated passes.


3. HOW TO RUN
---------------------------------------------------------------------
C* simulator (by hand):            or simply:  run_simulator.bat smooth
    cstar.exe
    *MPI ON
    *INPUT = image256_smooth.txt
    *OUTPUT = out_smooth.pgm
    *OPEN SpatialFilterHypercube.c
    *RUN
    *UTILIZATION 0:7
The filtered image and the program's performance summary (the lines
starting with '#') go to the OUTPUT file.  The simulator appends its own
SEQUENTIAL / PARALLEL EXECUTION TIME and SPEEDUP lines there as well.
("Runtime error 103" printed after *TERMINATE C* SYSTEM is the
simulator's own exit message when its input is redirected.  It also
appears with the supplied MPI-Hello.c and does not affect the results.)

Real hardware (after build.bat):
    mpiexec -n 8   bin\SpatialFilterMPI.exe image256_smooth.txt
    mpiexec -n 128 bin\SpatialFilterMPI.exe image256_smooth.txt
    mpiexec -n 8   bin\SpatialFilterMPI.exe -gen 8192 8192 smooth 100
    bin\SpatialFilterCUDA.exe -gen 8192 8192 smooth 100
Arguments: <input.txt> | -gen <rows> <cols> <filter>, then
[passes] [out.bmp]


4. PERFORMANCE SUMMARY
---------------------------------------------------------------------
(a) C* simulator - 256 x 256 image, 128 processors (7-D hypercube)
    (simulated time units; all three outputs verified pixel-exact)

    filter    sequential C*   parallel C*   overall   filter-phase  filter
              (1 processor)   (128 procs)   speedup   speedup       efficiency
    smooth      17,889,122     3,955,543     4.52 x    128.0 x      100.0 %
    edge        18,028,266     3,956,559     4.56 x    127.6 x       99.7 %
    sharpen     17,951,766     3,955,835     4.54 x    127.4 x       99.5 %

    Phase breakdown (smooth):
      header broadcast        1,184   (7 hypercube steps)
      strip distribution  1,183,056   (node 0 reads the file)
      halo exchange           7,968   (2 one-hop steps)
      filtering + reduce    114,337   (vs 14,111,360 on one node)
      collect + output    2,648,813   (node 0 writes the file)
    The filtering itself scales almost perfectly (128x on 128 nodes,
    because every halo exchange is one hop and the load is exactly
    balanced).  The overall speedup is limited by Amdahl's law: reading
    and printing the 65,536 pixels is inherently sequential through
    node 0's file I/O.  That is also why the simulator shows processor 0
    at 99% utilisation and the others at about 2%.

(b) Real MPI on the 8 CPU cores - 8192 x 8192 image, 100 passes
    ranks  hypercube  filter time  speedup  CPU usage (of 800%)
      1      0-D        12.38 s     1.00 x        99 %
      2      1-D         6.22 s     1.99 x       200 %
      4      2-D         3.14 s     3.94 x       395 %
      8      3-D         1.84 s     6.71 x       694 %
    Sustained 8-core run (1000 passes): 718% of 800% CPU, 3.6 Gpixel/s,
    99% load balance.  CPU usage = sum of the CPU time of all ranks /
    wall-clock time, so all 8 cores busy = 800%.  The rest of the 800%
    goes to halo-exchange waiting and to other programs running on the
    desktop at the same time.
    128 ranks (the full 7-D hypercube on 8 cores) also gives
    pixel-identical results.  The result checksum is the same for every
    rank count.

(c) CUDA on the RTX 5070 Ti - 8192 x 8192 image
    1 pass   : 0.86 ms kernel, about 78,000 Mpixel/s, verified pixel-exact
    100 pass : 87.7 ms (0.88 ms/pass).  Checksum 456994278fda5a34, the
               same as the MPI 100-pass run
    About 21x faster than all 8 CPU cores with MPI (18.4 ms/pass), and
    about 310x faster than the unoptimised single-core reference filter.
    Host<->device copies (about 3 + 6 ms) cost more than a single pass,
    so the GPU pays off most when several passes are run.


5. SCREEN CAPTURES (screenshots\)
---------------------------------------------------------------------
1_simulator_smooth.png       C* simulator, 128 nodes, smoothing filter
2_simulator_edge.png         C* simulator, 128 nodes, edge detection
3_mpi_sample_128ranks.png    real MPI, 7-D hypercube (128 ranks) + 8 ranks
4_mpi_scaling_1-8_cores.png  real MPI scaling 1/2/4/8 cores, CPU usage
5_mpi_sustained_8_cores.png  long 8-core run, CPU usage out of 800%
6_cuda_gpu.png               CUDA runs on the RTX 5070 Ti
7_input_vs_filtered_images.png  input image and filtered results
