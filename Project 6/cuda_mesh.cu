/*
 * Project 6 - CUDA version
 *
 * Trapezoidal integration of sqrt(4 - t^2) over [0, 2] (= pi) on the GPU,
 * organised as a 2-D mesh at both levels of the hardware:
 *
 *   - Each thread block is a 32 x 8 mesh of threads.  A block row is exactly
 *     one warp, so the row phase of the summation runs in registers with warp
 *     shuffles; the 8 row totals then go through shared memory for the
 *     column phase.
 *   - The grid is a GX x GY mesh of blocks.  Each block writes its partial
 *     sum to its own slot in a GX x GY array, and a second kernel (one block
 *     with one thread per slot) combines them with the same row-then-column
 *     summation.
 *
 * No atomics are used, so the result is bit-for-bit reproducible from run to
 * run.  Within a row the GPU uses a shuffle tree (log2 32 = 5 steps) instead
 * of the neighbour pipeline used in MeshIntegrate.c: on a GPU all lanes of a
 * warp exchange in one instruction, so the tree is the fast form.
 *
 * Each thread sums a contiguous run of points in float with Kahan
 * compensation; partial sums are combined in double.
 *
 * Usage:  ./cuda_mesh [n] [grid_side] [repeats]
 *         n defaults to 1e9 intervals, grid_side (GX = GY) to 32,
 *         repeats to 10 (use a large value to load the GPU for a while).
 */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define A 0.0
#define B 2.0
#define BX 32                 /* block mesh: BX x BY threads, BX = warp size */
#define BY 8

#define CHECK(call)                                                        \
    do {                                                                   \
        cudaError_t e_ = (call);                                           \
        if (e_ != cudaSuccess) {                                           \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(e_));                               \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

__host__ __device__ float f(float t)
{
    return (sqrtf(4 - t * t));
}

/* Row phase inside one warp: shuffle tree, total ends in lane 0. */
__device__ double warp_row_sum(double v)
{
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

/*
 * Kernel 1: every thread sums f at its own contiguous run of grid points
 * i in [first, last), then the block reduces row-then-column and writes one
 * partial per block into partials[by * GX + bx].
 */
__global__ void integrate(long long n, double h, long long per_thread,
                          double *partials)
{
    __shared__ double rowsum[BY];
    long long block = (long long)blockIdx.y * gridDim.x + blockIdx.x;
    long long tid = block * (BX * BY) + threadIdx.y * BX + threadIdx.x;
    long long first = tid * per_thread;
    long long last = first + per_thread < n ? first + per_thread : n;
    float t0 = (float)(A + first * h), hf = (float)h;
    float sum = 0.0f, comp = 0.0f, y, s;

    for (long long k = 0; first + k < last; k++) {
        y = f(t0 + (float)k * hf) - comp;    /* Kahan summation */
        s = sum + y;
        comp = (s - sum) - y;
        sum = s;
    }

    double v = warp_row_sum((double)sum);     /* row phase    */
    if (threadIdx.x == 0)
        rowsum[threadIdx.y] = v;
    __syncthreads();
    if (threadIdx.y == 0) {                   /* column phase */
        v = threadIdx.x < BY ? rowsum[threadIdx.x] : 0.0;
        v = warp_row_sum(v);
        if (threadIdx.x == 0)
            partials[block] = v;
    }
}

/*
 * Kernel 2: one block, one thread per block partial (gx x gy <= 1024).
 * Row phase across each mesh row, then column phase down column 0.
 */
__global__ void mesh_combine(const double *partials, int gx, int gy,
                             double h, double *result)
{
    __shared__ double cell[1024];
    int x = threadIdx.x, y = threadIdx.y;

    cell[y * gx + x] = partials[y * gx + x];
    __syncthreads();
    for (int off = 1; off < gx; off <<= 1) {     /* row phase */
        if (x % (2 * off) == 0 && x + off < gx)
            cell[y * gx + x] += cell[y * gx + x + off];
        __syncthreads();
    }
    for (int off = 1; off < gy; off <<= 1) {     /* column phase */
        if (x == 0 && y % (2 * off) == 0 && y + off < gy)
            cell[y * gx] += cell[(y + off) * gx];
        __syncthreads();
    }
    if (x == 0 && y == 0)
        *result = h * cell[0] + h / 2 * (f((float)B) - f((float)A));
}

int main(int argc, char *argv[])
{
    long long n = argc > 1 ? atoll(argv[1]) : 1000000000LL;
    int side = argc > 2 ? atoi(argv[2]) : 32;
    int repeats = argc > 3 ? atoi(argv[3]) : 10;
    int gx = side, gy = side;
    long long threads = (long long)gx * gy * BX * BY;
    long long per_thread = (n + threads - 1) / threads;
    double h = (B - A) / n, result, best = 1e30, ms_d;
    double *d_partials, *d_result;
    float ms;
    cudaEvent_t start, stop;
    cudaDeviceProp prop;

    if (n < 1 || side < 1 || gx * gy > 1024 || repeats < 1) {
        fprintf(stderr, "usage: %s [n] [grid_side <= 32] [repeats]\n", argv[0]);
        return 1;
    }
    CHECK(cudaGetDeviceProperties(&prop, 0));
    CHECK(cudaMalloc(&d_partials, gx * gy * sizeof(double)));
    CHECK(cudaMalloc(&d_result, sizeof(double)));
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    /* Repeat 1 is a warm-up (context/clock ramp); report the best. */
    for (int rep = 0; rep <= repeats; rep++) {
        CHECK(cudaEventRecord(start));
        integrate<<<dim3(gx, gy), dim3(BX, BY)>>>(n, h, per_thread, d_partials);
        mesh_combine<<<1, dim3(gx, gy)>>>(d_partials, gx, gy, h, d_result);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaGetLastError());
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (rep > 0 && ms < best)
            best = ms;
    }
    CHECK(cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    ms_d = best;

    printf("CUDA mesh on %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    printf("  grid     = %d x %d blocks of %d x %d threads (%lld threads)\n",
           gx, gy, BX, BY, threads);
    printf("  n        = %lld intervals (%lld per thread)\n", n, per_thread);
    printf("  result   = %.12f\n", result);
    printf("  error    = %.3e\n", fabs(result - M_PI));
    printf("  time     = %.6f s (best of %d, kernels only)\n", ms_d / 1e3, repeats);
    printf("  rate     = %.2f G evaluations/s\n", n / (ms_d / 1e3) / 1e9);
    printf("CSV,%lld,%d,%.12f,%.6f\n", n, side, result, ms_d / 1e3);

    cudaFree(d_partials);
    cudaFree(d_result);
    return 0;
}
