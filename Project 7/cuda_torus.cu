/*
 * Project 7 - CUDA version
 *
 * Cannon's algorithm mapped onto the GPU.  The grid is a G x G torus of
 * thread blocks (G = N / 128); block (r, c) owns the 128 x 128 tile C(r, c)
 * and keeps it in registers for the whole computation, like a torus node
 * that owns its C block.  The K dimension is split into G tiles, and block
 * (r, c) visits them in Cannon's order
 *
 *     step s uses  A(r, k) and B(k, c)  with  k = (r + c + s) mod G
 *
 * i.e. the alignment (k starts at r + c) followed by one rotation per step.
 * At every step the blocks of one grid row read G different A tiles and the
 * blocks of one grid column read G different B tiles, so - exactly as on
 * the torus - no tile is wanted by two blocks of the same row (column) at
 * the same moment, and the memory traffic is spread over all tiles instead
 * of every block hitting tile 0 first.  -n turns the skew off (every block
 * runs k = 0, 1, 2, ...) for comparison.
 *
 * Inside a block (256 threads) each thread computes an 8 x 8 sub-tile with
 * outer products; A and B tiles are staged through double-buffered shared
 * memory 8 columns of K at a time.  cuBLAS SGEMM on the same matrices is
 * timed as a reference for what the hardware can do.
 *
 * Matrix entries are small integers, so the result is checked exactly.
 *
 * Usage:  ./cuda_torus [-n] [N] [repeats]
 *         N defaults to 8192 (must be a multiple of 128), repeats to 10
 *         (use a large value to keep the GPU busy for a while).
 */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BM 128                /* C tile per thread block: BM x BN       */
#define BN 128
#define BK 8                  /* K columns per shared-memory stage      */
#define TM 8                  /* C sub-tile per thread: TM x TN         */
#define TN 8
#define THREADS 256

#define CHECK(call)                                                        \
    do {                                                                   \
        cudaError_t e_ = (call);                                           \
        if (e_ != cudaSuccess) {                                           \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(e_));                               \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

static float Aval(int r, int c) { return (float)((r * 7 + c * 3) % 11 - 5); }
static float Bval(int r, int c) { return (float)((r * 5 + c * 2) % 13 - 6); }

/* C = A x B, all N x N row-major.  skew = 1: Cannon order over K tiles. */
__global__ void __launch_bounds__(THREADS)
torus_sgemm(const float *__restrict__ A, const float *__restrict__ B,
            float *__restrict__ C, int N, int skew)
{
    __shared__ float As[2][BK][BM];     /* A tile, transposed: As[k][m] */
    __shared__ float Bs[2][BK][BN];

    const int r = blockIdx.y, c = blockIdx.x;   /* torus coordinates    */
    const int G = N / BM;                       /* torus side           */
    const int ksteps = N / BK;                  /* stages in total      */
    const int per_tile = BM / BK;               /* stages per K tile    */
    const int tid = threadIdx.x;
    const int tr = tid / (BN / TN), tc = tid % (BN / TN);

    /* loader layout: A tile is BM x BK (one float4 per thread),
     *                B tile is BK x BN (one float4 per thread)         */
    const int a_row = tid / (BK / 4), a_col = (tid % (BK / 4)) * 4;
    const int b_row = tid / (BN / 4), b_col = (tid % (BN / 4)) * 4;

    const float *Ablk = A + (size_t)r * BM * N;
    const float *Bblk = B + (size_t)c * BN;

    float acc[TM][TN] = {};
    float a_frag[TM], b_frag[TN];

    /* K offset of stage t in Cannon order: tile (r+c+t/per_tile) mod G */
    auto kofs = [&](int t) {
        int tile = t / per_tile, sub = t % per_tile;
        if (skew)
            tile = (tile + r + c) % G;
        return tile * BM + sub * BK;
    };

    auto load = [&](int buf, int t) {
        int k0 = kofs(t);
        float4 av = *(const float4 *)(Ablk + (size_t)a_row * N + k0 + a_col);
        As[buf][a_col + 0][a_row] = av.x;
        As[buf][a_col + 1][a_row] = av.y;
        As[buf][a_col + 2][a_row] = av.z;
        As[buf][a_col + 3][a_row] = av.w;
        *(float4 *)&Bs[buf][b_row][b_col] =
            *(const float4 *)(Bblk + (size_t)(k0 + b_row) * N + b_col);
    };

    load(0, 0);
    __syncthreads();
    for (int t = 0; t < ksteps; t++) {
        int cur = t & 1;
        if (t + 1 < ksteps)
            load(cur ^ 1, t + 1);       /* next stage while computing   */
#pragma unroll
        for (int k = 0; k < BK; k++) {
#pragma unroll
            for (int i = 0; i < TM; i += 4)
                *(float4 *)&a_frag[i] = *(const float4 *)&As[cur][k][tr * TM + i];
#pragma unroll
            for (int j = 0; j < TN; j += 4)
                *(float4 *)&b_frag[j] = *(const float4 *)&Bs[cur][k][tc * TN + j];
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();
    }

    float *Cblk = C + (size_t)(r * BM + tr * TM) * N + c * BN + tc * TN;
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j += 4)
            *(float4 *)(Cblk + (size_t)i * N + j) =
                make_float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
}

/* Exact check: every row sum and column sum of C, in double. */
static int check(const float *C, int N, double *total)
{
    double *SA = (double *)calloc(N, sizeof(double)), *SB = (double *)calloc(N, sizeof(double));
    double *rs = (double *)calloc(N, sizeof(double)), *cs = (double *)calloc(N, sizeof(double));
    int errors = 0;

    for (int k = 0; k < N; k++)
        for (int i = 0; i < N; i++) {
            SA[k] += Aval(i, k);
            SB[k] += Bval(k, i);
        }
    *total = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double v = C[(size_t)i * N + j];
            rs[i] += v;
            cs[j] += v;
            *total += v;
        }
    for (int i = 0; i < N; i++) {
        double er = rs[i], ec = cs[i];
        for (int k = 0; k < N; k++) {
            er -= Aval(i, k) * SB[k];
            ec -= SA[k] * Bval(k, i);
        }
        errors += (er != 0) + (ec != 0);
    }
    free(SA); free(SB); free(rs); free(cs);
    return errors;
}

int main(int argc, char *argv[])
{
    int ai = 1, skew = 1;

    if (ai < argc && strcmp(argv[ai], "-n") == 0) {
        skew = 0;
        ai++;
    }
    int N = ai < argc ? atoi(argv[ai++]) : 8192;
    int reps = ai < argc ? atoi(argv[ai++]) : 10;
    if (N < BM || N % BM || reps < 1) {
        fprintf(stderr, "usage: %s [-n] [N (multiple of %d)] [repeats]\n", argv[0], BM);
        return 1;
    }
    size_t bytes = (size_t)N * N * sizeof(float);
    float *hA = (float *)malloc(bytes), *hB = (float *)malloc(bytes), *hC = (float *)malloc(bytes);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            hA[(size_t)i * N + j] = Aval(i, j);
            hB[(size_t)i * N + j] = Bval(i, j);
        }

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMalloc(&dC, bytes));
    CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    dim3 grid(N / BN, N / BM);
    float ms, best = 1e30f, best_blas = 1e30f;

    torus_sgemm<<<grid, THREADS>>>(dA, dB, dC, N, skew);     /* warm-up  */
    CHECK(cudaGetLastError());
    for (int i = 0; i < reps; i++) {
        CHECK(cudaEventRecord(e0));
        torus_sgemm<<<grid, THREADS>>>(dA, dB, dC, N, skew);
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (ms < best)
            best = ms;
    }
    CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
    double total;
    int errors = check(hC, N, &total);

    /* cuBLAS reference (row-major C = A B is column-major C^T = B^T A^T) */
    cublasHandle_t h;
    const float one = 1.0f, zero = 0.0f;
    cublasCreate(&h);
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, dB, N, dA, N, &zero, dC, N);
    for (int i = 0; i < (reps < 3 ? reps : 3); i++) {
        CHECK(cudaEventRecord(e0));
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, dB, N, dA, N, &zero, dC, N);
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (ms < best_blas)
            best_blas = ms;
    }
    cublasDestroy(h);

    double flop = 2.0 * N * (double)N * N;
    printf("CUDA torus on %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    printf("  N        = %d, grid = %d x %d torus of %d-thread blocks, %d x %d tile each\n",
           N, N / BN, N / BM, THREADS, BM, BN);
    printf("  K order  = %s\n", skew ? "Cannon skew (k = r + c + s mod G)" : "unskewed (k = s)");
    printf("  sum(C)   = %.0f   C[0][0] = %.0f   C[N-1][N-1] = %.0f\n",
           total, hC[0], hC[(size_t)N * N - 1]);
    printf("  checks   = %d of %d row/column sums exact\n", 2 * N - errors, 2 * N);
    printf("  time     = %.6f s kernel (best of %d)\n", best / 1e3, reps);
    printf("  rate     = %.1f GFLOP/s   (cuBLAS SGEMM: %.1f GFLOP/s, %.0f%%)\n",
           flop / (best / 1e3) / 1e9, flop / (best_blas / 1e3) / 1e9, 100.0 * best_blas / best);
    printf("CSV,%d,%d,%.6f,%.6f,%d\n", N, skew, best / 1e3, best_blas / 1e3, errors);
    return errors != 0;
}
