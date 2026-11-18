/* SpatialFilterCUDA.cu - Project 8, GPU version (NVIDIA CUDA)
 *
 * Same 3x3 spatial filter and identical integer arithmetic as the C* and
 * MPI versions.  Each 32x8 thread block loads a (32+2)x(8+2) tile with a
 * one-pixel halo into shared memory, the mask lives in constant memory,
 * and multiple passes ping-pong between two device buffers.
 *
 * usage:
 *   SpatialFilterCUDA <input.txt> [passes] [out.bmp]
 *   SpatialFilterCUDA -gen <rows> <cols> <filter> [passes] [out.bmp]
 */
#include <cuda_runtime.h>
#include <chrono>
#include "image_io.h"

#define BX 32
#define BY 8

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

__constant__ int cK[9];
__constant__ int cDiv, cOff;

__global__ void filter3x3(const unsigned char *__restrict__ in, unsigned char *__restrict__ out,
                          int R, int C) {
    __shared__ unsigned char tile[BY + 2][BX + 2];
    int x = blockIdx.x * BX + threadIdx.x;
    int y = blockIdx.y * BY + threadIdx.y;

    /* cooperative load of the tile + halo, borders clamped */
    for (int ty = threadIdx.y; ty < BY + 2; ty += BY)
        for (int tx = threadIdx.x; tx < BX + 2; tx += BX) {
            int gy = blockIdx.y * BY + ty - 1, gx = blockIdx.x * BX + tx - 1;
            gy = gy < 0 ? 0 : gy >= R ? R - 1 : gy;
            gx = gx < 0 ? 0 : gx >= C ? C - 1 : gx;
            tile[ty][tx] = in[(size_t)gy * C + gx];
        }
    __syncthreads();
    if (x >= C || y >= R) return;

    int tx = threadIdx.x + 1, ty = threadIdx.y + 1;
    int s = cK[0] * tile[ty-1][tx-1] + cK[1] * tile[ty-1][tx] + cK[2] * tile[ty-1][tx+1]
          + cK[3] * tile[ty  ][tx-1] + cK[4] * tile[ty  ][tx] + cK[5] * tile[ty  ][tx+1]
          + cK[6] * tile[ty+1][tx-1] + cK[7] * tile[ty+1][tx] + cK[8] * tile[ty+1][tx+1];
    int v = s / cDiv + cOff;
    out[(size_t)y * C + x] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
}

int main(int argc, char **argv) {
    Image im = { 0 };
    int passes = 1, ai;
    const char *outBmp = NULL;
    char fname[64];
    if (argc >= 5 && strcmp(argv[1], "-gen") == 0) {
        const FilterDef *f = find_filter(argv[4]);
        if (!f) { fprintf(stderr, "unknown filter %s\n", argv[4]); return 1; }
        gen_image(&im, atoi(argv[2]), atoi(argv[3]), 0);
        set_filter(&im, f);
        snprintf(fname, sizeof fname, "%s (generated)", f->name);
        ai = 5;
    } else if (argc >= 2) {
        if (!read_image(argv[1], &im)) return 1;
        snprintf(fname, sizeof fname, "%.60s", argv[1]);
        ai = 2;
    } else {
        fprintf(stderr, "usage: SpatialFilterCUDA <input.txt>|-gen R C filter [passes] [out.bmp]\n");
        return 1;
    }
    if (argc > ai) passes = atoi(argv[ai]);
    if (argc > ai + 1) outBmp = argv[ai + 1];
    if (passes < 1) passes = 1;

    int R = im.rows, C = im.cols;
    size_t N = (size_t)R * C;

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    CK(cudaFree(0));   /* create the context outside the timed region */

    unsigned char *dA, *dB;
    CK(cudaMalloc(&dA, N));
    CK(cudaMalloc(&dB, N));
    CK(cudaMemcpyToSymbol(cK, im.kern, sizeof im.kern));
    CK(cudaMemcpyToSymbol(cDiv, &im.divisor, sizeof(int)));
    CK(cudaMemcpyToSymbol(cOff, &im.offset, sizeof(int)));

    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);

    dim3 block(BX, BY), grid((C + BX - 1) / BX, (R + BY - 1) / BY);
    unsigned char *result = (unsigned char *)malloc(N);

    /* untimed warm-up launch (module load, clocks ramp up) */
    filter3x3<<<grid, block>>>(dA, dB, R, C);
    CK(cudaDeviceSynchronize());

    cudaEventRecord(e0);
    CK(cudaMemcpy(dA, im.pix, N, cudaMemcpyHostToDevice));
    cudaEventRecord(e1);
    for (int p = 0; p < passes; p++) {
        filter3x3<<<grid, block>>>(dA, dB, R, C);
        unsigned char *t = dA; dA = dB; dB = t;
    }
    CK(cudaGetLastError());
    cudaEventRecord(e2);
    CK(cudaMemcpy(result, dA, N, cudaMemcpyDeviceToHost));
    cudaEventRecord(e3);
    cudaEventSynchronize(e3);

    float msH2D, msK, msD2H;
    cudaEventElapsedTime(&msH2D, e0, e1);
    cudaEventElapsedTime(&msK, e1, e2);
    cudaEventElapsedTime(&msD2H, e2, e3);

    /* single-core CPU reference for one pass (speedup + verification) */
    unsigned char *ref = (unsigned char *)malloc(N);
    auto c0 = std::chrono::steady_clock::now();
    filter_reference(&im, ref);
    double cpuPass = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();

    double gpuPass = msK / 1000.0 / passes;
    double mpix = (double)N / gpuPass / 1e6;
    printf("\n================ SPATIAL FILTER - CUDA GPU ======================\n");
    printf(" input            : %s\n", fname);
    printf(" image            : %d x %d pixels, 3x3 mask, %d pass(es)\n", R, C, passes);
    printf(" GPU              : %s (%d SMs, compute %d.%d)\n", prop.name,
           prop.multiProcessorCount, prop.major, prop.minor);
    printf(" launch           : grid %d x %d blocks of %d x %d threads, shared-memory tiles\n",
           grid.x, grid.y, BX, BY);
    printf("-----------------------------------------------------------------\n");
    printf(" host -> device   : %9.3f ms\n", msH2D);
    printf(" kernel (%d pass) : %9.3f ms  (%.4f ms per pass)\n", passes, msK, msK / passes);
    printf(" device -> host   : %9.3f ms\n", msD2H);
    printf(" throughput       : %9.1f Mpixel/s (%.1f GFLOP/s, 17 ops/pixel)\n", mpix, mpix * 17 / 1000);
    printf(" 1-core CPU pass  : %9.3f ms (unoptimised sequential reference)\n", cpuPass * 1000);
    printf(" kernel speedup   : %9.1f x  vs that single-core reference\n", cpuPass / gpuPass);
    printf("=================================================================\n");
    if (passes == 1) {
        size_t bad = 0;
        for (size_t i = 0; i < N; i++) bad += ref[i] != result[i];
        printf(" verify vs sequential reference: %zu pixels differ -> %s\n", bad, bad ? "FAIL" : "PASS");
    }
    printf(" result checksum  : %016llx\n", checksum(result, N));
    if (outBmp) { write_bmp(outBmp, result, R, C); printf(" wrote %s\n", outBmp); }

    cudaFree(dA); cudaFree(dB);
    free(ref); free(result);
    return 0;
}
