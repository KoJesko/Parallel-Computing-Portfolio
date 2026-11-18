/*
 * Project 7 - real CPU version (POSIX threads, all 32 logical CPUs)
 *
 * Cannon's algorithm on a Q x Q torus of "nodes", the same algorithm as
 * TorusMatMul.c.  Every node owns its own NB x NB blocks of A, B and C
 * (NB = N / Q) in separately allocated memory; nodes never read another
 * node's blocks.  The only inter-node traffic is the torus rotation: at
 * each step a node copies its current A block into the inbox of its west
 * neighbour and its B block into the inbox of its north neighbour (with
 * wraparound), which is the shared-memory form of sending a message along
 * one torus link.  The inbox becomes the current block at the next step
 * (double buffering), so the copies of step s overlap the multiplies of
 * step s and only one barrier per step is needed.
 *
 * Each node is served by W threads, so Q*Q*W threads run in total
 * (default Q = 4, W = 2: 32 threads on a 4 x 4 torus).  The i9-14900HX is
 * a hybrid part (8 P-cores with 2 hyperthreads each, 16 E-cores), so equal
 * static shares leave the fast cores idle at every step.  By default each
 * step's work is cut into (node, 48-row strip) units that threads take
 * from an atomic counter, so faster cores do more of each step; -s uses a
 * static split (each thread works only on its own node's rows) for
 * comparison.
 *
 * Inner kernel: AVX2/FMA register-blocked 6 x 16 micro-kernel, 48-row x
 * 256-wide panels of A and packed 256 x 16 slices of B, so both stay in
 * cache whatever the block size.
 *
 * Matrix entries are small integers, so every C entry is exact in float and
 * the result is checked exactly (all row and column sums, in double).
 *
 * Usage:  ./cpu_torus [-s] [Q] [W] [N]
 *         defaults Q = 4, W = (online CPUs) / Q^2, N = 4096
 */
#define _GNU_SOURCE
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MAX_THREADS 256
#define MAX_NODES 256
#define MR 6                /* micro-kernel rows                         */
#define NR 16               /* micro-kernel columns (2 AVX2 vectors)     */
#define KC 256              /* K panel width                             */
#define MC 48               /* rows per work unit (A panel stays in L2)  */
#define REPEATS 3

static int Q, W, N, NB, LD, nnodes, nthreads, use_static;
static float *Ablk[MAX_NODES][2], *Bblk[MAX_NODES][2], *Cblk[MAX_NODES];
static atomic_int next_unit[MAX_NODES];   /* one work counter per step */
static pthread_barrier_t barrier;
static int cpu_order[MAX_THREADS], ncpus;
static double t_start, t_end;

static float Aval(int r, int c) { return (float)((r * 7 + c * 3) % 11 - 5); }
static float Bval(int r, int c) { return (float)((r * 5 + c * 2) % 13 - 6); }

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* C[i0..i0+6) x [j0..j0+16) += A[i0.., 0..kn) * Bp[0..kn, 0..16)
 * A and C have row stride ld; Bp is a packed contiguous kn x 16 slice. */
static void micro_6x16(const float *A, const float *Bp, float *C,
                       int i0, int j0, int kn, int ld)
{
    __m256 c[MR][2];
    int r, k;

    for (r = 0; r < MR; r++) {
        c[r][0] = _mm256_loadu_ps(C + (i0 + r) * ld + j0);
        c[r][1] = _mm256_loadu_ps(C + (i0 + r) * ld + j0 + 8);
    }
    for (k = 0; k < kn; k++) {
        __m256 b0 = _mm256_load_ps(Bp + k * NR);
        __m256 b1 = _mm256_load_ps(Bp + k * NR + 8);
        for (r = 0; r < MR; r++) {
            __m256 a = _mm256_broadcast_ss(A + (i0 + r) * ld + k);
            c[r][0] = _mm256_fmadd_ps(a, b0, c[r][0]);
            c[r][1] = _mm256_fmadd_ps(a, b1, c[r][1]);
        }
    }
    for (r = 0; r < MR; r++) {
        _mm256_storeu_ps(C + (i0 + r) * ld + j0, c[r][0]);
        _mm256_storeu_ps(C + (i0 + r) * ld + j0 + 8, c[r][1]);
    }
}

/*
 * Rows [i0, i1) of one node's C block += A block x B block, i0..i1 at most
 * MC rows.  For each 256-wide K panel the A rows (MC x KC) stay in L2, and
 * each 256 x 16 slice of B is first packed into a contiguous 16 KB buffer
 * that stays in L1 while all MC rows use it.  Without packing the slice
 * spans 256 rows (256 pages for a 4096-wide block), and the TLB misses
 * make large blocks run ~40% slower.
 */
static void block_rows(const float *A, const float *B, float *C, int i0, int i1)
{
    static __thread float Bp[KC * NR] __attribute__((aligned(32)));
    int kk, kend, i, j, k, r;

    for (kk = 0; kk < NB; kk += KC) {
        kend = kk + KC < NB ? kk + KC : NB;
        for (j = 0; j + NR <= NB; j += NR) {
            for (k = kk; k < kend; k++)
                memcpy(Bp + (k - kk) * NR, B + k * LD + j, NR * sizeof(float));
            for (i = i0; i + MR <= i1; i += MR)
                micro_6x16(A + kk, Bp, C, i, j, kend - kk, LD);
            for (r = i; r < i1; r++)                 /* leftover rows  */
                for (k = kk; k < kend; k++)
                    for (int jj = j; jj < j + NR; jj++)
                        C[r * LD + jj] += A[r * LD + k] * B[k * LD + jj];
        }
        for (; j < NB; j++)                          /* leftover cols  */
            for (r = i0; r < i1; r++)
                for (k = kk; k < kend; k++)
                    C[r * LD + j] += A[r * LD + k] * B[k * LD + j];
    }
}

/* Rows [i0, i1) in MC-row pieces. */
static void node_rows(const float *A, const float *B, float *C, int i0, int i1)
{
    for (int i = i0; i < i1; i += MC)
        block_rows(A, B, C, i, i + MC < i1 ? i + MC : i1);
}

/* "Send" one block to a neighbour: copy rows [r0, r1) into its inbox. */
static void send_rows(const float *src, float *inbox, int r0, int r1)
{
    memcpy(inbox + (size_t)r0 * LD, src + (size_t)r0 * LD,
           (size_t)(r1 - r0) * LD * sizeof(float));
}

static void *worker(void *arg)
{
    int t = (int)(long)arg;
    int me = t / W, w = t % W;                 /* my node, my slot in it */
    int row = me / Q, col = me % Q;
    int west = row * Q + (col + Q - 1) % Q, north = ((row + Q - 1) % Q) * Q + col;
    int strips = (NB + MC - 1) / MC, units = nnodes * strips;
    int r0 = (int)((long)NB * w / W), r1 = (int)((long)NB * (w + 1) / W);
    int step, cur, u, i;
    cpu_set_t set;

    CPU_ZERO(&set);
    CPU_SET(cpu_order[t % ncpus], &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);

    /* ---- generate own blocks (each thread its share of rows) into the
     *      spare buffers; this is the input, so it is not timed --------- */
    for (i = r0; i < r1; i++)
        for (int j = 0; j < NB; j++) {
            Ablk[me][1][i * LD + j] = Aval(row * NB + i, col * NB + j);
            Bblk[me][1][i * LD + j] = Bval(row * NB + i, col * NB + j);
            Cblk[me][i * LD + j] = 0;
        }
    pthread_barrier_wait(&barrier);
    if (t == 0)
        t_start = now();

    /* ---- alignment: A(row,col) -> node (row, col-row),
     *                 B(row,col) -> node (row-col, col) ---------------- */
    send_rows(Ablk[me][1], Ablk[row * Q + (col + Q - row) % Q][0], r0, r1);
    send_rows(Bblk[me][1], Bblk[((row + Q - col) % Q) * Q + col][0], r0, r1);
    pthread_barrier_wait(&barrier);

    /* ---- Q steps: rotate (into the neighbours' inboxes) and multiply -- */
    for (step = 0; step < Q; step++) {
        cur = step & 1;
        if (step < Q - 1) {
            send_rows(Ablk[me][cur], Ablk[west][cur ^ 1], r0, r1);
            send_rows(Bblk[me][cur], Bblk[north][cur ^ 1], r0, r1);
        }
        if (use_static) {
            node_rows(Ablk[me][cur], Bblk[me][cur], Cblk[me], r0, r1);
        } else {
            while ((u = atomic_fetch_add(&next_unit[step], 1)) < units) {
                int node = u / strips, s = u % strips;
                int i1 = (s + 1) * MC < NB ? (s + 1) * MC : NB;
                block_rows(Ablk[node][cur], Bblk[node][cur], Cblk[node], s * MC, i1);
            }
        }
        pthread_barrier_wait(&barrier);
    }
    if (t == 0)
        t_end = now();
    return NULL;
}

static long read_long(const char *fmt, int cpu)
{
    char path[128];
    long v = -1;
    FILE *fp;

    snprintf(path, sizeof(path), fmt, cpu);
    if ((fp = fopen(path, "r")) != NULL) {
        if (fscanf(fp, "%ld", &v) != 1)
            v = -1;
        fclose(fp);
    }
    return v;
}

/*
 * Order logical CPUs for pinning: one hyperthread per physical core first,
 * fastest cores (highest max clock) first, then the second hyperthreads.
 */
static void build_cpu_order(void)
{
    long key[MAX_THREADS], tk;
    int i, j, t;

    ncpus = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpus > MAX_THREADS)
        ncpus = MAX_THREADS;
    for (i = 0; i < ncpus; i++) {
        long first = read_long("/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", i);
        long mhz = read_long("/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i) / 1000;
        cpu_order[i] = i;
        key[i] = (first == i ? 0 : 1000000) - (mhz > 0 ? mhz : 0);
    }
    for (i = 1; i < ncpus; i++)
        for (j = i; j > 0 && key[j - 1] > key[j]; j--) {
            tk = key[j]; key[j] = key[j - 1]; key[j - 1] = tk;
            t = cpu_order[j]; cpu_order[j] = cpu_order[j - 1]; cpu_order[j - 1] = t;
        }
}

/* Exact check: every row sum and column sum of C, in double. */
static int check(double *total, float *c00, float *cnn)
{
    double *SA = calloc(N, sizeof(double)), *SB = calloc(N, sizeof(double));
    double *rs = calloc(N, sizeof(double)), *cs = calloc(N, sizeof(double));
    int errors = 0, i, j, k, me;

    for (k = 0; k < N; k++)
        for (i = 0; i < N; i++) {
            SA[k] += Aval(i, k);
            SB[k] += Bval(k, i);
        }
    *total = 0;
    for (me = 0; me < nnodes; me++)
        for (i = 0; i < NB; i++)
            for (j = 0; j < NB; j++) {
                double v = Cblk[me][i * LD + j];
                rs[(me / Q) * NB + i] += v;
                cs[(me % Q) * NB + j] += v;
                *total += v;
            }
    for (i = 0; i < N; i++) {
        double er = rs[i], ec = cs[i];
        for (k = 0; k < N; k++) {
            er -= Aval(i, k) * SB[k];
            ec -= SA[k] * Bval(k, i);
        }
        errors += (er != 0) + (ec != 0);
    }
    *c00 = Cblk[0][0];
    *cnn = Cblk[nnodes - 1][(NB - 1) * LD + NB - 1];
    free(SA); free(SB); free(rs); free(cs);
    return errors;
}

int main(int argc, char *argv[])
{
    pthread_t tid[MAX_THREADS];
    double best = 1e30, total;
    float c00, cnn;
    int i, rep, ai = 1, errors;

    if (ai < argc && strcmp(argv[ai], "-s") == 0) {
        use_static = 1;
        ai++;
    }
    Q = ai < argc ? atoi(argv[ai++]) : 4;
    W = ai < argc ? atoi(argv[ai++]) : (int)sysconf(_SC_NPROCESSORS_ONLN) / (Q * Q);
    N = ai < argc ? atoi(argv[ai++]) : 4096;
    if (W < 1)
        W = 1;
    nnodes = Q * Q;
    nthreads = nnodes * W;
    if (Q < 1 || nnodes > MAX_NODES || nthreads > MAX_THREADS || N % Q || N < Q) {
        fprintf(stderr, "usage: %s [-s] [Q] [W] [N]   (N divisible by Q)\n", argv[0]);
        return 1;
    }
    NB = N / Q;
    LD = NB + 16;           /* pad rows: a power-of-two stride would map
                               every row of a panel to the same cache sets */

    for (i = 0; i < nnodes; i++) {          /* each node's private memory */
        size_t bytes = (size_t)NB * LD * sizeof(float);
        Ablk[i][0] = aligned_alloc(64, bytes); Ablk[i][1] = aligned_alloc(64, bytes);
        Bblk[i][0] = aligned_alloc(64, bytes); Bblk[i][1] = aligned_alloc(64, bytes);
        Cblk[i] = aligned_alloc(64, bytes);
    }
    build_cpu_order();

    for (rep = 0; rep < REPEATS; rep++) {
        for (i = 0; i < Q; i++)
            atomic_store(&next_unit[i], 0);
        pthread_barrier_init(&barrier, NULL, nthreads);
        for (i = 0; i < nthreads; i++)
            pthread_create(&tid[i], NULL, worker, (void *)(long)i);
        for (i = 0; i < nthreads; i++)
            pthread_join(tid[i], NULL);
        pthread_barrier_destroy(&barrier);
        if (t_end - t_start < best)
            best = t_end - t_start;
    }
    errors = check(&total, &c00, &cnn);

    printf("CPU torus %d x %d, %d threads per node (%d threads, %s), N = %d\n",
           Q, Q, W, nthreads, use_static ? "static" : "dynamic", N);
    printf("  block    = %d x %d per node, %d rotation steps\n", NB, NB, Q - 1);
    printf("  sum(C)   = %.0f   C[0][0] = %.0f   C[N-1][N-1] = %.0f\n", total, c00, cnn);
    printf("  checks   = %d of %d row/column sums exact\n", 2 * N - errors, 2 * N);
    printf("  time     = %.6f s (best of %d)\n", best, REPEATS);
    printf("  rate     = %.1f GFLOP/s\n", 2.0 * N * N * (double)N / best / 1e9);
    printf("CSV,%d,%d,%d,%d,%d,%.6f,%d\n", nthreads, Q, W, N, use_static, best, errors);
    return errors != 0;
}
