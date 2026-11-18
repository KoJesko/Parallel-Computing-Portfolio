/*
 * Project 6 - CPU-bound version (POSIX threads, up to 32 cores)
 *
 * Same algorithm as mesh_integrate.c, but the "multicomputer" is a set of
 * threads arranged as an R x C mesh and pinned to CPUs one physical core at
 * a time, fastest cores first, before any second hyperthread.  Threads
 * never touch a shared accumulator: each one owns a small mailbox per mesh
 * direction, and local sums are combined with the same neighbour-only
 * row-then-column sweep that converges on the centre node (~(R+C)/2 hops).
 *
 * Work distribution: the i9-14900HX is a hybrid part (P-cores ~5.6 GHz,
 * E-cores ~4.1 GHz), so a static equal split leaves the P-cores idle while
 * the E-cores finish.  By default threads instead pull fixed-size chunks of
 * intervals from an atomic counter (dynamic scheduling); pass -s to use the
 * static block split for comparison.
 *
 * Usage:  ./cpu_mesh [-s] [threads] [n] [rows cols]
 *         threads defaults to all online CPUs, n to 1000000000.
 */
#define _GNU_SOURCE
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define A 0.0
#define B 2.0
#define MAX_THREADS 256
#define CHUNK 262144L   /* intervals per dynamic work unit */
#define REPEATS 5

enum { FROM_LEFT, FROM_RIGHT, FROM_UP, FROM_DOWN, NDIR };

/* One mailbox slot, padded to its own cache line to avoid false sharing. */
typedef struct {
    _Atomic int ready;
    double val;
    char pad[64 - sizeof(int) - sizeof(double)];
} slot_t;

static slot_t box[MAX_THREADS][NDIR];
static int rows, cols, nthreads, use_static;
static long n;
static atomic_long next_chunk;
static pthread_barrier_t barrier;
static double result, t_reduce, t_done[MAX_THREADS];
static int cpu_order[MAX_THREADS], ncpus;

float f(float t)
{
    return (sqrt(4 - t * t));
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Trapezoidal rule over intervals [first, last) of the global n-interval grid. */
static double local_trapezoid(long first, long last)
{
    double h = (B - A) / n;
    double sum;
    long i;

    if (first >= last)
        return 0.0;
    sum = 0.5 * (f((float)(A + first * h)) + f((float)(A + last * h)));
    for (i = first + 1; i < last; i++)
        sum += f((float)(A + i * h));
    return sum * h;
}

static void send_to(int dest, int dir, double v)
{
    box[dest][dir].val = v;
    atomic_store_explicit(&box[dest][dir].ready, 1, memory_order_release);
}

static double recv_from(int me, int dir)
{
    int spins = 0;
    double v;

    while (!atomic_load_explicit(&box[me][dir].ready, memory_order_acquire))
        if (++spins > 1000) {
            sched_yield();
            spins = 0;
        }
    v = box[me][dir].val;
    atomic_store_explicit(&box[me][dir].ready, 0, memory_order_relaxed);
    return v;
}

/*
 * Combine values along one mesh line so the total lands at position mid.
 * lo/hi are the neighbour ids (-1 at the mesh edge); the dir_* arguments
 * say which mailbox a value arriving from that side goes into.
 */
static double line_reduce(int me, double val, int pos, int mid, int lo, int hi,
                          int dir_from_lo, int dir_from_hi)
{
    if (pos < mid) {
        if (lo >= 0)
            val += recv_from(me, dir_from_lo);
        send_to(hi, dir_from_lo, val);
    } else if (pos > mid) {
        if (hi >= 0)
            val += recv_from(me, dir_from_hi);
        send_to(lo, dir_from_hi, val);
    } else {
        if (lo >= 0)
            val += recv_from(me, dir_from_lo);
        if (hi >= 0)
            val += recv_from(me, dir_from_hi);
    }
    return val;
}

static void *node(void *arg)
{
    int me = (int)(long)arg;
    int r = me / cols, c = me % cols;
    int cr = (rows - 1) / 2, cc = (cols - 1) / 2;
    int left = c > 0 ? me - 1 : -1, right = c < cols - 1 ? me + 1 : -1;
    int up = r > 0 ? me - cols : -1, down = r < rows - 1 ? me + cols : -1;
    long chunk, extra, first, last, k, nchunks;
    double sum;
    cpu_set_t set;

    CPU_ZERO(&set);
    CPU_SET(cpu_order[me % ncpus], &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);

    pthread_barrier_wait(&barrier);   /* start signal from main */

    if (use_static) {
        chunk = n / nthreads;
        extra = n % nthreads;
        first = me * chunk + (me < extra ? me : extra);
        last = first + chunk + (me < extra ? 1 : 0);
        sum = local_trapezoid(first, last);
    } else {
        nchunks = (n + CHUNK - 1) / CHUNK;
        sum = 0.0;
        while ((k = atomic_fetch_add(&next_chunk, 1)) < nchunks) {
            first = k * CHUNK;
            last = first + CHUNK < n ? first + CHUNK : n;
            sum += local_trapezoid(first, last);
        }
    }

    t_done[me] = now();   /* published to the centre via the reduction chain */
    sum = line_reduce(me, sum, c, cc, left, right, FROM_LEFT, FROM_RIGHT);
    if (c == cc)
        sum = line_reduce(me, sum, r, cr, up, down, FROM_UP, FROM_DOWN);
    if (r == cr && c == cc) {
        /* Reduction cost = time after the last node finished computing. */
        double last_done = t_done[0];
        int j;

        for (j = 1; j < nthreads; j++)
            if (t_done[j] > last_done)
                last_done = t_done[j];
        result = sum;
        t_reduce = now() - last_done;
    }

    pthread_barrier_wait(&barrier);   /* finish signal to main */
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
 * Pinning thread i to CPU i would put threads 0 and 1 on the same core.
 */
static void build_cpu_order(void)
{
    long key[MAX_THREADS], tk;
    int i, j, t;

    ncpus = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpus > MAX_THREADS)
        ncpus = MAX_THREADS;
    for (i = 0; i < ncpus; i++) {
        /* the first sibling listed is the core's primary hyperthread */
        long first = read_long("/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", i);
        long mhz = read_long("/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i) / 1000;
        cpu_order[i] = i;
        key[i] = (first == i ? 0 : 1000000) - (mhz > 0 ? mhz : 0);
    }
    for (i = 1; i < ncpus; i++)          /* stable insertion sort on key */
        for (j = i; j > 0 && key[j - 1] > key[j]; j--) {
            tk = key[j]; key[j] = key[j - 1]; key[j - 1] = tk;
            t = cpu_order[j]; cpu_order[j] = cpu_order[j - 1]; cpu_order[j - 1] = t;
        }
}

/* Pick the most square R x C factorisation of p. */
static void mesh_dims(int p, int *r, int *c)
{
    int i;

    for (i = (int)sqrt((double)p); i > 1 && p % i; i--)
        ;
    *r = i;
    *c = p / i;
}

int main(int argc, char *argv[])
{
    pthread_t tid[MAX_THREADS];
    double t_start, t, best = 1e30, best_reduce = 0;
    int i, rep, ai = 1;

    if (ai < argc && strcmp(argv[ai], "-s") == 0) {
        use_static = 1;
        ai++;
    }
    nthreads = ai < argc ? atoi(argv[ai++]) : (int)sysconf(_SC_NPROCESSORS_ONLN);
    n = ai < argc ? atol(argv[ai++]) : 1000000000L;
    if (ai + 1 < argc) {
        rows = atoi(argv[ai]);
        cols = atoi(argv[ai + 1]);
    } else {
        mesh_dims(nthreads, &rows, &cols);
    }
    if (nthreads < 1 || nthreads > MAX_THREADS || rows * cols != nthreads || n < 1) {
        fprintf(stderr, "usage: %s [-s] [threads] [n] [rows cols]\n", argv[0]);
        return 1;
    }

    build_cpu_order();
    for (rep = 0; rep < REPEATS; rep++) {
        atomic_store(&next_chunk, 0);
        pthread_barrier_init(&barrier, NULL, nthreads + 1);
        for (i = 0; i < nthreads; i++)
            pthread_create(&tid[i], NULL, node, (void *)(long)i);
        t_start = now();
        pthread_barrier_wait(&barrier);
        pthread_barrier_wait(&barrier);
        t = now() - t_start;
        for (i = 0; i < nthreads; i++)
            pthread_join(tid[i], NULL);
        pthread_barrier_destroy(&barrier);
        if (t < best) {
            best = t;
            best_reduce = t_reduce;
        }
    }

    printf("CPU mesh %d x %d (%d threads, %s), n = %ld intervals\n",
           rows, cols, nthreads, use_static ? "static" : "dynamic", n);
    printf("  result   = %.12f\n", result);
    printf("  error    = %.3e\n", fabs(result - M_PI));
    printf("  time     = %.6f s (best of %d)\n", best, REPEATS);
    printf("  reduce   = %.2f us, %d neighbour steps\n", best_reduce * 1e6,
           cols / 2 + rows / 2);
    printf("CSV,%d,%d,%d,%ld,%.12f,%.6f\n", nthreads, rows, cols, n, result, best);
    return 0;
}
