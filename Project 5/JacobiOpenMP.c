/* =====================================================================
 *  JacobiOpenMP.c
 *
 *  Project 5 - Efficient parallel Jacobi relaxation on a multiprocessor,
 *              with convergence test and efficient barrier and
 *              aggregation functions.
 *
 *  This is the REAL HARDWARE companion to the C* simulator programs
 *  (JacobiParallel.c / JacobiNaiveBarrier.c / JacobiSequential.c).  It
 *  solves the same problem with the same algorithm and the same hand
 *  written synchronisation, but it runs on the actual multiprocessor,
 *  in double precision, on grids large enough to be interesting.
 *
 *  The barrier and the aggregation are written by hand on purpose:
 *  "#pragma omp barrier" and "reduction(max:)" are NOT used by the
 *  efficient version, because supplying them is the point of the
 *  project.  They are available as the "omp" sync mode purely as a
 *  reference point to measure the hand written versions against.
 *
 *  ------------------------------------------------------------------
 *  THE EFFICIENT BARRIER AND AGGREGATION
 *
 *  AggregateMax() is a DISSEMINATION all-reduce: in round k every
 *  thread publishes its partial maximum, then consumes the value
 *  published by the thread 2^k places behind it on a ring.  After
 *  ceil(log2 P) rounds every thread holds the maximum over all P
 *  threads, so the function is simultaneously
 *
 *      - the barrier that ends the Jacobi sweep, and
 *      - the reduction that the convergence test needs,
 *
 *  in O(log P) time with no locks, no atomic contention on a shared
 *  counter and no master thread.  (Dissemination is used rather than
 *  the textbook butterfly so that any thread count works, not only
 *  powers of two.  It is valid here because max is idempotent and
 *  commutative.)
 *
 *  Three details make it fast and correct on real hardware:
 *   1. CACHE LINE PADDING - every (thread, round) slot occupies its own
 *      64 byte line, so the spinning threads never share a line and
 *      there is no false sharing ping-pong.
 *   2. RELEASE / ACQUIRE ORDERING - the value is written first and the
 *      flag is published with a release store; the consumer reads the
 *      flag with an acquire load.  That is what guarantees a reader
 *      which sees the flag also sees the value, on a weakly ordered
 *      machine as well as on x86.
 *   3. MONOTONIC TICKETS + PARITY DOUBLE BUFFERING - the flag carries
 *      the sweep number, so flags never need to be cleared (no second
 *      "reset" barrier), and the slots are double buffered by sweep
 *      parity so a thread that runs ahead cannot overwrite a value a
 *      slower partner has not read yet.  A thread cannot get two
 *      sweeps ahead, because finishing sweep s+1 means it has heard,
 *      directly or transitively, from every other thread at sweep s+1.
 *  ------------------------------------------------------------------
 *
 *  ------------------------------------------------------------------
 *  ADAPTIVE LOAD BALANCING  (option -L)
 *
 *  A static equal row partition is only efficient on a UNIFORM
 *  multiprocessor.  On a hybrid machine (this one has 8 fast P-cores
 *  and 16 slower E-cores) the sweep time is set by the SLOWEST thread,
 *  so an equal partition throws away most of the fast cores' capacity.
 *
 *  With -L each thread times its own band, publishes the rate it
 *  achieved (rows per second), and at a few geometrically spaced
 *  sweeps the partition is recomputed so that the number of rows a
 *  thread owns is proportional to the speed it actually demonstrated.
 *  All threads then finish their band at the same moment and the
 *  barrier wait goes to nearly zero.
 *
 *  This changes only WHICH thread updates WHICH row, never the
 *  arithmetic, so the sweep count and the checksum are unchanged -
 *  which is exactly what the validation in benchmark-openmp.sh checks.
 *  ------------------------------------------------------------------
 *
 *  BUILD    make            (or see build.sh)
 *  USAGE    ./JacobiOpenMP [options]
 *    -n N     interior grid is N x N                  (default 1024)
 *    -p P     number of threads                       (default: all cores)
 *    -t TOL   convergence tolerance                   (default 1e-2)
 *    -i MAX   maximum sweeps                          (default 1000000)
 *    -s MODE  sync mode: tree | linear | omp          (default tree)
 *    -c K     run the convergence test every K sweeps (default 1)
 *    -r R     repeat the solve R times, report best    (default 1)
 *    -L       adaptive load balancing for hybrid/uneven cores
 *    -b       barrier microbenchmark: time the barrier alone
 *    -I N     iterations for -b (default 200000)
 *    -q       terse one-line output (for scripts)
 *    -h       help
 * ===================================================================== */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>
#include <sched.h>
#include <stdatomic.h>
#include <omp.h>

#if defined(__x86_64__) || defined(__i386__)
#  include <immintrin.h>
#  define CPU_RELAX() _mm_pause()
#elif defined(__aarch64__)
#  define CPU_RELAX() __asm__ __volatile__("yield" ::: "memory")
#else
#  define CPU_RELAX() ((void)0)
#endif

#define CACHELINE 64
#define MAXROUNDS 32                 /* enough for 2^32 threads */

/* ---------------------------------------------------------------------
 *  One synchronisation slot, alone on its own cache line.
 * ------------------------------------------------------------------- */
typedef struct {
    _Atomic uint64_t flag;           /* sweep number, published last    */
    double           val;            /* partial maximum, published first*/
    char             pad[CACHELINE - sizeof(_Atomic uint64_t) - sizeof(double)];
} slot_t;

/* ---------------------------------------------------------------------
 *  Shared state for the three synchronisation modes.
 * ------------------------------------------------------------------- */
typedef struct {
    int      P;                      /* threads                         */
    int      rounds;                 /* ceil(log2 P)                    */
    slot_t  *slots;                  /* [2][P][rounds] for AggregateMax */
    slot_t  *bslots;                 /* [2][P][rounds] for Barrier      */

    /* naive centralised barrier (the control experiment) */
    omp_lock_t        mutex;
    _Atomic int       count;
    _Atomic uint64_t  release;
    double            gmax[2];      /* parity buffered, see below       */

    /* reference mode: shared array combined after an omp barrier */
    double  *partial;                /* [P], cache line separated       */
    double   ompmax;

    /* adaptive load balancing */
    double  *rate;                   /* [P] cache line separated        */
    int     *band_r0;                /* [P] first interior row owned    */
    int     *band_rows;              /* [P] rows owned                  */
} sync_t;

static inline slot_t *slot_of(sync_t *s, int par, int tid, int k)
{
    return &s->slots[((size_t)par * s->P + tid) * s->rounds + k];
}

/*  Barrier() MUST NOT share its slots with AggregateMax().  If it did, a
 *  barrier carrying a higher ticket would leave flags that let the NEXT
 *  sweep's aggregation pass its spin immediately and read a stale value -
 *  losing both the synchronisation and the maximum.  Hence a second,
 *  independent set of slots, exactly as the C* program keeps bflag and
 *  bsync apart.                                                         */
static inline slot_t *bslot_of(sync_t *s, int par, int tid, int k)
{
    return &s->bslots[((size_t)par * s->P + tid) * s->rounds + k];
}

/* =====================================================================
 *  EFFICIENT AGGREGATION  (and barrier)  -  O(log P), lock free
 *
 *  Returns the maximum of "v" over all P threads, to every thread.
 *  Also acts as a full barrier.  "ticket" must increase on every call
 *  (the sweep number).
 * ===================================================================== */
static double AggregateMax(sync_t *s, double v, int me, uint64_t ticket)
{
    const int P = s->P;
    if (P == 1) return v;

    const int par = (int)(ticket & 1u);
    int step = 1;

    for (int k = 0; k < s->rounds; k++) {
        slot_t *mine = slot_of(s, par, me, k);
        mine->val = v;                                    /* value first */
        atomic_store_explicit(&mine->flag, ticket, memory_order_release);

        int src = me - step;                 /* ring predecessor 2^k back */
        if (src < 0) src += P;
        slot_t *his = slot_of(s, par, src, k);

        unsigned spins = 0;
        while (atomic_load_explicit(&his->flag, memory_order_acquire) < ticket) {
            CPU_RELAX();
            if (++spins == 4096u) { spins = 0; sched_yield(); }
        }
        double o = his->val;                              /* value after  */
        if (o > v) v = o;

        step <<= 1;
    }
    return v;
}

/* =====================================================================
 *  EFFICIENT BARRIER  -  the same pattern with no value to combine.
 * ===================================================================== */
static void Barrier(sync_t *s, int me, uint64_t ticket)
{
    const int P = s->P;
    if (P == 1) return;

    const int par = (int)(ticket & 1u);
    int step = 1;

    for (int k = 0; k < s->rounds; k++) {
        slot_t *mine = bslot_of(s, par, me, k);
        atomic_store_explicit(&mine->flag, ticket, memory_order_release);

        int src = me - step;
        if (src < 0) src += P;
        slot_t *his = bslot_of(s, par, src, k);

        unsigned spins = 0;
        while (atomic_load_explicit(&his->flag, memory_order_acquire) < ticket) {
            CPU_RELAX();
            if (++spins == 4096u) { spins = 0; sched_yield(); }
        }
        step <<= 1;
    }
}

/* =====================================================================
 *  NAIVE CONTROL  -  centralised counter barrier guarded by one lock,
 *  plus a global maximum built by P serialised critical sections.
 *  This is O(P) and every thread hammers the same two words.
 * ===================================================================== */
static double AggregateMaxLinear(sync_t *s, double v, int me, uint64_t ticket)
{
    (void)me; (void)ticket;
    const int P = s->P;
    if (P == 1) return v;

    /* --- every arrival serialises on the one lock: this is the O(P) --- */
    omp_set_lock(&s->mutex);
    uint64_t myrel = atomic_load_explicit(&s->release, memory_order_relaxed);
    int par = (int)(myrel & 1u);        /* parity buffer, so a thread    */
    if (s->count == 0) s->gmax[par] = -1.0;   /* that races ahead into   */
    if (v > s->gmax[par]) s->gmax[par] = v;   /* the next barrier cannot */
    int arrived = ++s->count;                 /* clobber this maximum    */
    if (arrived == P) {                        /* the last one releases  */
        s->count = 0;
        atomic_store_explicit(&s->release, myrel + 1, memory_order_release);
    }
    omp_unset_lock(&s->mutex);

    if (arrived != P) {
        unsigned spins = 0;
        while (atomic_load_explicit(&s->release, memory_order_acquire) == myrel) {
            CPU_RELAX();
            if (++spins == 4096u) { spins = 0; sched_yield(); }
        }
    }
    return s->gmax[par];
}

/* =====================================================================
 *  LIBRARY REFERENCE  -  OpenMP's own barrier plus a shared array that
 *  one thread combines.  Measured only for comparison.
 * ===================================================================== */
static double AggregateMaxOmp(sync_t *s, double v, int me)
{
    const int P = s->P;
    if (P == 1) return v;

    s->partial[(size_t)me * (CACHELINE / sizeof(double))] = v;
    #pragma omp barrier
    #pragma omp single
    {
        double m = s->partial[0];
        for (int i = 1; i < P; i++) {
            double x = s->partial[(size_t)i * (CACHELINE / sizeof(double))];
            if (x > m) m = x;
        }
        s->ompmax = m;
    }                        /* the implicit barrier of single publishes */
    return s->ompmax;
}

/* ------------------------------------------------------------------- */
typedef enum { SYNC_TREE, SYNC_LINEAR, SYNC_OMP } syncmode_t;

static const char *syncname(syncmode_t m)
{
    switch (m) {
        case SYNC_TREE:   return "tree";
        case SYNC_LINEAR: return "linear";
        default:          return "omp";
    }
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int ceil_log2i(int n)
{
    int r = 0;
    while ((1 << r) < n) r++;
    return r;
}

/* ---------------------------------------------------------------------
 *  Recompute the row partition so that each thread's share is
 *  proportional to the rate it actually achieved.  Called by one thread
 *  between two barriers, so every thread picks up the same partition.
 *
 *  "damp" keeps the adjustment from oscillating: the new share is a
 *  blend of the share a thread has now and the share its measured rate
 *  says it deserves.
 * ------------------------------------------------------------------- */
static void repartition(sync_t *s, int N, double damp)
{
    const int P = s->P;
    const size_t st = CACHELINE / sizeof(double);

    double tot = 0.0;
    for (int i = 0; i < P; i++) {
        double r = s->rate[(size_t)i * st];
        if (!(r > 0.0) || !isfinite(r)) r = 1.0;   /* ignore a bad sample */
        s->rate[(size_t)i * st] = r;
        tot += r;
    }
    if (!(tot > 0.0)) return;

    int    want[1024];
    int    assigned = 0;
    int    use = P < 1024 ? P : 1024;
    for (int i = 0; i < use; i++) {
        double ideal = (double)N * s->rate[(size_t)i * st] / tot;
        double blend = damp * (double)s->band_rows[i] + (1.0 - damp) * ideal;
        int    w     = (int)(blend + 0.5);
        if (w < 1) w = 1;
        want[i] = w;
        assigned += w;
    }
    /* ---- force the shares to add up to exactly N ---------------------- */
    while (assigned != N) {
        int best = 0;
        if (assigned > N) {                 /* take a row from the biggest */
            for (int i = 1; i < use; i++) if (want[i] > want[best]) best = i;
            if (want[best] <= 1) break;
            want[best]--; assigned--;
        } else {                            /* give a row to the fastest   */
            double bv = -1.0;
            for (int i = 0; i < use; i++) {
                double v = s->rate[(size_t)i * st] / (double)want[i];
                if (v > bv) { bv = v; best = i; }
            }
            want[best]++; assigned++;
        }
    }
    int at = 1;
    for (int i = 0; i < use; i++) {
        s->band_r0[i]   = at;
        s->band_rows[i] = want[i];
        at += want[i];
    }
}

/* =====================================================================
 *  The solver.
 *
 *  Row block partition: thread "me" owns a contiguous band of interior
 *  rows, so only two halo rows per thread are ever shared.  Ping pong
 *  buffers: a sweep reads only the old grid and writes only the new
 *  one, which is why ONE barrier per sweep is enough and why there is
 *  no copy back sweep.
 *
 *  First touch: every thread initialises the rows it owns, so on a NUMA
 *  machine each band is backed by memory local to the socket that will
 *  use it.
 * ===================================================================== */
typedef struct {
    double  seconds;
    long    sweeps;
    double  final_change;
    double  checksum;
    double  centre;
} result_t;

static result_t solve(int N, int P, double tol, long maxit,
                      syncmode_t mode, int checkevery, double hot,
                      bool balance)
{
    const size_t stride = (size_t)(N + 2 + 6) & ~(size_t)7;  /* pad rows  */
    const size_t cells  = stride * (size_t)(N + 2);

    double *const base0 = aligned_alloc(CACHELINE, cells * sizeof(double));
    double *const base1 = aligned_alloc(CACHELINE, cells * sizeof(double));
    if (!base0 || !base1) { fprintf(stderr, "out of memory\n"); exit(1); }
    double *g0 = base0, *g1 = base1;   /* g0 ends up holding the answer  */

    sync_t S;
    memset(&S, 0, sizeof S);
    S.P      = P;
    S.rounds = ceil_log2i(P); if (S.rounds == 0) S.rounds = 1;
    S.slots  = aligned_alloc(CACHELINE,
                   (size_t)2 * P * S.rounds * sizeof(slot_t));
    memset(S.slots, 0, (size_t)2 * P * S.rounds * sizeof(slot_t));
    S.bslots = aligned_alloc(CACHELINE,
                   (size_t)2 * P * S.rounds * sizeof(slot_t));
    memset(S.bslots, 0, (size_t)2 * P * S.rounds * sizeof(slot_t));
    S.partial = aligned_alloc(CACHELINE,
                   (size_t)P * CACHELINE);
    memset(S.partial, 0, (size_t)P * CACHELINE);
    S.rate = aligned_alloc(CACHELINE, (size_t)P * CACHELINE);
    memset(S.rate, 0, (size_t)P * CACHELINE);
    S.band_r0   = calloc((size_t)P, sizeof(int));
    S.band_rows = calloc((size_t)P, sizeof(int));
    atomic_store(&S.count, 0);
    atomic_store(&S.release, 0);
    omp_init_lock(&S.mutex);

    long   sweeps_out = 0;
    double change_out = 0.0;
    double t0 = 0.0, t1 = 0.0;

    #pragma omp parallel num_threads(P)
    {
        const int me = omp_get_thread_num();

        /* ---- row block partition (equal to start with) --------------- */
        const int base  = N / P, extra = N % P;
        int rows  = base + (me < extra ? 1 : 0);
        int r0    = 1 + me * base + (me < extra ? me : extra);
        int r1    = r0 + rows - 1;
        S.band_r0[me]   = r0;
        S.band_rows[me] = rows;
        const size_t rst = CACHELINE / sizeof(double);
        long next_rebal = balance ? 8 : 0;

        /* ---- first touch: each thread writes its own rows ------------ */
        for (int r = r0; r <= r1; r++)
            for (int c = 0; c <= N + 1; c++) {
                g0[(size_t)r * stride + c] = 0.0;
                g1[(size_t)r * stride + c] = 0.0;
            }
        #pragma omp single
        {
            for (int c = 0; c <= N + 1; c++) {      /* boundary rows      */
                g0[c] = hot;  g1[c] = hot;          /* top edge = hot     */
                g0[(size_t)(N + 1) * stride + c] = 0.0;
                g1[(size_t)(N + 1) * stride + c] = 0.0;
            }
        }

        double *cur = g0, *nxt = g1;
        long   sweep = 0;
        double gchange = 0.0;

        #pragma omp barrier
        if (me == 0) t0 = now_sec();

        for (;;) {
            sweep++;

            /* ---- 1. relax my own band, tracking my largest change ---- */
            double localmax = 0.0;
            const bool timing = balance && next_rebal &&
                                sweep >= next_rebal - 2;
            double tc0 = timing ? now_sec() : 0.0;
            for (int r = r0; r <= r1; r++) {
                const double *__restrict up = cur + (size_t)(r - 1) * stride;
                const double *__restrict mi = cur + (size_t)r       * stride;
                const double *__restrict dn = cur + (size_t)(r + 1) * stride;
                double       *__restrict ou = nxt + (size_t)r       * stride;
                double m = localmax;
                #pragma omp simd reduction(max:m)
                for (int c = 1; c <= N; c++) {
                    double v = 0.25 * (up[c] + dn[c] + mi[c - 1] + mi[c + 1]);
                    ou[c] = v;
                    double d = fabs(v - mi[c]);
                    m = d > m ? d : m;
                }
                localmax = m;
            }

            if (timing) {        /* rate this thread actually achieved  */
                double dt = now_sec() - tc0;
                S.rate[(size_t)me * rst] =
                    (dt > 0.0) ? (double)rows / dt : 0.0;
            }

            /* ---- 2. barrier + global maximum, one O(log P) operation -- */
            bool test = (sweep % checkevery == 0);
            if (mode == SYNC_TREE) {
                if (test) gchange = AggregateMax(&S, localmax, me,
                                                 (uint64_t)sweep);
                else      Barrier(&S, me, (uint64_t)sweep * 2u);
            } else if (mode == SYNC_LINEAR) {
                gchange = AggregateMaxLinear(&S, localmax, me,
                                             (uint64_t)sweep);
            } else {
                gchange = AggregateMaxOmp(&S, localmax, me);
            }

            /* ---- 2b. adaptive load balancing ------------------------- */
            if (balance && sweep == next_rebal) {
                /* the aggregation above was a barrier, so every rate is
                 * published; one thread repartitions and a second
                 * barrier publishes the new bands.                      */
                if (mode != SYNC_TREE) {
                    #pragma omp barrier
                } 
                if (me == 0) repartition(&S, N, 0.3);
                if (mode == SYNC_TREE) Barrier(&S, me, (uint64_t)sweep * 2u + 1u);
                else { 
                    #pragma omp barrier
                }
                rows = S.band_rows[me];
                r0   = S.band_r0[me];
                r1   = r0 + rows - 1;
                next_rebal = (next_rebal < 1024) ? next_rebal * 2 : 0;
            }

            /* ---- 3. exchange the grids: no copy back sweep ------------ */
            double *t = cur; cur = nxt; nxt = t;

            /* ---- 4. convergence test, identical in every thread ------- */
            if (test && gchange <= tol) break;
            if (sweep >= maxit) break;
        }

        if (me == 0) {
            t1 = now_sec();
            sweeps_out = sweep;
            change_out = gchange;
        }

        /* the final solution is in "cur" for every thread */
        #pragma omp barrier
        if (me == 0) { g0 = cur; g1 = nxt; }
    }

    /* ---- validation numbers ------------------------------------------ */
    double sum = 0.0;
    for (int r = 1; r <= N; r++)
        for (int c = 1; c <= N; c++)
            sum += g0[(size_t)r * stride + c];

    result_t res;
    res.seconds      = t1 - t0;
    res.sweeps       = sweeps_out;
    res.final_change = change_out;
    res.checksum     = sum;
    res.centre       = g0[(size_t)(N / 2) * stride + (N / 2)];

    omp_destroy_lock(&S.mutex);
    free(S.slots); free(S.bslots); free(S.partial); free(S.rate);
    free(S.band_r0); free(S.band_rows);
    free(base0); free(base1);
    return res;
}

/* =====================================================================
 *  Barrier microbenchmark: time the synchronisation alone, with no
 *  Jacobi work at all, to measure the cost of one barrier directly.
 * ===================================================================== */
static double barrier_cost_ns(int P, syncmode_t mode, long iters)
{
    sync_t S;
    memset(&S, 0, sizeof S);
    S.P      = P;
    S.rounds = ceil_log2i(P); if (S.rounds == 0) S.rounds = 1;
    S.slots  = aligned_alloc(CACHELINE,
                   (size_t)2 * P * S.rounds * sizeof(slot_t));
    memset(S.slots, 0, (size_t)2 * P * S.rounds * sizeof(slot_t));
    S.bslots = aligned_alloc(CACHELINE,
                   (size_t)2 * P * S.rounds * sizeof(slot_t));
    memset(S.bslots, 0, (size_t)2 * P * S.rounds * sizeof(slot_t));
    S.partial = aligned_alloc(CACHELINE, (size_t)P * CACHELINE);
    memset(S.partial, 0, (size_t)P * CACHELINE);
    atomic_store(&S.count, 0);
    atomic_store(&S.release, 0);
    omp_init_lock(&S.mutex);

    double t0 = 0.0, t1 = 0.0;
    volatile double sink = 0.0;

    #pragma omp parallel num_threads(P)
    {
        const int me = omp_get_thread_num();
        double acc = 0.0;

        for (long i = 1; i <= 1000; i++)                   /* warm up     */
            if (mode == SYNC_TREE)        acc += AggregateMax(&S, (double)me, me, (uint64_t)i);
            else if (mode == SYNC_LINEAR) acc += AggregateMaxLinear(&S, (double)me, me, (uint64_t)i);
            else                          acc += AggregateMaxOmp(&S, (double)me, me);

        #pragma omp barrier
        if (me == 0) t0 = now_sec();

        for (long i = 1001; i <= 1000 + iters; i++)
            if (mode == SYNC_TREE)        acc += AggregateMax(&S, (double)me, me, (uint64_t)i);
            else if (mode == SYNC_LINEAR) acc += AggregateMaxLinear(&S, (double)me, me, (uint64_t)i);
            else                          acc += AggregateMaxOmp(&S, (double)me, me);

        if (me == 0) t1 = now_sec();
        #pragma omp atomic
        sink += acc;
    }

    omp_destroy_lock(&S.mutex);
    free(S.slots); free(S.bslots); free(S.partial);
    (void)sink;
    return 1e9 * (t1 - t0) / (double)iters;
}

/* ------------------------------------------------------------------- */
static void usage(const char *p)
{
    printf("usage: %s [-n N] [-p P] [-t TOL] [-i MAXIT] [-s tree|linear|omp]\n"
           "          [-c EVERY] [-r REPS] [-L] [-b] [-I ITERS] [-q] [-h]\n", p);
}

int main(int argc, char **argv)
{
    int        N      = 1024;
    int        P      = omp_get_max_threads();
    double     tol    = 1e-2;
    long       maxit  = 1000000;
    syncmode_t mode   = SYNC_TREE;
    int        every  = 1;
    int        reps   = 1;
    bool       bench  = false;
    bool       terse  = false;
    bool       balance = false;
    long       biters = 200000;
    const double hot  = 100.0;

    int o;
    while ((o = getopt(argc, argv, "n:p:t:i:s:c:r:I:bqhL")) != -1) {
        switch (o) {
            case 'n': N = atoi(optarg); break;
            case 'p': P = atoi(optarg); break;
            case 't': tol = atof(optarg); break;
            case 'i': maxit = atol(optarg); break;
            case 'c': every = atoi(optarg); break;
            case 'r': reps = atoi(optarg); break;
            case 'b': bench = true; break;
            case 'L': balance = true; break;
            case 'I': biters = atol(optarg); break;
            case 'q': terse = true; break;
            case 's':
                if      (!strcmp(optarg, "tree"))   mode = SYNC_TREE;
                else if (!strcmp(optarg, "linear")) mode = SYNC_LINEAR;
                else if (!strcmp(optarg, "omp"))    mode = SYNC_OMP;
                else { fprintf(stderr, "unknown sync mode %s\n", optarg); return 2; }
                break;
            case 'h': default: usage(argv[0]); return (o == 'h') ? 0 : 2;
        }
    }
    if (P < 1) P = 1;
    if (N < P) N = P;
    if (every < 1) every = 1;
    if (reps < 1) reps = 1;

    if (bench) {
        if (biters < 100) biters = 100;
        double ns = barrier_cost_ns(P, mode, biters);
        if (terse) printf("%d %s %.1f\n", P, syncname(mode), ns);
        else printf("barrier+aggregate cost: P=%d  mode=%-6s  %8.1f ns/op  "
                    "(%d rounds)\n", P, syncname(mode), ns, ceil_log2i(P));
        return 0;
    }

    result_t best;
    memset(&best, 0, sizeof best);
    best.seconds = 1e300;
    for (int i = 0; i < reps; i++) {
        result_t r = solve(N, P, tol, maxit, mode, every, hot, balance);
        if (r.seconds < best.seconds) best = r;
    }

    /* 5 flops per interior point per sweep (4 adds, 1 multiply) +
     * 3 for the change and the running maximum                      */
    const double pts    = (double)N * (double)N * (double)best.sweeps;
    const double gflops = 8.0 * pts / best.seconds / 1e9;
    const double gbs    = 16.0 * pts / best.seconds / 1e9;

    if (terse) {
        printf("%d %d %s %.6f %ld %.6g %.6g %.3f %.3f\n",
               N, P, syncname(mode), best.seconds, best.sweeps,
               best.final_change, best.checksum, gflops, gbs);
        return 0;
    }

    char buf[32];
    printf("=======================================================\n");
    printf(" PARALLEL JACOBI RELAXATION on real hardware (OpenMP)\n");
    printf("=======================================================\n");
    printf(" Threads P            = %d\n", P);
    printf(" Grid (interior)      = %d x %d  (%s points)\n", N, N,
           (N * (double)N >= 1e6)
             ? (snprintf(buf, sizeof buf, "%.1f M", N*(double)N/1e6), buf)
             : (snprintf(buf, sizeof buf, "%.0f", N*(double)N), buf));
    printf(" Rows per thread      = %.1f\n", (double)N / P);
    printf(" Convergence tolerance= %g   (tested every %d sweep%s)\n",
           tol, every, every == 1 ? "" : "s");
    printf(" Synchronisation      = %s", syncname(mode));
    if (mode == SYNC_TREE)
        printf("   (dissemination, %d rounds, lock free)", ceil_log2i(P));
    else if (mode == SYNC_LINEAR)
        printf("   (centralised counter, %d serialised locks)", P);
    else
        printf("   (omp barrier + shared array)");
    printf("\n");
    printf(" Barriers per sweep   = 1\n");
    printf(" Load balancing       = %s\n\n",
           balance ? "adaptive (rows proportional to measured core speed)"
                   : "static equal row blocks");

    printf(" %s\n", best.sweeps >= maxit ? "SWEEP LIMIT REACHED" : "CONVERGED");
    printf(" Sweeps               = %ld\n", best.sweeps);
    printf(" Final max change     = %.6g\n", best.final_change);
    printf(" Solution checksum    = %.10g\n", best.checksum);
    printf(" Centre value         = %.6f\n", best.centre);
    if (N <= 64)
        printf("   the exact steady state centre value is 25, so this is the\n"
               "   physically correct answer to the tolerance requested\n\n");
    else
        printf("   NOTE: on a grid this large the max-change test halts long\n"
               "   before the plate reaches its global steady state, so the\n"
               "   centre has not warmed up yet.  That is a property of the\n"
               "   stopping rule, not an error: raise -n gently or tighten -t\n"
               "   to follow the solution further.\n\n");

    printf("----------------- PERFORMANCE SUMMARY -----------------\n");
    printf(" Wall clock time               = %.4f s\n", best.seconds);
    printf(" Time per sweep                = %.4f ms\n",
           1e3 * best.seconds / (double)best.sweeps);
    printf(" Point updates per second      = %.3f G/s\n", pts / best.seconds / 1e9);
    printf(" Arithmetic rate               = %.3f GFLOP/s\n", gflops);
    printf(" Grid data touched             = %.3f GB/s\n", gbs);
    printf("   (16 bytes per point update - the least a sweep must move;\n    a grid that fits in L3 is served by cache, not main memory)\n");
    printf(" Barriers executed             = %ld\n",
           best.sweeps / every + (every > 1 ? best.sweeps - best.sweeps / every : 0));
    printf("=======================================================\n");
    return 0;
}
