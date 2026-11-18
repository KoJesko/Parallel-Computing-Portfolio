/* SpatialFilterMPI.c - Project 8, real MPI version (MS-MPI on Windows)
 *
 * Same algorithm as SpatialFilterHypercube.c (C* simulator), but on real
 * hardware.  The P MPI ranks (P a power of two) are treated as the nodes of
 * a d-dimensional hypercube, d = log2(P):
 *     mpiexec -n 8   -> 3-D hypercube, one rank per core (all 8 cores)
 *     mpiexec -n 128 -> the full 7-D hypercube (ranks share the 8 cores)
 *
 *   1. strip s of the image belongs to rank Gray(s), so neighbouring strips
 *      sit on hypercube neighbours (ranks differing in exactly one bit)
 *   2. scatter = binomial tree along the hypercube dimensions (d steps,
 *      every message travels over one hypercube link)
 *   3. each pass: halo exchange with the two Gray-code neighbours
 *      (MPI_Sendrecv, one hop), then the 3x3 mask on the local strip
 *   4. gather = the reverse binomial tree
 *
 * usage:
 *   mpiexec -n 8 SpatialFilterMPI <input.txt> [passes] [out.bmp]
 *   mpiexec -n 8 SpatialFilterMPI -gen <rows> <cols> <filter> [passes] [out.bmp]
 *        filter = smooth | box | sharpen | edge | emboss
 */
#include <mpi.h>
#include <windows.h>
#include "image_io.h"

static int rank, P, dimHC;

static int gray(int s)    { return s ^ (s >> 1); }
static int invgray(int g) { int s = g; for (int t = g >> 1; t; t >>= 1) s ^= t; return s; }

/* rows of strip s when R rows are split over P strips */
static int strip_first(int s, int R) { return (int)((long long)s * R / P); }
static int strip_rows(int s, int R)  { return strip_first(s + 1, R) - strip_first(s, R); }

static double cpu_seconds(void) {   /* user + kernel time of this process */
    FILETIME c, e, k, u;
    GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
    ULARGE_INTEGER K = { { k.dwLowDateTime, k.dwHighDateTime } };
    ULARGE_INTEGER U = { { u.dwLowDateTime, u.dwHighDateTime } };
    return (double)(K.QuadPart + U.QuadPart) * 1e-7;
}

/* pixels held by ranks [lo, lo+count) in rank order */
static size_t block_pixels(int lo, int count, int R, int C) {
    size_t n = 0;
    for (int q = lo; q < lo + count; q++) n += (size_t)strip_rows(invgray(q), R) * C;
    return n;
}

/* one 3x3 pass over my strip.  in has (myrows+2) rows incl. halos. */
static void filter_strip(const unsigned char *in, unsigned char *out, int myrows, int C,
                         const int *k, int divisor, int offset) {
    for (int r = 0; r < myrows; r++) {
        const unsigned char *a = in + (size_t)r * C, *b = a + C, *d = b + C;
        unsigned char *o = out + (size_t)r * C;
        /* left/right border columns (clamped) */
        for (int pass = 0; pass < 2; pass++) {
            int x = pass ? C - 1 : 0;
            int xl = x > 0 ? x - 1 : 0, xr = x < C - 1 ? x + 1 : C - 1;
            int s = k[0]*a[xl] + k[1]*a[x] + k[2]*a[xr] + k[3]*b[xl] + k[4]*b[x] + k[5]*b[xr]
                  + k[6]*d[xl] + k[7]*d[x] + k[8]*d[xr];
            int v = s / divisor + offset;
            o[x] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
        /* interior: branch-free, auto-vectorised */
        for (int x = 1; x < C - 1; x++) {
            int s = k[0]*a[x-1] + k[1]*a[x] + k[2]*a[x+1] + k[3]*b[x-1] + k[4]*b[x] + k[5]*b[x+1]
                  + k[6]*d[x-1] + k[7]*d[x] + k[8]*d[x+1];
            int v = s / divisor + offset;
            v = v < 0 ? 0 : v;
            o[x] = (unsigned char)(v > 255 ? 255 : v);
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    for (dimHC = 0; (1 << dimHC) < P; dimHC++) {}
    if ((1 << dimHC) != P) {
        if (rank == 0) fprintf(stderr, "number of ranks must be a power of two (hypercube)\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double cpu0 = cpu_seconds();
    double t0 = MPI_Wtime();

    /* ---------- rank 0: obtain the image ---------- */
    Image im = { 0 };
    int passes = 1, ai = 1;
    const char *outBmp = NULL;
    char fname[64] = "file";
    if (rank == 0) {
        if (argc >= 5 && strcmp(argv[1], "-gen") == 0) {
            const FilterDef *f = find_filter(argv[4]);
            if (!f) { fprintf(stderr, "unknown filter %s\n", argv[4]); MPI_Abort(MPI_COMM_WORLD, 1); }
            gen_image(&im, atoi(argv[2]), atoi(argv[3]), 0);
            set_filter(&im, f);
            snprintf(fname, sizeof fname, "%s (generated)", f->name);
            ai = 5;
        } else if (argc >= 2) {
            if (!read_image(argv[1], &im)) MPI_Abort(MPI_COMM_WORLD, 1);
            snprintf(fname, sizeof fname, "%.60s", argv[1]);
            ai = 2;
        } else {
            fprintf(stderr, "usage: SpatialFilterMPI <input.txt>|-gen R C filter [passes] [out.bmp]\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (argc > ai) passes = atoi(argv[ai]);
        if (argc > ai + 1) outBmp = argv[ai + 1];
        if (passes < 1) passes = 1;
    }
    double tRead = MPI_Wtime();

    /* ---------- header broadcast along hypercube dimensions ---------- */
    int hdr[14];
    if (rank == 0) {
        hdr[0] = im.rows; hdr[1] = im.cols;
        memcpy(hdr + 2, im.kern, 9 * sizeof(int));
        hdr[11] = im.divisor; hdr[12] = im.offset; hdr[13] = passes;
    }
    for (int d = dimHC - 1; d >= 0; d--) {
        int bit = 1 << d;
        if ((rank & (2 * bit - 1)) == 0)        MPI_Send(hdr, 14, MPI_INT, rank + bit, 0, MPI_COMM_WORLD);
        else if ((rank & (2 * bit - 1)) == bit) MPI_Recv(hdr, 14, MPI_INT, rank - bit, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    int R = hdr[0], C = hdr[1], divisor = hdr[11], offset = hdr[12];
    int kern[9]; memcpy(kern, hdr + 2, sizeof kern);
    passes = hdr[13];
    if (R < P) { if (rank == 0) fprintf(stderr, "need at least one row per rank\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    int strip = invgray(rank);
    int myrows = strip_rows(strip, R);
    int up = strip > 0 ? gray(strip - 1) : MPI_PROC_NULL;
    int down = strip < P - 1 ? gray(strip + 1) : MPI_PROC_NULL;

    /* subtree of this rank in the binomial tree = ranks [rank, rank+span) */
    int span = rank ? (rank & -rank) : P;
    size_t treeN = block_pixels(rank, span, R, C);
    unsigned char *tree = (unsigned char *)malloc(treeN ? treeN : 1);

    /* ---------- scatter: root permutes strips into rank order ---------- */
    if (rank == 0) {
        size_t off = 0;
        for (int q = 0; q < P; q++) {
            int s = invgray(q);
            size_t n = (size_t)strip_rows(s, R) * C;
            memcpy(tree + off, im.pix + (size_t)strip_first(s, R) * C, n);
            off += n;
        }
    }
    for (int d = dimHC - 1; d >= 0; d--) {
        int bit = 1 << d;
        if ((rank & (2 * bit - 1)) == 0) {
            size_t keep = block_pixels(rank, bit, R, C);
            size_t give = block_pixels(rank + bit, bit, R, C);
            MPI_Send(tree + keep, (int)give, MPI_UNSIGNED_CHAR, rank + bit, 1, MPI_COMM_WORLD);
        } else if ((rank & (2 * bit - 1)) == bit) {
            MPI_Recv(tree, (int)treeN, MPI_UNSIGNED_CHAR, rank - bit, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    double tScatter = MPI_Wtime();

    /* local strip with halo rows 0 and myrows+1 */
    unsigned char *cur = (unsigned char *)malloc((size_t)(myrows + 2) * C);
    unsigned char *nxt = (unsigned char *)malloc((size_t)(myrows + 2) * C);
    memcpy(cur + C, tree, (size_t)myrows * C);

    /* ---------- passes: halo exchange + filter ---------- */
    double tHalo = 0, tComp = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double cpuF0 = cpu_seconds(), wallF0 = MPI_Wtime();
    for (int p = 0; p < passes; p++) {
        double h0 = MPI_Wtime();
        MPI_Sendrecv(cur + C, C, MPI_UNSIGNED_CHAR, up, 3,
                     cur + (size_t)(myrows + 1) * C, C, MPI_UNSIGNED_CHAR, down, 3,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(cur + (size_t)myrows * C, C, MPI_UNSIGNED_CHAR, down, 4,
                     cur, C, MPI_UNSIGNED_CHAR, up, 4,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (up == MPI_PROC_NULL)   memcpy(cur, cur + C, C);
        if (down == MPI_PROC_NULL) memcpy(cur + (size_t)(myrows + 1) * C, cur + (size_t)myrows * C, C);
        double h1 = MPI_Wtime();
        filter_strip(cur, nxt + C, myrows, C, kern, divisor, offset);
        unsigned char *t = cur; cur = nxt; nxt = t;
        tHalo += h1 - h0;
        tComp += MPI_Wtime() - h1;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double tFilter = MPI_Wtime();
    double cpuF = cpu_seconds() - cpuF0, wallF = tFilter - wallF0;

    /* ---------- gather: reverse binomial tree ---------- */
    memcpy(tree, cur + C, (size_t)myrows * C);
    for (int d = 0; d < dimHC; d++) {
        int bit = 1 << d;
        if ((rank & (2 * bit - 1)) == bit) {
            MPI_Send(tree, (int)block_pixels(rank, bit, R, C), MPI_UNSIGNED_CHAR, rank - bit, 5, MPI_COMM_WORLD);
        } else if ((rank & (2 * bit - 1)) == 0) {
            size_t keep = block_pixels(rank, bit, R, C);
            MPI_Recv(tree + keep, (int)block_pixels(rank + bit, bit, R, C), MPI_UNSIGNED_CHAR,
                     rank + bit, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    double tGather = MPI_Wtime();

    /* ---------- statistics ---------- */
    double cpu = cpu_seconds() - cpu0;
    double cpuSum, cpuFSum, compSum, compMax, haloMax;
    MPI_Reduce(&cpu, &cpuSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&cpuF, &cpuFSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tComp, &compSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tComp, &compMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tHalo, &haloMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double wall = MPI_Wtime() - t0;
        /* un-permute into image order */
        unsigned char *result = (unsigned char *)malloc((size_t)R * C);
        size_t off = 0;
        for (int q = 0; q < P; q++) {
            int s = invgray(q);
            size_t n = (size_t)strip_rows(s, R) * C;
            memcpy(result + (size_t)strip_first(s, R) * C, tree + off, n);
            off += n;
        }
        SYSTEM_INFO si; GetSystemInfo(&si);
        int cores = (int)si.dwNumberOfProcessors;
        double parWall = tGather - tRead;
        double mpix = (double)R * C * passes / (tFilter - tScatter) / 1e6;

        printf("\n================ SPATIAL FILTER - MPI HYPERCUBE ================\n");
        printf(" input            : %s\n", fname);
        printf(" image            : %d x %d pixels, 3x3 mask, %d pass(es)\n", R, C, passes);
        printf(" MPI ranks        : %d  = %d-dimensional hypercube (Gray-code strips)\n", P, dimHC);
        printf(" logical CPUs     : %d\n", cores);
        printf("----------------------------------------------------------------\n");
        printf(" read/generate    : %9.4f s  (rank 0, sequential I/O)\n", tRead - t0);
        printf(" bcast + scatter  : %9.4f s  (%d hypercube steps)\n", tScatter - tRead, dimHC);
        printf(" filter passes    : %9.4f s  (compute %.4f s, halo %.4f s max/rank)\n",
               tFilter - tScatter, compMax, haloMax);
        printf(" gather           : %9.4f s  (%d hypercube steps)\n", tGather - tFilter, dimHC);
        printf(" parallel total   : %9.4f s\n", parWall);
        printf(" throughput       : %9.1f Mpixel/s (%.2f GFLOP/s, 17 ops/pixel)\n", mpix, mpix * 17 / 1000);
        printf(" load balance     : %9.1f %%  (avg / max compute time per rank)\n", 100 * compSum / P / compMax);
        printf("----------------------------------------------------------------\n");
        /* Windows CPU-time counters tick every ~15.6 ms: too coarse for tiny runs */
        if (wallF >= 0.25)
            printf(" CPU USAGE        : %6.0f %%  out of %d %%  during the parallel filter phase\n",
                   100 * cpuFSum / wallF, cores * 100);
        else
            printf(" CPU USAGE        :    n/a    filter phase too short to measure (use -gen\n"
                   "                              8192 8192 smooth 100 for a CPU-usage run)\n");
        if (wall >= 0.25)
            printf("                    %6.0f %%  out of %d %%  whole run incl. serial I/O\n",
                   100 * cpuSum / wall, cores * 100);
        printf("                    (sum of CPU time of all %d ranks / wall-clock time,\n"
               "                     %d cores x 100%% = %d%% maximum)\n", P, cores, cores * 100);
        printf("================================================================\n");

        if (R * (size_t)C <= 4096u * 4096u && passes == 1) {
            unsigned char *ref = (unsigned char *)malloc((size_t)R * C);
            double s0 = MPI_Wtime();
            filter_reference(&im, ref);
            double s1 = MPI_Wtime();
            size_t bad = 0;
            for (size_t i = 0; i < (size_t)R * C; i++) bad += ref[i] != result[i];
            printf(" verify vs sequential reference: %zu pixels differ -> %s\n", bad, bad ? "FAIL" : "PASS");
            printf(" (unoptimised sequential reference took %.4f s)\n", s1 - s0);
            free(ref);
        }
        printf(" result checksum  : %016llx\n", checksum(result, (size_t)R * C));
        if (outBmp) {
            write_bmp(outBmp, result, R, C);
            printf(" wrote %s\n", outBmp);
        }
        free(result);
    }
    free(tree); free(cur); free(nxt);
    MPI_Finalize();
    return 0;
}
