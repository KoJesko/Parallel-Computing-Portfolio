/*
 * Project 6 - Parallel Numerical Integration on a 2-D Mesh
 *
 * Computes  integral_0^2 sqrt(4 - t^2) dt  (= pi)  with the trapezoidal
 * rule, distributed over P = R x C processes arranged as a 2-D mesh
 * (non-periodic Cartesian topology).
 *
 * The shared-memory version (Fig. 5.8) adds each local sum into one global
 * variable inside a critical section.  On a multicomputer that becomes P
 * messages funneling into one node, i.e. O(P) steps.  Here every message
 * travels only between mesh neighbours, and the local sums are combined in
 * two pipelined sweeps that converge on the centre of the mesh:
 *
 *   1. Row phase:    in every row, the left half pushes partial sums
 *                    rightward and the right half pushes leftward, so each
 *                    row's total lands in the middle column after
 *                    ~C/2 neighbour steps.
 *   2. Column phase: the middle column does the same vertically, so the
 *                    grand total lands at the centre node after ~R/2 steps.
 *
 * Total communication: ~(R + C)/2 = O(sqrt P) neighbour hops, versus
 * (R - 1) + (C - 1) for reducing to a corner, and P - 1 for the naive
 * "everyone sends to node 0" approach.  The interval count n is broadcast
 * outward from the centre using the reverse of the same pattern.
 *
 * Usage:  mpirun -np P ./mesh_integrate [n] [rows cols]
 *         n defaults to 100000000; rows x cols defaults to MPI_Dims_create.
 */
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define A 0.0
#define B 2.0
#define TAG_BCAST 1
#define TAG_SUM   2
#define TAG_RES   3

float f(float t)
{
    return (sqrt(4 - t * t));
}

/* Trapezoidal rule over intervals [first, last) of the global n-interval grid. */
static double local_trapezoid(long first, long last, long n)
{
    double h = (B - A) / n;
    double sum = 0.0;
    long i;

    if (first >= last)
        return 0.0;
    sum = 0.5 * (f((float)(A + first * h)) + f((float)(A + last * h)));
    for (i = first + 1; i < last; i++)
        sum += f((float)(A + i * h));
    return sum * h;
}

/*
 * Combine values along one line of the mesh (a row or a column) so the total
 * ends up at position mid.  pos is this node's position on the line, lo/hi are
 * its neighbours (MPI_PROC_NULL at the mesh edge).  Nodes left of mid forward
 * toward hi, nodes right of mid forward toward lo; both halves run at once.
 */
static double line_reduce(double val, int pos, int mid, int lo, int hi,
                          MPI_Comm comm)
{
    double in;

    if (pos < mid) {
        if (lo != MPI_PROC_NULL) {
            MPI_Recv(&in, 1, MPI_DOUBLE, lo, TAG_SUM, comm, MPI_STATUS_IGNORE);
            val += in;
        }
        MPI_Send(&val, 1, MPI_DOUBLE, hi, TAG_SUM, comm);
    } else if (pos > mid) {
        if (hi != MPI_PROC_NULL) {
            MPI_Recv(&in, 1, MPI_DOUBLE, hi, TAG_SUM, comm, MPI_STATUS_IGNORE);
            val += in;
        }
        MPI_Send(&val, 1, MPI_DOUBLE, lo, TAG_SUM, comm);
    } else {
        if (lo != MPI_PROC_NULL) {
            MPI_Recv(&in, 1, MPI_DOUBLE, lo, TAG_SUM, comm, MPI_STATUS_IGNORE);
            val += in;
        }
        if (hi != MPI_PROC_NULL) {
            MPI_Recv(&in, 1, MPI_DOUBLE, hi, TAG_SUM, comm, MPI_STATUS_IGNORE);
            val += in;
        }
    }
    return val;
}

/* Reverse of line_reduce: spread a value outward from position mid. */
static long line_bcast(long val, int pos, int mid, int lo, int hi,
                       MPI_Comm comm)
{
    if (pos < mid) {
        MPI_Recv(&val, 1, MPI_LONG, hi, TAG_BCAST, comm, MPI_STATUS_IGNORE);
        if (lo != MPI_PROC_NULL)
            MPI_Send(&val, 1, MPI_LONG, lo, TAG_BCAST, comm);
    } else if (pos > mid) {
        MPI_Recv(&val, 1, MPI_LONG, lo, TAG_BCAST, comm, MPI_STATUS_IGNORE);
        if (hi != MPI_PROC_NULL)
            MPI_Send(&val, 1, MPI_LONG, hi, TAG_BCAST, comm);
    } else {
        if (lo != MPI_PROC_NULL)
            MPI_Send(&val, 1, MPI_LONG, lo, TAG_BCAST, comm);
        if (hi != MPI_PROC_NULL)
            MPI_Send(&val, 1, MPI_LONG, hi, TAG_BCAST, comm);
    }
    return val;
}

int main(int argc, char *argv[])
{
    int world_size, rank, center_rank;
    int dims[2] = {0, 0}, periods[2] = {0, 0}, coords[2], center[2];
    int up, down, left, right;
    long n = 100000000L, first, last, chunk, extra;
    double t_start, t_comm, t_end, my_sum, row_sum, total;
    double t_compute, max_compute, max_total;
    MPI_Comm mesh;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc > 1)
        n = atol(argv[1]);
    if (argc > 3) {
        dims[0] = atoi(argv[2]);
        dims[1] = atoi(argv[3]);
    }
    MPI_Dims_create(world_size, 2, dims);
    if (dims[0] * dims[1] != world_size || n < 1) {
        fprintf(stderr, "invalid mesh %dx%d for %d processes or n=%ld\n",
                dims[0], dims[1], world_size, n);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Build the 2-D mesh; reorder lets MPI map it onto the hardware. */
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &mesh);
    MPI_Comm_rank(mesh, &rank);
    MPI_Cart_coords(mesh, rank, 2, coords);
    MPI_Cart_shift(mesh, 0, 1, &up, &down);
    MPI_Cart_shift(mesh, 1, 1, &left, &right);
    center[0] = (dims[0] - 1) / 2;
    center[1] = (dims[1] - 1) / 2;
    MPI_Cart_rank(mesh, center, &center_rank);

    MPI_Barrier(mesh);
    t_start = MPI_Wtime();

    /* Broadcast n from the centre: down the middle column, then along rows. */
    if (coords[1] == center[1])
        n = line_bcast(n, coords[0], center[0], up, down, mesh);
    n = line_bcast(n, coords[1], center[1], left, right, mesh);

    /* Block-partition the n intervals; the first `extra` ranks get one more. */
    chunk = n / world_size;
    extra = n % world_size;
    first = rank * chunk + (rank < extra ? rank : extra);
    last = first + chunk + (rank < extra ? 1 : 0);
    my_sum = local_trapezoid(first, last, n);

    t_comm = MPI_Wtime();

    /* Mesh reduction: rows into the middle column, then that column to centre. */
    row_sum = line_reduce(my_sum, coords[1], center[1], left, right, mesh);
    total = 0.0;
    if (coords[1] == center[1])
        total = line_reduce(row_sum, coords[0], center[0], up, down, mesh);

    t_end = MPI_Wtime();

    /* Report from rank 0 (outside the timed region). */
    t_compute = t_comm - t_start;
    t_end -= t_start;
    MPI_Reduce(&t_compute, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, mesh);
    MPI_Reduce(&t_end, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, mesh);
    if (rank == center_rank && rank != 0)
        MPI_Send(&total, 1, MPI_DOUBLE, 0, TAG_RES, mesh);
    if (rank == 0) {
        if (center_rank != 0)
            MPI_Recv(&total, 1, MPI_DOUBLE, center_rank, TAG_RES, mesh,
                     MPI_STATUS_IGNORE);
        printf("Mesh %d x %d (%d procs), n = %ld intervals\n",
               dims[0], dims[1], world_size, n);
        printf("  result   = %.12f\n", total);
        printf("  error    = %.3e\n", fabs(total - M_PI));
        printf("  compute  = %.6f s\n", max_compute);
        printf("  total    = %.6f s\n", max_total);
        printf("  reduce   = %d neighbour steps (vs %d for naive gather)\n",
               dims[1] / 2 + dims[0] / 2, world_size - 1);
        printf("CSV,%d,%d,%d,%ld,%.12f,%.6f,%.6f\n", world_size, dims[0],
               dims[1], n, total, max_compute, max_total);
    }

    MPI_Comm_free(&mesh);
    MPI_Finalize();
    return 0;
}
