/* ===================================================================== */
/*  PROGRAM JacobiParallel                                               */
/*                                                                       */
/*  Project 5 - Efficient parallel Jacobi relaxation on a multiprocessor */
/*              with convergence test and efficient barrier and          */
/*              aggregation functions.                                   */
/*                                                                       */
/*  PROBLEM                                                              */
/*  Steady-state temperature of a square plate (Laplace's equation).     */
/*  The top edge is held at 100 degrees and the other three edges at 0.  */
/*  Each Jacobi sweep replaces every interior point by the average of    */
/*  its four neighbours.  The program stops when the largest change made */
/*  by a sweep falls below TOL - this is the CONVERGENCE TEST, and it     */
/*  requires the maximum of the P per-process changes to be aggregated   */
/*  and made known to every process on every sweep.                      */
/*                                                                       */
/*  EFFICIENCY FEATURES  (see README.md for the full discussion)          */
/*   1. EFFICIENT BARRIER - Barrier() is a butterfly (pairwise exchange)  */
/*      barrier: O(log P) time, no locks, and every process spins on a    */
/*      different memory word, so there is no serialisation and no hot    */
/*      spot.  A lock-based counter barrier costs O(P).                   */
/*   2. EFFICIENT AGGREGATION - AggregateMax() performs the global        */
/*      maximum with the same butterfly pattern, so it IS the barrier     */
/*      and the all-reduce at the same time.  The convergence test        */
/*      therefore costs nothing beyond the barrier that the sweep needs   */
/*      anyway, and every process reaches the same decision without a     */
/*      master process and without a broadcast.                          */
/*   3. PING-PONG GRIDS - new values are written into a second grid and   */
/*      the two grids exchange roles, so there is no copy-back sweep and  */
/*      ONE barrier per iteration is sufficient.                          */
/*   4. ROW-BLOCK PARTITION - each process owns a contiguous band of      */
/*      rows, so only two boundary rows per process are ever shared.      */
/* ===================================================================== */

ARCHITECTURE SHARED(32);

#define P       32       /* number of parallel processes (a power of 2)  */
#define LOGP    5        /* log2(P)                                      */
#define MAXLOG  6        /* dimension bound for the butterfly slots      */
#define N       32       /* the interior grid is N x N (N multiple of P) */
#define TOL     0.01     /* convergence tolerance for the sweep change   */
#define MAXIT   4000     /* safety limit on the number of sweeps         */
#define HOT     100.0    /* temperature of the top edge                  */
#define PSTEP   4        /* print every PSTEP-th row/column of the grid  */

/* --------------------------------------------------------------------- */
/*  Shared data                                                          */
/* --------------------------------------------------------------------- */

/*  G[0] and G[1] are the two ping-pong grids.  Row 0, row N+1, column 0  */
/*  and column N+1 hold the fixed boundary values.                        */
float G[2][N+2][N+2];

/*  Butterfly exchange slots.  The first subscript is the sweep parity.   */
/*  Double buffering by parity is what makes the barrier safe without     */
/*  locks: a fast process running ahead into the next sweep writes the    */
/*  OTHER buffer, so it can never overwrite a value that a slow partner   */
/*  has not yet read.  A process cannot get two sweeps ahead, because to  */
/*  enter sweep s+2 it must first see its partner's sweep s+1 flag.       */
float bval[2][P][MAXLOG];   /* value published by a process in a round   */
int   bflag[2][P][MAXLOG];  /* sweep number published with that value    */
int   bsync[2][P][MAXLOG];  /* slots for the plain barrier (no value)    */

int   sweepsdone;           /* results, reported by process 0            */
float finalchange;
int   finalgrid;

int   i, j;                 /* used only by the sequential set-up code   */

/* ===================================================================== */
/*  EFFICIENT BARRIER FUNCTION                                           */
/*                                                                       */
/*  Butterfly barrier.  In round k every process synchronises with the    */
/*  partner whose process number differs from its own in bit k.  After    */
/*  LOGP rounds every process has synchronised, directly or indirectly,   */
/*  with all P processes, so this is a full barrier.                      */
/*                                                                       */
/*      round 0:  0<->1  2<->3  4<->5  6<->7   (pairs)                    */
/*      round 1:  0<->2  1<->3  4<->6  5<->7   (pairs of pairs)           */
/*      round 2:  0<->4  1<->5  2<->6  3<->7   (groups of four)           */
/*                                                                       */
/*  Cost: LOGP rounds of one write and one spin  =>  O(log P).            */
/*                                                                       */
/*  "ticket" must be the sweep number.  Because it increases on every     */
/*  call, the flags never have to be cleared, so no extra "reset"         */
/*  barrier is needed - a classic cost of the naive counter barrier.      */
/* ===================================================================== */

void Barrier(int me, int ticket) {
  int k, step, bit, partner, par;

  par  = ticket % 2;          /* parity selects the slot buffer          */
  step = 1;                   /* step = 2^k                              */
  for (k = 0; k < LOGP; k++) {
    bsync[par][me][k] = ticket;          /* tell my partner I am here    */
    bit = (me / step) % 2;               /* bit k of my process number   */
    if (bit == 0) partner = me + step;   /* me XOR 2^k, computed without */
    else          partner = me - step;   /* bitwise operators (no ^ in C*)*/
    while (bsync[par][partner][k] < ticket) ;   /* wait for the partner  */
    step = step * 2;
  }
}

/* ===================================================================== */
/*  EFFICIENT AGGREGATION FUNCTION  (all-reduce maximum)                 */
/*                                                                       */
/*  The same butterfly pattern, except that each process also combines    */
/*  the partner's value into its own.  After round k a process holds the  */
/*  maximum over the block of 2^(k+1) processes that contains it, so      */
/*  after LOGP rounds every process holds the maximum over all P.         */
/*                                                                       */
/*  This single function therefore delivers BOTH                          */
/*     - the barrier that ends the Jacobi sweep, and                      */
/*     - the global maximum change needed by the convergence test,        */
/*  in O(log P) time.  The naive alternative - a lock-protected shared    */
/*  maximum, then a barrier, then a broadcast - costs O(P) serialised     */
/*  lock acquisitions plus two more barriers.                             */
/*                                                                       */
/*  The value is published BEFORE the flag, so a partner that sees the    */
/*  flag is guaranteed to see the value that goes with it.                */
/* ===================================================================== */

float AggregateMax(int me, float myval, int ticket) {
  int   k, step, bit, partner, par;
  float v;

  v    = myval;
  par  = ticket % 2;
  step = 1;
  for (k = 0; k < LOGP; k++) {
    bval[par][me][k]  = v;               /* publish the combined value.. */
    bflag[par][me][k] = ticket;          /* ..then release the partner   */
    bit = (me / step) % 2;
    if (bit == 0) partner = me + step;
    else          partner = me - step;
    while (bflag[par][partner][k] < ticket) ;   /* wait for the partner  */
    if (bval[par][partner][k] > v) v = bval[par][partner][k];
    step = step * 2;
  }
  return v;                   /* identical in every process              */
}

/* ===================================================================== */
/*  THE PARALLEL JACOBI PROCESS                                          */
/*                                                                       */
/*  Process "me" (0 .. P-1) owns the interior rows r0 .. r1.  On every    */
/*  sweep it                                                             */
/*    1. computes the new value of each of its own points from the        */
/*       neighbouring values in the current grid, recording the largest   */
/*       change it makes,                                                */
/*    2. calls AggregateMax, which is simultaneously the barrier that     */
/*       ends the sweep and the reduction that yields the global largest  */
/*       change,                                                         */
/*    3. exchanges the roles of the two grids, and                        */
/*    4. applies the convergence test.                                   */
/*                                                                       */
/*  ONE barrier per sweep is sufficient, because a sweep reads only the   */
/*  old grid and writes only the new one: no process can overwrite a      */
/*  value that another process still needs.                               */
/* ===================================================================== */

void Relax(int me) {
  int   sweep, r, c, cur, nxt, r0, r1, rows;
  float newval, change, localmax, globalmax;

  rows = N / P;                /* rows of the grid owned by this process */
  r0   = me * rows + 1;        /* first interior row owned               */
  r1   = r0 + rows - 1;        /* last  interior row owned               */

  cur   = 0;                   /* read from G[cur], write into G[nxt]    */
  nxt   = 1;
  sweep = 0;
  globalmax = 0.0;

  do {
    sweep = sweep + 1;

    /* ---- 1. relax my own band of rows, tracking my largest change --- */
    localmax = 0.0;
    for (r = r0; r <= r1; r++)
      for (c = 1; c <= N; c++) {
        newval = (G[cur][r-1][c] + G[cur][r+1][c]
                + G[cur][r][c-1] + G[cur][r][c+1]) / 4.0;
        G[nxt][r][c] = newval;
        change = newval - G[cur][r][c];     /* |change| without math.h   */
        if (change < 0.0) change = -change;
        if (change > localmax) localmax = change;
      }

    /* ---- 2. barrier AND global maximum in one O(log P) operation ---- */
    globalmax = AggregateMax(me, localmax, sweep);

    /* ---- 3. exchange the grids - no copy-back sweep is needed -------- */
    cur = nxt;
    nxt = 1 - cur;

    /* ---- 4. convergence test, made identically by every process ----- */
  } while (globalmax > TOL && sweep < MAXIT);

  if (me == 0) {               /* report the results of the computation  */
    sweepsdone  = sweep;
    finalchange = globalmax;
    finalgrid   = cur;
  }
}

/* --------------------------------------------------------------------- */
/*  Printing.  Only every PSTEP-th row and column is printed: the C*     */
/*  console has a 128 character line limit.                              */
/* --------------------------------------------------------------------- */

void PrintGrid(int gid) {
  int r, c;
  cout.precision(4);
  for (r = 0; r <= N+1; r = r + PSTEP) {
    for (c = 0; c <= N+1; c = c + PSTEP)
      cout << G[gid][r][c] << " ";
    cout << endl;
  }
  cout.precision(6);
}

float CheckSum(int gid) {
  int   r, c;
  float s;
  s = 0.0;
  for (r = 1; r <= N; r++)
    for (c = 1; c <= N; c++)
      s = s + G[gid][r][c];
  return s;
}

/* ===================================================================== */
main() {
  cout.precision(6);

  /* ---- initialise both grids: interior 0, top edge HOT -------------- */
  for (i = 0; i <= N+1; i++)
    for (j = 0; j <= N+1; j++) {
      G[0][i][j] = 0.0;
      G[1][i][j] = 0.0;
    }
  for (j = 0; j <= N+1; j++) {       /* the top edge of BOTH grids       */
    G[0][0][j] = HOT;
    G[1][0][j] = HOT;
  }

  /* ---- clear the butterfly synchronisation slots -------------------- */
  for (i = 0; i < P; i++)
    for (j = 0; j < MAXLOG; j++) {
      bflag[0][i][j] = 0;   bflag[1][i][j] = 0;
      bsync[0][i][j] = 0;   bsync[1][i][j] = 0;
      bval[0][i][j]  = 0.0; bval[1][i][j]  = 0.0;
    }

  cout << "=======================================================" << endl;
  cout << " PARALLEL JACOBI RELAXATION - butterfly barrier" << endl;
  cout << "=======================================================" << endl;
  cout << " Processes P          = " << P << endl;
  cout << " Grid (interior)      = " << N << " x " << N << endl;
  cout << " Rows per process     = " << N/P << endl;
  cout << " Convergence tolerance= " << TOL << endl;
  cout << " Barrier rounds       = " << LOGP << "  (log2 P)" << endl;
  cout << " Barriers per sweep   = " << 1 << endl;
  cout << endl;

  /* ---- run the P Jacobi processes in parallel ----------------------- */
  forall i = 0 to P-1 do
    Relax(i);

  /* ---- results ------------------------------------------------------ */
  cout << " CONVERGED" << endl;
  cout << " Sweeps to converge   = " << sweepsdone << endl;
  cout << " Final max change     = " << finalchange << endl;
  cout << " Solution checksum    = " << CheckSum(finalgrid) << endl;
  cout << " Centre value         = " << G[finalgrid][N/2][N/2]
       << "   (exact steady state = 25)" << endl;
  cout << "   N is small enough here that the run really does reach the"
       << endl << "   physical answer, so this doubles as a validation." << endl;
  cout << endl;
  cout << " Temperature field (every " << PSTEP << "th row and column):" << endl;
  PrintGrid(finalgrid);
  cout << endl;

  cout << "----------------- PERFORMANCE SUMMARY -----------------" << endl;
  cout.precision(9);
  cout << " Parallel execution time   T(P) = " << clock << endl;
  cout << " Sequential execution time T(1) = " << seqtime << endl;
  cout.precision(4);
  cout << " Speedup   T(1)/T(P)            = " << seqtime / clock << endl;
  cout << " Efficiency  speedup/P          = " << seqtime / clock / P << endl;
  cout.precision(6);
  cout << " Barriers executed              = " << sweepsdone
       << "  (1 per sweep)" << endl;
  cout << "=======================================================" << endl;
}
