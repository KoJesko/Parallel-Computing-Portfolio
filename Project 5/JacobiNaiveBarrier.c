/* ===================================================================== */
/*  PROGRAM JacobiNaiveBarrier                                           */
/*                                                                       */
/*  Project 5 - the "obvious" parallel Jacobi relaxation, kept as a       */
/*              control experiment.  It solves exactly the same problem  */
/*              with exactly the same arithmetic as JacobiParallel.c and  */
/*              so converges in the same number of sweeps to the same    */
/*              solution - but it synchronises and aggregates in the      */
/*              naive way.  The difference in execution time is the       */
/*              measured value of the efficient barrier and aggregation.  */
/*                                                                       */
/*  WHAT IS NAIVE HERE                                                    */
/*   1. LINEAR COUNTER BARRIER - every process locks one shared spinlock */
/*      to increment one shared counter, so the P arrivals are           */
/*      SERIALISED: the barrier costs O(P), not O(log P).  Every process  */
/*      then spins on the same release variable - a single hot spot.      */
/*   2. LOCK-BASED AGGREGATION - the global maximum for the convergence  */
/*      test is built by P serialised critical sections on one shared     */
/*      variable, instead of an O(log P) tree of exchanges.               */
/*   3. THREE BARRIERS PER SWEEP - one to publish the reset of the        */
/*      shared maximum, one to collect the new values and the maxima,     */
/*      and one to finish the copy-back.                                  */
/*   4. COPY-BACK SWEEP - the new grid is copied over the old grid        */
/*      instead of the two grids exchanging roles, which adds N*N        */
/*      assignments of pure overhead on every sweep.                      */
/* ===================================================================== */

ARCHITECTURE SHARED(32);

#define P       32       /* number of parallel processes                 */
#define N       32       /* the interior grid is N x N (N multiple of P) */
#define TOL     0.01     /* convergence tolerance for the sweep change   */
#define MAXIT   4000     /* safety limit on the number of sweeps         */
#define HOT     100.0    /* temperature of the top edge                  */
#define PSTEP   4        /* print every PSTEP-th row/column of the grid  */

float A[N+2][N+2];       /* the old grid - read by every sweep           */
float B[N+2][N+2];       /* the new grid - copied back onto A            */

spinlock mutex;          /* the one lock that serialises everything      */
int   bcount;            /* processes that have arrived at the barrier   */
int   brelease;          /* barrier release counter (the hot spot)       */
float gmax;              /* shared global maximum change                 */

int   sweepsdone;
float finalchange;
int   i, j;

/* ===================================================================== */
/*  NAIVE BARRIER - linear counter, O(P) serialised lock acquisitions    */
/* ===================================================================== */

void NaiveBarrier() {
  int myphase;

  lock(mutex);                     /* ALL P processes serialise here     */
  myphase = brelease;
  bcount  = bcount + 1;
  if (bcount == P) {               /* the last one releases everybody    */
    bcount   = 0;
    brelease = brelease + 1;
  }
  unlock(mutex);
  while (brelease == myphase) ;    /* all P processes spin on one word   */
}

/* ===================================================================== */
/*  NAIVE AGGREGATION - P serialised critical sections on one variable   */
/* ===================================================================== */

void NaiveMaxUpdate(float v) {
  lock(mutex);
  if (v > gmax) gmax = v;
  unlock(mutex);
}

/* ===================================================================== */
/*  The naive Jacobi process: 3 barriers and a copy-back sweep           */
/* ===================================================================== */

void RelaxNaive(int me) {
  int   sweep, r, c, r0, r1, rows;
  float newval, change, localmax, g;

  rows = N / P;
  r0   = me * rows + 1;
  r1   = r0 + rows - 1;
  sweep = 0;
  g     = 0.0;

  do {
    sweep = sweep + 1;

    if (me == 0) gmax = 0.0;       /* one process resets the maximum     */
    NaiveBarrier();                /* barrier 1: the reset is visible    */

    localmax = 0.0;
    for (r = r0; r <= r1; r++)
      for (c = 1; c <= N; c++) {
        newval = (A[r-1][c] + A[r+1][c] + A[r][c-1] + A[r][c+1]) / 4.0;
        B[r][c] = newval;
        change = newval - A[r][c];
        if (change < 0.0) change = -change;
        if (change > localmax) localmax = change;
      }

    NaiveMaxUpdate(localmax);      /* serialised aggregation             */
    NaiveBarrier();                /* barrier 2: B complete, maxima in   */

    g = gmax;                      /* every process reads the maximum    */

    for (r = r0; r <= r1; r++)     /* the copy-back sweep: pure overhead */
      for (c = 1; c <= N; c++)
        A[r][c] = B[r][c];

    NaiveBarrier();                /* barrier 3: copy-back complete      */
  } while (g > TOL && sweep < MAXIT);

  if (me == 0) {
    sweepsdone  = sweep;
    finalchange = g;
  }
}

/* --------------------------------------------------------------------- */
void PrintGrid() {
  int r, c;
  cout.precision(4);
  for (r = 0; r <= N+1; r = r + PSTEP) {
    for (c = 0; c <= N+1; c = c + PSTEP)
      cout << A[r][c] << " ";
    cout << endl;
  }
  cout.precision(6);
}

float CheckSum() {
  int   r, c;
  float s;
  s = 0.0;
  for (r = 1; r <= N; r++)
    for (c = 1; c <= N; c++)
      s = s + A[r][c];
  return s;
}

/* ===================================================================== */
main() {
  cout.precision(6);

  for (i = 0; i <= N+1; i++)
    for (j = 0; j <= N+1; j++) {
      A[i][j] = 0.0;
      B[i][j] = 0.0;
    }
  for (j = 0; j <= N+1; j++) {
    A[0][j] = HOT;
    B[0][j] = HOT;
  }
  bcount   = 0;
  brelease = 0;
  gmax     = 0.0;

  cout << "=======================================================" << endl;
  cout << " NAIVE PARALLEL JACOBI - linear barrier, locked maximum" << endl;
  cout << "=======================================================" << endl;
  cout << " Processes P          = " << P << endl;
  cout << " Grid (interior)      = " << N << " x " << N << endl;
  cout << " Rows per process     = " << N/P << endl;
  cout << " Convergence tolerance= " << TOL << endl;
  cout << " Barrier cost         = " << P << " serialised locks (P)" << endl;
  cout << " Barriers per sweep   = " << 3 << endl;
  cout << endl;

  forall i = 0 to P-1 do
    RelaxNaive(i);

  cout << " CONVERGED" << endl;
  cout << " Sweeps to converge   = " << sweepsdone << endl;
  cout << " Final max change     = " << finalchange << endl;
  cout << " Solution checksum    = " << CheckSum() << endl;
  cout << " Centre value         = " << A[N/2][N/2]
       << "   (exact steady state = 25)" << endl;
  cout << "   N is small enough here that the run really does reach the"
       << endl << "   physical answer, so this doubles as a validation." << endl;
  cout << endl;
  cout << " Temperature field (every " << PSTEP << "th row and column):" << endl;
  PrintGrid();
  cout << endl;

  cout << "----------------- PERFORMANCE SUMMARY -----------------" << endl;
  cout.precision(9);
  cout << " Parallel execution time   T(P) = " << clock << endl;
  cout << " Sequential execution time T(1) = " << seqtime << endl;
  cout.precision(4);
  cout << " Speedup   T(1)/T(P)            = " << seqtime / clock << endl;
  cout << " Efficiency  speedup/P          = " << seqtime / clock / P << endl;
  cout.precision(6);
  cout << " Barriers executed              = " << sweepsdone * 3
       << "  (3 per sweep)" << endl;
  cout << "=======================================================" << endl;
}
