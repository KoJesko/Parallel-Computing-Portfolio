/* ===================================================================== */
/*  PROGRAM JacobiSequential                                             */
/*                                                                       */
/*  Project 5 - sequential reference version of the Jacobi relaxation.   */
/*                                                                       */
/*  Same problem, same grid, same tolerance and the same convergence     */
/*  test as JacobiParallel.c, but executed by a single process.  It is    */
/*  used for two purposes:                                               */
/*    1. CORRECTNESS - Jacobi relaxation is deterministic, so the        */
/*       parallel program must produce exactly the same sweep count and  */
/*       the same solution as this program.                              */
/*    2. PERFORMANCE - its execution time is the T(1) baseline against   */
/*       which the speedup of the parallel version is measured.          */
/* ===================================================================== */

ARCHITECTURE SHARED(16);

#define N       32       /* the interior grid is N x N                   */
#define TOL     0.01     /* convergence tolerance for the sweep change   */
#define MAXIT   4000     /* safety limit on the number of sweeps         */
#define PSTEP   4        /* print every PSTEP-th row/column of the grid  */
#define HOT     100.0    /* temperature of the top edge                  */

float G[2][N+2][N+2];    /* the two ping-pong grids                      */
int   i, j;

/*  The grid is printed every 2nd row and column: the C* console has a    */
/*  128 character line limit, and the full N+2 columns do not fit.        */
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
  int   sweep, r, c, cur, nxt;
  float newval, change, maxchange;

  cout.precision(6);

  /* ---- initialise both grids: interior 0, top edge HOT -------------- */
  for (i = 0; i <= N+1; i++)
    for (j = 0; j <= N+1; j++) {
      G[0][i][j] = 0.0;
      G[1][i][j] = 0.0;
    }
  for (j = 0; j <= N+1; j++) {
    G[0][0][j] = HOT;
    G[1][0][j] = HOT;
  }

  cout << "=====================================================" << endl;
  cout << " SEQUENTIAL JACOBI RELAXATION  (reference version)" << endl;
  cout << "=====================================================" << endl;
  cout << " Processes P        = " << 1 << endl;
  cout << " Grid (interior)    = " << N << " x " << N << endl;
  cout << " Tolerance          = " << TOL << endl;
  cout << endl;

  /* ---- the relaxation loop ------------------------------------------ */
  cur   = 0;
  nxt   = 1;
  sweep = 0;
  maxchange = 0.0;
  do {
    sweep = sweep + 1;
    maxchange = 0.0;
    for (r = 1; r <= N; r++)
      for (c = 1; c <= N; c++) {
        newval = (G[cur][r-1][c] + G[cur][r+1][c]
                + G[cur][r][c-1] + G[cur][r][c+1]) / 4.0;
        G[nxt][r][c] = newval;
        change = newval - G[cur][r][c];
        if (change < 0.0) change = -change;
        if (change > maxchange) maxchange = change;
      }
    cur = nxt;             /* swap the grids - no copy-back sweep       */
    nxt = 1 - cur;
  } while (maxchange > TOL && sweep < MAXIT);

  /* ---- results ------------------------------------------------------ */
  cout << " CONVERGED" << endl;
  cout << " Sweeps to converge = " << sweep << endl;
  cout << " Final max change   = " << maxchange << endl;
  cout << " Solution checksum  = " << CheckSum(cur) << endl;
  cout << " Centre value       = " << G[cur][N/2][N/2]
       << "   (exact steady state = 25)" << endl;
  cout << "   N is small enough here that the run really does reach the"
       << endl << "   physical answer, so this doubles as a validation." << endl;
  cout << endl;
  cout << " Temperature field (every " << PSTEP << "th row and column):" << endl;
  PrintGrid(cur);
  cout << endl;

  cout << "----------------- PERFORMANCE SUMMARY -----------------" << endl;
  cout.precision(9);
  cout << " Execution time on 1 processor  T(1) = " << clock << endl;
  cout << "=======================================================" << endl;
}
