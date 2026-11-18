/* PROGRAM TorusMatMul                                                      */
/*                                                                          */
/*  Project 7 - Efficient parallel matrix multiplication on a multicomputer */
/*  with a torus interconnection network.                                   */
/*                                                                          */
/*  Computes C = A x B for N x N matrices on a Q x Q torus of P = Q*Q       */
/*  processors using Cannon's algorithm.  Process i runs on processor i,    */
/*  at row i / Q, column i % Q.  Each processor owns one NB x NB block      */
/*  (NB = N / Q) of A, of B and of C, so no processor ever holds more than  */
/*  3 blocks and the matrices are never collected in one place.            */
/*                                                                          */
/*  CANNON'S ALGORITHM                                                      */
/*   1. Alignment: block row r of A is rotated left r places and block      */
/*      column c of B is rotated up c places, so processor (r,c) then       */
/*      holds A(r, r+c) and B(r+c, c) (indices mod Q).                      */
/*   2. Q steps: multiply the two blocks held into the local C block, then  */
/*      rotate every A block one place left and every B block one place up. */
/*      After Q steps processor (r,c) has added A(r,k) x B(k,c) for every   */
/*      k, which is block C(r,c).                                           */
/*                                                                          */
/*  WHY THE TORUS                                                           */
/*  The rotations are cyclic: the block leaving column 0 must re-enter at   */
/*  column Q-1.  On a torus that is a wraparound link, so every rotation    */
/*  message in step 2 travels exactly ONE link, every processor sends and   */
/*  receives exactly one A block and one B block per step, and no link      */
/*  carries more than one message per direction.  On a plain mesh the same  */
/*  message crosses Q-1 links and shares them with Q-1 other messages.      */
/*                                                                          */
/*  EFFICIENCY MEASURES                                                     */
/*   - A whole block is sent as one message (a struct), so each rotation    */
/*     costs one message start-up instead of NB*NB.                         */
/*   - Communication is overlapped with computation: the blocks for the     */
/*     next step are sent before the local multiply starts, and received    */
/*     after it finishes.                                                   */
/*   - The alignment sends each block straight to its destination, and is  */
/*     skipped for row 0 of A and column 0 of B, which do not move.         */
/*   - Each processor generates its own blocks of A and B, so there is no   */
/*     distribution phase through a single host node.                       */
/*                                                                          */
/*  T(P) is the time the last processor finishes its C block.  The checks   */
/*  and the reduction that carries them to processor 0 come afterwards.     */

ARCHITECTURE TORUS(4);

#define Q        4          /* torus is Q x Q; must match TORUS(Q) above   */
#define P        16         /* number of processes = Q * Q                 */
#define N        96         /* matrix size N x N                           */
#define NB       24         /* block size = N / Q                          */
#define SEQTIME4 3649.9233  /* T(1) / 10000 from MatMulSequential.c       */
#define OVERLAP  1          /* 1: send next blocks before the multiply;    */
                            /* 0: send after it (for comparison only)      */

typedef struct { float v[NB][NB]; } block;
typedef struct { float sum, errors, tfin, corner; } report;

block  stream Aalign[P];    /* alignment: A block arriving at processor    */
block  stream Balign[P];    /* alignment: B block arriving at processor    */
block  stream Arot[P];      /* rotation: A block arriving from the east    */
block  stream Brot[P];      /* rotation: B block arriving from the south   */
report stream fromeast[P];  /* check reduction, arriving from the east     */
report stream fromwest[P];  /* check reduction, arriving from the west     */
report stream fromsouth[P]; /* check reduction, arriving from the south    */
report stream fromnorth[P]; /* check reduction, arriving from the north    */
int i;

float Aval(int row, int col) { return( (row * 7 + col * 3) % 11 - 5 ); }
float Bval(int row, int col) { return( (row * 5 + col * 2) % 13 - 6 ); }

void addreport(report *x, report y) {
  x->sum = x->sum + y.sum;
  x->errors = x->errors + y.errors;
  if (y.tfin > x->tfin) x->tfin = y.tfin;
  x->corner = x->corner + y.corner;
}

void Multiply(int me) {
  block a, b;
  float cb[NB][NB], colsum[N];
  float aik, s;
  report rep, in;
  int row, col, left, up, step, ii, jj, k, g;

  row  = me / Q;
  col  = me % Q;
  left = row * Q + (col + Q - 1) % Q;     /* west neighbour (wraps)       */
  up   = ((row + Q - 1) % Q) * Q + col;   /* north neighbour (wraps)      */

  /* ---- generate own blocks A(row,col), B(row,col); clear C(row,col) -- */
  for (ii = 0; ii < NB; ii++)
    for (jj = 0; jj < NB; jj++) {
      a.v[ii][jj] = Aval(row * NB + ii, col * NB + jj);
      b.v[ii][jj] = Bval(row * NB + ii, col * NB + jj);
      cb[ii][jj] = 0;
    }

  /* ---- 1. alignment: A left by row, B up by col (direct send) -------- */
  if (row > 0) send(Aalign[row * Q + (col + Q - row) % Q], a);
  if (col > 0) send(Balign[((row + Q - col) % Q) * Q + col], b);
  if (row > 0) recv(Aalign[me], a);
  if (col > 0) recv(Balign[me], b);

  /* ---- 2. Q multiply steps, rotating one link per step -------------- */
  for (step = 1; step <= Q; step++) {
    if (OVERLAP == 1 && step < Q) {     /* start next step's transfers  */
      send(Arot[left], a);
      send(Brot[up], b);
    }
    for (ii = 0; ii < NB; ii++)         /* C(row,col) += A-block x B-block */
      for (k = 0; k < NB; k++) {
        aik = a.v[ii][k];
        for (jj = 0; jj < NB; jj++)
          cb[ii][jj] = cb[ii][jj] + aik * b.v[k][jj];
      }
    if (OVERLAP == 0 && step < Q) {
      send(Arot[left], a);
      send(Brot[up], b);
    }
    if (step < Q) {
      recv(Arot[me], a);
      recv(Brot[me], b);
    }
  }
  rep.tfin = clock;

  /* ---- check this block: every row sum and column sum, exactly ------- */
  rep.sum = 0;
  rep.errors = 0;
  for (k = 0; k < N; k++) {             /* colsum[k] = sum of B(k, my cols)  */
    colsum[k] = 0;
    for (jj = 0; jj < NB; jj++) colsum[k] = colsum[k] + Bval(k, col * NB + jj);
  }
  for (ii = 0; ii < NB; ii++) {
    s = 0;
    for (jj = 0; jj < NB; jj++) s = s + cb[ii][jj];
    rep.sum = rep.sum + s;
    for (k = 0; k < N; k++) s = s - Aval(row * NB + ii, k) * colsum[k];
    if (s != 0) rep.errors = rep.errors + 1;
  }
  for (k = 0; k < N; k++) {             /* colsum[k] = sum of A(my rows, k)  */
    colsum[k] = 0;
    for (ii = 0; ii < NB; ii++) colsum[k] = colsum[k] + Aval(row * NB + ii, k);
  }
  for (jj = 0; jj < NB; jj++) {
    s = 0;
    for (ii = 0; ii < NB; ii++) s = s + cb[ii][jj];
    for (k = 0; k < N; k++) s = s - colsum[k] * Bval(k, col * NB + jj);
    if (s != 0) rep.errors = rep.errors + 1;
  }
  rep.corner = 0;
  if (me == P - 1) rep.corner = cb[NB - 1][NB - 1];

  /* ---- reduce the reports to processor 0 over torus links ----------- */
  /*  Row phase: columns 1..Q/2 pass west, columns Q/2+1..Q-1 pass east   */
  /*  across the wraparound link, so column 0 has the row total after     */
  /*  Q/2 one-hop steps.  Column phase: the same in column 0, north/south. */
  g = Q / 2;
  if (col >= 1 && col + 1 <= g) { recv(fromeast[me], in); addreport(&rep, in); }
  if (col > g + 1)              { recv(fromwest[me], in); addreport(&rep, in); }
  if (col >= 1 && col <= g) send(fromeast[me - 1], rep);
  if (col > g)              send(fromwest[row * Q + (col + 1) % Q], rep);
  if (col == 0) {
    if (Q >= 2) { recv(fromeast[me], in); addreport(&rep, in); }
    if (Q >= 3) { recv(fromwest[me], in); addreport(&rep, in); }

    if (row >= 1 && row + 1 <= g) { recv(fromsouth[me], in); addreport(&rep, in); }
    if (row > g + 1)              { recv(fromnorth[me], in); addreport(&rep, in); }
    if (row >= 1 && row <= g) send(fromsouth[me - Q], rep);
    if (row > g)              send(fromnorth[((row + 1) % Q) * Q], rep);
    if (row == 0) {
      if (Q >= 2) { recv(fromsouth[me], in); addreport(&rep, in); }
      if (Q >= 3) { recv(fromnorth[me], in); addreport(&rep, in); }

      /* ---- processor 0: report --------------------------------------- */
      cout << "=======================================================" << endl;
      cout << " MATRIX MULTIPLICATION ON A TORUS - Cannon's algorithm" << endl;
      cout << "=======================================================" << endl;
      cout << " Matrix size N        = " << N << " x " << N << endl;
      cout << " Torus                = " << Q << " x " << Q
           << "  (P = " << P << " processors)" << endl;
      cout << " Block per processor  = " << NB << " x " << NB << endl;
      cout << " Rotation steps       = " << Q - 1
           << "  (2 block messages per processor, 1 hop each)" << endl;
      if (OVERLAP == 1)
        cout << " Communication        = overlapped with the multiply" << endl;
      else
        cout << " Communication        = after the multiply (not overlapped)" << endl;
      cout.precision(12);
      cout << endl << " Sum of all C entries = " << rep.sum << endl;
      cout << " C[0][0], C[N-1][N-1] = " << cb[0][0] << ", " << rep.corner << endl;
      cout << " Row/column checks    = " << 2 * N - (int) rep.errors << " of "
           << 2 * N << " correct" << endl;
      cout << endl << "----------------- PERFORMANCE SUMMARY -----------------" << endl;
      cout.precision(9);
      cout << " Parallel execution time   T(P) = " << rep.tfin << endl;
      if (SEQTIME4 > 0) {
        cout << " Sequential execution time T(1) = " << SEQTIME4 * 10000 << endl;
        cout.precision(4);
        cout << " Speedup   T(1)/T(P)            = " << SEQTIME4 * 10000 / rep.tfin << endl;
        cout << " Efficiency  speedup/P          = " << SEQTIME4 * 10000 / rep.tfin / P << endl;
      }
      cout << "=======================================================" << endl;
    }
  }
}

main() {
  forall i = 0 to P-1 @ i do
    Multiply(i);
}
