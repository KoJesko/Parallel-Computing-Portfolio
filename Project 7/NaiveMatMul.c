/* PROGRAM NaiveMatMul                                                      */
/*                                                                          */
/*  Project 7 - control version for comparison with TorusMatMul.c.          */
/*                                                                          */
/*  The direct message-passing translation of the course program            */
/*  MatrixMultiplyParallel.c, on the same Q x Q torus.  In the shared-      */
/*  memory original every process reads row i of A and column j of B       */
/*  straight from the shared arrays.  A multicomputer has no shared         */
/*  memory, so here processor 0 holds A and B and sends each processor the  */
/*  NB x N strip of A and the N x NB strip of B that its C block needs.     */
/*  Each processor computes its NB x NB block of C and sends it back.       */
/*                                                                          */
/*  Processor 0 sends 2(P - 1) strips and receives P - 1 blocks one after   */
/*  another, every message enters or leaves the network through the links  */
/*  next to processor 0, and in total (P - 1) * 2 * N * NB floats leave     */
/*  it, compared with 2 * NB * NB per processor per step in Cannon's        */
/*  algorithm.  The arithmetic (N^3 multiply-adds, i-k-j kernel) is the     */
/*  same as TorusMatMul.c.                                                  */
/*                                                                          */
/*  T(P) is the time processor 0 has received every block of C.            */

ARCHITECTURE TORUS(4);

#define Q        4          /* torus is Q x Q; must match TORUS(Q) above   */
#define P        16         /* number of processes = Q * Q                 */
#define N        96         /* matrix size N x N                           */
#define NB       24         /* block size = N / Q                          */
#define SEQTIME4 3649.9233  /* T(1) / 10000 from MatMulSequential.c       */

typedef struct { float v[NB][N]; } rowstrip;   /* NB rows of A            */
typedef struct { float v[N][NB]; } colstrip;   /* NB columns of B         */
typedef struct { int who; float v[NB][NB]; } block; /* a C block + sender */

rowstrip stream Ain[P];     /* strip of A sent by processor 0              */
colstrip stream Bin[P];     /* strip of B sent by processor 0              */
block    stream Cback;      /* finished C blocks, back to processor 0      */
int i;

float Aval(int row, int col) { return( (row * 7 + col * 3) % 11 - 5 ); }
float Bval(int row, int col) { return( (row * 5 + col * 2) % 13 - 6 ); }

/* C block = A strip x B strip: same i-k-j kernel as TorusMatMul.c */
void BlockProduct(rowstrip *a, colstrip *b, block *cb) {
  float aik;
  int ii, jj, k;

  for (ii = 0; ii < NB; ii++)
    for (jj = 0; jj < NB; jj++)
      cb->v[ii][jj] = 0;
  for (ii = 0; ii < NB; ii++)
    for (k = 0; k < N; k++) {
      aik = a->v[ii][k];
      for (jj = 0; jj < NB; jj++)
        cb->v[ii][jj] = cb->v[ii][jj] + aik * b->v[k][jj];
    }
}

/* Processors 1 .. P-1: receive two strips, return one block. */
void Worker(int me) {
  rowstrip a;
  colstrip b;
  block cb;

  recv(Ain[me], a);
  recv(Bin[me], b);
  BlockProduct(&a, &b, &cb);
  cb.who = me;
  send(Cback, cb);
}

/* Processor 0: the only one that holds all of A, B and C. */
void Host() {
  rowstrip a;
  colstrip b;
  block cb;
  float AA[N][N], BB[N][N], C[N][N], SA[N], SB[N];
  float total, tdone, s;
  int dest, ii, jj, k, errors, who;

  /* ---- build A and B once, then send every strip -------------------- */
  for (ii = 0; ii < N; ii++)
    for (jj = 0; jj < N; jj++) {
      AA[ii][jj] = Aval(ii, jj);
      BB[ii][jj] = Bval(ii, jj);
    }
  for (dest = P - 1; dest >= 0; dest--) {
    for (ii = 0; ii < NB; ii++)
      for (jj = 0; jj < N; jj++)
        a.v[ii][jj] = AA[(dest / Q) * NB + ii][jj];
    for (ii = 0; ii < N; ii++)
      for (jj = 0; jj < NB; jj++)
        b.v[ii][jj] = BB[ii][(dest % Q) * NB + jj];
    if (dest > 0) {
      send(Ain[dest], a);
      send(Bin[dest], b);
    }
  }

  /* ---- own block (strips for dest = 0 are left in a and b) ---------- */
  BlockProduct(&a, &b, &cb);

  /* ---- collect all of C --------------------------------------------- */
  for (ii = 0; ii < NB; ii++)
    for (jj = 0; jj < NB; jj++)
      C[ii][jj] = cb.v[ii][jj];
  for (dest = 1; dest < P; dest++) {
    recv(Cback, cb);
    who = cb.who;
    for (ii = 0; ii < NB; ii++)
      for (jj = 0; jj < NB; jj++)
        C[(who / Q) * NB + ii][(who % Q) * NB + jj] = cb.v[ii][jj];
  }
  tdone = clock;

  /* ---- check: every row sum and column sum of C, plus the total ---- */
  for (k = 0; k < N; k++) {
    SA[k] = 0;
    SB[k] = 0;
    for (ii = 0; ii < N; ii++) SA[k] = SA[k] + AA[ii][k];
    for (jj = 0; jj < N; jj++) SB[k] = SB[k] + BB[k][jj];
  }
  errors = 0;
  total = 0;
  for (ii = 0; ii < N; ii++) {
    s = 0;
    for (jj = 0; jj < N; jj++) s = s + C[ii][jj];
    total = total + s;
    for (k = 0; k < N; k++) s = s - AA[ii][k] * SB[k];
    if (s != 0) errors++;
  }
  for (jj = 0; jj < N; jj++) {
    s = 0;
    for (ii = 0; ii < N; ii++) s = s + C[ii][jj];
    for (k = 0; k < N; k++) s = s - SA[k] * BB[k][jj];
    if (s != 0) errors++;
  }

  cout << "=======================================================" << endl;
  cout << " MATRIX MULTIPLICATION ON A TORUS - naive host version" << endl;
  cout << "=======================================================" << endl;
  cout << " Matrix size N        = " << N << " x " << N << endl;
  cout << " Torus                = " << Q << " x " << Q
       << "  (P = " << P << " processors)" << endl;
  cout << " Block per processor  = " << NB << " x " << NB << endl;
  cout << " Processor 0 sends    = " << 2 * (P - 1) << " strips of "
       << NB << " x " << N << ", receives " << P - 1 << " blocks" << endl;
  cout.precision(12);
  cout << endl << " Sum of all C entries = " << total << endl;
  cout << " C[0][0], C[N-1][N-1] = " << C[0][0] << ", " << C[N-1][N-1] << endl;
  cout << " Row/column checks    = " << 2 * N - errors << " of " << 2 * N
       << " correct" << endl;
  cout << endl << "----------------- PERFORMANCE SUMMARY -----------------" << endl;
  cout.precision(9);
  cout << " Parallel execution time   T(P) = " << tdone << endl;
  if (SEQTIME4 > 0) {
    cout << " Sequential execution time T(1) = " << SEQTIME4 * 10000 << endl;
    cout.precision(4);
    cout << " Speedup   T(1)/T(P)            = " << SEQTIME4 * 10000 / tdone << endl;
    cout << " Efficiency  speedup/P          = " << SEQTIME4 * 10000 / tdone / P << endl;
  }
  cout << "=======================================================" << endl;
}

main() {
  forall i = 0 to P-1 @ i do
    if (i == 0) Host();
    else Worker(i);
}
