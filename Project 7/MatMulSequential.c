/* PROGRAM MatMulSequential                                                */
/*                                                                          */
/*  Project 7 - sequential reference for TorusMatMul.col.                     */
/*                                                                          */
/*  Computes C = A x B for N x N matrices on one processor, using the same  */
/*  matrix entries, the same inner kernel and the same correctness checks   */
/*  as the torus program, so its execution time is the T(1) used for        */
/*  speedup and efficiency.                                                 */
/*                                                                          */
/*  Matrix entries are small integers (from a formula), so every product   */
/*  and sum is exact in float and the result can be checked exactly.        */

#define N 96                /* matrix size N x N                            */

float A[N][N], B[N][N], C[N][N];
float SA[N], SB[N];          /* column sums of A, row sums of B (checking) */
int i;

float Aval(int row, int col) { return( (row * 7 + col * 3) % 11 - 5 ); }
float Bval(int row, int col) { return( (row * 5 + col * 2) % 13 - 6 ); }

main() {
  int row, col, k, errors;
  float aik, s, total, tdone;

  for (row = 0; row < N; row++)
    for (col = 0; col < N; col++) {
      A[row][col] = Aval(row, col);
      B[row][col] = Bval(row, col);
      C[row][col] = 0;
    }

  /* ---- C = A x B (same i-k-j kernel as every torus node) ------------- */
  for (row = 0; row < N; row++)
    for (k = 0; k < N; k++) {
      aik = A[row][k];
      for (col = 0; col < N; col++)
        C[row][col] = C[row][col] + aik * B[k][col];
    }
  tdone = clock;

  /* ---- check: every row sum and column sum of C, plus the total ------ */
  /*  row sum r of C = sum_k A[r][k] * (row sum k of B), and similarly     */
  /*  for columns, so the check costs O(N^2) instead of another N^3.       */
  for (k = 0; k < N; k++) {
    SA[k] = 0;
    SB[k] = 0;
    for (row = 0; row < N; row++) SA[k] = SA[k] + A[row][k];
    for (col = 0; col < N; col++) SB[k] = SB[k] + B[k][col];
  }
  errors = 0;
  total = 0;
  for (row = 0; row < N; row++) {
    s = 0;
    for (col = 0; col < N; col++) s = s + C[row][col];
    total = total + s;
    for (k = 0; k < N; k++) s = s - A[row][k] * SB[k];
    if (s != 0) errors++;
  }
  for (col = 0; col < N; col++) {
    s = 0;
    for (row = 0; row < N; row++) s = s + C[row][col];
    for (k = 0; k < N; k++) s = s - SA[k] * B[k][col];
    if (s != 0) errors++;
  }

  cout << "=======================================================" << endl;
  cout << " MATRIX MULTIPLICATION C = A x B - sequential" << endl;
  cout << "=======================================================" << endl;
  cout << " Matrix size N        = " << N << " x " << N << endl;
  cout << " Multiply-adds        = N^3" << endl;
  cout.precision(12);
  cout << " Sum of all C entries = " << total << endl;
  cout << " C[0][0], C[N-1][N-1] = " << C[0][0] << ", " << C[N-1][N-1] << endl;
  cout << " Row/column checks    = " << 2 * N - errors << " of " << 2 * N
       << " correct" << endl;
  cout << endl << "----------------- PERFORMANCE SUMMARY -----------------" << endl;
  cout.precision(9);
  cout << " Sequential execution time T(1) = " << tdone << endl;
  cout << "=======================================================" << endl;
}
