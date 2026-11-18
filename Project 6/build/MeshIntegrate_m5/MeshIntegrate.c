/* PROGRAM MeshIntegrate                                                   */
/*                                                                          */
/*  Project 6 - Numerical integration on a 2-D mesh multicomputer.          */
/*  Conversion of the numerical integration program of Figure 5.8.          */
/*                                                                          */
/*  Computes  integral from a = 0 to b = 2 of sqrt(4 - t*t) dt  =  pi       */
/*  with the trapezoidal rule, using P = M x M processes, one per           */
/*  processor of an M x M mesh (process i runs on processor i, which is     */
/*  row i / M, column i % M).                                               */
/*                                                                          */
/*  WHAT CHANGED FROM FIGURE 5.8                                            */
/*  Figure 5.8 adds each local sum into a shared variable inside a lock.    */
/*  A multicomputer has no shared memory, and funnelling all P sums into    */
/*  one node costs P - 1 sequential receives at that node, with messages    */
/*  crossing up to 2(M - 1) links.                                          */
/*                                                                          */
/*  MESH SUMMATION (every message goes to a direct mesh neighbour)          */
/*   1. Row phase: in each row the left half passes a running sum to the    */
/*      right and the right half passes one to the left.  Both halves run   */
/*      at the same time and meet at the middle column after M/2 steps.     */
/*   2. Column phase: the middle column does the same vertically, so the    */
/*      grand total arrives at the centre processor after M/2 more steps.   */
/*  That is M = sqrt(P) steps in total, every message is one hop, and no    */
/*  node receives more than 3 messages.  A gather to processor 0 needs      */
/*  P - 1 receives there.                                                   */
/*                                                                          */
/*  The centre processor adds the endpoint correction and prints the        */
/*  answer and the performance summary.                                     */

ARCHITECTURE MESH2(5);

#include <math.h>

#define M        5          /* mesh is M x M; must match MESH2(M) above    */
#define P        25         /* number of processes = M * M                 */
#define N        28800      /* total number of trapezoids (all processes)  */
#define SEQTIMEK 1238.584   /* T(1) / 1000 from IntegrateSequential.c     */

float stream fromleft[P];   /* value arriving from the west neighbour      */
float stream fromright[P];  /* value arriving from the east neighbour      */
float stream fromup[P];     /* value arriving from the north neighbour     */
float stream fromdown[P];   /* value arriving from the south neighbour     */
int i;

float f(float t) {
  return( sqrt(4 - t*t) );
}

void Integrate(int me) {
  float a, b, w, t, localsum, rowsum, total, inval, err;
  int row, col, mid, numpoints, j;

  a = 0.0;
  b = 2.0;
  w = (b - a) / N;
  numpoints = N / P;
  row = me / M;
  col = me % M;
  mid = (M - 1) / 2;

  /* ---- local computation: identical to Figure 5.8 ------------------- */
  localsum = 0.0;
  t = a + me * numpoints * w;
  for (j = 1; j <= numpoints; j++) {
    localsum = localsum + f(t);
    t = t + w;
  }
  localsum = w * localsum;

  /* ---- row phase: both halves of the row converge on column mid ----- */
  rowsum = localsum;
  if (col < mid) {
    if (col > 0) {
      recv(fromleft[me], inval);
      rowsum = rowsum + inval;
    }
    send(fromleft[me + 1], rowsum);
  }
  if (col > mid) {
    if (col < M - 1) {
      recv(fromright[me], inval);
      rowsum = rowsum + inval;
    }
    send(fromright[me - 1], rowsum);
  }
  if (col == mid) {
    if (col > 0) {
      recv(fromleft[me], inval);
      rowsum = rowsum + inval;
    }
    if (col < M - 1) {
      recv(fromright[me], inval);
      rowsum = rowsum + inval;
    }

    /* ---- column phase: the middle column converges on row mid ------- */
    total = rowsum;
    if (row < mid) {
      if (row > 0) {
        recv(fromup[me], inval);
        total = total + inval;
      }
      send(fromup[me + M], total);
    }
    if (row > mid) {
      if (row < M - 1) {
        recv(fromdown[me], inval);
        total = total + inval;
      }
      send(fromdown[me - M], total);
    }
    if (row == mid) {
      if (row > 0) {
        recv(fromup[me], inval);
        total = total + inval;
      }
      if (row < M - 1) {
        recv(fromdown[me], inval);
        total = total + inval;
      }

      /* ---- centre processor: finish and report ---------------------- */
      total = total + w / 2 * (f(b) - f(a));
      cout << "=======================================================" << endl;
      cout << " NUMERICAL INTEGRATION ON A 2-D MESH - mesh summation" << endl;
      cout << "=======================================================" << endl;
      cout << " f(t) = sqrt(4 - t*t),  a = 0,  b = 2" << endl;
      cout << " Mesh                 = " << M << " x " << M << endl;
      cout << " Processes P          = " << P << endl;
      cout << " Trapezoids N         = " << N << endl;
      cout << " Points per process   = " << numpoints << endl;
      cout << " Summation steps      = " << M / 2 + M / 2
           << "  (row M/2 + column M/2, 1 hop each)" << endl;
      cout << " Result collected at  = processor " << me
           << " (row " << row << ", col " << col << ")" << endl;
      cout.precision(12);
      cout << endl << " Integral             = " << total << endl;
      cout << " pi                   = " << 3.14159265358979 << endl;
      err = total - 3.14159265358979;
      if (err < 0) err = -err;
      cout << " Absolute error       = " << err << endl;
      cout << endl << "----------------- PERFORMANCE SUMMARY -----------------" << endl;
      cout.precision(9);
      cout << " Parallel execution time   T(P) = " << clock << endl;
      if (SEQTIMEK > 0) {
        cout << " Sequential execution time T(1) = " << SEQTIMEK * 1000 << endl;
        cout.precision(4);
        cout << " Speedup   T(1)/T(P)            = " << SEQTIMEK * 1000 / clock << endl;
        cout << " Efficiency  speedup/P          = " << SEQTIMEK * 1000 / clock / P << endl;
      }
      cout << "=======================================================" << endl;
    }
  }
}

main() {
  forall i = 0 to P-1 @ i do
    Integrate(i);
}
