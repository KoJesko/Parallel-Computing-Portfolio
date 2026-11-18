/* PROGRAM MeshIntegrateNaive                                              */
/*                                                                          */
/*  Project 6 - control version for comparison with MeshIntegrate.c.        */
/*                                                                          */
/*  The direct message-passing translation of Figure 5.8: the shared        */
/*  "globalsum += localsum" inside a lock becomes "every process sends its  */
/*  local sum to processor 0".  Processor 0 must perform P - 1 receives one */
/*  after another, and messages from the far corner cross 2(M - 1) links.   */
/*  The local computation is identical to MeshIntegrate.c.                  */

ARCHITECTURE MESH2(5);

#include <math.h>

#define M        5          /* mesh is M x M; must match MESH2(M) above    */
#define P        25         /* number of processes = M * M                 */
#define N        28800      /* total number of trapezoids (all processes)  */
#define SEQTIMEK 1238.584   /* T(1) / 1000 from IntegrateSequential.c     */

float stream tozero;        /* every local sum is sent here                */
int i;

float f(float t) {
  return( sqrt(4 - t*t) );
}

void Integrate(int me) {
  float a, b, w, t, localsum, total, inval, err;
  int numpoints, j, k;

  a = 0.0;
  b = 2.0;
  w = (b - a) / N;
  numpoints = N / P;

  localsum = 0.0;
  t = a + me * numpoints * w;
  for (j = 1; j <= numpoints; j++) {
    localsum = localsum + f(t);
    t = t + w;
  }
  localsum = w * localsum;

  if (me > 0)
    send(tozero, localsum);
  else {
    total = localsum;
    for (k = 1; k <= P - 1; k++) {
      recv(tozero, inval);
      total = total + inval;
    }
    total = total + w / 2 * (f(b) - f(a));

    err = total - 3.14159265358979;
    if (err < 0) err = -err;
    cout << "=======================================================" << endl;
    cout << " NUMERICAL INTEGRATION ON A 2-D MESH - naive gather" << endl;
    cout << "=======================================================" << endl;
    cout << " f(t) = sqrt(4 - t*t),  a = 0,  b = 2" << endl;
    cout << " Mesh                 = " << M << " x " << M << endl;
    cout << " Processes P          = " << P << endl;
    cout << " Trapezoids N         = " << N << endl;
    cout << " Points per process   = " << numpoints << endl;
    cout << " Summation            = " << P - 1
         << " receives at processor 0" << endl;
    cout.precision(12);
    cout << endl << " Integral             = " << total << endl;
    cout << " pi                   = " << 3.14159265358979 << endl;
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

main() {
  forall i = 0 to P-1 @ i do
    Integrate(i);
}
