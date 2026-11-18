/* PROGRAM IntegrateSequential                                             */
/*                                                                          */
/*  Project 6 - sequential reference for MeshIntegrate.c.                   */
/*  One process evaluates all N trapezoids with exactly the same            */
/*  arithmetic, giving the T(1) baseline and the reference answer.          */

#include <math.h>

#define N        28800      /* total number of trapezoids                  */

float f(float t) {
  return( sqrt(4 - t*t) );
}

main() {
  float a, b, w, t, sum, err;
  int j;

  a = 0.0;
  b = 2.0;
  w = (b - a) / N;
  sum = 0.0;
  t = a;
  for (j = 1; j <= N; j++) {
    sum = sum + f(t);
    t = t + w;
  }
  sum = w * sum + w / 2 * (f(b) - f(a));

  err = sum - 3.14159265358979;
  if (err < 0) err = -err;
  cout << "=======================================================" << endl;
  cout << " NUMERICAL INTEGRATION - sequential reference" << endl;
  cout << "=======================================================" << endl;
  cout << " f(t) = sqrt(4 - t*t),  a = 0,  b = 2" << endl;
  cout << " Trapezoids N         = " << N << endl;
  cout.precision(12);
  cout << endl << " Integral             = " << sum << endl;
  cout << " pi                   = " << 3.14159265358979 << endl;
  cout << " Absolute error       = " << err << endl;
  cout.precision(9);
  cout << endl << " Sequential execution time T(1) = " << clock << endl;
  cout << "=======================================================" << endl;
}
