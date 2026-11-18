/* PROGRAM SieveParallel                                                */
/* Programming Project 1, page 144-145: Parallel Prime Number Sieve.    */
/*                                                                      */
/* The array Prime[1..n] is partitioned into p equal size portions and  */
/* one process is created for each portion. Process 1 owns the first    */
/* portion, which must contain every number up to sqrt(n). It searches  */
/* its portion for the next TRUE value (the next prime "num") and       */
/* passes num down a pipeline of streams: process 1 sends to process 2, */
/* process 2 forwards to process 3, and so on. Each process steps       */
/* through its own portion in jumps of size num, changing multiples of  */
/* num to FALSE. A value of 0 sent down the pipeline means "finished".  */
/*                                                                      */
/* Input: n (2..32000) and the number of processes p (1..MAXP).         */
/* C* integers are limited to 32767, which sets the upper limit on n.   */

#include <stdlib.h>
#define MAXN 32000
#define MAXP 200
#define TRUE 1
#define FALSE 0

int Prime[MAXN+1];
int stream pipe[MAXP+2];      /* pipe[k] carries step numbers into process k */
int PortionCount[MAXP+1];     /* number of primes in portion k */
int n, p, size, limit, i, count, shown;
float parstart, seqstart, partime, seqwork;

void Sieve(int k) {
  int first, last, num, loc, remainder, j, c;

  first = (k - 1) * size + 1;
  if (k == p) last = n;       /* last portion takes any leftover elements */
  else last = k * size;

  for (j = first; j <= last; j++)   /* each process initializes its own */
    Prime[j] = TRUE;                /* portion in parallel               */

  if (k == 1) {
    /* First process: almost identical to the sequential program, but  */
    /* the while loop stops at the end of the first portion.           */
    for (num = 2; num <= limit; num++)
      if (Prime[num]) {
        if (p > 1) send(pipe[2], num);   /* pass num down the pipeline */
        loc = num + num;
        while (loc <= last) {
          Prime[loc] = FALSE;
          loc = loc + num;
        }
      }
    if (p > 1) send(pipe[2], 0);         /* tell the others to stop */
  }
  else {
    recv(pipe[k], num);
    while (num != 0) {
      if (k < p) send(pipe[k+1], num);   /* forward before working */
      remainder = first % num;           /* find the starting point */
      if (remainder == 0) loc = first;
      else loc = (first / num + 1) * num;
      while (loc <= last) {
        Prime[loc] = FALSE;
        loc = loc + num;
      }
      recv(pipe[k], num);
    }
    if (k < p) send(pipe[k+1], 0);       /* forward the stop signal */
  }

  if (k == 1) Prime[1] = FALSE;          /* 1 is not a prime number */
  c = 0;                                 /* count primes in own portion */
  for (j = first; j <= last; j++)
    if (Prime[j]) c++;
  PortionCount[k] = c;
}

main() {
  cout.precision(8);   /* show times as plain numbers, not E format */
  cout << "Enter n (2.." << MAXN << ") and number of processes: ";
  cin >> n >> p;

  limit = 1;                        /* limit = integer square root of n */
  while ((limit + 1) * (limit + 1) <= n)
    limit++;
  size = n / p;
  if (n < 2 || n > MAXN || p < 1 || p > MAXP || size < limit) {
    cout << endl << "Invalid input: need 2 <= n <= " << MAXN
         << ", 1 <= p <= " << MAXP
         << ", and n/p >= sqrt(n) = " << limit << endl;
    exit(0);
  }
  cout << endl << "n = " << n << ", processes = " << p
       << ", portion size = " << size << ", sqrt(n) = " << limit << endl;

  parstart = clock();
  seqstart = seqtime();
  forall i = 1 to p do
    Sieve(i);
  partime = clock() - parstart;
  seqwork = seqtime() - seqstart;

  count = 0;
  for (i = 1; i <= p; i++)
    count = count + PortionCount[i];

  if (n <= 1000) {
    cout << "Primes:" << endl;
    shown = 0;
    for (i = 2; i <= n; i++)
      if (Prime[i]) {
        cout << i << " ";
        shown++;
        if (shown % 15 == 0) cout << endl;
      }
    cout << endl;
  }
  else {
    cout << "Largest 10 primes found:";
    shown = 0;
    i = n;
    while (shown < 10) {
      if (Prime[i]) {
        cout << " " << i;
        shown++;
      }
      i--;
    }
    cout << endl;
  }
  cout << "Number of primes from 1 to " << n << ": " << count << endl;
  cout << "Sieve + count time (parallel): " << partime << endl;
  cout << "Sieve work (all processes, run one at a time): " << seqwork << endl;
  cout << "Sieve speedup: " << seqwork / partime << endl;
}
