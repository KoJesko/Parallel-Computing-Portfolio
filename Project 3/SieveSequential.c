/* PROGRAM SieveSequential                                              */
/* The sequential Sieve of Eratosthenes from page 144, used as the      */
/* baseline for measuring the speedup of SieveParallel.c.               */
/* Input: n (2..32000). C* integers are limited to 32767.               */

#define MAXN 32000
#define TRUE 1
#define FALSE 0

int Prime[MAXN+1];
int n, i, num, loc, count;
float start, seqtime_sieve;

main() {
  cout.precision(8);   /* show times as plain numbers, not E format */
  cout << "Enter n (2.." << MAXN << "): ";
  cin >> n;
  cout << endl << "n = " << n << " (sequential)" << endl;

  start = clock();
  for (i = 1; i <= n; i++)
    Prime[i] = TRUE;
  for (num = 2; num * num <= n; num++)    /* num <= Sqrt(n) */
    if (Prime[num]) {
      loc = num + num;
      while (loc <= n) {
        Prime[loc] = FALSE;
        loc = loc + num;
      }
    }
  Prime[1] = FALSE;                       /* 1 is not a prime number */
  count = 0;
  for (i = 1; i <= n; i++)
    if (Prime[i]) count++;
  seqtime_sieve = clock() - start;

  cout << "Number of primes from 1 to " << n << ": " << count << endl;
  cout << "Sieve + count time (sequential): " << seqtime_sieve << endl;
}
