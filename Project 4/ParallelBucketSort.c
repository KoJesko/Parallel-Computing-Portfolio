/* PROGRAM ParallelBucketSort */
/* Programming Project 2, pages 175-176: Parallel Bucket Sort of n integers in range [0, m-1] */
#include <stdlib.h>
/* NOTE: C* identifiers are case-insensitive, so macros must not share a name with any variable */
#define n 2000            /* number of values to be sorted */
#define m 1000            /* each number is in range [0, m-1] */
#define buckets 10        /* number of buckets */
#define procs 10          /* number of processes in the distribution phase (n must be divisible by procs) */
#define width 100         /* size of the value range covered by each bucket (m / buckets) */
#define skewed 0          /* 0 = uniform data; 1 = about 80% of values land in bucket 0 (load imbalance test) */
int data[n+1], sortedData[n+1];  /* data to be sorted, and data sorted */
int bucket[buckets][n+1]; /* bucket[b][0] = # of values in bucket b; bucket[b][1..] holds the values */
spinlock L[buckets];      /* L[b] protects the slot counter bucket[b][0] */
int stream start[buckets+1]; /* start[b] carries the first index in sortedData for bucket b */
int i, ok;

/* sort d[1:d[0]] with selection sort, same as SequentialBucketSort.c */
void sort(int d[]) {
  int j, k, max, max_index, temp;
  for (j = d[0]; j > 1; j--) {
    max = d[j];
    max_index = j;
    for (k = 1; k < j; k++) {
      if (d[k] > max) {
        max = d[k];
        max_index = k;
      }
    }
    temp = d[j];
    d[j] = max;
    d[max_index] = temp;
  }
}

/* Phase 1: process p distributes its slice data[lo:hi] into the shared buckets */
void Distribute(int p) {
  int k, v, b, index, lo, hi;
  lo = (p - 1) * (n / procs) + 1;
  hi = p * (n / procs);
  for (k = lo; k <= hi; k++) {
    v = data[k];
    b = v / width;
    Lock(L[b]);              /* exclusion: only one process may claim a slot at a time */
    index = ++bucket[b][0];
    Unlock(L[b]);            /* release before the store to keep the critical section short */
    bucket[b][index] = v;
  }
}

/* Phases 2+3: process b learns where its bucket goes, sorts it, and copies it into place */
void SortAndAppend(int b) {
  int j, pos, size;
  size = bucket[b][0];
  recv(start[b], pos);           /* first index = 1 + sizes of all predecessor buckets */
  send(start[b+1], pos + size);  /* pass the running sum on to bucket b+1 */
  sort(bucket[b]);
  for (j = 1; j <= size; j++)
    sortedData[pos + j - 1] = bucket[b][j];
}

main() {
  seqoff;  /* exclude setup from the timing (Appendix C, page 540) */
  for (i = 1; i <= n; i++)
    if (skewed == 1 && rand() % 10 < 8)
      data[i] = rand() % width;
    else
      data[i] = rand() % m;
  seqon;

  forall i = 1 to procs do
    Distribute(i);

  send(start[0], 1);  /* bucket 0 starts at sortedData[1] */
  forall i = 0 to buckets - 1 do
    SortAndAppend(i);

  seqoff;  /* exclude printing and checking from the timing */
  for (i = 1; i <= n; i++) {
    cout << sortedData[i] << " ";
    if (i % 20 == 0) cout << endl;
  }
  ok = 1;
  for (i = 2; i <= n; i++)
    if (sortedData[i-1] > sortedData[i]) ok = 0;
  cout << endl << "Bucket sizes:";
  for (i = 0; i < buckets; i++) {
    if (i % 10 == 0) cout << endl << "  ";
    cout << " " << bucket[i][0];
  }
  cout << endl;
  if (ok) cout << "VERIFIED: all " << n << " values are in sorted order" << endl;
  else cout << "ERROR: output is NOT sorted" << endl;
  seqon;
}
