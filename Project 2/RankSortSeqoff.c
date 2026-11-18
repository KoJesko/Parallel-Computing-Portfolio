/* PROGRAM RankSortSeqoff                                               */
/* Figure 2.3 Parallel Rank Sort, modified to demonstrate the seqoff    */
/* and seqon primitives from Appendix C section C.9.3.                  */
/*                                                                      */
/* NOTE ON CORRECTNESS OF THIS EXCLUSION: C.9.3 intends seqoff/seqon    */
/* for code a sequential version of the program would not execute at    */
/* all, such as the Barrier procedure of Chapter 6. Rank Sort has no    */
/* such code — it uses no barrier, spinlock, or stream, and every line  */
/* of PutinPlace is comparison work a sequential rank sort would also   */
/* perform. The cin loop excluded below is likewise work a sequential   */
/* version would do, so the SPEEDUP this program reports is inflated.   */
/* It is run here to show what the primitives do to the statistic, not  */
/* to claim a legitimate measurement. Compare its SPEEDUP against the   */
/* unmodified RankSortParallel.c to see the size of the effect.         */

#define n 100

int values[n+1], final[n+1];
int i;

void PutinPlace( int src ) {
  int testval, j, rank;
  testval = values[src];
  j = src;              /*j moves through the whole array*/
  rank = 1;
  do {
    j = j % n + 1;
    if (testval > values[j] ||
        (testval == values[j] && src > j))
      rank = rank + 1;
  } while (j != src);
  final[rank] = testval;   /*put into position*/
}

main() {
  seqoff;                  /*stop charging this process's work to  */
                           /*the SEQUENTIAL time statistic         */

  for (i = 1; i <= n; i++)
    cin >> values[i];      /*initialize values*/

  seqon;                   /*resume charging SEQUENTIAL time*/

  forall i = 1 to n do
    PutinPlace(i);         /*put values[i] in place*/
}
