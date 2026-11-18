/* PROGRAM RankSortMesh                                              */
/* Figure 2.3 Parallel Rank Sort, run on a                              */
/* 8 by 8 two-dimensional mesh multicomputer (64 processors).        */
/* The ARCHITECTURE specification must be the first non-comment line    */
/* in the file, ahead of the global declarations (Appendix C, C.11.1).  */
/* Compare the end-of-run statistics against an unmodified run on the   */
/* default SHARED(1024) architecture.                                   */

ARCHITECTURE MESH2(8);

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
  for (i = 1; i <= n; i++)
    cin >> values[i];      /*initialize values*/

  forall i = 1 to n do
    PutinPlace(i);         /*put values[i] in place*/
}
