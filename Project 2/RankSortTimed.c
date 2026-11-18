/* PROGRAM RankSortTimed                                                */
/* Figure 2.3 Parallel Rank Sort, modified per Appendix C section C.9.1 */
/* to time the sorting phase on its own. The cin loop that loads the    */
/* array is sequential and dominates the whole-program speedup, so the  */
/* clock() and seqtime() marks are taken after it and the speedup is    */
/* computed across the forall only.                                     */

#define n 100

int values[n+1], final[n+1];
int i;
float parstart, seqstart;

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

  parstart = clock();
  seqstart = seqtime();

  forall i = 1 to n do
    PutinPlace(i);         /*put values[i] in place*/

  cout << "Speedup is ";
  cout << (seqtime() - seqstart) / (clock() - parstart);
  cout << endl;
}
