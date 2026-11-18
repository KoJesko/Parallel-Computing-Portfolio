/* PROGRAM Summation of integers from 1 to 100 */
#include <stdlib.h>
#define n 100
int i;
int sum;

main() {
    sum = 0;
    for (i = 1; i <= n; i++)
        sum += i;
    cout << "sum = " << sum << "\n";
}
