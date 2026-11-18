/* PROGRAM Factorial of a user-supplied positive integer */
#include <stdlib.h>
int n;
int i;
int fact;

main() {
    cout << "Enter a positive integer n: ";
    cin >> n;
    fact = 1;
    for (i = 2; i <= n; i++)
        fact *= i;
    cout << n << "! = " << fact << "\n";
}
