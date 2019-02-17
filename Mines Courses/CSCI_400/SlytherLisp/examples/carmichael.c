/**
 * Equivalent algorithm to carmichael.scm
 * ... if you want to compare your SlytherLisp
 * implementation to something else
 *
 * Note: this is **most certainly** not an optimal
 * algorithm, just what is equivalent to carmichael.scm
 *
 * Compile with:
 * gcc carmichael.c -o carmichael -lm -O9
 */
#include <stdio.h>
#include <math.h>

char divides(int a, int b) {
    return b % a == 0;
}

int isqrt(int n) {
    double guess = n / 2.0;
    double next = (guess + (n / guess)) / 2;
    while (fabs(next - guess) >= 1) {
        guess = next;
        next = (guess + (n / guess)) / 2;
    }
    return floor(next);
}

char is_prime(int n) {
    int stop = isqrt(n);
    if (n > 3) {
        if (divides(2, n)) return 0;
        for (int i = 3; i <= stop; i += 2) {
            if (divides(i, n)) return 0;
        }
        return 1;
    }
    if (n >= 2) return 1;
    return 0;
}

int powmod(int base, int exp, int modulo) {
    long long result = 1;
    for (int i = 0; i < exp; i++) {
        result *= base;
        result %= modulo;
    }
    return result % modulo;
}

char congruent(int a, int b, int m) {
    return a % m == b % m;
}

char fermat_prime(int n) {
    for (int b = 2; b < n; b++) {
        int p = powmod(b, n, n);
        if (!congruent(p, b, n)) return 0;
    }
    return 1;
}

void main(void) {
    for (int x = 5;; x += 2) {
        if (fermat_prime(x) && !is_prime(x)) {
            printf("%d\n", x);
        }
    }
}
