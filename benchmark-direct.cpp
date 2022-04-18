#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cmath> // For: fabs

#include <cblas.h>
#include <stdio.h>

extern "C" {
    static void dws_conv(double*, double*, double*, double*, int, int, int, int, int, int, int, int, int, int, int, int);
}

void fill(double* p, int n) {
    static std::random_device rd;
    static std::default_random_engine gen(rd());
    static std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < n; ++i)
        p[i] = 2 * dis(gen) - 1;
}

void fillDet(double* p, int n){
    for (int i = 0; i < n; ++i){
	    p[i] = i + 1.0;
    }
}

void fillConst(double* p, int n, double k){
    for (int i = 0; i < n; ++i) {
        p[i] = k;
	}
}

void fillZero(double* p, int n){
    fillConst(p, n, 0.0);
}

void printMatrix(double* p, int n) {
    for (int i = 0; i < n * n; ++i) {
        if (i > 0 && i % n == 0) {
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "%f ", p[i]);
	}
}

/* The benchmarking program */
int main(int argc, char** argv) {
    std::cout << std::fixed << std::setprecision(2);
    return 0;
}