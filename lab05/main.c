/*
    BONUS:
        Testare timpi de executie pentru
    diferite moduri de inmultire a matricilor
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef int* vector;
typedef vector* matrix;

const int N = 500;

// ==============================================

matrix new(int N, int val)
{
    matrix M = malloc(N * sizeof(vector));

    for (int i = 0; i < N; ++i)
    {
        M[i] = malloc(N * sizeof(int));
        for (int j = 0; j < N; ++j)
            M[i][j] = val;
    }

    return M;
}

// ==============================================

void delete(matrix M)
{
    for (int i = 0; i < N; ++i)
        free(M[i]);
    free(M);
}

// ==============================================

matrix ijk(matrix A, matrix B)
{
    matrix C = new(N, 0);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

// ==============================================

matrix ikj(matrix A, matrix B)
{
    matrix C = new(N, 0);

    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k)
            for (int j = 0; j < N; ++j)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

// ==============================================

matrix jik(matrix A, matrix B)
{
    matrix C = new(N, 0);

    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
            for (int k = 0; k < N; ++k)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

// ==============================================

matrix jki(matrix A, matrix B)
{
    matrix C = new(N, 0);

    for (int j = 0; j < N; ++j)
        for (int k = 0; k < N; ++k)
            for (int i = 0; i < N; ++i)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

// ==============================================

matrix kij(matrix A, matrix B)
{
    matrix C = new(N, 0);

    for (int k = 0; k < N; ++k)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

// ==============================================

matrix kji(matrix A, matrix B)
{
    matrix C = new(N, 0);

    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

// ==============================================

int main()
{
    double begin, end, elapsed;
    matrix A = new(N, 2), B = new(N, 3), C;

    // ------------------------------------------

    begin = clock();
    C = ijk(A, B);
    delete(C);
    end = clock();
    elapsed = (end - begin) / CLOCKS_PER_SEC;
    printf("i-j-k: %fs\n", elapsed);

    // ------------------------------------------

    begin = clock();
    C = ikj(A, B);
    delete(C);
    end = clock();
    elapsed = (end - begin) / CLOCKS_PER_SEC;
    printf("i-k-j: %fs\n", elapsed);

    // ------------------------------------------

    begin = clock();
    C = jik(A, B);
    delete(C);
    end = clock();
    elapsed = (end - begin) / CLOCKS_PER_SEC;
    printf("j-i-k: %fs\n", elapsed);

    // ------------------------------------------

    begin = clock();
    C = jki(A, B);
    delete(C);
    end = clock();
    elapsed = (end - begin) / CLOCKS_PER_SEC;
    printf("j-k-i: %fs\n", elapsed);

    // ------------------------------------------

    begin = clock();
    C = kij(A, B);
    delete(C);
    end = clock();
    elapsed = (end - begin) / CLOCKS_PER_SEC;
    printf("k-i-j: %fs\n", elapsed);

    // ------------------------------------------

    begin = clock();
    C = kji(A, B);
    delete(C);
    end = clock();
    elapsed = (end - begin) / CLOCKS_PER_SEC;
    printf("k-j-i: %fs\n", elapsed);

    // ------------------------------------------

    return 0;
}