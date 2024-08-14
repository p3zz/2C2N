#ifndef __COMMON_H__
#define __COMMON_H__

typedef struct{
    int rows_n;
    int cols_n;
    float** values;
} matrix2d_t;

typedef struct{
    int depth;
    matrix2d_t* layers;
} matrix3d_t;

void cross_correlation(const matrix2d_t* const m1, const matrix2d_t* const m2, matrix2d_t* result, int padding, int stride);
matrix2d_t create_matrix(int rows_n, int cols_n);

#endif