#ifndef __COMMON_H__
#define __COMMON_H__

typedef struct{
    int rows_n;
    int cols_n;
    float** values;
} matrix2d_t;

typedef struct {
    matrix2d_t* layers;
    int depth;
} matrix3d_t;

void cross_correlation(const matrix2d_t* const m1, const matrix2d_t* const m2, matrix2d_t* result, int padding, int stride);
int max_pooling(const matrix2d_t* const mat, int kernel_size, matrix2d_t* result, int padding, int stride);
int avg_pooling(const matrix2d_t* const mat, int kernel_size, matrix2d_t* result, int padding, int stride);

void create_matrix2d(matrix2d_t* m, int rows_n, int cols_n);
void create_matrix3d(matrix3d_t* m, int rows_n, int cols_n, int depth);

#endif