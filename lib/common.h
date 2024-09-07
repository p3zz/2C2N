#ifndef __COMMON_H__
#define __COMMON_H__

#include "stdbool.h"

typedef struct{
    int rows_n;
    int cols_n;
    float** values;
} matrix2d_t;

typedef struct {
    matrix2d_t* layers;
    int depth;
} matrix3d_t;


// matrix2d
void matrix2d_init(matrix2d_t* m, int rows_n, int cols_n);
void matrix2d_destroy(matrix2d_t* m);
void matrix2d_print(const matrix2d_t* const m);
void matrix2d_sum_inplace(const matrix2d_t* const m, matrix2d_t* result);
void matrix2d_relu(const matrix2d_t* const m, matrix2d_t* result);
void matrix2d_relu_inplace(const matrix2d_t* const m);
void matrix2d_sigmoid(const matrix2d_t* const m, matrix2d_t* result);
void matrix2d_sigmoid_inplace(const matrix2d_t* const m);
void matrix2d_copy(const matrix2d_t* const input, matrix2d_t* output);
void matrix2d_copy_inplace(const matrix2d_t* const input, const matrix2d_t* output);
void matrix2d_randomize(matrix2d_t* input);
void matrix2d_rotate180(const matrix2d_t* const input, matrix2d_t* output);
void matrix2d_rotate180_inplace(const matrix2d_t* const input);
void matrix2d_submatrix(const matrix2d_t* const input, matrix2d_t* output, int row_start, int row_end, int col_start, int col_end);
void matrix2d_mul(const matrix2d_t* const m1, const matrix2d_t* const m2, matrix2d_t* result);
void matrix2d_mul_inplace(const matrix2d_t* const m1, const matrix2d_t* const m2);
void matrix2d_erase(matrix2d_t* input);
void matrix2d_reshape(const matrix2d_t* const m, matrix2d_t* result, int rows_n, int cols_n);
void matrix2d_tanh_inplace(const matrix2d_t* const m);

// matrix3d
void matrix3d_init(matrix3d_t* m, int rows_n, int cols_n, int depth);
void matrix3d_destroy(matrix3d_t* m);
void matrix3d_print(const matrix3d_t* const m);
void matrix3d_erase(matrix3d_t* input);
void matrix3d_copy(const matrix3d_t* const input, matrix3d_t* output);
void matrix3d_copy_inplace(const matrix3d_t* const input, const matrix3d_t* output);
void matrix3d_randomize(matrix3d_t* input);

// math
void full_cross_correlation(const matrix2d_t* const m1, const matrix2d_t* const m2, matrix2d_t* result, int padding, int stride);
void max_pooling(const matrix2d_t* const mat, matrix2d_t* result, matrix3d_t* indexes, int kernel_size, int padding, int stride);
void avg_pooling(const matrix2d_t* const mat, matrix2d_t* result, int kernel_size, int padding, int stride);
float cross_correlation(const matrix2d_t* const m1, const matrix2d_t* const m2, float result);
void convolution(const matrix2d_t* const m1, const matrix2d_t* const m2, matrix2d_t* result, int padding);

#endif