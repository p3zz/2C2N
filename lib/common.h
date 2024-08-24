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
void max_pooling(const matrix2d_t* const mat, matrix2d_t* result, int kernel_size, int padding, int stride);
void avg_pooling(const matrix2d_t* const mat, matrix2d_t* result, int kernel_size, int padding, int stride);

void create_matrix2d(matrix2d_t* m, int rows_n, int cols_n);
void create_matrix3d(matrix3d_t* m, int rows_n, int cols_n, int depth);

void destroy_matrix3d(matrix3d_t* m);
void destroy_matrix2d(matrix2d_t* m);

void matrix2d_sum_inplace(const matrix2d_t* const m, matrix2d_t* result);

void matrix2d_print(const matrix2d_t* const m);
void matrix3d_print(const matrix3d_t* const m);

void matrix2d_relu(const matrix2d_t* const m, matrix2d_t* result);
void matrix2d_relu_inplace(const matrix2d_t* const m);
void matrix2d_sigmoid(const matrix2d_t* const m, matrix2d_t* result);
void matrix2d_sigmoid_inplace(const matrix2d_t* const m);
void matrix2d_copy(const matrix2d_t* const input, matrix2d_t* output);

#endif