#ifndef __COMMON_H__
#define __COMMON_H__

#include "stdbool.h"
#include "utils.h"

typedef struct {
  int rows_n;
  int cols_n;
  float *values;
} matrix2d_t;

typedef struct {
  int rows_n;
  int cols_n;
  int depth;
  float *values;
} matrix3d_t;

const float *matrix2d_get_elem_as_ref(const matrix2d_t *const m, int row_idx,
                                      int col_idx);
float *matrix2d_get_elem_as_mut_ref(const matrix2d_t *const m, int row_idx,
                                    int col_idx);
float matrix2d_get_elem(const matrix2d_t *const m, int row_idx, int col_idx);
void matrix2d_set_elem(const matrix2d_t *const m, int row_idx, int col_idx,
                       float value);
const float *matrix3d_get_elem_as_ref(const matrix3d_t *const m, int row_idx,
                                      int col_idx, int z_idx);
float *matrix3d_get_elem_as_mut_ref(const matrix3d_t *const m, int row_idx,
                                    int col_idx, int z_idx);
float matrix3d_get_elem(const matrix3d_t *const m, int row_idx, int col_idx,
                        int z_idx);
void matrix3d_set_elem(const matrix3d_t *const m, int row_idx, int col_idx,
                       int z_idx, float value);
void matrix3d_get_slice_as_mut_ref(const matrix3d_t *m, matrix2d_t *result,
                                   int z_idx);

// matrix2d
void matrix2d_init(matrix2d_t *m, int rows_n, int cols_n);
void matrix2d_destroy(const matrix2d_t *m);
void matrix2d_print(const matrix2d_t *const m);
void matrix2d_sum_inplace(const matrix2d_t *const m, const matrix2d_t * const result);
void matrix2d_relu(const matrix2d_t *const m, matrix2d_t *result);
void matrix2d_relu_inplace(const matrix2d_t *const m);
void matrix2d_sigmoid(const matrix2d_t *const m, matrix2d_t *result);
void matrix2d_sigmoid_inplace(const matrix2d_t *const m);
void matrix2d_copy(const matrix2d_t *const input, matrix2d_t *output);
void matrix2d_copy_inplace(const matrix2d_t *const input,
                           const matrix2d_t *output);
void matrix2d_randomize(matrix2d_t *input);
void matrix2d_rotate180(const matrix2d_t *const input, matrix2d_t *output);
void matrix2d_rotate180_inplace(const matrix2d_t *const input);
void matrix2d_submatrix(const matrix2d_t *const input, matrix2d_t *output,
                        int row_start, int row_end, int col_start, int col_end);
void matrix2d_element_wise_product_inplace(const matrix2d_t *const m1,
                                           const matrix2d_t *const m2);
void matrix2d_erase(matrix2d_t *input);
void matrix2d_reshape(const matrix2d_t *const m, matrix2d_t *result, int rows_n,
                      int cols_n);
void matrix2d_tanh_inplace(const matrix2d_t *const m);
void matrix2d_softmax_inplace(const matrix2d_t * const m);
void matrix2d_load(matrix2d_t *m, int rows_n, int cols_n,
                   float *const base_address);
void matrix2d_activate_inplace(const matrix2d_t *const m, activation_type type);

// matrix3d
void matrix3d_init(matrix3d_t *m, int rows_n, int cols_n, int depth);
void matrix3d_destroy(const matrix3d_t *m);
void matrix3d_print(const matrix3d_t *const m);
void matrix3d_erase(matrix3d_t *input);
void matrix3d_copy(const matrix3d_t *const input, matrix3d_t *output);
void matrix3d_copy_inplace(const matrix3d_t *const input,
                           const matrix3d_t *output);
void matrix3d_randomize(matrix3d_t *input);
void matrix3d_reshape(const matrix3d_t *const m, matrix3d_t *result);
void matrix3d_load(matrix3d_t *m, int rows_n, int cols_n, int depth,
                   float *const base_address);

// math
void full_cross_correlation(const matrix2d_t *const m1,
                            const matrix2d_t *const m2, matrix2d_t *result,
                            int padding, int stride);
void max_pooling(const matrix2d_t *const mat, const matrix2d_t * const result,
                 const matrix3d_t *const indexes, int kernel_size, int padding, int stride);
void avg_pooling(const matrix2d_t *const mat, const matrix2d_t *const result,
                 int kernel_size, int padding, int stride);
void convolution(const matrix2d_t *const m1, const matrix2d_t *const m2,
                 matrix2d_t *result, int padding);
void cross_entropy_loss_derivative(const matrix2d_t *const output,
                                   const matrix2d_t *const target_output,
                                   const matrix2d_t *const result);
void mean_squared_error_derivative(const matrix2d_t *const output,
                                   const matrix2d_t *const target_output,
                                   const matrix2d_t *const result);
float cross_entropy_loss(const matrix2d_t *const output,
                         const matrix2d_t *const target_output);
float mean_squared_error(const matrix2d_t *const output,
                         const matrix2d_t *const target_output);

void parse_line(char *line, int length, matrix2d_t *image, float *label);
void zero_pad(const matrix2d_t *const m, matrix2d_t *result, int padding);

#endif