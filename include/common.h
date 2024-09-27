#ifndef __COMMON_H__
#define __COMMON_H__

#include "matrix.h"

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