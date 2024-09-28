/**
 * @brief This library provides a basic implementation of the most common operations performed in Convolutional Neural Networks (CNNs),
 * such as:
 * - Cross-correlation
 * - Convolution
 * - Max/Average pooling
 * - Cross entropy/mean squared error and their derivatives
 */

#ifndef __COMMON_H__
#define __COMMON_H__

#include "matrix.h"

/**
 * @brief Perform the cross-correlation between two 2D matrices,
 * and stores the result inside a 2D result matrix.
 * @param m1: pointer to the 1st matrix
 * @param m2: pointer to the 2nd matrix
 * @param result: pointer to the result matrix
 * @param padding: padding used to perform the cross-correlation
 * @param stride: stride used to perform the cross-correlation
 */
void cross_correlation(const matrix2d_t *const m1,
                            const matrix2d_t *const m2, matrix2d_t *result,
                            int padding, int stride);

/**
 * @brief Perform the convolution between two 2D matrices,
 * and stores the result inside a 2D result matrix.
 * @param m1: pointer to the 1st matrix
 * @param m2: pointer to the 2nd matrix
 * @param result: pointer to the result matrix
 * @param padding: padding used to perform the cross-correlation
 * @param stride: stride used to perform the cross-correlation
 */
void convolution(const matrix2d_t *const m1, const matrix2d_t *const m2,
                 matrix2d_t *result, int padding, int stride);

/**
 * @brief Perform the max pooling of a 2D input matrix using a squared kernel,
 * and stores the result inside a 2D result matrix.
 * @param input: pointer to the input matrix
 * @param output: pointer to the output matrix
 * @param indexes: pointer to the indeces matrix
 * @param kernel_size: length of the kernel side
 * @param padding: padding used to perform the cross-correlation
 * @param stride: stride used to perform the cross-correlation
 */
void max_pooling(const matrix2d_t *const input, const matrix2d_t *const output,
                 const matrix3d_t *const indexes, int kernel_size, int padding,
                 int stride);

/**
 * @brief Perform the average pooling of a 2D input matrix using a squared kernel,
 * and stores the result inside a 2D result matrix.
 * @param input: pointer to the input matrix
 * @param output: pointer to the output matrix
 * @param kernel_size: length of the kernel side
 * @param padding: padding used to perform the cross-correlation
 * @param stride: stride used to perform the cross-correlation
 */
void avg_pooling(const matrix2d_t *const mat, const matrix2d_t *const result,
                 int kernel_size, int padding, int stride);

/**
 * @brief Compute the cross entropy loss of a 2D input matrix, using a
 * target 2D matrix.
 * @param m: pointer to the input matrix
 * @param target: pointer to the target matrix
 * @return the cross entropy loss
 */
float cross_entropy_loss(const matrix2d_t *const m,
                         const matrix2d_t *const target);

/**
 * @brief Compute the mean squared error of a 2D input matrix, using a
 * target 2D matrix.
 * @param m: pointer to the input matrix
 * @param target: pointer to the target matrix
 * @return the cross entropy loss
 */
float mean_squared_error(const matrix2d_t *const output,
                         const matrix2d_t *const target_output);

/**
 * @brief Compute the derivative of a 2D input matrix w.r.t. the cross entropy loss
 * @param m: pointer to the input matrix
 * @param target: pointer to the target matrix
 * @param result: pointer to the derivative matrix
 * @return the cross entropy loss
 */
void cross_entropy_loss_derivative(const matrix2d_t *const m,
                                   const matrix2d_t *const target,
                                   const matrix2d_t *const result);
/**
 * @brief Compute the derivative of a 2D input matrix w.r.t. the mean squared error
 * @param m: pointer to the input matrix
 * @param target: pointer to the target matrix
 * @param result: pointer to the derivative matrix
 * @return the cross entropy loss
 */
void mean_squared_error_derivative(const matrix2d_t *const m,
                                   const matrix2d_t *const target,
                                   const matrix2d_t *const result);

// void parse_line(char *line, int length, matrix2d_t *image, float *label);

#endif