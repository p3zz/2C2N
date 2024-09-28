/**
 * @brief This library provides useful functions used in Convolutional Neural
 * Networks (CNNs), such as activation functions and their derivative, the
 * gradient descent computations and a float number generator
 */

#ifndef __UTILS_H__
#define __UTILS_H__

/**
 * @enum activation_type
 * @brief this enum contains a list of well known activation functions
 */
typedef enum {
  ACTIVATION_TYPE_RELU,
  ACTIVATION_TYPE_SIGMOID,
  ACTIVATION_TYPE_TANH,
  ACTIVATION_TYPE_IDENTITY,
} activation_type;

/**
 * @enum pooling_type
 * @brief this enum contains a list of well known pooling functions
 */
typedef enum { POOLING_TYPE_AVERAGE, POOLING_TYPE_MAX } pooling_type;

/**
 * @brief Compute the output of the sigmoid function.
 * @param x: the input value
 * @return the result
 */
float sigmoid(float x);

/**
 * @brief Compute the output of the derivative of the sigmoid function w.r.t.
 * the input.
 * @param x: the input value
 * @return the result
 */
float sigmoid_derivative(float x);

/**
 * @brief Compute the output of the ReLU function.
 * @param x: the input value
 * @return the result
 */
float relu(float x);

/**
 * @brief Compute the output of the derivative of the ReLU function w.r.t. the
 * input.
 * @param x: the input value
 * @return the result
 */
float relu_derivative(float x);

/**
 * @brief Compute the gradient descent of an input value using the slope rate
 * and the error.
 * @param x: the input value
 * @param rate: the slope
 * @param dx: the error
 * @return the result
 */
float gradient_descent(float x, float rate, float dx);

/**
 * @brief Generate a random float value between 0 and 1
 * @return the result
 */
float generate_random(void);

/**
 * @brief Compute the activated value of the input using a given activation type
 * @param x: the input value
 * @param type: the activation type
 * @return the result
 */
float activate(float x, activation_type type);

/**
 * @brief Compute the activated value of the input using the derivative of a
 * given activation type w.r.t. the input
 * @param x: the input value
 * @param type: the activation type
 * @return the result
 */
float d_activate(float x, activation_type type);

/**
 * @brief Compute the height and width of a 2D matrix that represent the result
 * of a convolution
 * @param input_height: the height of the input matrix
 * @param input_width: the width of the input matrix
 * @param kernel_size: the length of the kernel side
 * @param padding: padding used to perform the convolution
 * @param stride: stride used to perform the convolution
 * @param output_height: the resulting height
 * @param output_width: the resulting width
 */
void compute_output_size(int input_height, int input_width, int kernel_size,
                         int padding, int stride, int *output_height,
                         int *output_width);

#endif