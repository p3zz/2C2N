#ifndef __UTILS_H__
#define __UTILS_H__

#include "stdint.h"

#define SUCCESS 0
#define ERR 1

typedef enum {
  ACTIVATION_TYPE_RELU,
  ACTIVATION_TYPE_SIGMOID,
  ACTIVATION_TYPE_TANH,
  ACTIVATION_TYPE_IDENTITY,
} activation_type;

typedef float (*activation_function)(float);

float sigmoid(float x);
float sigmoid_derivative(float x);

float relu(float x);
float relu_derivative(float x);

float gradient_descent(float x, float rate, float dx);
float generate_random(void);
float activate(float x, activation_type type);
float d_activate(float x, activation_type type);
void compute_output_size(int input_height, int input_width, int kernel_size,
                         int padding, int stride, int *output_width,
                         int *output_height);
uint32_t quantize_f32_to_u32(float x);

#endif