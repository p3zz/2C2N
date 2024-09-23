#include "utils.h"
#include "float.h"
#include "math.h"
#include "stdlib.h"

float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

float sigmoid_derivative(float x) {
  float res = sigmoid(x);
  return res * (1 - res);
}

float relu(float x) {
  if (x < 0) {
    return 0;
  } else {
    return x;
  }
}

float relu_derivative(float x) {
  if (x < 0) {
    return 0;
  } else {
    return 1;
  }
}

float tanh_derivative(float x) {
  float res = tanh(x);
  return 1.0 - res * res;
}

float activate(float x, activation_type type) {
  float result = x;
  switch (type) {
  case ACTIVATION_TYPE_RELU:
    result = relu(x);
    break;
  case ACTIVATION_TYPE_SIGMOID:
    result = sigmoid(x);
    break;
  case ACTIVATION_TYPE_TANH:
    result = tanhf(x);
  default:
    break;
  }
  return result;
}

float d_activate(float x, activation_type type) {
  float result = x;
  switch (type) {
  case ACTIVATION_TYPE_RELU:
    result = relu_derivative(x);
    break;
  case ACTIVATION_TYPE_SIGMOID:
    result = sigmoid_derivative(x);
    break;
  case ACTIVATION_TYPE_TANH:
    result = tanh_derivative(x);
    break;
  case ACTIVATION_TYPE_IDENTITY:
    result = 1;
    break;
  default:
    break;
  }
  return result;
}

float gradient_descent(float x, float rate, float dx) {
  return x - (rate * dx);
}

float generate_random(void) { return ((float)rand()) / ((float)RAND_MAX); }

void compute_output_size(int input_height, int input_width, int kernel_size,
                         int padding, int stride, int *output_height,
                         int *output_width) {
  *output_height = (input_height - kernel_size + 2 * padding) / stride + 1;
  *output_width = (input_width - kernel_size + 2 * padding) / stride + 1;
}

uint32_t quantize_f32_to_u32(float x) {
  if (x > 0.f) {
    return (x * UINT32_MAX) / FLT_MAX;
  } else {
    return 0u;
  }
}
