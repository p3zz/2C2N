#ifndef UTILS_H
#define UTILS_H

#define SUCCESS 0
#define ERR 1

typedef enum{
	ACTIVATION_TYPE_RELU,
	ACTIVATION_TYPE_SIGMOID
}activation_type;

typedef float(*activation_function)(float);

float sigmoid(float x);
float sigmoid_derivative(float x);

float relu(float x);
float relu_derivative(float x);

float gradient_descent(float x, float rate, float dx);
float generate_random(void);
float activate(float x, activation_type type);
float d_activate(float x, activation_type type);
void compute_output_size(int input_height, int input_width, int kernel_size, int padding, int stride, int* output_width, int* output_height);

#endif