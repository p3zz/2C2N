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

float update_output(float input, float weight, float bias);

float gradient_descent(float x, float rate, float dx);
float generate_random(void);
float activate(float x, activation_type type);
float d_activate(float x, activation_type type);

#endif