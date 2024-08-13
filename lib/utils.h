#ifndef UTILS_H
#define UTILS_H

#define SUCCESS 0
#define ERR 1

typedef float(*activation_function)(float);

float sigmoid(float x);
float sigmoid_derivative(float x);

float relu(float x);
float relu_derivative(float x);

float update_output(float input, float weight, float bias);

float gradient_descent(float x, float rate, float dx);

#endif