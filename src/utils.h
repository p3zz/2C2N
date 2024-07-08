#ifndef UTILS_H
#define UTILS_H

#define SUCCESS 0
#define ERR 1

float sigmoid(float x);
float sigmoid_derivative(float x);

float relu(float x);
float relu_derivative(float x);

float update_output(float input, float weight, float bias);
float update_weight(float weight, float learning_rate, float correction);
float update_bias(float bias, float learning_rate, float correction);

#endif