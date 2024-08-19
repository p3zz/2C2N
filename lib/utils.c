#include <math.h>
#include "utils.h"
#include <stdlib.h>

float sigmoid(float x){
    return 1.f / (1.f + expf(-x));
}

float sigmoid_derivative(float x){
    return x * (1-x);
}

float relu(float x){
    if(x < 0){
        return 0;
    }
    else{
        return x;
    }
}

float relu_derivative(float x){
    if(x < 0){
        return 0;
    }
    else{
        return 1;
    }
}

float update_output(float input, float weight, float bias){
    return input * weight + bias;
}

float gradient_descent(float x, float rate, float dx){
    return x - (rate * dx);
}

float generate_random(void){
    return ((float)rand())/((float)RAND_MAX);
}