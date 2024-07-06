#include <math.h>

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

float update_weight(float weight, float learning_rate, float correction){
    return weight - (learning_rate * correction);
}

float update_bias(float bias, float learning_rate, float correction){
    return update_weight(bias, learning_rate, correction);
}
