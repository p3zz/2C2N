#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"

typedef struct{
    layer_t* layers;
    int layers_num;
} network_t;

int create_network(int layers_num, const int* neurons_per_layer);

#endif