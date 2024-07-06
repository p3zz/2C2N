#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"

typedef struct{
    layer_t* layers;
    int layers_num;
} network_t;

#endif