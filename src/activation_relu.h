#ifndef ACTIVATION_RELU_H
#define ACTIVATION_RELU_H 1

#include "tensor.h"

typedef struct Activation_ReLU Activation_ReLU;

struct Activation_ReLU {
    /*
     * Forward pass
     */
    Tensor *inputs;
    Tensor *output;

    /*
     * Backward pass
     */
    Tensor *dinputs;

};

Activation_ReLU *activation_relu_init(void);

int activation_relu_destroy(Activation_ReLU *activation_relu);

int activation_relu_forward(Activation_ReLU *activation_relu, Tensor *inputs);
int activation_relu_backward(Activation_ReLU *activation_relu, Tensor *dvalues);

#endif
