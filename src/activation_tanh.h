#ifndef ACTIVATION_TANH_H
#define ACTIVATION_TANH_H 1

#include "tensor.h"

typedef struct Activation_Tanh Activation_Tanh;

struct Activation_Tanh {
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

Activation_Tanh *activation_tanh_init(void);

int activation_tanh_destroy(Activation_Tanh *activation_tanh);

int activation_tanh_forward(Activation_Tanh *activation_tanh, Tensor *inputs);
int activation_tanh_backward(Activation_Tanh *activation_tanh, Tensor *dvalues);

Tensor *activation_tanh_tanh(Tensor *inputs);
#endif
