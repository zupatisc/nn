/*
 * This header defines the functions pertaining to the creation
 * and handling of a dense layer for an artifical neural network.
 */
#ifndef LAYER_DENSE_H
#define LAYER_DENSE_H 1

#include "tensor.h"

typedef struct layer_dense layer_dense;

struct layer_dense {
    /*
     * Forward pass
     */
    tensor *weights;
    tensor *biases;
    tensor *output;

    /*
     * Backward pass
     */
    tensor *inputs;
    tensor *dinputs;
    tensor *dvalues;
    tensor *dweights;
};

//Init function
layer_dense *layer_dense_init(unsigned n_inputs, unsigned n_neurons);
//forward pass
void layer_dense_forward(layer_dense *layer_dense, tensor *inputs);
//backward pass
void layer_dense_backward(layer_dense *layer_dense, tensor *dvalues);
#endif
