/*
 * This header defines the functions pertaining to the creation
 * and handling of a dense layer for an artifical neural network.
 */
#ifndef LAYER_DENSE_H
#define LAYER_DENSE_H 1

#include "tensor.h"

typedef struct Layer_Dense Layer_Dense;

struct Layer_Dense {
    /*
     * Forward pass
     */
    Tensor *weights;
    Tensor *biases;
    Tensor *output;

    /*
     * Backward pass
     */
    Tensor *inputs;
    Tensor *dinputs;
    Tensor *dvalues;
    Tensor *dweights;
};

//Init function
Layer_Dense *layer_dense_init(unsigned n_inputs, unsigned n_neurons);
//forward pass
void layer_dense_forward(Layer_Dense *layer_dense, Tensor *inputs);
//backward pass
void layer_dense_backward(Layer_Dense *layer_dense, Tensor *dvalues);
#endif
