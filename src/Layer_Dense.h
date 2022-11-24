/*
 * This header defines the functions pertaining to the creation
 * and handling of a dense layer for an artifical neural network.
 */
#ifndef LAYER_DENSE_H
#define LAYER_DENSE_H 1

#include <stdbool.h>

typedef struct layer_dense layer_dense;

struct layer_dense {
    unsigned n_inputs;
    unsigned n_neurons;

    double weights;
    double biases;

    double inputs;
    double output;

    // Gradients on parameters
    double dweights;
    double dbiases;
    // Gradients on input values
    double dinputs;
};

// Initialize weights and biases
layer_dense layer_dense_init(unsigned n_inputs, unsigned n_neurons);
// Forward pass
bool layer_dense_forward(layer_dense* layer, double* inputs);
// Backward pass
bool layer_dense_backward(layer_dense* layer, double* dvalues);

#endif
