#include "layer_dense.h"
#include <stdio.h>
#include <stdlib.h>

Layer_Dense *layer_dense_init(unsigned int n_inputs, unsigned int n_neurons) {
    Layer_Dense *layer_denseptr = malloc(sizeof(Layer_Dense));

    layer_denseptr->weights = tensor_rinit(n_inputs, n_neurons);
    layer_denseptr->biases = tensor_init(1, n_neurons, 0);

    return layer_denseptr;
}

int layer_dense_destroy(Layer_Dense *layer_dense) {
    tensor_destroy(layer_dense->weights);
    tensor_destroy(layer_dense->biases);
    tensor_destroy(layer_dense->output);
    tensor_destroy(layer_dense->inputs);
    tensor_destroy(layer_dense->dinputs);
    tensor_destroy(layer_dense->dvalues);
    tensor_destroy(layer_dense->dweights);

    free(layer_dense);

    return EXIT_SUCCESS;
}

int layer_dense_forward(Layer_Dense *layer_dense, Tensor *inputs) {
    if (layer_dense->weights == NULL || layer_dense->biases == NULL)
        return ELDFNULL;
    if (inputs == NULL)
        return ELDFNULL;

    // Since we don't know the size of output before getting the inputs
    // we need to create it now.
    layer_dense->output = tensor_init(inputs->dim[0], layer_dense->weights->dim[1], 0);
    tensor_print(layer_dense->output); //TODO: Remove printing

    layer_dense->inputs = inputs;
    int rval = tensor_dot(layer_dense->output, inputs, layer_dense->weights);
    if (rval != EXIT_SUCCESS) {
        printf("Panic: tensor_dot - Error: %d\n", rval);
        return rval;
    }
    tensor_print(layer_dense->output); //TODO: Remove printing

    rval = tensor_add(layer_dense->output, layer_dense->output, layer_dense->biases);
    if (rval != EXIT_SUCCESS) {
        printf("Panic: tensor_add - Error: %d\n", rval);
        return rval;
    }
    tensor_print(layer_dense->output); //TODO: Remove printing

    return EXIT_SUCCESS;
}

int layer_dense_backward(Layer_Dense *layer_dense, Tensor *dvalues) {

    return EXIT_SUCCESS;
}
