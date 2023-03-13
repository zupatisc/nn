#include "layer_dense.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>

Layer_Dense *layer_dense_init(unsigned int n_inputs, unsigned int n_neurons) {
    Layer_Dense *layer_denseptr = malloc(sizeof(Layer_Dense));

    layer_denseptr->weights = tensor_rinit(n_inputs, n_neurons);
    layer_denseptr->biases = tensor_init(1, n_neurons, 0);
    layer_denseptr->output = NULL;

    /*
     * Backward pass
     */
    layer_denseptr->inputs = NULL;
    layer_denseptr->dinputs = NULL;
    layer_denseptr->dvalues = NULL;
    layer_denseptr->dbiases = NULL;
    layer_denseptr->dweights = NULL;

    return layer_denseptr;
}

int layer_dense_destroy(Layer_Dense *layer_dense) {
    tensor_destroy(layer_dense->weights);
    tensor_destroy(layer_dense->biases);
    tensor_destroy(layer_dense->output);
    // tensor_destroy(layer_dense->inputs);

    tensor_destroy(layer_dense->dinputs);
    tensor_destroy(layer_dense->dvalues);
    tensor_destroy(layer_dense->dweights);
    tensor_destroy(layer_dense->dbiases);

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
    if (layer_dense->output == NULL) {
        layer_dense->output = tensor_init(inputs->dim[0], layer_dense->weights->dim[1], 0);
    } else if (layer_dense->output->dim[0] != inputs->dim[0]) {
        tensor_destroy(layer_dense->output);
        layer_dense->output = tensor_init(inputs->dim[0], layer_dense->weights->dim[1], 0);
    }

    if (layer_dense->inputs != NULL && layer_dense->inputs != inputs) { // Clear tensor if not NULL
        tensor_destroy(layer_dense->inputs);
        layer_dense->inputs = inputs;
    } else if (layer_dense->inputs == NULL) {
        layer_dense->inputs = inputs;
    }

    int rval = tensor_dot(layer_dense->output, inputs, layer_dense->weights);
    if (rval != EXIT_SUCCESS) {
        printf("Panic: tensor_dot - Error: %d\n", rval);
        return rval;
    }

    rval = tensor_add(layer_dense->output, layer_dense->output, layer_dense->biases);
    if (rval != EXIT_SUCCESS) {
        printf("Panic: tensor_add - Error: %d\n", rval);
        return rval;
    }

    return EXIT_SUCCESS;
}

int layer_dense_backward(Layer_Dense *layer_dense, Tensor *dvalues) {
    /* Compute the transpose here as maintaining an up to date copy would be
     * more of a headache than the speedup is worth
     */
    Tensor *inputs_transp = tensor_transpose(layer_dense->inputs);
    Tensor *weights_transp = tensor_transpose(layer_dense->weights);

    if (layer_dense->dweights == NULL)
        layer_dense->dweights = tensor_like(layer_dense->weights, 0);
    if (layer_dense->dbiases == NULL)
        layer_dense->dbiases = tensor_like(layer_dense->biases, 0);
    if (layer_dense->dinputs == NULL)
        layer_dense->dinputs = tensor_like(layer_dense->inputs, 0);

    tensor_dot(layer_dense->dweights, inputs_transp, dvalues);
    tensor_sum(layer_dense->dbiases, dvalues, 0);
    tensor_dot(layer_dense->dinputs, dvalues, weights_transp);

    tensor_destroy(inputs_transp);
    tensor_destroy(weights_transp);

    return EXIT_SUCCESS;
}
