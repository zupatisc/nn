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

void layer_dense_forward(Layer_Dense *layer_dense, Tensor *inputs) {
    if (layer_dense->weights == NULL || layer_dense->biases == NULL)
        printf("Panic"); //TODO: Fail better
    if (inputs == NULL)
        printf("Also panic"); //TODO: Fail better

    //Since we don't know the size of output before getting the inputs we need
    //to creat it now.
    layer_dense->output = tensor_init(inputs->dim[0], layer_dense->weights->dim[1], 0);
    tensor_print(layer_dense->output);

    layer_dense->inputs = inputs;
    if(tensor_dot(layer_dense->output, inputs, layer_dense->weights))
        printf("Panic");
    tensor_print(layer_dense->output);
    if(tensor_add(layer_dense->output, layer_dense->output, layer_dense->biases))
        printf("Panic");
    tensor_print(layer_dense->output);

    return;
}

void layer_dense_backward(Layer_Dense *layer_dense, Tensor *dvalues) {

}
