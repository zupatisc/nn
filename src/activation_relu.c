#include "activation_relu.h"
#include "tensor.h"


Activation_ReLU *activation_relu_init(void) {
    Activation_ReLU *activation_relu = malloc(sizeof(Activation_ReLU));

    activation_relu->inputs = NULL;
    activation_relu->output = NULL;
    activation_relu->dinputs = NULL;

    return activation_relu;
}

int activation_relu_destroy(Activation_ReLU *activation_relu) {
    tensor_destroy(activation_relu->output);
    tensor_destroy(activation_relu->dinputs);

    if (activation_relu)
        free(activation_relu);

    return EXIT_SUCCESS;
}

int activation_relu_forward(Activation_ReLU *activation_relu, Tensor *inputs) {
    activation_relu->inputs = inputs;

    if (!activation_relu->output) {
        activation_relu->output = tensor_like(inputs, 0);
    } else if (activation_relu->output->dim[0] != inputs->dim[0] || activation_relu->output->dim[1] != inputs->dim[1]) {
        tensor_destroy(activation_relu->output);
        activation_relu->output = tensor_like(inputs, 0);
    }

    for (unsigned ic = 0; ic < inputs->dim[0]; ic++) {
        for (unsigned ir = 0; ir < inputs->dim[1]; ir++) {
            activation_relu->output->matrix[ic][ir] = fmax(0., inputs->matrix[ic][ir]);
        }
    }

    return EXIT_SUCCESS;
}

int activation_relu_backward(Activation_ReLU *activation_relu, Tensor *dvalues) {
    if (!activation_relu->dinputs) {
        activation_relu->dinputs = tensor_like(dvalues, 0);
    } else if (activation_relu->dinputs->dim[0] != dvalues->dim[0] || activation_relu->dinputs->dim[1] != dvalues->dim[1]) {
        tensor_destroy(activation_relu->dinputs);
        activation_relu->dinputs = tensor_like(dvalues, 0);
    }

    for (unsigned ic = 0; ic < dvalues->dim[0]; ic++) {
        for (unsigned ir = 0; ir < dvalues->dim[1]; ir++) {
            if (activation_relu->inputs->matrix[ic][ir] > 0)
                activation_relu->dinputs->matrix[ic][ir] = dvalues->matrix[ic][ir];
        }
    }

    return EXIT_SUCCESS;
}
