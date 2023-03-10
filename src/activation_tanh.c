#include "activation_tanh.h"
#include "tensor.h"
#include <math.h>
#include <stdlib.h>

Activation_Tanh *activation_tanh_init(void) {
    Activation_Tanh *activation_tanh = malloc(sizeof(Activation_Tanh));

    activation_tanh->inputs = NULL;
    activation_tanh->output = NULL;
    activation_tanh->dinputs = NULL;

    return activation_tanh;
}

int activation_tanh_destroy(Activation_Tanh *activation_tanh) {
    if (activation_tanh != NULL)
        free(activation_tanh);

    return EXIT_SUCCESS;
}

int activation_tanh_forward(Activation_Tanh *activation_tanh, Tensor *inputs) {
    if (inputs == NULL)
        return ETENNULL;

    if (activation_tanh->inputs == NULL) {
        activation_tanh->inputs = inputs;
    } else if (activation_tanh->inputs != NULL && activation_tanh->inputs != inputs) {
        tensor_destroy(activation_tanh->inputs);
        activation_tanh->inputs = inputs;
    }

    if (activation_tanh->output != NULL) {
        tensor_destroy(activation_tanh->output);
    }

    activation_tanh->output = activation_tanh_tanh(inputs);

    return EXIT_SUCCESS;
}

int activation_tanh_backward(Activation_Tanh *activation_tanh, Tensor *dvalues) {
    if (dvalues == NULL)
        return ETENNULL;

    if (activation_tanh->dinputs != NULL) {
        tensor_destroy(activation_tanh->dinputs);
    }
    //TODO: Do this better?
    Tensor *temp_tensor = activation_tanh_tanh(activation_tanh->inputs);
    tensor_pow(temp_tensor, temp_tensor, 2);
    Tensor *dinputs = tensor_init(temp_tensor->dim[0], temp_tensor->dim[1], 1);
    tensor_sub(dinputs, dinputs, temp_tensor);
    tensor_mult(dinputs, dinputs, dvalues);
    activation_tanh->dinputs = dinputs;

    tensor_destroy(temp_tensor);

    return EXIT_SUCCESS;
}

Tensor *activation_tanh_tanh(Tensor *inputs) {
    Tensor *tensor = tensor_init(inputs->dim[0], inputs->dim[1], 0);

    for (unsigned ir = 0; ir < tensor->dim[0]; ir++) {
        for (unsigned ic = 0; ic < tensor->dim[1]; ic++) {
            tensor->matrix[ir][ic] = tanh(inputs->matrix[ir][ic]);
        }
    }

    return tensor;
}
