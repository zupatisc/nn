#include "optimizer_sgd.h"
#include <stdio.h>

Optimizer_SGD *optimizer_sgd_init(double learning_rate, double decay) {
    Optimizer_SGD *optim = malloc(sizeof(Optimizer_SGD));

    optim->learning_rate = learning_rate;
    optim->current_learning_rate = learning_rate;
    optim->decay = decay;
    optim->iterations = 0;

    return optim;
}

void optimizer_sgd_destroy(Optimizer_SGD *optim) {
    free(optim);
}

int optimizer_sgd_pre_update_params(Optimizer_SGD *optim) {
    if (!optim)
        return EXIT_FAILURE;

    if (optim->decay)
        optim->current_learning_rate = optim->learning_rate * (1. / (1. + optim->decay * optim->iterations));

    return EXIT_SUCCESS;
}

int optimizer_sgd_update_params(Optimizer_SGD *optim, Layer_Dense *layer_dense) {
    if (!optim || !layer_dense)
        return EXIT_FAILURE;

    double scol = {-optim->current_learning_rate};
    double *scolp = &scol;
    double **lr_matrix = {&scolp};
    Tensor lr_tensor = {
        .dim = {1, 1},
        .matrix = lr_matrix,
    };

    if(tensor_mult(layer_dense->dweights, layer_dense->dweights, &lr_tensor) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    if(tensor_add(layer_dense->weights, layer_dense->weights, layer_dense->dweights) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    if(tensor_mult(layer_dense->dbiases, layer_dense->dbiases, &lr_tensor) != EXIT_SUCCESS)
        return EXIT_FAILURE;
    if(tensor_add(layer_dense->biases, layer_dense->biases, layer_dense->dbiases) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}

int optimizer_sgd_post_update_params(Optimizer_SGD *optim) {
    if (!optim)
        return EXIT_FAILURE;

    optim->iterations++;
    return EXIT_SUCCESS;
}
