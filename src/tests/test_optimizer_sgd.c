#include "test_utils.h"

#include "../tensor.h"
#include "../layer_dense.h"
#include "../optimizer_sgd.h"
#include <assert.h>
#include <stdio.h>

int test_basic(void) {
    MSG_START;

    Optimizer_SGD *optim = optimizer_sgd_init(1.0, 0.);
    assert(optim->learning_rate == 1.0);
    assert(optim->current_learning_rate == 1.0);
    assert(optim->decay == 0.0);
    assert(optim->iterations == 0);

    optimizer_sgd_destroy(optim);
    MSG_STOP;
    return EXIT_SUCCESS;
}

int test_pre_update_params(void) {
    MSG_START;

    Optimizer_SGD *optim = optimizer_sgd_init(1.2, 0.5);
    assert(optim->current_learning_rate == 1.2);

    optim->iterations = 4;
    optimizer_sgd_pre_update_params(optim);
    assert((optim->current_learning_rate - 0.4) < 1e-3);

    optimizer_sgd_destroy(optim);
    MSG_STOP;
    return EXIT_SUCCESS;
}

int test_update_params(void) {
    MSG_START;

    Optimizer_SGD *optim = optimizer_sgd_init(1.2, 0.0);
    Layer_Dense *test_layer = layer_dense_init(1, 2);
    test_layer->weights->matrix[0][0] = 2;
    test_layer->weights->matrix[0][1] = 3;
    test_layer->biases->matrix[0][0] = 4;
    test_layer->biases->matrix[0][1] = 5;
    test_layer->dweights = tensor_like(test_layer->weights, 0);
    test_layer->dweights->matrix[0][0] = 6;
    test_layer->dweights->matrix[0][1] = 6;
    test_layer->dbiases = tensor_like(test_layer->biases, 0);
    test_layer->dbiases->matrix[0][0] = 4;
    test_layer->dbiases->matrix[0][1] = 4;

    Tensor *result_weights_tensor = tensor_like(test_layer->weights, 0);
    result_weights_tensor->matrix[0][0] = -5.2;
    result_weights_tensor->matrix[0][1] = -4.2;
    Tensor *result_biases_tensor = tensor_like(test_layer->biases, 0);
    result_biases_tensor->matrix[0][0] = -0.8;
    result_biases_tensor->matrix[0][1] = 0.2;

    optimizer_sgd_update_params(optim, test_layer);
    if (PRINT) {
        COMMENT("Weights tensor and ground truth");
        tensor_print(test_layer->weights);
        tensor_print(result_weights_tensor);
        COMMENT("Biases tensor and ground truth");
        tensor_print(test_layer->biases);
        tensor_print(result_biases_tensor);
    }
    assert(tensor_cmp(test_layer->weights, result_weights_tensor) == true);
    assert(tensor_cmp(test_layer->biases, result_biases_tensor) == true);


    optimizer_sgd_destroy(optim);
    layer_dense_destroy(test_layer);
    tensor_destroy(result_weights_tensor);
    tensor_destroy(result_biases_tensor);
    MSG_STOP;
    return EXIT_SUCCESS;
}

int main(void) {
    test_basic();
    test_pre_update_params();
    test_update_params();

    printf("Optimizer SGD Test success!\n\n");
}
