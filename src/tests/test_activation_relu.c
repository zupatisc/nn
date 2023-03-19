#include <stdio.h>
#include <assert.h>

#include "test_utils.h"
#include "../activation_relu.h"


int test_basic(void) {
    MSG_START;

    Activation_ReLU *test_relu = activation_relu_init();
    assert(test_relu != NULL);

    assert(test_relu->inputs == NULL);
    assert(test_relu->output == NULL);
    assert(test_relu->dinputs == NULL);

    activation_relu_destroy(test_relu);

    MSG_STOP;
    return EXIT_SUCCESS;
}

int test_forward(void) {
    MSG_START;

    Activation_ReLU *test_relu = activation_relu_init();
    Tensor *test_tensor = tensor_init(3, 2, 2);

    activation_relu_forward(test_relu, test_tensor);
    assert(test_relu->inputs == test_tensor);
    assert(tensor_cmp(test_relu->output, test_tensor) == true);

    test_tensor->matrix[1][0] = -4;
    test_tensor->matrix[2][1] = 0;

    Tensor *result_tensor = tensor_like(test_tensor, 2);
    result_tensor->matrix[1][0] = 0;
    result_tensor->matrix[2][1] = 0;

    activation_relu_forward(test_relu, test_tensor);
    assert(tensor_cmp(test_relu->output, result_tensor) == true);

    activation_relu_destroy(test_relu);
    tensor_destroy(test_tensor);
    tensor_destroy(result_tensor);

    MSG_STOP;
    return EXIT_SUCCESS;
}

int test_backward(void) {
    MSG_START;

    Activation_ReLU *test_relu = activation_relu_init();
    Tensor *input_tensor = tensor_init(3, 3, 2);
    input_tensor->matrix[0][0] = -4;
    input_tensor->matrix[1][1] = -3;
    input_tensor->matrix[2][2] = -7;

    Tensor *dvalues_tensor = tensor_like(input_tensor, 20);
    Tensor *result_tensor = tensor_like(dvalues_tensor, 20);
    result_tensor->matrix[0][0] = 0;
    result_tensor->matrix[1][1] = 0;
    result_tensor->matrix[2][2] = 0;

    if (PRINT) {
        puts("input_tensor:");
        tensor_print(input_tensor);
        puts("dvalues_tensor:");
        tensor_print(dvalues_tensor);
        puts("result_tensor:");
        tensor_print(result_tensor);
    }

    activation_relu_forward(test_relu, input_tensor);
    activation_relu_backward(test_relu, dvalues_tensor);
    if (PRINT) tensor_print(test_relu->dinputs);
    assert(tensor_cmp(test_relu->dinputs, result_tensor) == true);

    activation_relu_destroy(test_relu);
    tensor_destroy(input_tensor);
    tensor_destroy(dvalues_tensor);
    tensor_destroy(result_tensor);

    MSG_STOP;
    return EXIT_SUCCESS;
}

int main(void) {
    test_basic();
    test_forward();
    test_backward();

    printf("Activation_relu Test success!\n\n");
    return EXIT_SUCCESS;
}
