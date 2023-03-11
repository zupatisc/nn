#include <stdio.h>
#include <assert.h>

#include "test_utils.h"
#include "../activation_tanh.h"

int test_forward(void) {
    MSG_START;

    Tensor *test_tensor = tensor_init(2, 2, 1);
    Activation_Tanh *test_tanh = activation_tanh_init();

    if (PRINT)
        tensor_print(test_tensor);

    assert(activation_tanh_forward(test_tanh, test_tensor) == EXIT_SUCCESS);

    assert(test_tanh->dinputs == NULL);
    assert(test_tanh->output != NULL);
    if (PRINT)
        tensor_print(test_tanh->output);
    assert(test_tanh->inputs != NULL);
    if (PRINT)
        tensor_print(test_tanh->inputs);

    Tensor *cmpr_tensor = tensor_init(2, 2, 0);
    /*
     * ([[0.76159416, 0.76159416],
     * [0.76159416, 0.76159416]])
     */
    cmpr_tensor->matrix[0][0] = 0.76159416;
    cmpr_tensor->matrix[0][1] = 0.76159416;
    cmpr_tensor->matrix[1][0] = 0.76159416;
    cmpr_tensor->matrix[1][1] = 0.76159416;

    assert(tensor_cmp(test_tanh->output, cmpr_tensor) == true);

    assert(test_tanh->inputs == test_tensor);

    tensor_destroy(test_tensor);
    tensor_destroy(cmpr_tensor);
    activation_tanh_destroy(test_tanh);

    MSG_STOP;
    return EXIT_SUCCESS;
}

int test_backward(void) {
    MSG_START;

    Tensor *input_tensor = tensor_init(2, 2, 3.8);
    Tensor *dvalues_tensor = tensor_init(2, 2, 2.3);
    Tensor *output_tensor = tensor_init(2, 2, 0);
    Tensor *dinputs_tensor = tensor_init(2, 2, 0);
    Activation_Tanh *test_activation = activation_tanh_init();

    if (PRINT) {
        tensor_print(input_tensor);
        tensor_print(dvalues_tensor);
    }
    /*
     * [[0.9989996 0.9989996]
     * [0.9989996 0.9989996]]
     */
    output_tensor->matrix[0][0] = 0.9989996;
    output_tensor->matrix[0][1] = 0.9989996;
    output_tensor->matrix[1][0] = 0.9989996;
    output_tensor->matrix[1][1] = 0.9989996;

    assert(activation_tanh_backward(test_activation, dvalues_tensor) == ETENNULL);

    assert(activation_tanh_forward(test_activation, input_tensor) == EXIT_SUCCESS);
    assert(tensor_cmp(test_activation->output, output_tensor) == true);

    /*
     * [[0.00459955 0.00459955]
     * [0.00459955 0.00459955]]
     */
    dinputs_tensor->matrix[0][0] = 0.00459955;
    dinputs_tensor->matrix[0][1] = 0.00459955;
    dinputs_tensor->matrix[1][0] = 0.00459955;
    dinputs_tensor->matrix[1][1] = 0.00459955;

    assert(activation_tanh_backward(test_activation, dvalues_tensor) == EXIT_SUCCESS);
    assert(tensor_cmp(test_activation->dinputs, dinputs_tensor) == true);

    tensor_destroy(input_tensor);
    tensor_destroy(dvalues_tensor);
    tensor_destroy(output_tensor);
    tensor_destroy(dinputs_tensor);
    activation_tanh_destroy(test_activation);
    MSG_STOP;
    return EXIT_SUCCESS;
}

int test_basic(void) {
    MSG_START;

    Activation_Tanh *test_tanh = activation_tanh_init();
    assert(test_tanh != NULL);

    assert(test_tanh->inputs == NULL);
    assert(test_tanh->output == NULL);
    assert(test_tanh->dinputs == NULL);

    activation_tanh_destroy(test_tanh);

    MSG_STOP;
    return EXIT_SUCCESS;
}

int main(void) {
    test_basic();
    test_forward();
    test_backward();

    printf("Activation_tanh Test success!\n\n");
    return EXIT_SUCCESS;
}
