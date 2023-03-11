#include <stdio.h>
#include <assert.h>

#include "test_utils.h"
#include "../layer_dense.h"

static int test_layer_dense(void) {
    MSG_START;

    Layer_Dense *test_layer = layer_dense_init(1, 8);

    // tensor_print(test_layer->weights);
    assert(test_layer->weights->dim[0] == 1);
    assert(test_layer->weights->dim[1] == 8);

    // tensor_print(test_layer->biases);
    Tensor *biases_reference = tensor_init(1, 8, 0);
    assert(tensor_cmp(test_layer->biases, biases_reference));


    Tensor *test_tensor = tensor_init(4, 1, 1);
    // tensor_print(test_tensor);

    assert(layer_dense_forward(test_layer, test_tensor) == EXIT_SUCCESS);

    layer_dense_destroy(test_layer);

    tensor_destroy(biases_reference);
    tensor_destroy(test_tensor);

    MSG_STOP;
    return EXIT_SUCCESS;
}

static int double_forward_test(void) {
    MSG_START;

    Layer_Dense *test_layer = layer_dense_init(2, 5);

    test_layer->weights->matrix[0][0] = 1;
    test_layer->weights->matrix[0][1] = 1;
    test_layer->weights->matrix[0][2] = 1;
    test_layer->weights->matrix[0][3] = 1;
    test_layer->weights->matrix[0][4] = 1;
    test_layer->weights->matrix[1][0] = 1;
    test_layer->weights->matrix[1][1] = 1;
    test_layer->weights->matrix[1][2] = 1;
    test_layer->weights->matrix[1][3] = 1;
    test_layer->weights->matrix[1][4] = 1;

    assert(test_layer->weights->dim[0] == 2);
    assert(test_layer->weights->dim[1] == 5);
    assert(test_layer->biases->dim[0] == 1);
    assert(test_layer->biases->dim[1] == 5);

    assert(test_layer->output == NULL);
    assert(test_layer->inputs == NULL);
    assert(test_layer->dinputs == NULL);
    assert(test_layer->dvalues == NULL);
    assert(test_layer->dbiases == NULL);
    assert(test_layer->dweights == NULL);

    /*
     * Test the first forward pass
     */
    Tensor *input_tensor = tensor_init(8, 2, 3);
    Tensor *reference_output = tensor_init(8, 5, 6);
    assert(layer_dense_forward(test_layer, input_tensor) == EXIT_SUCCESS);

    assert(test_layer->output != NULL);
    assert(test_layer->output->dim[0] == 8);
    assert(test_layer->output->dim[1] == 5);
    assert(tensor_cmp(test_layer->output, reference_output) == true);

    assert(test_layer->inputs != NULL);
    assert(test_layer->inputs == test_layer->inputs);
    assert(test_layer->dinputs == NULL);
    assert(test_layer->dvalues == NULL);
    assert(test_layer->dbiases == NULL);
    assert(test_layer->dweights == NULL);

    /*
     * Test the second forward pass but with an input of the same shape
     */
    Tensor *input_tensor_2 = tensor_init(8, 2, 2);
    Tensor *reference_output_2 = tensor_init(8, 5, 4);
    assert(layer_dense_forward(test_layer, input_tensor_2) == EXIT_SUCCESS);

    assert(test_layer->output != NULL);
    assert(test_layer->output->dim[0] == 8);
    assert(test_layer->output->dim[1] == 5);
    assert(tensor_cmp(test_layer->output, reference_output_2) == true);

    assert(test_layer->inputs != NULL);
    assert(test_layer->inputs == test_layer->inputs);
    assert(test_layer->dinputs == NULL);
    assert(test_layer->dvalues == NULL);
    assert(test_layer->dbiases == NULL);
    assert(test_layer->dweights == NULL);

    /*
     * Test the third forward pass but this time with an input of a different shape
     */
    Tensor *input_tensor_3 = tensor_init(4, 2, 2);
    Tensor *reference_output_3 = tensor_init(4, 5, 4);
    assert(layer_dense_forward(test_layer, input_tensor_3) == EXIT_SUCCESS);

    assert(test_layer->output != NULL);
    assert(test_layer->output->dim[0] == 4);
    assert(test_layer->output->dim[1] == 5);
    assert(tensor_cmp(test_layer->output, reference_output_3) == true);

    assert(test_layer->inputs != NULL);
    assert(test_layer->inputs == test_layer->inputs);
    assert(test_layer->dinputs == NULL);
    assert(test_layer->dvalues == NULL);
    assert(test_layer->dbiases == NULL);
    assert(test_layer->dweights == NULL);


    layer_dense_destroy(test_layer);
    tensor_destroy(reference_output);
    tensor_destroy(reference_output_2);
    tensor_destroy(reference_output_3);

    MSG_STOP;
    return EXIT_SUCCESS;
}

static int backward_test(void) {
    MSG_START;
    // TODO: Test Backward function
    MSG_STOP;
    return EXIT_SUCCESS;
}

int main(void) {
    test_layer_dense();
    double_forward_test();
    backward_test();

    printf("Layer_Dense Test success!\n\n");

    return EXIT_SUCCESS;
}
