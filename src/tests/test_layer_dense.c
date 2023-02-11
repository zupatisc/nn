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
    tensor_print(test_tensor);

    assert(layer_dense_forward(test_layer, test_tensor) == EXIT_SUCCESS);

    layer_dense_destroy(test_layer);

    tensor_destroy(biases_reference);
    // tensor_destroy(test_tensor);

    MSG_STOP;
    return EXIT_SUCCESS;
}

int main(void) {
    test_layer_dense();

    printf("Layer_Dense Test success!\n");

    return EXIT_SUCCESS;
}
