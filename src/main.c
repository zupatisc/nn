// #include <stdlib.h>
#include <stdio.h>

#include "tensor.h"
#include "layer_dense.h"


static int test_layer_dense(void) {

    Layer_Dense *test_layer = layer_dense_init(1, 8);
    tensor_print(test_layer->weights);
    tensor_print(test_layer->biases);

    Tensor *test_tensor = tensor_init(4, 1, 1);
    tensor_print(test_tensor);

    /* if(tensor_dot(test_layer->output, test_tensor, test_layer->weights))
        printf("EXIT FAILURE");
    tensor_print(test_layer->output); */
    layer_dense_forward(test_layer, test_tensor);

    layer_dense_destroy(test_layer);
    tensor_destroy(test_tensor);

    printf("Layer Test success!\n");
    return EXIT_SUCCESS;
}

int main(void) {

    puts("Sanity test");

    test_layer_dense();


    return EXIT_SUCCESS;
}
