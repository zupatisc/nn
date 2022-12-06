// #include <stdlib.h>
#include <stdio.h>

#include "tensor.h"
#include "layer_dense.h"

int test_tensors() {
    /*
     * Very basic tensor test
     */
    int row = 8, col = 4;

    Tensor *test_tensor = tensor_init(row, col, 0);

    printf("Dims: (%d, %d)\n", test_tensor->dim[0], test_tensor->dim[1]);

    tensor_print(test_tensor);

    tensor_destroy(test_tensor);

    /*
     * Matrix multiplication test
     */
    Tensor *first_tensor = tensor_init(2, 3, 1);
    Tensor *second_tensor = tensor_init(3, 2, 1);
    Tensor *target_tensor = tensor_init(2, 2, 0);

    first_tensor->matrix[0][1] = 2;
    first_tensor->matrix[0][2] = 3;
    first_tensor->matrix[1][0] = 4;
    first_tensor->matrix[1][1] = 5;
    first_tensor->matrix[1][2] = 6;

    second_tensor->matrix[0][0] = 3;
    second_tensor->matrix[0][1] = 4;
    second_tensor->matrix[1][0] = 7;
    second_tensor->matrix[1][1] = 8;
    second_tensor->matrix[2][0] = 9;
    second_tensor->matrix[2][1] = 10;

    tensor_dot(target_tensor, first_tensor, second_tensor);
    tensor_print(target_tensor);

    printf("tensor_iter Test: %f\n", tensor_iter(target_tensor, 3));

    puts("\n");
    Tensor *rnd_tensor = tensor_rinit(2, 3);
    tensor_print(rnd_tensor);

    tensor_destroy(first_tensor);
    tensor_destroy(second_tensor);
    tensor_destroy(target_tensor);
    tensor_destroy(rnd_tensor);


    printf("Tensor Test success!\n");
    return EXIT_SUCCESS;
}

int test_layer_dense() {

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

int main() {

    puts("Sanity test");

    // test_tensors();
    test_layer_dense();


    return EXIT_SUCCESS;
}
