// #include <stdlib.h>
#include <stdio.h>

#include "tensor.h"


int main() {

    puts("Sanity test");

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

    tensor_matmul(target_tensor, first_tensor, second_tensor);
    tensor_print(target_tensor);

    printf("tensor_iter Test: %f\n", tensor_iter(target_tensor, 3));

    Tensor *rnd_tensor = tensor_rinit(2, 3);
    tensor_print(rnd_tensor);

    tensor_destroy(first_tensor);
    tensor_destroy(second_tensor);
    tensor_destroy(target_tensor);

    return EXIT_SUCCESS;
}
