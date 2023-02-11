#include <stdio.h>
#include <assert.h>

#include "test_utils.h"
#include "../tensor.h"

static int basic_test(void) {
    /*
     * Very basic tensor test
     */
    MSG_START;
    unsigned row = 8, col = 4;
    Tensor *test_tensor = tensor_init(row, col, 0);

    // printf("Dims: (%d, %d)\n", test_tensor->dim[0], test_tensor->dim[1]);
    assert(test_tensor->dim[0] == row);
    assert(test_tensor->dim[1] == col);
    // tensor_print(test_tensor);

    tensor_destroy(test_tensor);

    MSG_STOP;
    return EXIT_SUCCESS;
}

static int dotproduct_test(void) {
    /*
     * Matrix multiplication test
     */
    MSG_START;
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

    assert(tensor_dot(target_tensor, first_tensor, second_tensor) == EXIT_SUCCESS);

    // tensor_print(target_tensor);
    assert((int)target_tensor->matrix[0][0] == 44);
    assert((int)target_tensor->matrix[0][1] == 50);
    assert((int)target_tensor->matrix[1][0] == 101);
    assert((int)target_tensor->matrix[1][1] == 116);


    tensor_destroy(first_tensor);
    tensor_destroy(second_tensor);
    tensor_destroy(target_tensor);

    MSG_STOP;
    return EXIT_SUCCESS;
}

static int iteration_test(void) {
    /*
     * Test iteration over a Tensor
     */
    MSG_START;
    Tensor *iter_tensor = tensor_init(2, 2, 3);
    for (unsigned i = 0; i < (2 * 2); i++) {
        assert(tensor_iter(iter_tensor, i) == 3);
    }
    tensor_destroy(iter_tensor);
    /* printf("tensor_iter Test: %f\n", tensor_iter(target_tensor, 3));
    puts("\n"); */

    MSG_STOP;
    return EXIT_SUCCESS;
}

static int rinit_test(void) {
    /*
     * Test random init of values in Tensor
     */
    MSG_START;

    /*
     * At least one value should be non-zero
     */
    Tensor *rnd_tensor = tensor_rinit(2, 3);
    for (unsigned i = 0; i < (2 * 3); i++) {
        if (tensor_iter(rnd_tensor, i) != 0) {
            break;
        } else if (i == ((2 * 3) - 1)) {
            assert(0);
        }
    }
    // tensor_print(rnd_tensor);

    tensor_destroy(rnd_tensor);

    MSG_STOP;
    return EXIT_SUCCESS;
}

static int broadcast_test(void) {
    MSG_START;

    Tensor *first_tensor = tensor_init(1, 3, 1);
    first_tensor->matrix[0][1] = 2;
    first_tensor->matrix[0][2] = 3;
    Tensor *second_tensor = tensor_init(1, 1, 2);

    Tensor *target_tensor = tensor_init(1, 3, 0);
    assert(tensor_dot(target_tensor, first_tensor, second_tensor) == EXIT_SUCCESS);
    assert(target_tensor->dim[0] == 1);
    assert(target_tensor->dim[1] == 3);
    assert(target_tensor->matrix[0][0] == 2);
    assert(target_tensor->matrix[0][1] == 4);
    assert(target_tensor->matrix[0][2] == 6);


    tensor_destroy(first_tensor);
    tensor_destroy(second_tensor);
    tensor_destroy(target_tensor);

    MSG_STOP;
    return EXIT_SUCCESS;
}

static int dotproduct_create_target_tensor_test(void) {
    MSG_START;
    Tensor *first_tensor = tensor_init(1, 3, 1);
    Tensor *second_tensor = tensor_init(3, 1, 2);
    Tensor *target_tensor = NULL;

    assert(tensor_dot(target_tensor, first_tensor, second_tensor) == EXIT_SUCCESS);
    assert(target_tensor != NULL);
    assert(target_tensor->dim[0] == 1);
    assert(target_tensor->dim[1] == 1);
    assert(target_tensor->matrix[0][0] == 6);

    tensor_destroy(target_tensor);
    tensor_destroy(first_tensor);
    tensor_destroy(second_tensor);

    MSG_STOP;
    return EXIT_SUCCESS;
}

static int tensor_cmp_test(void) {
    MSG_START;

    Tensor *first_tensor = tensor_init(3, 2, 1);
    Tensor *second_tensor = tensor_init(3, 2, 1);

    assert(tensor_cmp(first_tensor, second_tensor) == true);

    MSG_STOP;
    return EXIT_SUCCESS;
}

static int tensor_add_test(void) {
    MSG_START;

    Tensor *first_tensor = tensor_init(3, 2, 3);
    Tensor *second_tensor = tensor_init(3, 2, 2);
    Tensor *reference_tensor = tensor_init(3, 2, 5);
    Tensor *reference_tensor_2 = tensor_init(3, 2, 7);

    assert(tensor_add(first_tensor, first_tensor, second_tensor) == EXIT_SUCCESS);
    // tensor_print(first_tensor);
    assert(tensor_cmp(reference_tensor, first_tensor) == true);

    Tensor *target_tensor = tensor_init(3, 2, 0);
    assert(tensor_add(target_tensor, first_tensor, second_tensor) == EXIT_SUCCESS);
    assert(tensor_cmp(target_tensor, reference_tensor_2) == true);

    tensor_destroy(first_tensor);
    tensor_destroy(second_tensor);
    tensor_destroy(reference_tensor);
    tensor_destroy(reference_tensor_2);

    // Broadcasting test
    Tensor *bias_tensor = tensor_init(1, 8, 4);
    Tensor *values_tensor = tensor_init(4, 8, 16);
    Tensor *reference_tensor_outputs = tensor_init(4, 8, 20);

    assert(tensor_add(values_tensor, values_tensor, bias_tensor) == EXIT_SUCCESS);
    assert(tensor_cmp(values_tensor, reference_tensor_outputs) == true);

    tensor_destroy(bias_tensor);
    tensor_destroy(values_tensor);
    tensor_destroy(reference_tensor_outputs);

    MSG_STOP;
    return EXIT_SUCCESS;
}

int main(void) {
    basic_test();
    tensor_add_test();
    dotproduct_test();
    iteration_test();
    rinit_test();
    tensor_cmp_test();
    // dotproduct_create_target_tensor_test();
    // broadcast_test();

    printf("Tensor Test success!\n");

    return EXIT_SUCCESS;
}
