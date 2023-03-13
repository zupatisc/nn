#include <stdio.h>
#include <assert.h>

#include "test_utils.h"
#include "../loss_mse.h"

int test_forward(void) {
    MSG_START;

    Loss_MSE *mse_test = loss_mse_init();
    Tensor *pred_tensor = tensor_init(6, 1, 3);
    Tensor *true_tensor = tensor_init(6, 1, 7);
    Tensor *result_tensor = tensor_init(1, 1, 8);

    loss_mse_forward(mse_test, pred_tensor, true_tensor);
    assert(tensor_cmp(mse_test->deviations, result_tensor) == true);

    loss_mse_destroy(mse_test);
    tensor_destroy(pred_tensor);
    tensor_destroy(true_tensor);
    tensor_destroy(result_tensor);
    MSG_STOP;
    return EXIT_SUCCESS;
}

int test_backward(void) {
    MSG_START;

    Loss_MSE *mse_test = loss_mse_init();
    Tensor *dvalues = tensor_init(2, 2, 0);
    /*
     * [[2.3 2.3]
     * [2.3 2.3]]
     */
    dvalues->matrix[0][0] = 2.3;
    dvalues->matrix[0][1] = 2.3;
    dvalues->matrix[1][0] = 2.3;
    dvalues->matrix[1][1] = 2.3;

    Tensor *y_true = tensor_like(dvalues, 0);
    y_true->matrix[0][0] = 1.3;
    y_true->matrix[0][1] = 2.3;
    y_true->matrix[1][0] = 3.3;
    y_true->matrix[1][1] = 4.3;

    Tensor *result_tensor = tensor_like(y_true, 0);
    /*
     * [ 0.5, -0. ],
     * [-0.5, -1. ]]
     */
    result_tensor->matrix[0][0] = 0.5;
    result_tensor->matrix[0][1] = -0.;
    result_tensor->matrix[1][0] = -0.5;
    result_tensor->matrix[1][1] = -1;

    assert(loss_mse_backward(mse_test, dvalues, y_true) == EXIT_SUCCESS);
    assert(tensor_cmp(mse_test->dinputs, result_tensor));

    loss_mse_destroy(mse_test);
    tensor_destroy(dvalues);
    tensor_destroy(y_true);
    tensor_destroy(result_tensor);
    MSG_STOP;
    return EXIT_SUCCESS;
}

int main(void) {
    test_forward();
    test_backward();

    /* double col = {1};
    double *colp = &col;
    double **matrix = {&colp};

    printf("%f\n", matrix[0][0]);

    Tensor num_samples = {
        .dim = {1, 1},
        .matrix = matrix,
    };
    printf("%f\n", num_samples.matrix[0][0]);
    num_samples.matrix[0][0] = 3;
    printf("%f\n", num_samples.matrix[0][0]); */

    printf("Loss_MSE Test success!\n\n");
    return EXIT_SUCCESS;
}
