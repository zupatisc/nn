#include "loss_mse.h"
#include "tensor.h"
#include <stdlib.h>

Loss_MSE *loss_mse_init(void) {
    Loss_MSE *loss_mse = malloc(sizeof(Loss_MSE));

    loss_mse->dinputs = NULL;
    loss_mse->deviations = NULL;

    return loss_mse;
}

int loss_mse_destroy(Loss_MSE *loss_mse) {
    tensor_destroy(loss_mse->deviations);
    tensor_destroy(loss_mse->dinputs);
    free(loss_mse);

    return EXIT_SUCCESS;
}

Tensor *loss_mse_forward(Loss_MSE *loss_mse, Tensor *y_pred, Tensor *y_true) {

    double scol = {y_pred->dim[0]};
    double *scolp = &scol;
    double **samples_cmatrix = {&scolp};
    Tensor num_samples = {
        .dim = {1, 1},
        .matrix = samples_cmatrix,
    };

    double fcol = {0.5};
    double *fcolp = &fcol;
    double **flip_cmatrix = {&fcolp};
    Tensor scalar = {
        .dim = {1, 1},
        .matrix = flip_cmatrix,
    };

    Tensor *tmp_tensor = tensor_like(y_pred, 0);
    tensor_sub(tmp_tensor, y_pred, y_true);
    tensor_pow(tmp_tensor, tmp_tensor, 2);

    loss_mse->deviations = tensor_init(1, 1, 0);
    tensor_sum(loss_mse->deviations, tmp_tensor, 0);
    tensor_mult(loss_mse->deviations, loss_mse->deviations, &scalar);
    tensor_div(loss_mse->deviations, loss_mse->deviations, &num_samples);

    tensor_destroy(tmp_tensor);
    return loss_mse->deviations;
}

int loss_mse_backward(Loss_MSE *loss_mse, Tensor *dvalues, Tensor *y_true) {

    // Tensor *num_samples = tensor_init(1, 1, dvalues->dim[0]);
    /*
     * There is probably a fancy oneliner for this but this
     * is also just easier to step through and understand.
     * I just want a fixed Tensor object on the stack to avoid malloc calls
     */
    double scol = {dvalues->dim[0]};
    double *scolp = &scol;
    double **samples_cmatrix = {&scolp};
    Tensor num_samples = {
        .dim = {1, 1},
        .matrix = samples_cmatrix,
    };

    loss_mse->dinputs = tensor_like(dvalues, 0);
    tensor_sub(loss_mse->dinputs, y_true, dvalues);
    tensor_div(loss_mse->dinputs, loss_mse->dinputs, &num_samples);

    // Tensor *flip = tensor_init(1, 1, -1);
    /*
     * Same reason as above, the Tensor is static in size
     */
    double fcol = {-1};
    double *fcolp = &fcol;
    double **flip_cmatrix = {&fcolp};
    Tensor flip = {
        .dim = {1, 1},
        .matrix = flip_cmatrix,
    };
    tensor_mult(loss_mse->dinputs, loss_mse->dinputs, &flip);

    return EXIT_SUCCESS;
}
