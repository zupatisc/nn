#ifndef LOSS_MSE_H
#define LOSS_MSE_H 1

#include "tensor.h"

typedef struct Loss_MSE Loss_MSE;

struct Loss_MSE {
    /*
     * Forward pass
     */
    Tensor *deviations;

    /*
     * Backward pass
     */
    Tensor *dinputs;

};

Loss_MSE *loss_mse_init(void);

int loss_mse_destroy(Loss_MSE *loss_mse);

Tensor *loss_mse_forward(Loss_MSE *loss_mse, Tensor *y_pred, Tensor *y_true);

int loss_mse_backward(Loss_MSE *loss_mse, Tensor *dvalues, Tensor *y_true);

#endif
