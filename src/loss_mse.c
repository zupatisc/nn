#include "loss_mse.h"

Loss_MSE *loss_mse_init(void);

int loss_mse_destroy(Loss_MSE *loss_mse);

Tensor *loss_mse_forward(Loss_MSE *loss_mse, Tensor *y_pred, Tensor *y_true);

int loss_mse_backward(Loss_MSE *loss_mse, Tensor *dvalues, Tensor *y_true);
