// #include <stdlib.h>
#include <stdio.h>

#include "utils.h"
#include "tensor.h"
#include "layer_dense.h"
#include "activation_tanh.h"
#include "loss_mse.h"
#include "optimizer_sgd.h"

#define NUM_DATA_POINTS 1000
#define LOWER_BOUND 0.
#define UPPER_BOUND 8.
#define NUM_EPOCHS 10000

int sinus_network(void) {

    /* unsigned data_points = NUM_DATA_POINTS;

    Tensor *X = tensor_init(data_points, 1, 0);
    for (unsigned i = 0; i < X->dim[0]; i++) {
        X->matrix[i][0] = (i+1) * (UPPER_BOUND/data_points);
    }
    Tensor *Y = tensor_init(data_points, 1, 0);
    for (unsigned i = 0; i < Y->dim[0]; i++) {
        Y->matrix[i][0] = sin(X->matrix[i][0]);
    } */

    Tensor *X = read_tensor("sin_data.csv", 0);
    Tensor *Y = read_tensor("sin_data.csv", 1);

    Layer_Dense *dense1 = layer_dense_init(1, 8);
    Activation_Tanh *activation1 = activation_tanh_init();
    Layer_Dense *dense2 = layer_dense_init(8, 1);
    Loss_MSE *loss_activation = loss_mse_init();
    Optimizer_SGD *optimizer = optimizer_sgd_init(0.05, 5e-7);


    for (unsigned i = 0; i < NUM_EPOCHS; i++) {

        // Forward pass
        layer_dense_forward(dense1, X);
        activation_tanh_forward(activation1, dense1->output);
        layer_dense_forward(dense2, activation1->output);
        Tensor *loss = loss_mse_forward(loss_activation, dense2->output, Y);

        if (i % 100 == 0) {
            printf("Epoch: %d\nLoss: %f\nlr: %f\n", i, loss->matrix[0][0], optimizer->current_learning_rate);
        }

        // Backward pass
        loss_mse_backward(loss_activation, dense2->output, Y);
        layer_dense_backward(dense2, loss_activation->dinputs);
        activation_tanh_backward(activation1, dense2->dinputs);
        layer_dense_backward(dense1, activation1->dinputs);

        // Update weights and biases
        optimizer_sgd_pre_update_params(optimizer);
        optimizer_sgd_update_params(optimizer, dense1);
        optimizer_sgd_update_params(optimizer, dense2);
        optimizer_sgd_post_update_params(optimizer);

    }

    /*
     * Single forward pass then combine X and y_pred
     * and write to file
     */
    layer_dense_forward(dense1, X);
    activation_tanh_forward(activation1, dense1->output);
    layer_dense_forward(dense2, activation1->output);
    Tensor *result_tensor = tensor_init(dense2->output->dim[0], 2, 0);
    for (unsigned ic = 0; ic < result_tensor->dim[0]; ic++) {
        result_tensor->matrix[ic][0] = X->matrix[ic][0];
        result_tensor->matrix[ic][1] = dense2->output->matrix[ic][0];
    }
    write_tensor(result_tensor, "pred_sin_data.csv");
    tensor_destroy(result_tensor);


    tensor_destroy(X);
    tensor_destroy(Y);

    layer_dense_destroy(dense1);
    activation_tanh_destroy(activation1);
    layer_dense_destroy(dense2);
    loss_mse_destroy(loss_activation);
    optimizer_sgd_destroy(optimizer);
    return EXIT_SUCCESS;
}

int write_sin(void) {

    unsigned data_points = NUM_DATA_POINTS;

    Tensor *data_tensor = tensor_init(data_points, 2, 0);

    for (unsigned i = 0; i < data_tensor->dim[0]; i++) {
        data_tensor->matrix[i][0] = (i+1) * (UPPER_BOUND/data_points);
    }
    for (unsigned i = 0; i < data_tensor->dim[0]; i++) {
        data_tensor->matrix[i][1] = sin(data_tensor->matrix[i][0]);
    }

    write_tensor(data_tensor, "sin_data.csv");

    tensor_destroy(data_tensor);
    return EXIT_SUCCESS;
}

int main(void) {

    write_sin();
    sinus_network();
    /* test_read();
    test_write(); */

    return EXIT_SUCCESS;
}
