// #include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "tensor.h"
#include "layer_dense.h"
#include "activation_tanh.h"
#include "activation_relu.h"
#include "loss_mse.h"
#include "optimizer_sgd.h"

#define NUM_DATA_POINTS_SIN 1000
#define LOWER_BOUND 0.
#define UPPER_BOUND 7.
#define NUM_EPOCHS_SIN 10001

#define NUM_DATA_POINTS_XOR 3000
#define NUM_EPOCHS_XOR 10001

/*
 * Network for learning the sin() function
 */
int sinus_network(void) {

    Tensor *X = read_tensor("sin_data.csv", 0);
    Tensor *Y = read_tensor("sin_data.csv", 1);

    Layer_Dense *dense1 = layer_dense_init(1, 8);
    Activation_Tanh *activation1 = activation_tanh_init();
    Layer_Dense *dense2 = layer_dense_init(8, 1);
    Loss_MSE *loss_activation = loss_mse_init();
    Optimizer_SGD *optimizer = optimizer_sgd_init(0.05, 5e-7);


    for (unsigned i = 0; i < NUM_EPOCHS_SIN; i++) {

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

    unsigned data_points = NUM_DATA_POINTS_SIN;

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

/*
 * Network for learning the xor() function
 */
int xor_network(void) {

    Tensor *X_1 = read_tensor("xor_data.csv", 0);
    Tensor *X_2 = read_tensor("xor_data.csv", 1);

    Tensor *X = tensor_init(X_1->dim[0], 2, 0);
    for (unsigned i = 0; i < X->dim[0]; i++) {
        X->matrix[i][0] = X_1->matrix[i][0];
        X->matrix[i][1] = X_2->matrix[i][0];
    }
    Tensor *Y = read_tensor("xor_data.csv", 2);

    /* tensor_print(X);
    tensor_print(Y); */

    tensor_destroy(X_1);
    tensor_destroy(X_2);

    Layer_Dense *dense1 = layer_dense_init(2, 2);
    Activation_ReLU *activation1 = activation_relu_init();
    Layer_Dense *dense2 = layer_dense_init(2, 1);
    Loss_MSE *loss_activation = loss_mse_init();
    Optimizer_SGD *optimizer = optimizer_sgd_init(0.05, 5e-7);


    for (unsigned i = 0; i < NUM_EPOCHS_XOR; i++) {

        // Forward pass
        layer_dense_forward(dense1, X);
        activation_relu_forward(activation1, dense1->output);
        layer_dense_forward(dense2, activation1->output);
        Tensor *loss = loss_mse_forward(loss_activation, dense2->output, Y);

        if (i % 100 == 0) {
            printf("Epoch: %d\nLoss: %f\nlr: %f\n", i, loss->matrix[0][0], optimizer->current_learning_rate);
        }

        // Backward pass
        loss_mse_backward(loss_activation, dense2->output, Y);
        layer_dense_backward(dense2, loss_activation->dinputs);
        activation_relu_backward(activation1, dense2->dinputs);
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
    activation_relu_forward(activation1, dense1->output);
    layer_dense_forward(dense2, activation1->output);
    Tensor *result_tensor = tensor_init(dense2->output->dim[0], 3, 0);
    for (unsigned ic = 0; ic < result_tensor->dim[0]; ic++) {
        result_tensor->matrix[ic][0] = X->matrix[ic][0];
        result_tensor->matrix[ic][1] = X->matrix[ic][1];
        result_tensor->matrix[ic][2] = dense2->output->matrix[ic][0];
    }
    write_tensor(result_tensor, "pred_xor_data.csv");
    tensor_destroy(result_tensor);


    tensor_destroy(X);
    tensor_destroy(Y);

    layer_dense_destroy(dense1);
    activation_relu_destroy(activation1);
    layer_dense_destroy(dense2);
    loss_mse_destroy(loss_activation);
    optimizer_sgd_destroy(optimizer);
    return EXIT_SUCCESS;
}

static inline int xor(int x_1, int x_2) {
    return (x_1 + x_2) * (!(x_1 * x_2));
}

int write_xor(void) {

    unsigned data_points = NUM_DATA_POINTS_XOR;

    Tensor *data_tensor = tensor_init(data_points, 3, 0);

    for (unsigned ic = 0; ic < data_tensor->dim[0]; ic++) {
        unsigned x_1 = 0, x_2 = 0, y = 0;

        x_1 = rand() % 2;
        x_2 = rand() % 2;
        y = xor(x_1, x_2);

        data_tensor->matrix[ic][0] = x_1;
        data_tensor->matrix[ic][1] = x_2;
        data_tensor->matrix[ic][2] = y;
    }

    write_tensor(data_tensor, "xor_data.csv");

    tensor_destroy(data_tensor);

    return EXIT_SUCCESS;
}

int test_read(void) {
    // Tensor *X = read_tensor("sin_data.csv", 0);
    Tensor *Y = read_tensor("sin_data.csv", 1);

    /* puts("X Tensor");
    tensor_print(X); */
    puts("Y Tensor");
    tensor_print(Y);

    // tensor_destroy(X);
    tensor_destroy(Y);

    return EXIT_SUCCESS;
}

int main(void) {

    write_sin();
    sinus_network();

    write_xor();
    xor_network();

    return EXIT_SUCCESS;
}
