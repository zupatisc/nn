#include "test_utils.h"

#include "../tensor.h"
#include "../layer_dense.h"
#include "../activation_tanh.h"
#include "../loss_mse.h"
#include <assert.h>
#include <stdio.h>

static int test_layer_activation(void) {
    MSG_START;

    Layer_Dense *test_layer = layer_dense_init(2, 4);
    Activation_Tanh *test_activation = activation_tanh_init();

    /*
     * [[0.00548814 0.00715189 0.00602763 0.00544883]
     * [0.00423655 0.00645894 0.00437587 0.00891773]]
     */
    test_layer->weights->matrix[0][0] = 0.00548814;
    test_layer->weights->matrix[0][1] = 0.00715189;
    test_layer->weights->matrix[0][2] = 0.00602763;
    test_layer->weights->matrix[0][3] = 0.00544883;
    test_layer->weights->matrix[1][0] = 0.00423655;
    test_layer->weights->matrix[1][1] = 0.00645894;
    test_layer->weights->matrix[1][2] = 0.00437587;
    test_layer->weights->matrix[1][3] = 0.00891773;

    Tensor *test_input = tensor_init(5, 2, 6);

    if (PRINT) {
        tensor_print(test_layer->weights);
        tensor_print(test_layer->biases);
        tensor_print(test_input);
    }

    layer_dense_forward(test_layer, test_input);
    if(PRINT)
        tensor_print(test_layer->output);
    activation_tanh_forward(test_activation, test_layer->output);
    if(PRINT)
        tensor_print(test_activation->output);

    Tensor *result_tensor = tensor_init(5, 4, 0);
    /*
     * [[0.05828197 0.08148395 0.06234009 0.08598651]
     * [0.05828197 0.08148395 0.06234009 0.08598651]
     * [0.05828197 0.08148395 0.06234009 0.08598651]
     * [0.05828197 0.08148395 0.06234009 0.08598651]
     * [0.05828197 0.08148395 0.06234009 0.08598651]]
     */
    result_tensor->matrix[0][0] = 0.05828197;
    result_tensor->matrix[0][1] = 0.08148395;
    result_tensor->matrix[0][2] = 0.06234009;
    result_tensor->matrix[0][3] = 0.08598651;

    result_tensor->matrix[1][0] = 0.05828197;
    result_tensor->matrix[1][1] = 0.08148395;
    result_tensor->matrix[1][2] = 0.06234009;
    result_tensor->matrix[1][3] = 0.08598651;

    result_tensor->matrix[2][0] = 0.05828197;
    result_tensor->matrix[2][1] = 0.08148395;
    result_tensor->matrix[2][2] = 0.06234009;
    result_tensor->matrix[2][3] = 0.08598651;

    result_tensor->matrix[3][0] = 0.05828197;
    result_tensor->matrix[3][1] = 0.08148395;
    result_tensor->matrix[3][2] = 0.06234009;
    result_tensor->matrix[3][3] = 0.08598651;

    result_tensor->matrix[4][0] = 0.05828197;
    result_tensor->matrix[4][1] = 0.08148395;
    result_tensor->matrix[4][2] = 0.06234009;
    result_tensor->matrix[4][3] = 0.08598651;

    assert(tensor_cmp(test_activation->output, result_tensor) == true);

    layer_dense_destroy(test_layer);
    activation_tanh_destroy(test_activation);
    tensor_destroy(test_input);
    tensor_destroy(result_tensor);

    MSG_STOP;
    return EXIT_SUCCESS;
}

int test_full_forward(void) {
    MSG_START;

    Layer_Dense *dense1 = layer_dense_init(1, 8);
    /*
     * [[0.00548814 0.00715189 0.00602763 0.00544883 0.00423655 0.00645894
     * 0.00437587 0.00891773]]
     */
    dense1->weights->matrix[0][0] = 0.00548814;
    dense1->weights->matrix[0][1] = 0.00715189;
    dense1->weights->matrix[0][2] = 0.00602763;
    dense1->weights->matrix[0][3] = 0.00544883;
    dense1->weights->matrix[0][4] = 0.00423655;
    dense1->weights->matrix[0][5] = 0.00645894;
    dense1->weights->matrix[0][6] = 0.00437587;
    dense1->weights->matrix[0][7] = 0.00891773;

    Activation_Tanh *activation1 = activation_tanh_init();

    Layer_Dense *dense2 = layer_dense_init(8, 1);
    /*
     * [[0.00963663]
     * [0.00383442]
     * [0.00791725]
     * [0.00528895]
     * [0.00568045]
     * [0.00925597]
     * [0.00071036]
     * [0.00087129]]
     */
    dense2->weights->matrix[0][0] = 0.00963663;
    dense2->weights->matrix[1][0] = 0.00383442;
    dense2->weights->matrix[2][0] = 0.00791725;
    dense2->weights->matrix[3][0] = 0.00528895;
    dense2->weights->matrix[4][0] = 0.00568045;
    dense2->weights->matrix[5][0] = 0.00925597;
    dense2->weights->matrix[6][0] = 0.00071036;
    dense2->weights->matrix[7][0] = 0.00087129;

    Loss_MSE *loss_activation = loss_mse_init();

    Tensor *loss;
    Tensor *input_tensor = tensor_init(6, 1, 3);
    Tensor *true_tensor = tensor_init(6, 1, 0);
    true_tensor->matrix[0][0] = 1;
    true_tensor->matrix[1][0] = 2;
    true_tensor->matrix[2][0] = 3;
    true_tensor->matrix[3][0] = 4;
    true_tensor->matrix[4][0] = 5;
    true_tensor->matrix[5][0] = 6;

    if (PRINT) {
        tensor_print(input_tensor);
        tensor_print(true_tensor);
    }

    Tensor *result_loss_tensor = tensor_init(1, 1, 7.5806923284734475);
    Tensor *result_pred_tensor = tensor_init(6, 1, 0);

    /*
     * [[0.00075465]
     * [0.00075465]
     * [0.00075465]
     * [0.00075465]
     * [0.00075465]
     * [0.00075465]]
     */
    result_pred_tensor->matrix[0][0] = 0.00075465;
    result_pred_tensor->matrix[1][0] = 0.00075465;
    result_pred_tensor->matrix[2][0] = 0.00075465;
    result_pred_tensor->matrix[3][0] = 0.00075465;
    result_pred_tensor->matrix[4][0] = 0.00075465;
    result_pred_tensor->matrix[5][0] = 0.00075465;


    COMMENT("dense1");
    layer_dense_forward(dense1, input_tensor);

    COMMENT("activation1");
    activation_tanh_forward(activation1, dense1->output);

    COMMENT("dense2");
    layer_dense_forward(dense2, activation1->output);
    assert(tensor_cmp(dense2->output, result_pred_tensor) == true);

    COMMENT("loss_activation");
    loss = loss_mse_forward(loss_activation, dense2->output, true_tensor);

    assert(tensor_cmp(loss, result_loss_tensor) == true);

    if (PRINT) {
        printf("Loss: %f\n", loss->matrix[0][0]);
        puts("Prediction Tensor:\n");
        tensor_print(dense2->output);
    }

    layer_dense_destroy(dense1);
    activation_tanh_destroy(activation1);
    layer_dense_destroy(dense2);
    loss_mse_destroy(loss_activation);
    tensor_destroy(input_tensor);
    tensor_destroy(true_tensor);
    tensor_destroy(result_loss_tensor);
    tensor_destroy(result_pred_tensor);
    MSG_STOP;
    return EXIT_SUCCESS;
}

int test_full_backward(void) {
    MSG_START;

    Layer_Dense *dense1 = layer_dense_init(1, 8);
    /*
     * [[0.00548814 0.00715189 0.00602763 0.00544883 0.00423655 0.00645894
     * 0.00437587 0.00891773]]
     */
    dense1->weights->matrix[0][0] = 0.00548814;
    dense1->weights->matrix[0][1] = 0.00715189;
    dense1->weights->matrix[0][2] = 0.00602763;
    dense1->weights->matrix[0][3] = 0.00544883;
    dense1->weights->matrix[0][4] = 0.00423655;
    dense1->weights->matrix[0][5] = 0.00645894;
    dense1->weights->matrix[0][6] = 0.00437587;
    dense1->weights->matrix[0][7] = 0.00891773;

    Activation_Tanh *activation1 = activation_tanh_init();

    Layer_Dense *dense2 = layer_dense_init(8, 1);
    /*
     * [[0.00963663]
     * [0.00383442]
     * [0.00791725]
     * [0.00528895]
     * [0.00568045]
     * [0.00925597]
     * [0.00071036]
     * [0.00087129]]
     */
    dense2->weights->matrix[0][0] = 0.00963663;
    dense2->weights->matrix[1][0] = 0.00383442;
    dense2->weights->matrix[2][0] = 0.00791725;
    dense2->weights->matrix[3][0] = 0.00528895;
    dense2->weights->matrix[4][0] = 0.00568045;
    dense2->weights->matrix[5][0] = 0.00925597;
    dense2->weights->matrix[6][0] = 0.00071036;
    dense2->weights->matrix[7][0] = 0.00087129;

    Loss_MSE *loss_activation = loss_mse_init();

    Tensor *true_tensor = tensor_init(6, 1, 0);
    true_tensor->matrix[0][0] = 1;
    true_tensor->matrix[1][0] = 2;
    true_tensor->matrix[2][0] = 3;
    true_tensor->matrix[3][0] = 4;
    true_tensor->matrix[4][0] = 5;
    true_tensor->matrix[5][0] = 6;

    Tensor *input_tensor = tensor_init(6, 1, 3);
    Tensor *result_pred_tensor = tensor_init(6, 1, 0);

    /*
     * [[0.00075465]
     * [0.00075465]
     * [0.00075465]
     * [0.00075465]
     * [0.00075465]
     * [0.00075465]]
     */
    result_pred_tensor->matrix[0][0] = 0.00075465;
    result_pred_tensor->matrix[1][0] = 0.00075465;
    result_pred_tensor->matrix[2][0] = 0.00075465;
    result_pred_tensor->matrix[3][0] = 0.00075465;
    result_pred_tensor->matrix[4][0] = 0.00075465;
    result_pred_tensor->matrix[5][0] = 0.00075465;

    COMMENT("FORWARD: dense1");
    layer_dense_forward(dense1, input_tensor);

    COMMENT("FORWARD: activation1");
    activation_tanh_forward(activation1, dense1->output);

    COMMENT("FORWARD: dense2");
    layer_dense_forward(dense2, activation1->output);
    assert(tensor_cmp(dense2->output, result_pred_tensor) == true);

    COMMENT("FORWARD: loss_activation");
    loss_mse_forward(loss_activation, dense2->output, true_tensor);

    /*
     * Ground truth tensors
     */
    Tensor *result_loss_dinputs = tensor_init(6, 1, 0);
    /*
     * [[-0.16654089]
     * [-0.33320756]
     * [-0.49987422]
     * [-0.66654089]
     * [-0.83320756]
     * [-0.99987422]]
     */
    result_loss_dinputs->matrix[0][0] = -0.16654089;
    result_loss_dinputs->matrix[1][0] = -0.33320756;
    result_loss_dinputs->matrix[2][0] = -0.49987422;
    result_loss_dinputs->matrix[3][0] = -0.66654089;
    result_loss_dinputs->matrix[4][0] = -0.83320756;
    result_loss_dinputs->matrix[5][0] = -0.99987422;

    Tensor *result_dense2_dweights = tensor_init(8, 1, 0);
    /*[[-0.05760778]
     * [-0.07506718]
     * [-0.06326962]
     * [-0.05719531]
     * [-0.04447177]
     * [-0.06779578]
     * [-0.04593411]
     * [-0.09359365]]
     */
    result_dense2_dweights->matrix[0][0] = -0.05760778;
    result_dense2_dweights->matrix[1][0] = -0.07506718;
    result_dense2_dweights->matrix[2][0] = -0.06326962;
    result_dense2_dweights->matrix[3][0] = -0.05719531;
    result_dense2_dweights->matrix[4][0] = -0.04447177;
    result_dense2_dweights->matrix[5][0] = -0.06779578;
    result_dense2_dweights->matrix[6][0] = -0.04593411;
    result_dense2_dweights->matrix[7][0] = -0.09359365;

    Tensor *result_dense2_dbiases = tensor_init(1, 1, 0);
    /*
     * [[-3.49924535]]
     */
    result_dense2_dbiases->matrix[0][0] = -3.49924535;

    Tensor *result_dense1_dweights = tensor_init(1, 8, 0);
    /*
     * [[-0.10113536 -0.04023415 -0.08308604 -0.05550716 -0.05962219 -0.09713022
     * -0.00745589 -0.00914006]]
     */
    result_dense1_dweights->matrix[0][0] = -0.10113536;
    result_dense1_dweights->matrix[0][1] = -0.04023415;
    result_dense1_dweights->matrix[0][2] = -0.08308604;
    result_dense1_dweights->matrix[0][3] = -0.05550716;
    result_dense1_dweights->matrix[0][4] = -0.05962219;
    result_dense1_dweights->matrix[0][5] = -0.09713022;
    result_dense1_dweights->matrix[0][6] = -0.00745589;
    result_dense1_dweights->matrix[0][7] = -0.00914006;

    Tensor *result_dense1_dbiases = tensor_init(1, 8, 0);
    /*
     * [[-0.03371178 -0.01341138 -0.02769534 -0.01850239 -0.01987406 -0.03237674
     * -0.0024853  -0.00304669]]
     */
    result_dense1_dbiases->matrix[0][0] = -0.03371178;
    result_dense1_dbiases->matrix[0][1] = -0.01341138;
    result_dense1_dbiases->matrix[0][2] = -0.02769534;
    result_dense1_dbiases->matrix[0][3] = -0.01850239;
    result_dense1_dbiases->matrix[0][4] = -0.01987406;
    result_dense1_dbiases->matrix[0][5] = -0.03237674;
    result_dense1_dbiases->matrix[0][6] = -0.0024853;
    result_dense1_dbiases->matrix[0][7] = -0.00304669;

    Tensor *result_dense1_dinputs = tensor_init(6, 1, 0);
    /*
     * [[-4.1884414e-05]
     *  [-8.3800456e-05]
     *  [-1.2571650e-04]
     *  [-1.6763255e-04]
     *  [-2.0954860e-04]
     *  [-2.5146463e-04]]
     */
    result_dense1_dinputs->matrix[0][0] = -4.1884414e-05;
    result_dense1_dinputs->matrix[1][0] = -8.3800456e-05;
    result_dense1_dinputs->matrix[2][0] = -1.2571650e-04;
    result_dense1_dinputs->matrix[3][0] = -1.6763255e-04;
    result_dense1_dinputs->matrix[4][0] = -2.0954860e-04;
    result_dense1_dinputs->matrix[5][0] = -2.5146463e-04;

    /*
     * BACKWARD PASS
     */
    COMMENT("BACKWARD: loss_activation");
    loss_mse_backward(loss_activation, result_pred_tensor, true_tensor);
    if (PRINT) {
        COMMENT("prediction, true, and dinputs tensors");
        tensor_print(result_pred_tensor);
        tensor_print(true_tensor);
        tensor_print(loss_activation->dinputs);
    }
    assert(tensor_cmp(loss_activation->dinputs, result_loss_dinputs) == true);

    COMMENT("BACKWARD: dense2");
    layer_dense_backward(dense2, loss_activation->dinputs);
    if (PRINT) {
        tensor_print(dense2->dweights);
        tensor_print(result_dense2_dweights);
    }
    assert(tensor_cmp(dense2->dweights, result_dense2_dweights) == true);
    assert(tensor_cmp(dense2->dbiases, result_dense2_dbiases) == true);

    COMMENT("BACKWARD: activation1");
    activation_tanh_backward(activation1, dense2->dinputs);

    COMMENT("BACKWARD: dense1");
    layer_dense_backward(dense1, activation1->dinputs);
    assert(tensor_cmp(dense1->dweights, result_dense1_dweights) == true);
    assert(tensor_cmp(dense1->dbiases, result_dense1_dbiases) == true);
    assert(tensor_cmp(dense1->dinputs, result_dense1_dinputs) == true);



    layer_dense_destroy(dense1);
    activation_tanh_destroy(activation1);
    layer_dense_destroy(dense2);
    loss_mse_destroy(loss_activation);

    tensor_destroy(true_tensor);
    tensor_destroy(result_pred_tensor);
    tensor_destroy(input_tensor);
    tensor_destroy(result_loss_dinputs);
    tensor_destroy(result_dense2_dweights);
    tensor_destroy(result_dense2_dbiases);
    tensor_destroy(result_dense1_dweights);
    tensor_destroy(result_dense1_dbiases);
    tensor_destroy(result_dense1_dinputs);
    MSG_STOP;
    return EXIT_SUCCESS;
}

int main(void) {
    test_layer_activation();
    test_full_forward();
    test_full_backward();

    printf("Network Test success!\n\n");
}
