#include "test_utils.h"

#include "../tensor.h"
#include "../layer_dense.h"
#include "../activation_tanh.h"
#include <assert.h>

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

    MSG_STOP;
    return EXIT_SUCCESS;
}

int main(void) {
    test_layer_activation();

    printf("Network Test success!\n\n");
}
