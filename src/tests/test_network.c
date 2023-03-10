#include "test_utils.h"

#include "../layer_dense.h"
#include "../activation_tanh.h"
#include <assert.h>

#define PRINT 1

static int test_layer_activation(void) {
    MSG_START;

    Layer_Dense *test_layer = layer_dense_init(2, 4);
    Activation_Tanh *test_activation = activation_tanh_init();

    Tensor *test_input = tensor_init(5, 2, 6);

    layer_dense_forward(test_layer, test_input);
    if(PRINT)
        tensor_print(test_layer->output);
    activation_tanh_forward(test_activation, test_layer->output);
    if(PRINT)
        tensor_print(test_activation->output);

    layer_dense_destroy(test_layer);
    activation_tanh_destroy(test_activation);

    MSG_STOP;
    return EXIT_SUCCESS;
}

int main(void) {
    test_layer_activation();

    printf("Network Test success!\n\n");
}
