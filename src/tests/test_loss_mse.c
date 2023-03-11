#include <stdio.h>
#include <assert.h>

#include "test_utils.h"
#include "../loss_mse.h"

int main(void) {

    double col = {1};
    double *colp = &col;
    double **matrix = {&colp};

    printf("%f\n", matrix[0][0]);

    Tensor num_samples = {
        .dim = {1, 1},
        .matrix = matrix,
    };
    printf("%f\n", num_samples.matrix[0][0]);
    num_samples.matrix[0][0] = 3;
    printf("%f\n", num_samples.matrix[0][0]);

    printf("Loss_MSE Test success!\n\n");
    return EXIT_SUCCESS;
}
