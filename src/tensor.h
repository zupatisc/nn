/*
 * This header defines the custom tensor struct as well as the
 * functions to operate on it.
 * It should all be matrices in this case :)
 */
#ifndef TENSOR_H
#define TENSOR_H 1

#include <stdlib.h>
#include <stdio.h>

#define frand() (((double) rand() / (RAND_MAX + 1.0)) - 0.5)

typedef struct Tensor Tensor;

struct Tensor {
    unsigned dim[2];
    double** matrix;
};

//Needed functions:
//init with dimensions and default value
Tensor *tensor_init(unsigned row, unsigned col, double default_val);
//random init
Tensor *tensor_rinit(unsigned row, unsigned col);
//destroy tensor
int tensor_destroy(Tensor *tensor);
//Matrix multiplication/dot product
int tensor_matmul(Tensor *tensor_trgt, Tensor *tensor_1, Tensor *tensor_2);
// Print tensor
void tensor_print(Tensor *tensor);
//sum over specific axis
//np.max(a, b)
//Transpose matrix
//iterate through tensor and check for conidtion
//iterate through tensor and apply operation
//Iterate through tensor in a flat manner and return value of item
double tensor_iter(Tensor *tensor, unsigned iter);
double tensor_set(Tensor *tensor, unsigned iter, double val);
#endif
