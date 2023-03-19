/*
 * This header defines the custom tensor struct as well as the
 * functions to operate on it.
 * It should all be matrices in this case :)
 */
#ifndef TENSOR_H
#define TENSOR_H 1

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#define frand() (((double) rand() / (RAND_MAX + 1.0)) - 0.5)

#define ETENNULL 15 // Critical Tensor was NULL
#define ETENMIS 16 // Tensors mismatched and could not be broadcast
#define EDIM 17

#define CMPPREC  1e-6

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

Tensor *tensor_like(Tensor *tensor, double default_val);

//destroy tensor
int tensor_destroy(Tensor *tensor);

//Matrix multiplication/dot product
int tensor_dot(Tensor *tensor_trgt, Tensor *tensor_1, Tensor *tensor_2);

//Elementwise addition
int tensor_add(Tensor *tensor_trgt, Tensor *tensor_1, Tensor*restrict tensor_2);

// Elementwise substraction
int tensor_sub(Tensor *tensor_trgt, Tensor *tensor_1, Tensor*restrict tensor_2);

// Print tensor
void tensor_print(Tensor *tensor);

// Print shapes of tensor (rows, columns)
void tensor_shapes(Tensor *tensor);

// Compare tensors by value
bool tensor_cmp(Tensor *tensor_1, Tensor *tensor_2);

// Return new tensor which is transpose of input
Tensor *tensor_transpose(Tensor *tensor);

// sum over specific axis
int tensor_sum(Tensor *tensor_trgt, Tensor *tensor_1, unsigned dim);

// Elementwise multiplication
int tensor_mult(Tensor *tensor_trgt, Tensor *tensor_1, Tensor*restrict tensor_2);

// Elementwise division
int tensor_div(Tensor *tensor_trgt, Tensor *tensor_1, Tensor*restrict tensor_2);

// Elementwise power
int tensor_pow(Tensor *tensor_trgt, Tensor *tensor_1, double exponent);

//iterate through tensor and check for conidtion
//iterate through tensor and apply operation
//Iterate through tensor in a flat manner and return value of item
double tensor_iter(Tensor *tensor, unsigned iter);
double tensor_set(Tensor *tensor, unsigned iter, double val);
#endif
