#ifndef OPTIMIZER_SGD_H
#define OPTIMIZER_SGD_H 1

#include "tensor.h"
#include "layer_dense.h"

typedef struct Optimizer_SGD Optimizer_SGD;

struct Optimizer_SGD {
    double learning_rate;
    double current_learning_rate;
    double decay;
    unsigned iterations;
};

Optimizer_SGD *optimizer_sgd_init(double learning_rate, double decay);

void optimizer_sgd_destroy(Optimizer_SGD *optim);

int optimizer_sgd_pre_update_params(Optimizer_SGD *optim);

int optimizer_sgd_update_params(Optimizer_SGD *optim, Layer_Dense *layer_dense);

int optimizer_sgd_post_update_params(Optimizer_SGD *optim);

#endif
