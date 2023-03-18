#ifndef UTILS_H
#define UTILS_H 1

#include <stdio.h>

#include "tensor.h"


#define BUFFER_SIZE 32

/*
 * Write tensor to CSV file
 */
int write_tensor(Tensor *tensor, char *file_name);

/*
 * Read specified column from CSV file
 */
Tensor *read_tensor(char *file_name, unsigned col);

#endif
