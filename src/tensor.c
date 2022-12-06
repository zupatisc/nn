#include "tensor.h"

Tensor *tensor_init(unsigned int row, unsigned int col, double default_val) {
    double **matrix = malloc(row * sizeof *matrix);
    if (matrix == NULL) {
        //TODO: Error handling
    }

    for (unsigned int i = 0; i < row; i++) {
         double *colp = malloc(col * sizeof *colp);
         if (colp == NULL) {
             //TODO: Error handling
         }
         for (int u = 0; u < col; u++)
             *(colp + u) = default_val;
         matrix[i] = colp;
    }

    Tensor *tensorp = malloc(sizeof(Tensor));
    tensorp->matrix = matrix;
    tensorp->dim[0] = row;
    tensorp->dim[1] = col;

    return tensorp;
}

Tensor *tensor_rinit(unsigned int row, unsigned int col) {
    Tensor *new_tensor = tensor_init(row, col, 0);

    for (int i = 0; i < new_tensor->dim[0] * new_tensor->dim[1]; i++) {
        tensor_set(new_tensor, i, frand());
    }

    return new_tensor;
}

int tensor_destroy(Tensor *tensor) { //TODO: return useful info
    if (tensor) {
        for(int i = 0; i < tensor->dim[0]; i++) {
            free(tensor->matrix[i]);
        }
        free(tensor->matrix);
    }
    free(tensor);

    return 0;
}

int tensor_dot(Tensor *tensor_trgt, Tensor *tensor_1, Tensor *tensor_2) {
    if (tensor_trgt == NULL || tensor_1 == NULL || tensor_2 == NULL)
        return EXIT_FAILURE;

     if (tensor_trgt->dim[0] != tensor_1->dim[0] || tensor_trgt->dim[1] != tensor_2->dim[1])
         return EXIT_FAILURE;

     //Taken from my python work
     for (int ir = 0; ir < tensor_trgt->dim[0]; ir++) {
         for (int ic = 0; ic < tensor_trgt->dim[1]; ic++) {
             double val = 0;
             for (int i = 0; i < tensor_1->dim[1]; i++) {
                 val += tensor_1->matrix[ir][i] * tensor_2->matrix[i][ic];
             }
             tensor_trgt->matrix[ir][ic] = val;
         }
     }

    return EXIT_SUCCESS;
}

int tensor_add(Tensor *tensor_trgt, Tensor *tensor_1, Tensor*restrict tensor_2) {
    if (tensor_trgt == NULL || tensor_1 == NULL || tensor_2 == NULL)
        return EXIT_FAILURE;

    if (tensor_trgt->dim[0] != tensor_1->dim[0] || tensor_trgt->dim[1] != tensor_1->dim[1])
        return EXIT_FAILURE;

    if (tensor_trgt->dim[0] != tensor_2->dim[0] || tensor_trgt->dim[1] != tensor_2->dim[1])
        return EXIT_FAILURE;

    for(int ir = 0; ir < tensor_trgt->dim[0]; ir++) {
        for (int ic = 0; ic < tensor_trgt->dim[1]; ic++) {
            tensor_trgt->matrix[ir][ic] = tensor_1->matrix[ir][ic] + tensor_2->matrix[ir][ic];
        }
    }

    return EXIT_SUCCESS;
}

void tensor_print(Tensor *tensor) {
    int ir = 0;
    while (ir < tensor->dim[0]) { //Rows
        int ic = 0;
        while (ic < tensor->dim[1]) { //Columns
            printf("%f ", tensor->matrix[ir][ic]);
            ic++;
        }
        puts("\n");
        ir++;
    }
    puts("\n");
}

double tensor_iter(Tensor *tensor, unsigned iter) {
    int ir = 0, target = 0;
    while (ir < tensor->dim[0]) { //Rows
        int ic = 0;
        while (ic < tensor->dim[1]) { //Columns
            if (target == iter) {
                return tensor->matrix[ir][ic];
            }
            ic++;
            target++;
        }
        ir++;
    }
    return 0;
}

double tensor_set(Tensor *tensor, unsigned iter, double val) {
    int ir = 0, target = 0;
    while (ir < tensor->dim[0]) { //Rows
        int ic = 0;
        while (ic < tensor->dim[1]) { //Columns
            if (target == iter) {
                tensor->matrix[ir][ic] = val;
                return val;
            }
            ic++;
            target++;
        }
        ir++;
    }
    return 0;
}
