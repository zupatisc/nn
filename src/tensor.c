#include "tensor.h"
#include <math.h>
#include <stdio.h>

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
         for (unsigned u = 0; u < col; u++)
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

    for (unsigned i = 0; i < new_tensor->dim[0] * new_tensor->dim[1]; i++) {
        tensor_set(new_tensor, i, frand());
    }

    return new_tensor;
}

Tensor *tensor_like(Tensor *tensor, double default_val) {
    Tensor *new_tensor = tensor_init(tensor->dim[0], tensor->dim[1], default_val);
    return new_tensor;
}

int tensor_destroy(Tensor *tensor) { //TODO: return useful info
    if (tensor) {
        for(unsigned i = 0; i < tensor->dim[0]; i++) {
            free(tensor->matrix[i]);
        }
        free(tensor->matrix);
    }
    free(tensor);

    return 0;
}

static inline int tensor_broadcast_T2d1(Tensor *tensor_trgt, Tensor *tensor_1, Tensor *tensor_2) {

     //Taken from my python work
     for (unsigned ir = 0; ir < tensor_trgt->dim[0]; ir++) {
         for (unsigned ic = 0; ic < tensor_trgt->dim[1]; ic++) {
             double val = 0;
             for (unsigned i = 0; i < tensor_1->dim[1]; i++) {
                 val += tensor_1->matrix[ir][i] * tensor_2->matrix[i][0];
             }
             tensor_trgt->matrix[ir][ic] = val;
         }
     }

    return EXIT_SUCCESS;
}

int tensor_dot(Tensor *tensor_trgt, Tensor *tensor_1, Tensor *tensor_2) {
    if (tensor_1 == NULL || tensor_2 == NULL || tensor_trgt == NULL)
        return ETENNULL;

    // TODO: Do I really need broadcasting?
    /* if (tensor_1->dim[1] != tensor_2->dim[1]) {
        puts("dim 1 are unequal");
        if (tensor_1->dim[1] == 1) {
            puts("but tensor_1 dim 1 is 1");
        } else if (tensor_2->dim[1] == 1) {
            puts("but tensor_2 dim 1 is 1");
            tensor_broadcast_T2d1(tensor_trgt, tensor_1, tensor_2);
            return EXIT_SUCCESS;
        }
    } else if (tensor_1->dim[0] != tensor_2->dim[0]) {
        puts("dim 0 are unequal");
        if (tensor_1->dim[0] == 1) {
            puts("but tensor_1 dim 0 is 1");
        } else if (tensor_2->dim[0] == 1) {
            puts("but tensor_2 dim 0 is 1");
        }
    } */

    if (tensor_trgt->dim[0] != tensor_1->dim[0] || tensor_trgt->dim[1] != tensor_2->dim[1])
        return ETENMIS;

     //Taken from my python work
     for (unsigned ir = 0; ir < tensor_trgt->dim[0]; ir++) {
         for (unsigned ic = 0; ic < tensor_trgt->dim[1]; ic++) {
             double val = 0;
             for (unsigned i = 0; i < tensor_1->dim[1]; i++) {
                 val += tensor_1->matrix[ir][i] * tensor_2->matrix[i][ic];
             }
             tensor_trgt->matrix[ir][ic] = val;
         }
     }

    return EXIT_SUCCESS;
}

int tensor_add(Tensor *tensor_trgt, Tensor *tensor_1, Tensor*restrict tensor_2) {
    if (tensor_trgt == NULL || tensor_1 == NULL || tensor_2 == NULL)
        return ETENNULL;

    if (tensor_trgt->dim[0] != tensor_1->dim[0] || tensor_trgt->dim[1] != tensor_1->dim[1])
        return ETENMIS;

    unsigned ir = 0, ic = ir;
    unsigned *ir2 = &ir;
    unsigned *ic2 = &ic;
    unsigned hold_dim = 0;

    if (tensor_trgt->dim[0] != tensor_2->dim[0] || tensor_trgt->dim[1] != tensor_2->dim[1]) {
        if (tensor_2->dim[0] == 1 && tensor_2->dim[1] != 1) {
            puts("Dim 0 is 1"); //TODO: Remove printing
            ir2 = &hold_dim;
        } else if (tensor_2->dim[0] != 1 && tensor_2->dim[1] == 1) {
            puts("Dim 1 is 1"); //TODO: Remove printing
            ic2 = &hold_dim;
        } else {
            return ETENMIS;
        }
    }

    for(; ir < tensor_trgt->dim[0]; ir++) {
        for (; ic < tensor_trgt->dim[1]; ic++) {
            tensor_trgt->matrix[ir][ic] = tensor_1->matrix[ir][ic] + tensor_2->matrix[*ir2][*ic2];
        }
        ic = 0;
    }

    return EXIT_SUCCESS;
}

int tensor_sub(Tensor *tensor_trgt, Tensor *tensor_1, Tensor*restrict tensor_2) {
    if (tensor_trgt == NULL || tensor_1 == NULL || tensor_2 == NULL)
        return ETENNULL;

    if ((tensor_1->dim[0] != tensor_trgt->dim[0] && tensor_1->dim[0] != 1) || 
            (tensor_1->dim[1] != tensor_trgt->dim[1] && tensor_1->dim[1] != 1))
        return ETENMIS;
    if ((tensor_2->dim[0] != tensor_trgt->dim[0] && tensor_2->dim[0] != 1) || 
            (tensor_2->dim[1] != tensor_trgt->dim[1] && tensor_2->dim[1] != 1))
        return ETENMIS;


    unsigned irt = 0, ict = irt;
    unsigned *ir1 = &irt, *ic1 = &ict;
    unsigned *ir2 = &irt, *ic2 = &ict;
    unsigned hold_dim = 0;

    if (tensor_1->dim[0] == 1)
        ir1 = &hold_dim;
    if (tensor_1->dim[1] == 1)
        ic1 = &hold_dim;
    if (tensor_2->dim[0] == 1)
        ir2 = &hold_dim;
    if (tensor_2->dim[1] == 1)
        ic2 = &hold_dim;

    for (; irt < tensor_trgt->dim[0]; irt++) {
        for (; ict < tensor_trgt->dim[1]; ict++) {
            tensor_trgt->matrix[irt][ict] = tensor_1->matrix[*ir1][*ic1] - tensor_2->matrix[*ir2][*ic2];
        }
        ict = 0;
    }
    return EXIT_SUCCESS;
}

Tensor *tensor_transpose(Tensor *tensor) {
    Tensor *transpose = tensor_init(tensor->dim[1], tensor->dim[0], 0);

    for (unsigned ir = 0; ir < tensor->dim[0]; ir++) {
        for (unsigned ic = 0; ic < tensor->dim[1]; ic++) {
            transpose->matrix[ic][ir] = tensor->matrix[ir][ic];
        }
    }

    return transpose;
}

int tensor_sum(Tensor *tensor_trgt, Tensor *tensor_1, unsigned dim) {
    if (tensor_trgt == NULL || tensor_1 == NULL)
        return ETENMIS;

    switch (dim) {
        case 0:
            if (!(tensor_trgt->dim[0] == 1 && tensor_trgt->dim[1] == tensor_1->dim[1]))
                return ETENMIS;
            for (unsigned ic = 0; ic < tensor_1->dim[1]; ic++) {
                for (unsigned ir = 0; ir < tensor_1->dim[0]; ir++) {
                    tensor_trgt->matrix[0][ic] += tensor_1->matrix[ir][ic];
                }
            }
            break;
        case 1:
            if (!(tensor_trgt->dim[0] == tensor_1->dim[0] && tensor_trgt->dim[1] == 1))
                return ETENMIS;
            for (unsigned ir = 0; ir < tensor_1->dim[0]; ir++) {
                for (unsigned ic = 0; ic < tensor_1->dim[1]; ic++) {
                    tensor_trgt->matrix[ir][0] += tensor_1->matrix[ir][ic];
                }
            }
            break;
        default:
            return EDIM;
    }

    return EXIT_SUCCESS;
}

int tensor_mult(Tensor *tensor_trgt, Tensor *tensor_1, Tensor*restrict tensor_2) {
    if (tensor_trgt == NULL || tensor_1 == NULL || tensor_2 == NULL)
        return ETENNULL;

    if ((tensor_1->dim[0] != tensor_trgt->dim[0] && tensor_1->dim[0] != 1) || 
            (tensor_1->dim[1] != tensor_trgt->dim[1] && tensor_1->dim[1] != 1))
        return ETENMIS;
    if ((tensor_2->dim[0] != tensor_trgt->dim[0] && tensor_2->dim[0] != 1) || 
            (tensor_2->dim[1] != tensor_trgt->dim[1] && tensor_2->dim[1] != 1))
        return ETENMIS;


    unsigned irt = 0, ict = irt;
    unsigned *ir1 = &irt, *ic1 = &ict;
    unsigned *ir2 = &irt, *ic2 = &ict;
    unsigned hold_dim = 0;

    if (tensor_1->dim[0] == 1)
        ir1 = &hold_dim;
    if (tensor_1->dim[1] == 1)
        ic1 = &hold_dim;
    if (tensor_2->dim[0] == 1)
        ir2 = &hold_dim;
    if (tensor_2->dim[1] == 1)
        ic2 = &hold_dim;

    for (; irt < tensor_trgt->dim[0]; irt++) {
        for (; ict < tensor_trgt->dim[1]; ict++) {
            tensor_trgt->matrix[irt][ict] = tensor_1->matrix[*ir1][*ic1] * tensor_2->matrix[*ir2][*ic2];
        }
        ict = 0;
    }
    return EXIT_SUCCESS;
}

int tensor_pow(Tensor *tensor_trgt, Tensor *tensor_1, double exponent) {
    if (tensor_trgt == NULL || tensor_1 == NULL)
        return ETENNULL;
    if (tensor_1->dim[0] != tensor_trgt->dim[0] || tensor_1->dim[1] != tensor_trgt->dim[1])
        return ETENMIS;

    for (unsigned ir = 0; ir < tensor_trgt->dim[0]; ir++) {
        for (unsigned ic = 0; ic < tensor_trgt->dim[1]; ic++) {
            tensor_trgt->matrix[ir][ic] = pow(tensor_1->matrix[ir][ic], exponent);
        }
    }
    return EXIT_SUCCESS;
}

void tensor_print(Tensor *tensor) {
    unsigned ir = 0;
    while (ir < tensor->dim[0]) { //Rows
        unsigned ic = 0;
        while (ic < tensor->dim[1]) { //Columns
            printf("%f ", tensor->matrix[ir][ic]);
            ic++;
        }
        puts("\n");
        ir++;
    }
    puts("\n");
}

void tensor_shapes(Tensor *tensor) {
    printf("Shape: (%d, %d)\n", tensor->dim[0], tensor->dim[1]);
}

bool tensor_cmp(Tensor *tensor_1, Tensor *tensor_2) {
    if (tensor_1 == NULL || tensor_2 == NULL)
        return false;

    if (tensor_1->dim[1] != tensor_2->dim[1] || tensor_1->dim[0] != tensor_2->dim[0])
        return false;

    for (unsigned ir = 0; ir < tensor_1->dim[0]; ir++) {
        for (unsigned ic = 0; ic < tensor_1->dim[1]; ic++) {
            if (fabs(tensor_1->matrix[ir][ic] - tensor_2->matrix[ir][ic]) > CMPPREC)
                return false;
        }
    }
    return true;
}

double tensor_iter(Tensor *tensor, unsigned iter) {
    unsigned ir = 0, target = 0;
    while (ir < tensor->dim[0]) { //Rows
        unsigned ic = 0;
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
    unsigned ir = 0, target = 0;
    while (ir < tensor->dim[0]) { //Rows
        unsigned ic = 0;
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
