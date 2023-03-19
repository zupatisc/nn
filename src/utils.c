#include "utils.h"
#include <stdio.h>

int write_tensor(Tensor *tensor, char *file_name) {

    FILE *csv_file = fopen(file_name, "w");

    if (!csv_file) {
        perror("fopen failed");
        return EXIT_FAILURE;
    }

    for (unsigned ic = 0; ic < tensor->dim[0]; ic++) {
        for (unsigned ir = 0; ir < tensor->dim[1]; ir++) {
            fprintf(csv_file, "%f", tensor->matrix[ic][ir]);
            if (ir < (tensor->dim[1]-1))
                fprintf(csv_file, ",");
        }
        fprintf(csv_file, "\n");
    }

    fclose(csv_file);

    return EXIT_SUCCESS;
}

Tensor *read_tensor(char *file_name, unsigned col) {
    char file_buffer[BUFFER_SIZE] = { 0 };
    const char break_char = ',';

    unsigned tensor_buffer_size = BUFFER_SIZE;
    char *tensor_buffer = malloc(sizeof(char) * tensor_buffer_size);
    unsigned it = 0; // iterator for the current tensor_buffer pos.
    unsigned number_count = 0;

    FILE *csv_file = fopen(file_name, "r");

    if (csv_file) {
        while (fgets(file_buffer, BUFFER_SIZE, csv_file)) {

            for (unsigned i = 0, c = 0; i < BUFFER_SIZE; i++) {
                if (it >= tensor_buffer_size) {
                    tensor_buffer_size += tensor_buffer_size;
                    tensor_buffer = realloc(tensor_buffer, sizeof(char) * tensor_buffer_size);
                    if (!tensor_buffer) return NULL;
                }

                if (file_buffer[i] == break_char)
                    c++;

                if (file_buffer[i] == '\n') {
                    tensor_buffer[it] = ' ';
                    number_count++;
                    it++;
                    break;
                } else if (c > col) {
                    tensor_buffer[it] = ' ';
                    number_count++;
                    it++;
                    break;
                } else if (c == col && file_buffer[i] != break_char) {
                    tensor_buffer[it] = file_buffer[i];
                    it++;
                }
            }
        }
        fclose(csv_file);
    } else {
        perror("fopen failed");
        free(tensor_buffer);
        return NULL;
    }

    /* for (int i = 0; i < it; i++) {
        printf("%c", tensor_buffer[i]);
    }
    puts(""); */

    Tensor *new_tensor = tensor_init(number_count, 1, 0);
    char *pStart = tensor_buffer;
    char *pEnd = NULL;

    for (unsigned i = 0; i < number_count; i++) {
        new_tensor->matrix[i][0] = strtod(pStart, &pEnd);
        pStart = pEnd;
    }

    free(tensor_buffer);
    return new_tensor;
}
