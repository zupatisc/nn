#include "utils.h"
#include <stdio.h>

int write_tensor(Tensor *tensor, char *file_name) {

    return EXIT_SUCCESS;
}

Tensor *read_tensor(char *file_name, unsigned col) {
    char file_buffer[BUFFER_SIZE] = { 0 };
    const char break_char = ',';

    unsigned tensor_buffer_size = BUFFER_SIZE;
    char *tensor_buffer = malloc(sizeof(char) * tensor_buffer_size);
    unsigned it = 0; // iterator for the current tensor_buffer pos.

    FILE *csv_file = fopen(file_name, "r");

    if (csv_file) {
        while (fgets(file_buffer, BUFFER_SIZE, csv_file)) {

            for (unsigned i = 0, c = 0; i < BUFFER_SIZE; i++) {
                if (file_buffer[i] == '\n')
                    break;
                if (file_buffer[i] == break_char)
                    c++;
                if (c > col) {
                    break;
                } else if (c == col && file_buffer[i] != break_char) {
                    tensor_buffer[it] = file_buffer[i];
                    tensor_buffer[it+1] = ' ';
                    it+=2;
                    if (it >= tensor_buffer_size) {
                        tensor_buffer_size += tensor_buffer_size;
                        tensor_buffer = realloc(tensor_buffer, sizeof(char) * tensor_buffer_size);
                        if (!tensor_buffer) return NULL;
                    }
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

    Tensor *new_tensor = tensor_init((it/2), 1, 0);
    char *pStart = tensor_buffer;
    char *pEnd = NULL;

    for (unsigned i = 0; i < (it/2); i++) {
        new_tensor->matrix[i][0] = strtod(pStart, &pEnd);
        pStart = pEnd;
    }

    free(tensor_buffer);
    return new_tensor;
}
