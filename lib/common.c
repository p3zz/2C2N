#include "common.h"
#include "stdlib.h"
#include "utils.h"
#include "stdio.h"

void zero_pad(const matrix2d_t* const m, matrix2d_t* result, int padding){
    int output_rows = m->rows_n + 2 * padding;
    int output_cols = m->cols_n + 2 * padding;
    create_matrix2d(result, output_rows, output_cols, false);

    for (int i = 0; i < m->rows_n; i++) {
        for (int j = 0; j < m->cols_n; j++) {
            int row = i + padding;
            int col = j + padding;
            result->values[row][col] = m->values[i][j];
        }
    }
}

void full_cross_correlation(const matrix2d_t* const m1, const matrix2d_t* const m2, matrix2d_t* result, int padding, int stride){
    matrix2d_t m1_pad = {0};
    zero_pad(m1, &m1_pad, padding);

    int output_rows = (m1_pad.rows_n - m2->rows_n) / stride + 1;
    int output_cols = (m1_pad.cols_n - m2->cols_n) / stride + 1;

    create_matrix2d(result, output_rows, output_cols, true);

    for(int i=0;i<result->rows_n;i++){
        for(int j=0;j<result->cols_n;j++){
            float sum = 0;
            for(int m=0;m<m2->rows_n;m++){
                for(int n=0;n<m2->cols_n;n++){
                    int row = i * stride + m;
                    int col = j * stride + n;
                    sum += (m1_pad.values[row][col] * m2->values[m][n]);
                }
            }
            result->values[i][j] = sum;
        }
    }

    destroy_matrix2d(&m1_pad);
}

void convolution(const matrix2d_t* const m1, const matrix2d_t* const m2, matrix2d_t* result, int padding){
    matrix2d_t m2_rot = {0};
    matrix2d_rotate180(m2, &m2_rot);
    full_cross_correlation(m1, &m2_rot, result, padding, 1);
    destroy_matrix2d(&m2_rot);
}

float cross_correlation(const matrix2d_t* const m1, const matrix2d_t* const m2, float result){
    float sum = 0;
    for(int i=0;i<m1->rows_n;i++){
        for(int j=0;j<m1->rows_n;j++){
            sum += (m1->values[i][j] * m2->values[i][j]);
        }
    }
    return sum;
}

void matrix2d_mul(const matrix2d_t* const m1, const matrix2d_t* const m2, matrix2d_t* result){
    create_matrix2d(result, m1->rows_n, m1->cols_n, false);
    for(int i=0;i<m1->rows_n;i++){
        for(int j=0;j<m1->cols_n;j++){
            result->values[i][j] = (m1->values[i][j] * m2->values[i][j]);
        }
    }
}

void matrix2d_mul_inplace(const matrix2d_t* const m1, const matrix2d_t* const m2){
    for(int i=0;i<m1->rows_n;i++){
        for(int j=0;j<m1->cols_n;j++){
            m1->values[i][j] *= m2->values[i][j];
        }
    }
}

void create_matrix2d(matrix2d_t* m, int rows_n, int cols_n, bool random){
    m->rows_n = rows_n;
    m->cols_n = cols_n;
    m->values = (float**)malloc(m->rows_n * sizeof(float*));
    for(int i=0;i<m->rows_n;i++){
        m->values[i] = (float*)malloc(m->cols_n * sizeof(float));
        for(int j=0;j<m->cols_n;j++){
            if(random){
                m->values[i][j] = generate_random();
            }
            else{
                m->values[i][j] = 0.f;
            }
		}
    }
}

void matrix2d_copy(const matrix2d_t* const input, matrix2d_t* output){
	create_matrix2d(output, input->rows_n, input->cols_n, true);
    for(int i=0;i<input->rows_n;i++){
        for(int j=0;j<input->cols_n;j++){
            output->values[i][j] = input->values[i][j];
        }
    }
}

void matrix2d_erase(matrix2d_t* input){
    for(int i=0;i<input->rows_n;i++){
        for(int j=0;j<input->cols_n;j++){
            input->values[i][j] = 0.f;
        }
    }
}

void matrix3d_erase(matrix3d_t* input){
    for(int i=0;i<input->depth;i++){
        matrix2d_erase(&input->layers[i]);
    }
}

void matrix3d_copy(const matrix3d_t* const input, matrix3d_t* output){
    output->depth = input->depth;
    output->layers = (matrix2d_t*)malloc(output->depth * sizeof(matrix2d_t));
    for(int i=0;i<output->depth;i++){
        matrix2d_copy(&input->layers[i], &output->layers[i]);
    }
}

void matrix2d_rotate180(const matrix2d_t* const input, matrix2d_t* output){
    create_matrix2d(output, input->rows_n, input->cols_n, false);
    for(int i=0;i<input->rows_n;i++){
        for(int j=0;j<input->cols_n;j++){
            output->values[i][j] = input->values[input->rows_n - i - 1][input->cols_n - j - 1];
        }
    }
}

void matrix2d_rotate180_inplace(const matrix2d_t* const input){
    float aux = 0;
    int i_opposite = 0;
    int j_opposite = 0;
    for(int i=0;i<input->rows_n;i++){
        for(int j=0;j<input->cols_n;j++){
            i_opposite = input->rows_n - i - 1;
            j_opposite = input->cols_n - j - 1;
            aux = input->values[i][j];
            input->values[i][j] = input->values[i_opposite][j_opposite];
            input->values[i_opposite][j_opposite] = aux;
        }
    }
}

void matrix2d_submatrix(const matrix2d_t* const input, matrix2d_t* output, int row_start, int row_end, int col_start, int col_end){
    int output_rows = row_end - row_start + 1;
    int output_cols = col_end - col_start + 1;
    create_matrix2d(output, output_rows, output_cols, false);

    for(int i=0;i<output_rows;i++){
        for(int j=0;j<output_cols;j++){
            output->values[i][j] = input->values[i+output_rows-1][j+output_cols-1];
        }
    }
}

void destroy_matrix2d(matrix2d_t* m){
    for(int i=0;i<m->rows_n;i++){
        free(m->values[i]);
    }
    free(m->values);
}

void create_matrix3d(matrix3d_t* m, int rows_n, int cols_n, int depth){
    m->depth = depth;
    m->layers = (matrix2d_t*)malloc(m->depth * sizeof(matrix2d_t));
    for(int i=0;i<m->depth;i++){
        create_matrix2d(&m->layers[i], rows_n, cols_n, true);
    }
}

void destroy_matrix3d(matrix3d_t* m){
    for(int i=0;i<m->depth;i++){
        destroy_matrix2d(&m->layers[i]);
    }
    free(m->layers);
}

void max_pooling(const matrix2d_t* const mat, matrix2d_t* result, int kernel_size, int padding, int stride){
    int output_rows = (mat->rows_n - kernel_size + 2 * padding) / stride + 1;
    int output_cols = (mat->cols_n - kernel_size + 2 * padding) / stride + 1;

    create_matrix2d(result, output_rows, output_cols, true);

    for(int i=0;i<result->rows_n;i++){
        for(int j=0;j<result->cols_n;j++){
            float max = 0;
            for(int m=0;m<kernel_size;m++){
                for(int n=0;n<kernel_size;n++){
                    int row = i*stride + m - padding;
                    int col = j*stride + n - padding;
                    if(row >= 0 && row < mat->rows_n && col >= 0 && col < mat->cols_n){
                        if(mat->values[row][col] > max){
                            max = mat->values[row][col];
                        }
                    }
                }
            }
            result->values[i][j] = max;
        }
    }
}

void avg_pooling(const matrix2d_t* const mat, matrix2d_t* result, int kernel_size, int padding, int stride){
    int output_rows = (mat->rows_n - kernel_size + 2 * padding) / stride + 1;
    int output_cols = (mat->cols_n - kernel_size + 2 * padding) / stride + 1;

    create_matrix2d(result, output_rows, output_cols, true);

    for(int i=0;i<result->rows_n;i++){
        for(int j=0;j<result->cols_n;j++){
            float sum = 0;
            int values_len = 0;
            for(int m=0;m<kernel_size;m++){
                for(int n=0;n<kernel_size;n++){
                    int row = i*stride + m - padding;
                    int col = j*stride + n - padding;
                    if(row >= 0 && row < mat->rows_n && col >= 0 && col < mat->cols_n){
                        sum += mat->values[row][col];
                        values_len++;
                    }
                }
            }
            result->values[i][j] = sum/values_len;
        }
    }
}

void matrix2d_relu(const matrix2d_t* const m, matrix2d_t* result){
    create_matrix2d(result, m->rows_n, m->cols_n, true);
    for(int i=0;i<m->rows_n;i++){
        for(int j=0;j<m->rows_n;j++){
            result->values[i][j] = relu(m->values[i][j]);
        }
    }
}

void matrix2d_relu_inplace(const matrix2d_t* const m){
    for(int i=0;i<m->rows_n;i++){
        for(int j=0;j<m->rows_n;j++){
            m->values[i][j] = relu(m->values[i][j]);
        }
    }
}

void matrix2d_sigmoid(const matrix2d_t* const m, matrix2d_t* result){
    create_matrix2d(result, m->rows_n, m->cols_n, true);
    for(int i=0;i<m->rows_n;i++){
        for(int j=0;j<m->rows_n;j++){
            result->values[i][j] = sigmoid(m->values[i][j]);
        }
    }
}

void matrix2d_sigmoid_inplace(const matrix2d_t* const m){
    for(int i=0;i<m->rows_n;i++){
        for(int j=0;j<m->rows_n;j++){
            m->values[i][j] = sigmoid(m->values[i][j]);
        }
    }
}

void matrix2d_sum_inplace(const matrix2d_t* const m, matrix2d_t* result){
    for(int i=0;i<m->rows_n;i++){
        for(int j=0;j<m->cols_n;j++){
            result->values[i][j] += m->values[i][j];
        }
    }
}

void matrix2d_print(const matrix2d_t* const m){
    for(int i=0;i<m->rows_n;i++){
        printf("|");
        for(int j=0;j<m->cols_n;j++){
            printf("\t%.3f\t|", m->values[i][j]);
        }
        printf("\n");
    }
}

void matrix3d_print(const matrix3d_t* const m){
    for(int i=0;i<m->depth;i++){
        printf("[Layer %d]\n", i);
        matrix2d_print(&m->layers[i]);
    }
}

void matrix2d_flatten(const matrix2d_t* const m, matrix2d_t* result){
    int idx = 0;
    create_matrix2d(result, 1, m->rows_n * m->cols_n, true);
    for(int i=0;i<m->rows_n;i++){
        for(int j=0;j<m->cols_n;j++){
            result->values[0][idx] = m->values[i][j];
            idx++;
        }
    }
}