#include "common.h"
#include "stdlib.h"
#include "utils.h"
#include "stdio.h"
#include "math.h"

void full_cross_correlation(const matrix2d_t* const m1, const matrix2d_t* const m2, matrix2d_t* result, int padding, int stride){
    for(int i=0;i<result->rows_n;i++){
        for(int j=0;j<result->cols_n;j++){
            float sum = 0;
            for(int m=0;m<m2->rows_n;m++){
                for(int n=0;n<m2->cols_n;n++){
                    int row = i * stride + m - padding;
                    int col = j * stride + n - padding;
                    if(row >= 0 && row < m1->rows_n && col >= 0 && col < m1->cols_n){
                        sum += (m1->values[row][col] * m2->values[m][n]);
                    }
                }
            }
            result->values[i][j] = sum;
        }
    }
}

void convolution(const matrix2d_t* const m1, const matrix2d_t* const m2, matrix2d_t* result, int padding){
    matrix2d_rotate180_inplace(m2);
    full_cross_correlation(m1, m2, result, padding, 1);
    // restore the rotated matrix
    matrix2d_rotate180_inplace(m2);
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
    matrix2d_init(result, m1->rows_n, m1->cols_n);
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

void matrix2d_init(matrix2d_t* m, int rows_n, int cols_n){
    m->rows_n = rows_n;
    m->cols_n = cols_n;
    m->values = (float**)malloc(m->rows_n * sizeof(float*));
    for(int i=0;i<m->rows_n;i++){
        m->values[i] = (float*)malloc(m->cols_n * sizeof(float));
        for(int j=0;j<m->cols_n;j++){
            m->values[i][j] = 0.f;
		}
    }
}

void matrix2d_copy(const matrix2d_t* const input, matrix2d_t* output){
	matrix2d_init(output, input->rows_n, input->cols_n);
    for(int i=0;i<input->rows_n;i++){
        for(int j=0;j<input->cols_n;j++){
            output->values[i][j] = input->values[i][j];
        }
    }
}

void matrix2d_randomize(matrix2d_t* input){
    for(int i=0;i<input->rows_n;i++){
        for(int j=0;j<input->cols_n;j++){
            input->values[i][j] = generate_random();
        }
    }
}

void matrix3d_randomize(matrix3d_t* input){
    for(int i=0;i<input->depth;i++){
        matrix2d_randomize(&input->layers[i]);
    }
}

void matrix2d_copy_inplace(const matrix2d_t* const input, const matrix2d_t* output){
    if(input->rows_n != output->rows_n || input->cols_n != output->cols_n){
        return;
    }
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

void matrix3d_copy_inplace(const matrix3d_t* const input, const matrix3d_t* output){
    for(int i=0;i<output->depth;i++){
        matrix2d_copy_inplace(&input->layers[i], &output->layers[i]);
    }
}

void matrix2d_rotate180(const matrix2d_t* const input, matrix2d_t* output){
    matrix2d_init(output, input->rows_n, input->cols_n);
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
    matrix2d_init(output, output_rows, output_cols);

    for(int i=0;i<output_rows;i++){
        for(int j=0;j<output_cols;j++){
            output->values[i][j] = input->values[i+row_start][j+col_start];
        }
    }
}

void matrix2d_destroy(matrix2d_t* m){
    for(int i=0;i<m->rows_n;i++){
        free(m->values[i]);
    }
    free(m->values);
}

void matrix3d_init(matrix3d_t* m, int rows_n, int cols_n, int depth){
    m->depth = depth;
    m->layers = (matrix2d_t*)malloc(m->depth * sizeof(matrix2d_t));
    for(int i=0;i<m->depth;i++){
        matrix2d_init(&m->layers[i], rows_n, cols_n);
    }
}

void matrix3d_destroy(matrix3d_t* m){
    for(int i=0;i<m->depth;i++){
        matrix2d_destroy(&m->layers[i]);
    }
    free(m->layers);
}

void max_pooling(const matrix2d_t* const mat, matrix2d_t* result, matrix3d_t* indexes, int kernel_size, int padding, int stride){
    for(int i=0;i<result->rows_n;i++){
        for(int j=0;j<result->cols_n;j++){
            float max = 0;
            int max_i = 0;
            int max_j = 0;
            for(int m=0;m<kernel_size;m++){
                for(int n=0;n<kernel_size;n++){
                    int row = i*stride + m - padding;
                    int col = j*stride + n - padding;
                    if(row >= 0 && row < mat->rows_n && col >= 0 && col < mat->cols_n){
                        if(mat->values[row][col] > max){
                            max_i = row;
                            max_j = col;
                            max = mat->values[max_i][max_j];
                        }
                    }
                }
            }
            result->values[i][j] = max;
            indexes->layers[0].values[i][j] = max_i;
            indexes->layers[1].values[i][j] = max_j;
        }
    }
}

void avg_pooling(const matrix2d_t* const mat, matrix2d_t* result, int kernel_size, int padding, int stride){
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
    if(m->rows_n != result->rows_n || m->cols_n != result->cols_n){
        return;
    }
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
    if(m->rows_n != result->rows_n || m->cols_n != result->cols_n){
        return;
    }
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

void matrix2d_tanh_inplace(const matrix2d_t* const m){
    for(int i=0;i<m->rows_n;i++){
        for(int j=0;j<m->rows_n;j++){
            m->values[i][j] = tanhf(m->values[i][j]);
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

void matrix2d_reshape(const matrix2d_t* const m, matrix2d_t* result, int rows_n, int cols_n){
    int m_elems_n = m->rows_n * m->cols_n;
    int result_elems_n = rows_n * cols_n;
    if(m_elems_n != result_elems_n){
        return;
    }
    matrix2d_init(result, rows_n, cols_n);
    for (int i = 0; i < result_elems_n; i++) {
        result->values[i / result->cols_n][i % result->cols_n] = m->values[i / m->cols_n][i % m->cols_n];
    }
}

void compute_cost_derivative(const matrix2d_t* const output, const matrix2d_t* const target_output, matrix2d_t* result){
	for(int i=0;i<output->rows_n;i++){
		for(int j=0;j<output->cols_n;j++){
     	   result->values[i][j] = 2*(output->values[i][j] - target_output->values[i][j]);
    	}
	}
}
