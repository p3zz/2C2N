#include "common.h"
#include "stdlib.h"
#include "utils.h"
#include "stdio.h"

void cross_correlation(const matrix2d_t* const m1, const matrix2d_t* const m2, matrix2d_t* result, int padding, int stride){
    int output_rows = (m1->rows_n - m2->rows_n + 2 * padding) / stride + 1;
    int output_cols = (m1->cols_n - m2->cols_n + 2 * padding) / stride + 1;

    create_matrix2d(result, output_rows, output_cols);

    for(int i=0;i<result->rows_n;i++){
        for(int j=0;j<result->cols_n;j++){
            float sum = 0;
            for(int m=0;m<m2->rows_n;m++){
                for(int n=0;n<m2->cols_n;n++){
                    int row = i*stride + m - padding;
                    int col = j*stride + n - padding;
                    if(row >= 0 && row < m1->rows_n && col >= 0 && col < m1->cols_n){
                        sum += (m1->values[row][col] * m2->values[m][n]);
                    }
                }
            }
            result->values[i][j] = sum;
        }
    }
}

void create_matrix2d(matrix2d_t* m, int rows_n, int cols_n){
    m->rows_n = rows_n;
    m->cols_n = cols_n;
    m->values = (float**)malloc(m->rows_n * sizeof(float*));
    for(int i=0;i<m->rows_n;i++){
        m->values[i] = (float*)malloc(m->cols_n * sizeof(float));
        for(int j=0;j<m->cols_n;j++){
            m->values[i][j] = generate_random();
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
        create_matrix2d(&m->layers[i], rows_n, cols_n);
    }
}

void destroy_matrix3d(matrix3d_t* m){
    for(int i=0;i<m->depth;i++){
        destroy_matrix2d(&m->layers[i]);
    }
    free(m->layers);
}

int max_pooling(const matrix2d_t* const mat, int kernel_size, matrix2d_t* result, int padding, int stride){
    int output_rows = (mat->rows_n - kernel_size + 2 * padding) / stride + 1;
    int output_cols = (mat->cols_n - kernel_size + 2 * padding) / stride + 1;

    create_matrix2d(result, output_rows, output_cols);

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

int avg_pooling(const matrix2d_t* const mat, int kernel_size, matrix2d_t* result, int padding, int stride){
    int output_rows = (mat->rows_n - kernel_size + 2 * padding) / stride + 1;
    int output_cols = (mat->cols_n - kernel_size + 2 * padding) / stride + 1;

    create_matrix2d(result, output_rows, output_cols);

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
    create_matrix2d(result, m->rows_n, m->cols_n);
    for(int i=0;i<m->rows_n;i++){
        for(int j=0;j<m->rows_n;j++){
            result->values[i][j] = relu(m->values[i][j]);
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