#include "common.h"
#include "stdlib.h"
#include "utils.h"
#include "stdio.h"
#include "math.h"
#include "string.h"

inline const float* matrix2d_get_elem_as_ref(const matrix2d_t* const m, int row_idx, int col_idx){
    return matrix2d_get_elem_as_mut_ref(m, row_idx, col_idx);
}

inline float* matrix2d_get_elem_as_mut_ref(const matrix2d_t* const m, int row_idx, int col_idx){
    return &m->values[row_idx * m->cols_n + col_idx];
}

inline float matrix2d_get_elem(const matrix2d_t* const m, int row_idx, int col_idx){
    return *matrix2d_get_elem_as_ref(m, row_idx, col_idx);
}

inline void matrix2d_set_elem(const matrix2d_t* m, int row_idx, int col_idx, float value){
    *matrix2d_get_elem_as_mut_ref(m, row_idx, col_idx) = value;
}

inline const float* matrix3d_get_elem_as_ref(const matrix3d_t* const m, int row_idx, int col_idx, int z_idx){
    return matrix3d_get_elem_as_mut_ref(m, row_idx, col_idx, z_idx);
}

inline float* matrix3d_get_elem_as_mut_ref(const matrix3d_t* const m, int row_idx, int col_idx, int z_idx){
    return &m->values[(z_idx * m->rows_n * m->cols_n) + (row_idx * m->cols_n) + col_idx];
}

inline float matrix3d_get_elem(const matrix3d_t* const m, int row_idx, int col_idx, int z_idx){
    return *matrix3d_get_elem_as_ref(m, row_idx, col_idx, z_idx);
}

inline void matrix3d_set_elem(const matrix3d_t* const m, int row_idx, int col_idx, int z_idx, float value){
    *matrix3d_get_elem_as_mut_ref(m, row_idx, col_idx, z_idx) = value;
}

void matrix3d_get_slice_as_mut_ref(const matrix3d_t* m, matrix2d_t* result, int z_idx){
    result->rows_n = m->rows_n;
    result->cols_n = m->cols_n;
    result->values = &m->values[result->rows_n * result->cols_n * z_idx];
}

void zero_pad(const matrix2d_t* const m, matrix2d_t* result, int padding){
    for (int i = 0; i < m->rows_n; i++) {
        for (int j = 0; j < m->cols_n; j++) {
            int row = i + padding;
            int col = j + padding;
            float v = matrix2d_get_elem(m, i, j);
            matrix2d_set_elem(result, row, col, v);
        }
    }
}

void full_cross_correlation(const matrix2d_t* const m1, const matrix2d_t* const m2, matrix2d_t* result, int padding, int stride){
    for(int i=0;i<result->rows_n;i++){
        for(int j=0;j<result->cols_n;j++){
            float sum = 0;
            for(int m=0;m<m2->rows_n;m++){
                for(int n=0;n<m2->cols_n;n++){
                    int row = i * stride + m - padding;
                    int col = j * stride + n - padding;
                    if(row >= 0 && row < m1->rows_n && col >= 0 && col < m1->cols_n){
                        float v1 = matrix2d_get_elem(m1, row, col);
                        float v2 = matrix2d_get_elem(m2, m, n);
                        sum += (v1 * v2);
                    }
                }
            }
            matrix2d_set_elem(result, i, j, sum);
        }
    }
}

void convolution(const matrix2d_t* const m1, const matrix2d_t* const m2, matrix2d_t* result, int padding){
    matrix2d_rotate180_inplace(m2);
    full_cross_correlation(m1, m2, result, padding, 1);
    // restore the rotated matrix
    matrix2d_rotate180_inplace(m2);
}

void matrix2d_element_wise_product_inplace(const matrix2d_t* const m1, const matrix2d_t* const m2){
    for(int i=0;i<m1->rows_n;i++){
        for(int j=0;j<m1->cols_n;j++){
            *matrix2d_get_elem_as_mut_ref(m1, i, j) *= matrix2d_get_elem(m2, i, j);
        }
    }
}

void matrix2d_init(matrix2d_t* m, int rows_n, int cols_n){
    m->rows_n = rows_n;
    m->cols_n = cols_n;
    m->values = (float*)malloc(m->rows_n * cols_n);
    for(int i=0;i<m->rows_n;i++){
        for(int j=0;j<m->cols_n;j++){
            matrix2d_set_elem(m, i, j, 0.f);
        }
    }
}

void matrix2d_randomize(matrix2d_t* input){
    for(int i=0;i<input->rows_n;i++){
        for(int j=0;j<input->cols_n;j++){
            float v = generate_random();
            matrix2d_set_elem(input, i, j, v);
        }
    }
}

void matrix3d_randomize(matrix3d_t* input){
    matrix2d_t layer = {0};
    for(int i=0;i<input->depth;i++){
        matrix3d_get_slice_as_mut_ref(input, &layer, i);
        matrix2d_randomize(&layer);
    }
}

void matrix2d_copy_inplace(const matrix2d_t* const input, const matrix2d_t* output){
    if(input->rows_n != output->rows_n || input->cols_n != output->cols_n){
        return;
    }
    for(int i=0;i<input->rows_n;i++){
        for(int j=0;j<input->cols_n;j++){
            *matrix2d_get_elem_as_mut_ref(output, i, j) = matrix2d_get_elem(input, i, j);
        }
    }
}

void matrix3d_copy_inplace(const matrix3d_t* const input, const matrix3d_t* output){
    matrix2d_t i_layer = {0};
    matrix2d_t o_layer = {0};
    for(int i=0;i<output->depth;i++){
        matrix3d_get_slice_as_mut_ref(input, &i_layer, i);
        matrix3d_get_slice_as_mut_ref(output, &o_layer, i);
        matrix2d_copy_inplace(&i_layer, &o_layer);
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
            aux = matrix2d_get_elem(input, i, j);
            *matrix2d_get_elem_as_mut_ref(input, i, j) = matrix2d_get_elem(input, i_opposite, j_opposite);
            matrix2d_set_elem(input, i_opposite, j_opposite, aux);
        }
    }
}

void matrix2d_destroy(matrix2d_t* m){
    free(m->values);
}

void matrix3d_init(matrix3d_t* m, int rows_n, int cols_n, int depth){
    m->rows_n = rows_n;
    m->cols_n = cols_n;
    m->depth = depth;
    m->values = (float*)malloc(rows_n * cols_n * depth * sizeof(float));
    for(int i=0;i<m->rows_n;i++){
        for(int j=0;j<m->cols_n;j++){
            for(int z=0;z<m->depth;z++){
                *matrix3d_get_elem_as_mut_ref(m, i, j, z) = 0.f;
            }
        }
    }
}

void matrix3d_destroy(matrix3d_t* m){
    free(m->values);
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
                        float val = matrix2d_get_elem(mat, row, col);
                        if(val > max){
                            max_i = row;
                            max_j = col;
                            max = val;
                        }
                    }
                }
            }
            matrix2d_set_elem(result, i, j, max);
            matrix3d_set_elem(indexes, i, j, 0, max_i);
            matrix3d_set_elem(indexes, i, j, 1, max_j);
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
                        sum += matrix2d_get_elem(mat, row, col);
                        values_len++;
                    }
                }
            }
            matrix2d_set_elem(mat, i, j, sum/values_len);
        }
    }
}

void matrix2d_relu_inplace(const matrix2d_t* const m){
    for(int i=0;i<m->rows_n;i++){
        for(int j=0;j<m->cols_n;j++){
            float* elem = matrix2d_get_elem_as_mut_ref(m, i, j);
            *elem = relu(*elem);
        }
    }
}

void matrix2d_sigmoid_inplace(const matrix2d_t* const m){
    for(int i=0;i<m->rows_n;i++){
        for(int j=0;j<m->cols_n;j++){
            float* elem = matrix2d_get_elem_as_mut_ref(m, i, j);
            *elem = sigmoid(*elem);
        }
    }
}

void matrix2d_tanh_inplace(const matrix2d_t* const m){
    for(int i=0;i<m->rows_n;i++){
        for(int j=0;j<m->cols_n;j++){
            float* elem = matrix2d_get_elem_as_mut_ref(m, i, j);
            *elem = tanhf(*elem);
        }
    }
}

void matrix2d_softmax_inplace(matrix2d_t* m){
    float sum = 0.0;
    float maxInput = matrix2d_get_elem(m, 0, 0);

    // normalize to avoid overflow on exp
    for (int i = 1; i < m->cols_n; i++) {
        float m_val = matrix2d_get_elem(m, 0, i);
        if (m_val > maxInput) {
            maxInput = m_val;
        }
    }

    // Calculate the exponentials of each input element and the sum of exponentials
    for (int i = 0; i < m->cols_n; i++) {
        float* m_val = matrix2d_get_elem_as_mut_ref(m, 0, i);
        *m_val = exp(*m_val / maxInput);
        sum += *m_val;
    }

    // Normalize the values to make the sum equal to 1 (probabilities)
    for (int i = 0; i < m->cols_n; i++) {
        *matrix2d_get_elem_as_mut_ref(m, 0, i) /= sum;
    }
}

void matrix2d_sum_inplace(const matrix2d_t* const m, matrix2d_t* result){
    for(int i=0;i<m->rows_n;i++){
        for(int j=0;j<m->cols_n;j++){
            *matrix2d_get_elem_as_mut_ref(result, i, j) += matrix2d_get_elem(m, i, j);
        }
    }
}

void matrix2d_print(const matrix2d_t* const m){
    for(int i=0;i<m->rows_n;i++){
        printf("|");
        for(int j=0;j<m->cols_n;j++){
            printf(" %.2f |", matrix2d_get_elem(m, i, j));
        }
        printf("\n");
    }
}

void matrix3d_print(const matrix3d_t* const m){
    matrix2d_t slice = {0};
    for(int i=0;i<m->depth;i++){
        printf("[Layer %d]\n", i);
        matrix3d_get_slice_as_mut_ref(m, &slice, i);
        matrix2d_print(&slice);
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
        *matrix2d_get_elem_as_mut_ref(result, i / result->cols_n, i % result->cols_n) = matrix2d_get_elem(m, i / m->cols_n, i % m->cols_n);
    }
}

void matrix3d_reshape(const matrix3d_t* const m, matrix3d_t* result){
    int m_elems_n = m->rows_n * m->cols_n * m->depth;
    int result_elems_n = result->rows_n * result->cols_n * result->depth;
    if(m_elems_n != result_elems_n){
        return;
    }
    for (int i = 0; i < result_elems_n; i++) {
        int m_i = i / (m->rows_n * m->cols_n);
        int m_j = (i % (m->rows_n * m->cols_n)) / m->cols_n;
        int m_k = i % m->cols_n;

        int result_i = i / (result->rows_n * result->cols_n);
        int result_j = (i % (result->rows_n * result->cols_n)) / result->cols_n;
        int result_k = i % result->cols_n;
        *matrix3d_get_elem_as_mut_ref(result, result_j, result_k, result_i) = matrix3d_get_elem(m, m_j, m_k, m_i);
    }
}

void matrix2d_load(matrix2d_t* m, int rows_n, int cols_n, const float* base_address){
    m->rows_n = rows_n;
    m->cols_n = cols_n;
    m->values = base_address;
}

void matrix3d_load(matrix3d_t* m, int rows_n, int cols_n, int depth, const float* base_address){
    m->rows_n = rows_n;
    m->cols_n = cols_n;
    m->depth = depth;
    m->values = base_address;
}

void mean_squared_error_derivative(const matrix2d_t* const output, const matrix2d_t* const target_output, matrix2d_t* result){
    for(int i=0;i<output->cols_n;i++){
        float out_val = matrix2d_get_elem(output, 0, i);
        float target_out_val = matrix2d_get_elem(target_output, 0, i);
        float res = 2*(out_val - target_out_val) / output->cols_n;
        matrix2d_set_elem(result, 0, i, res);
	}
}

void cross_entropy_loss_derivative(const matrix2d_t* const output, const matrix2d_t* const target_output, matrix2d_t* result){
    for(int i=0;i<output->cols_n;i++){
        float n0 = 1 - matrix2d_get_elem(target_output, 0, i);
        float d0 = 1 - matrix2d_get_elem(output, 0, i);
        float n1 = matrix2d_get_elem(target_output, 0, i);
        float d1 = matrix2d_get_elem(output, 0, i);
        float res;
        if(d0 < 1e-4 || d1 < 1e-4){
            res = 0.f;    
        }
        else{
            res = (n0/d0 - n1/d1) / output->cols_n;
        }
        matrix2d_set_elem(result, 0, i, res);
    }
}

float mean_squared_error(const matrix2d_t* const output, const matrix2d_t* const target_output){
    float sum = 0.f;
    for(int i=0;i<output->cols_n;i++){
        float delta = matrix2d_get_elem(target_output, 0, i) - matrix2d_get_elem(output, 0, i);
        sum += (delta * delta);
    }
    return sum / output->cols_n;
}

float cross_entropy_loss(const matrix2d_t* const output, const matrix2d_t* const target_output){
    float sum = 0.f;
    for(int i=0;i<output->cols_n;i++){
        float target_out = matrix2d_get_elem(target_output, 0, i);
        float out = matrix2d_get_elem(output, 0, i);
        float d = (target_out * logf(out)) + ((1-target_out) * logf(1 - out));
        if(isnan(d)){
            d = 0.f;
        }
        sum += d;
    }
    return -sum / output->cols_n;
}

void parse_line(char* line, int length, matrix2d_t* image, float* label){
    char n[4] = {0};
    int n_idx = 0;
    int numbers_n = 0;
    bool label_read = false;
    for(int i=0;i<length;i++){
        // check if numbers length exceeds the capacity of the image matrix
        if(numbers_n == image->rows_n * image->cols_n){
            break;
        }
        if(line[i] == ',' || i == length - 1){
            n[n_idx] = '\0';
            float val = atof(n);
            if(!label_read){
                *label = val;
                label_read = true;
            }
            else{
                int row = numbers_n / image->cols_n;
                int col = numbers_n % image->cols_n;
                matrix2d_set_elem(image, row, col, val);
                numbers_n++;
            }
            n_idx = 0;
        }
        else{
            n[n_idx] = line[i];
            n_idx++;
        }
    }   
}
