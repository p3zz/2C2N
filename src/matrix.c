#include "matrix.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

const float *matrix2d_get_elem_as_ref(const matrix2d_t *const m, int row_idx,
                                      int col_idx) {
  return matrix2d_get_elem_as_mut_ref(m, row_idx, col_idx);
}

float *matrix2d_get_elem_as_mut_ref(const matrix2d_t *const m, int row_idx,
                                    int col_idx) {
  if (row_idx < m->rows_n && col_idx < m->cols_n) {
    return &m->values[row_idx * m->cols_n + col_idx];
  }
  return NULL;
}

float matrix2d_get_elem(const matrix2d_t *const m, int row_idx, int col_idx) {
  const float *addr = matrix2d_get_elem_as_ref(m, row_idx, col_idx);
  if (addr != NULL) {
    return *addr;
  }
  return 0.f;
}

void matrix2d_set_elem(const matrix2d_t *const m, int row_idx, int col_idx,
                       float value) {
  float *addr = matrix2d_get_elem_as_mut_ref(m, row_idx, col_idx);
  if (addr != NULL) {
    *addr = value;
  }
}

const float *matrix3d_get_elem_as_ref(const matrix3d_t *const m, int row_idx,
                                      int col_idx, int z_idx) {
  return matrix3d_get_elem_as_mut_ref(m, row_idx, col_idx, z_idx);
}

float *matrix3d_get_elem_as_mut_ref(const matrix3d_t *const m, int row_idx,
                                    int col_idx, int z_idx) {
  if (row_idx < m->rows_n && col_idx < m->cols_n && z_idx < m->depth) {
    return &m->values[(z_idx * m->rows_n * m->cols_n) + (row_idx * m->cols_n) +
                      col_idx];
  }
  return NULL;
}

float matrix3d_get_elem(const matrix3d_t *const m, int row_idx, int col_idx,
                        int z_idx) {
  const float *addr = matrix3d_get_elem_as_ref(m, row_idx, col_idx, z_idx);
  if (addr != NULL) {
    return *addr;
  }
  return 0.f;
}

void matrix3d_set_elem(const matrix3d_t *const m, int row_idx, int col_idx,
                       int z_idx, float value) {
  float *addr = matrix3d_get_elem_as_mut_ref(m, row_idx, col_idx, z_idx);
  if (addr != NULL) {
    *addr = value;
  }
}

void matrix3d_get_slice_as_mut_ref(const matrix3d_t *const m,
                                   matrix2d_t *const result, int z_idx) {
  if (z_idx >= m->depth) {
    return;
  }
  result->rows_n = m->rows_n;
  result->cols_n = m->cols_n;
  result->values = &m->values[result->rows_n * result->cols_n * z_idx];
}

void matrix2d_element_wise_product_inplace(const matrix2d_t *const m1,
                                           const matrix2d_t *const m2) {
  for (int i = 0; i < m1->rows_n; i++) {
    for (int j = 0; j < m1->cols_n; j++) {
      *matrix2d_get_elem_as_mut_ref(m1, i, j) *= matrix2d_get_elem(m2, i, j);
    }
  }
}

void matrix2d_init(matrix2d_t *m, int rows_n, int cols_n) {
  m->loaded = false;
  m->rows_n = rows_n;
  m->cols_n = cols_n;
  m->values = (float *)malloc(rows_n * cols_n * sizeof(float));
  for (int i = 0; i < m->rows_n; i++) {
    for (int j = 0; j < m->cols_n; j++) {
      matrix2d_set_elem(m, i, j, 0.f);
    }
  }
}

void matrix2d_randomize(matrix2d_t *const input) {
  for (int i = 0; i < input->rows_n; i++) {
    for (int j = 0; j < input->cols_n; j++) {
      float v = generate_random();
      matrix2d_set_elem(input, i, j, v);
    }
  }
}

void matrix3d_randomize(matrix3d_t *const input) {
  matrix2d_t layer = {0};
  for (int i = 0; i < input->depth; i++) {
    matrix3d_get_slice_as_mut_ref(input, &layer, i);
    matrix2d_randomize(&layer);
  }
}

void matrix2d_copy_inplace(const matrix2d_t *const input,
                           const matrix2d_t *const output) {
  if (input->rows_n != output->rows_n || input->cols_n != output->cols_n) {
    return;
  }
  memcpy((void *)output->values, (void *)input->values,
         output->rows_n * output->cols_n * sizeof(float));
}

void matrix3d_copy_inplace(const matrix3d_t *const input,
                           const matrix3d_t *const output) {
  if (input->rows_n != output->rows_n || input->cols_n != output->cols_n ||
      input->depth != output->depth) {
    return;
  }
  memcpy((void *)output->values, (void *)input->values,
         output->rows_n * output->cols_n * output->depth * sizeof(float));
}

void matrix2d_rotate180_inplace(const matrix2d_t *const input) {
  float aux = 0;
  int i_opposite = 0;
  int j_opposite = 0;
  for (int i = 0; i < input->rows_n; i++) {
    for (int j = 0; j < input->cols_n; j++) {
      i_opposite = input->rows_n - i - 1;
      j_opposite = input->cols_n - j - 1;
      aux = matrix2d_get_elem(input, i, j);
      *matrix2d_get_elem_as_mut_ref(input, i, j) =
          matrix2d_get_elem(input, i_opposite, j_opposite);
      matrix2d_set_elem(input, i_opposite, j_opposite, aux);
    }
  }
}

void matrix2d_destroy(const matrix2d_t *m) {
  if (!m->loaded) {
    free(m->values);
  }
}

void matrix3d_init(matrix3d_t *const m, int rows_n, int cols_n, int depth) {
  m->loaded = false;
  m->rows_n = rows_n;
  m->cols_n = cols_n;
  m->depth = depth;
  m->values = (float *)malloc(rows_n * cols_n * depth * sizeof(float));
  for (int i = 0; i < m->rows_n; i++) {
    for (int j = 0; j < m->cols_n; j++) {
      for (int z = 0; z < m->depth; z++) {
        *matrix3d_get_elem_as_mut_ref(m, i, j, z) = 0.f;
      }
    }
  }
}

void matrix3d_destroy(const matrix3d_t *m) {
  if (!m->loaded) {
    free(m->values);
  }
}

void matrix2d_activate_inplace(const matrix2d_t *const m,
                               activation_type type) {
  for (int i = 0; i < m->rows_n; i++) {
    for (int j = 0; j < m->cols_n; j++) {
      float *elem = matrix2d_get_elem_as_mut_ref(m, i, j);
      switch (type) {
      case ACTIVATION_TYPE_RELU:
        *elem = relu(*elem);
        break;
      case ACTIVATION_TYPE_SIGMOID:
        *elem = sigmoid(*elem);
        break;
      case ACTIVATION_TYPE_TANH:
        *elem = tanh(*elem);
        break;
      case ACTIVATION_TYPE_IDENTITY:
        break;
      default:
        break;
      }
    }
  }
}

void matrix2d_softmax_inplace(const matrix2d_t *const m) {
  float sum = 0.0;
  float max_input = matrix2d_get_elem(m, 0, 0);

  // normalize to avoid overflow on exp
  for (int i = 0; i < m->rows_n; i++) {
    for (int j = 0; j < m->cols_n; j++) {
      float m_val = matrix2d_get_elem(m, i, j);
      if (m_val > max_input) {
        max_input = m_val;
      }
    }
  }

  // Calculate the exponentials of each input element and the sum of
  // exponentials
  for (int i = 0; i < m->rows_n; i++) {
    for (int j = 0; j < m->cols_n; j++) {
      float *m_val = matrix2d_get_elem_as_mut_ref(m, i, j);
      *m_val = exp(*m_val / max_input);
      sum += *m_val;
    }
  }

  // Normalize the values to make the sum equal to 1 (probabilities)
  for (int i = 0; i < m->rows_n; i++) {
    for (int j = 0; j < m->cols_n; j++) {
      *matrix2d_get_elem_as_mut_ref(m, i, j) /= sum;
    }
  }
}

void matrix2d_sum_inplace(const matrix2d_t *const m,
                          const matrix2d_t *const result) {
  for (int i = 0; i < m->rows_n; i++) {
    for (int j = 0; j < m->cols_n; j++) {
      *matrix2d_get_elem_as_mut_ref(result, i, j) += matrix2d_get_elem(m, i, j);
    }
  }
}

void matrix2d_print(const matrix2d_t *const m) {
  for (int i = 0; i < m->rows_n; i++) {
    printf("|");
    for (int j = 0; j < m->cols_n; j++) {
      printf(" %.2f |", matrix2d_get_elem(m, i, j));
    }
    printf("\n");
  }
}

void matrix3d_print(const matrix3d_t *const m) {
  matrix2d_t slice = {0};
  for (int i = 0; i < m->depth; i++) {
    printf("[Layer %d]\n", i);
    matrix3d_get_slice_as_mut_ref(m, &slice, i);
    matrix2d_print(&slice);
  }
}

void matrix2d_reshape(const matrix2d_t *const m, matrix2d_t *result, int rows_n,
                      int cols_n) {
  int m_elems_n = m->rows_n * m->cols_n;
  int result_elems_n = rows_n * cols_n;
  if (m_elems_n != result_elems_n) {
    return;
  }
  matrix2d_init(result, rows_n, cols_n);
  for (int i = 0; i < result_elems_n; i++) {
    *matrix2d_get_elem_as_mut_ref(result, i / result->cols_n,
                                  i % result->cols_n) =
        matrix2d_get_elem(m, i / m->cols_n, i % m->cols_n);
  }
}

void matrix3d_reshape(const matrix3d_t *const m, matrix3d_t *result) {
  int m_elems_n = m->rows_n * m->cols_n * m->depth;
  int result_elems_n = result->rows_n * result->cols_n * result->depth;
  if (m_elems_n != result_elems_n) {
    return;
  }
  for (int i = 0; i < result_elems_n; i++) {
    int m_i = i / (m->rows_n * m->cols_n);
    int m_j = (i % (m->rows_n * m->cols_n)) / m->cols_n;
    int m_k = i % m->cols_n;

    int result_i = i / (result->rows_n * result->cols_n);
    int result_j = (i % (result->rows_n * result->cols_n)) / result->cols_n;
    int result_k = i % result->cols_n;
    *matrix3d_get_elem_as_mut_ref(result, result_j, result_k, result_i) =
        matrix3d_get_elem(m, m_j, m_k, m_i);
  }
}

void matrix2d_load(matrix2d_t *m, int rows_n, int cols_n,
                   float *const base_address) {
  m->loaded = true;
  m->rows_n = rows_n;
  m->cols_n = cols_n;
  m->values = base_address;
}

void matrix3d_load(matrix3d_t *m, int rows_n, int cols_n, int depth,
                   float *const base_address) {
  m->loaded = true;
  m->rows_n = rows_n;
  m->cols_n = cols_n;
  m->depth = depth;
  m->values = base_address;
}
