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
  if (row_idx < m->height && col_idx < m->width) {
    return &m->values[row_idx * m->width + col_idx];
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
  if (row_idx < m->height && col_idx < m->width && z_idx < m->depth) {
    return &m->values[(z_idx * m->height * m->width) + (row_idx * m->width) +
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
  result->height = m->height;
  result->width = m->width;
  result->values = &m->values[result->height * result->width * z_idx];
}

void matrix2d_element_wise_product_inplace(const matrix2d_t *const m1,
                                           const matrix2d_t *const m2) {
  for (int i = 0; i < m1->height; i++) {
    for (int j = 0; j < m1->width; j++) {
      *matrix2d_get_elem_as_mut_ref(m1, i, j) *= matrix2d_get_elem(m2, i, j);
    }
  }
}

void matrix2d_init(matrix2d_t *m, int height, int width) {
  m->loaded = false;
  m->height = height;
  m->width = width;
  m->values = (float *)malloc(height * width * sizeof(float));
  for (int i = 0; i < m->height; i++) {
    for (int j = 0; j < m->width; j++) {
      matrix2d_set_elem(m, i, j, 0.f);
    }
  }
}

void matrix2d_randomize(matrix2d_t *const input) {
  for (int i = 0; i < input->height; i++) {
    for (int j = 0; j < input->width; j++) {
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

void matrix2d_copy_content(const matrix2d_t *const input,
                           const matrix2d_t *const output) {
  if (input->height != output->height || input->width != output->width) {
    return;
  }
  memcpy((void *)output->values, (void *)input->values,
         output->height * output->width * sizeof(float));
}

void matrix3d_copy_content(const matrix3d_t *const input,
                           const matrix3d_t *const output) {
  if (input->height != output->height || input->width != output->width ||
      input->depth != output->depth) {
    return;
  }
  memcpy((void *)output->values, (void *)input->values,
         output->height * output->width * output->depth * sizeof(float));
}

void matrix2d_rotate180_inplace(const matrix2d_t *const input) {
  float aux = 0;
  int i_opposite = 0;
  int j_opposite = 0;
  for (int i = 0; i < input->height; i++) {
    for (int j = 0; j < input->width; j++) {
      i_opposite = input->height - i - 1;
      j_opposite = input->width - j - 1;
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

void matrix3d_init(matrix3d_t *const m, int height, int width, int depth) {
  m->loaded = false;
  m->height = height;
  m->width = width;
  m->depth = depth;
  m->values = (float *)malloc(height * width * depth * sizeof(float));
  for (int i = 0; i < m->height; i++) {
    for (int j = 0; j < m->width; j++) {
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

void matrix2d_sum_inplace(const matrix2d_t *const m1,
                          const matrix2d_t *const m2) {
  if(m1->height != m2->height || m1->width != m2->width){
    return;
  }

  for (int i = 0; i < m1->height; i++) {
    for (int j = 0; j < m1->width; j++) {
      *matrix2d_get_elem_as_mut_ref(m2, i, j) += matrix2d_get_elem(m1, i, j);
    }
  }
}

void matrix2d_print(const matrix2d_t *const m) {
  for (int i = 0; i < m->height; i++) {
    printf("|");
    for (int j = 0; j < m->width; j++) {
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

void matrix3d_reshape(const matrix3d_t *const m, matrix3d_t *result) {
  int m_elems_n = m->height * m->width * m->depth;
  int result_elems_n = result->height * result->width * result->depth;
  if (m_elems_n != result_elems_n) {
    return;
  }
  for (int i = 0; i < result_elems_n; i++) {
    int m_i = i / (m->height * m->width);
    int m_j = (i % (m->height * m->width)) / m->width;
    int m_k = i % m->width;

    int result_i = i / (result->height * result->width);
    int result_j = (i % (result->height * result->width)) / result->width;
    int result_k = i % result->width;
    *matrix3d_get_elem_as_mut_ref(result, result_j, result_k, result_i) =
        matrix3d_get_elem(m, m_j, m_k, m_i);
  }
}

void matrix2d_load(matrix2d_t *m, int height, int width,
                   float *const base_address) {
  m->loaded = true;
  m->height = height;
  m->width = width;
  m->values = base_address;
}

void matrix3d_load(matrix3d_t *m, int height, int width, int depth,
                   float *const base_address) {
  m->loaded = true;
  m->height = height;
  m->width = width;
  m->depth = depth;
  m->values = base_address;
}
