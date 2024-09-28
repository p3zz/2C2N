#include "common.h"
#include "math.h"
#include "stdlib.h"

void cross_correlation(const matrix2d_t *const m1,
                            const matrix2d_t *const m2,
                            matrix2d_t *const result, int padding, int stride) {
  for (int i = 0; i < result->height; i++) {
    for (int j = 0; j < result->width; j++) {
      float sum = 0;
      for (int m = 0; m < m2->height; m++) {
        for (int n = 0; n < m2->width; n++) {
          int row = i * stride + m - padding;
          int col = j * stride + n - padding;
          if (row >= 0 && row < m1->height && col >= 0 && col < m1->width) {
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

void convolution(const matrix2d_t *const m1, const matrix2d_t *const m2,
                 matrix2d_t *const result, int padding, int stride) {
  matrix2d_rotate180_inplace(m2);
  cross_correlation(m1, m2, result, padding, stride);
  // restore the rotated matrix
  matrix2d_rotate180_inplace(m2);
}

void max_pooling(const matrix2d_t *const mat, const matrix2d_t *const result,
                 const matrix3d_t *const indexes, int kernel_size, int padding,
                 int stride) {
  for (int i = 0; i < result->height; i++) {
    for (int j = 0; j < result->width; j++) {
      float max = 0;
      int max_i = 0;
      int max_j = 0;
      for (int m = 0; m < kernel_size; m++) {
        for (int n = 0; n < kernel_size; n++) {
          int row = i * stride + m - padding;
          int col = j * stride + n - padding;
          if (row >= 0 && row < mat->height && col >= 0 && col < mat->width) {
            float val = matrix2d_get_elem(mat, row, col);
            if (val > max) {
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

void avg_pooling(const matrix2d_t *const mat, const matrix2d_t *const result,
                 int kernel_size, int padding, int stride) {
  for (int i = 0; i < result->height; i++) {
    for (int j = 0; j < result->width; j++) {
      float sum = 0;
      int values_len = 0;
      for (int m = 0; m < kernel_size; m++) {
        for (int n = 0; n < kernel_size; n++) {
          int row = i * stride + m - padding;
          int col = j * stride + n - padding;
          if (row >= 0 && row < mat->height && col >= 0 && col < mat->width) {
            sum += matrix2d_get_elem(mat, row, col);
            values_len++;
          }
        }
      }
      matrix2d_set_elem(result, i, j, sum / values_len);
    }
  }
}

void mean_squared_error_derivative(const matrix2d_t *const output,
                                   const matrix2d_t *const target_output,
                                   const matrix2d_t *const result) {
  for (int i = 0; i < output->height; i++) {
    for (int j = 0; j < output->width; j++) {
      float out_val = matrix2d_get_elem(output, i, j);
      float target_out_val = matrix2d_get_elem(target_output, i, j);
      float res = 2 * (out_val - target_out_val);
      matrix2d_set_elem(result, i, j, res);
    }
  }
}

void cross_entropy_loss_derivative(const matrix2d_t *const output,
                                   const matrix2d_t *const target_output,
                                   const matrix2d_t *const result) {
  for (int i = 0; i < output->height; i++) {
    for (int j = 0; j < output->width; j++) {
      float n0 = 1 - matrix2d_get_elem(target_output, i, j);
      float d0 = 1 - matrix2d_get_elem(output, i, j);
      float n1 = matrix2d_get_elem(target_output, i, j);
      float d1 = matrix2d_get_elem(output, i, j);
      float res;
      if (d0 < 1e-4 || d1 < 1e-4) {
        res = 0.f;
      } else {
        res = (n0 / d0 - n1 / d1);
      }
      matrix2d_set_elem(result, i, j, res);
    }
  }
}

float mean_squared_error(const matrix2d_t *const output,
                         const matrix2d_t *const target_output) {
  float sum = 0.f;
  for (int i = 0; i < output->height; i++) {
    for (int j = 0; j < output->width; j++) {
      float delta = matrix2d_get_elem(target_output, i, j) -
                    matrix2d_get_elem(output, i, j);
      sum += (delta * delta);
    }
  }
  return sum / output->width;
}

float cross_entropy_loss(const matrix2d_t *const output,
                         const matrix2d_t *const target_output) {
  float sum = 0.f;
  for (int i = 0; i < output->height; i++) {
    for (int j = 0; j < output->width; j++) {
      float target_out = matrix2d_get_elem(target_output, i, j);
      float out = matrix2d_get_elem(output, i, j);
      float d = (target_out * logf(out)) + ((1 - target_out) * logf(1 - out));
      if (isnan(d)) {
        d = 0.f;
      }
      sum += d;
    }
  }
  return -sum / output->width;
}

// void parse_line(char *line, int length, matrix2d_t *image, float *label) {
//   char n[4] = {0};
//   int n_idx = 0;
//   int numbers_n = 0;
//   bool label_read = false;
//   for (int i = 0; i < length; i++) {
//     // check if numbers length exceeds the capacity of the image matrix
//     if (numbers_n == image->height * image->width) {
//       break;
//     }
//     if (line[i] == ',' || i == length - 1) {
//       n[n_idx] = '\0';
//       float val = atof(n);
//       if (!label_read) {
//         *label = val;
//         label_read = true;
//       } else {
//         int row = numbers_n / image->width;
//         int col = numbers_n % image->width;
//         matrix2d_set_elem(image, row, col, val);
//         numbers_n++;
//       }
//       n_idx = 0;
//     } else {
//       n[n_idx] = line[i];
//       n_idx++;
//     }
//   }
// }
