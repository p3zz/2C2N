/**
 * @brief library provides a basic implementation of a 2D/3D matrix, along with the most common
 * manipulation/computation functions used in Convolutional Neural Networks (CNNs).
 * Every function (except for initialization functions) has been implemented such that every operation is performed
 * in place, without the need to use auxiliary dinamically allocated structures. This can save space and perform a
 * cleaner prediction of the memory footprint.
 */

#ifndef __MATRIX_H__
#define __MATRIX_H__

#include "stdbool.h"
#include "utils.h"

/**
 * @struct matrix2d_t
 * @brief Implementation of a 2D matrix
 * @param height: n. of rows of the matrix
 * @param width: n. of columns of the matrix
 * @param values: linear array that stores the data of the matrix.
 * the data must be stored such that each row is stored consecutively
 * in memory, with one row directly following the previous one. 
 * @param loaded: flag used to keep track of the origin of the data.
 * e.g. if loaded is true, the pointer to the data has been externally set from
 * a caller of matrix2d_load(matrix2d_t *m, int height, int width).
 * In this case, the caller has the duty to free the pointed data (if dinamically
 * allocated, otherwise in statically allocated do nothing),
 * while false means that the data has been dinamically allocated by 
 * the matrix2d_init(matrix2d_t *m, int height, int width)
 * In this case, the caller needs to call matrix3d_destroy(matrix3d_t* m) in order
 * to free the matrix
*/
typedef struct {
  int height;
  int width;
  float *values;
  bool loaded;
} matrix2d_t;

/**
 * @struct matrix3d_t
 * Implementation of a 3D matrix
 * @param height: n. of rows of the matrix
 * @param width: n. of columns of the matrix
 * @param depth: n. of slices of the matrix
 * @param values: linear array that stores the data of the matrix.
 * the data must be stored such that each row of each slice is stored consecutively
 * in memory, with one row directly following the previous one, and one
 * slice following the previous one.
 * @param loaded: flag used to keep track of the origin of the data.
 * e.g. if loaded is true, the pointer to the data has been set from
 * a caller of matrix3d_load(matrix3d_t *m, int height, int width, int depth).
 * In this case, the caller has the duty to free the pointed data (if dinamically
 * allocated, otherwise in statically allocated do nothing),
 * while false means that the data has been dinamically allocated by 
 * the matrix3d_init(matrix3d_t *m, int height, int width, int depth).
 * In this case, the caller needs to call matrix3d_destroy(matrix3d_t* m) in order
 * to free the matrix
*/
typedef struct {
  int height;
  int width;
  int depth;
  float *values;
  bool loaded;
} matrix3d_t;


/**
 * @brief Returns a non-mutable pointer to a specific cell of a 2D matrix
 * @param m: the target matrix
 * @param row_idx: the index of the row
 * @param col_idx: the index of the column
 * @return a non-mutable pointer to the cell if indexes belong to the matrix,
 * otherwise NULL
 */
const float *matrix2d_get_elem_as_ref(const matrix2d_t *const m, int row_idx,
                                      int col_idx);
/**
 * @brief Returns a mutable pointer to a specific cell of a 2D matrix
 * @param m: the target matrix
 * @param row_idx: the index of the row
 * @param col_idx: the index of the column
 * @return a mutable pointer to the cell if indexes belong to the matrix,
 * otherwise NULL
 */
float *matrix2d_get_elem_as_mut_ref(const matrix2d_t *const m, int row_idx,
                                    int col_idx);
/**
 * @brief Returns the value of a specific cell of a 2D matrix
 * @param m: the target matrix
 * @param row_idx: the index of the row
 * @param col_idx: the index of the column
 * @return the value of the cell if indexes belong to the matrix,
 * otherwise 0.f
 */
float matrix2d_get_elem(const matrix2d_t *const m, int row_idx, int col_idx);

/**
 * @brief Returns the value of a specific cell of a 2D matrix
 * @param m: the target matrix
 * @param row_idx: the index of the row
 * @param col_idx: the index of the column
 * @param value: the new value of the cell
 */
void matrix2d_set_elem(const matrix2d_t *const m, int row_idx, int col_idx,
                       float value);

/**
 * @brief Initialize a 2D matrix. The data of the matrix is dinamically allocated
 * and its length will be equal to (height * width), and the "loaded" flag will be set to false
 * @param m: the target matrix
 * @param height: n. of rows of the matrix
 * @param width: n. of columns of the matrix
 */
void matrix2d_init(matrix2d_t *m, int height, int width);

/**
 * @brief Destroy a 2D matrix. The data will be deallocated only if "loaded" is set
 * to false, otherwise does nothing
 * @param m: the target matrix
 */
void matrix2d_destroy(const matrix2d_t *m);

/**
 * @brief Initialize a 2D matrix. The data of the matrix is set through the base_address,
 * and the "loaded" flag will be set to true
 * @param m: the target matrix
 * @param height: n. of rows of the matrix
 * @param width: n. of columns of the matrix
 * @param base_address: an existing memory address which will be the starting address
 * of the data of the target matrix
 */
void matrix2d_load(matrix2d_t *m, int height, int width,
                   float *const base_address);
/**
 * @brief Write to stdout a 2D matrix in a rectangle-shaped form.
 * e.g. if the matrix is 3x2, and the data is {1.f, 2.f, 3,f, 4.f, 5.f, 6.f},
 * the function will print:
 * | 1.f | 2.f |
 * | 3.f | 4.f |
 * | 5.f | 6.f |
 * @param m: the target matrix
 */
void matrix2d_print(const matrix2d_t *const m);

/**
 * @brief Performs the element-wise sum of two 2D matrices, and stores
 * the result inside the 2nd matrix
 * @param m1: the first matrix
 * @param m2: the second matrix, in which the result will be stored
 */
void matrix2d_sum_inplace(const matrix2d_t *const m1,
                          const matrix2d_t *const m2);

/**
 * @brief Copy the "values" of a 2D input matrix inside a 2D output matrix.
 * The output matrix must be allocated before calling this function
 * @param input: the first matrix
 * @param output: the second matrix, in which the result will be stored
 */
void matrix2d_copy_content(const matrix2d_t *const input,
                           const matrix2d_t *output);
/**
 * @brief Randomize the content of a 2D matrix.
 * @param input: the target matrix
 */
void matrix2d_randomize(matrix2d_t *input);

/**
 * @brief Perform a 180° rotation of a 2D matrix, and stores the result inside the the same matrix.
 * @param input: the target matrix
 */
void matrix2d_rotate180_inplace(const matrix2d_t *const input);

/**
 * @brief Performs an element-wise product between two 2D matrices, and stores the
 * result inside the 1st matrix.
 * @param m1: the 1st matrix, in which the result will be stored
 * @param m2: the 2nd matrix
 */
void matrix2d_element_wise_product_inplace(const matrix2d_t *const m1,
                                           const matrix2d_t *const m2);

/**
 * @brief Performs a softmax operation of a 2D matrix, and stores the
 * result inside the same matrix. The function first performs a normalization
 * of the content of the matrix.
 * @param input: the target matrix
 */
void matrix2d_softmax_inplace(const matrix2d_t *const input);

/**
 * @brief Performs an element-wise activation operation of a 2D matrix, and stores the
 * result inside the same matrix.
 * @param input: the target matrix
 * @param type: the activation type
 */
void matrix2d_activate_inplace(const matrix2d_t *const m, activation_type type);

/**
 * @brief Returns a non-mutable pointer to a specific cell of a 3D matrix
 * @param m: the target matrix
 * @param row_idx: the index of the row
 * @param col_idx: the index of the column
 * @param z_idx: the index of the slice
 * @return a non-mutable pointer to the cell if the indexes belongs to the matrix,
 * otherwise NULL
 */
const float *matrix3d_get_elem_as_ref(const matrix3d_t *const m, int row_idx,
                                      int col_idx, int z_idx);

/**
 * @brief Returns a mutable pointer to a specific cell of a 3D matrix
 * @param m: the target matrix
 * @param row_idx: the index of the row
 * @param col_idx: the index of the column
 * @param z_idx: the index of the slice
 * @return a mutable pointer to the cell if the indexes belongs to the matrix,
 * otherwise NULL
 */
float *matrix3d_get_elem_as_mut_ref(const matrix3d_t *const m, int row_idx,
                                    int col_idx, int z_idx);

/**
 * @brief Returns the value of a specific cell of a 3D matrix
 * @param m: the target matrix
 * @param row_idx: the index of the row
 * @param col_idx: the index of the column
 * @param z_idx: the index of the slice
 * @return the value of the cell if the indexes belongs to the matrix,
 * otherwise 0.f
 */
float matrix3d_get_elem(const matrix3d_t *const m, int row_idx, int col_idx,
                        int z_idx);

/**
 * @brief Set the value of a specific cell of a 3D matrix
 * @param m: the target matrix
 * @param row_idx: the index of the row
 * @param col_idx: the index of the column
 * @param z_idx: the index of the column
 * @param value: the new value of the cell
 */
void matrix3d_set_elem(const matrix3d_t *const m, int row_idx, int col_idx,
                       int z_idx, float value);

/**
 * @brief Retrieve a slice at a specific index of a 3D input matrix and stores it in the result 2D matrix
 * @param m: the target matrix
 * @param result: the 2D matrix in which the slice will be stored
 * @param z_idx: the index of the slice
 */
void matrix3d_get_slice_as_mut_ref(const matrix3d_t *m, matrix2d_t *result,
                                   int z_idx);

/**
 * @brief Initialize a 3D matrix. The data of the matrix is dinamically allocated
 * and its length will be equal to (height * width * depth)
 * @param m: the target matrix
 * @param height: n. of rows of the matrix
 * @param width: n. of columns of the matrix
 * @param depth: n. of slices of the matrix
 */
void matrix3d_init(matrix3d_t *m, int height, int width, int depth);

/**
 * @brief Initialize the 3D matrix. The data of the matrix is set through the base_address
 * @param m: the target matrix
 * @param height: n. of rows of the matrix
 * @param width: n. of columns of the matrix
 * @param depth: n. of slices of the matrix
 * @param base_address: an existing memory address which will be the starting address
 * of the data of the target matrix
 */
void matrix3d_load(matrix3d_t *m, int height, int width, int depth,
                   float *const base_address);

/**
 * @brief Destroy a 3D matrix. The "values" member will be deallocated only if "loaded" is set
 * to false, otherwise does nothing.
 * @param m: the target matrix
 */
void matrix3d_destroy(const matrix3d_t *m);

/**
 * @brief Write to stdout a 3D matrix in a rectangle-shaped form.
 * e.g. if the matrix is 2x2x2, and the data is {1.f, 2.f, 3,f, 4.f, 5.f, 6.f, 7.f, 8.f},
 * the function will print:
 * Layer 0
 * | 1.f | 2.f |
 * | 3.f | 4.f |
 * Layer 1
 * | 5.f | 6.f |
 * | 7.f | 8.f |
 * 
 * @param m: the target matrix
 */
void matrix3d_print(const matrix3d_t *const m);

/**
 * @brief Copy the "values" of a 3D input matrix inside a 3D output matrix.
 * The output matrix must be allocated before calling this function.
 * @param input: the first matrix
 * @param output: the second matrix, in which the result will be stored
 */
void matrix3d_copy_inplace(const matrix3d_t *const input,
                           const matrix3d_t *output);

/**
 * @brief Randomize the content of a 3D matrix.
 * @param input: the target matrix
 */
void matrix3d_randomize(matrix3d_t *input);

/**
 * @brief Reshape a 3D input matrix using the width/height/depth of a 3D output matrix,
 * and stores the values result into the output matrix
 * @param input: the 3D input matrix
 * @param output: the 3D output matrix, in which the result will be stored
 */
void matrix3d_reshape(const matrix3d_t *const input, matrix3d_t *output);

#endif