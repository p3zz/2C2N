#include "layer.h"
#include <stdlib.h>
#include "utils.h"
#include "common.h"
#include "stdbool.h"

// ------------------------------ INIT ------------------------------
void pool_layer_init(pool_layer_t* layer, int input_height, int input_width, int input_depth, int kernel_size, int padding, int stride, pooling_type type){
	if(input_width == 0 || input_height == 0 || input_depth == 0 || kernel_size == 0 || stride == 0){
		return;
	}

	layer->kernel_size = kernel_size;
	layer->padding = padding;
	layer->stride = stride;
	layer->type = type;
	
	layer->input = (matrix3d_t*)malloc(sizeof(matrix3d_t));
	matrix3d_init(layer->input, input_height, input_width , input_depth);
	layer->d_input = (matrix3d_t*)malloc(sizeof(matrix3d_t));
	matrix3d_init(layer->d_input, input_height, input_width, input_depth);
	
	int output_rows = 0;
    int output_cols = 0;
	compute_output_size(layer->input->rows_n, layer->input->cols_n, kernel_size, padding, stride, &output_rows, &output_cols);

	layer->output = (matrix3d_t*)malloc(sizeof(matrix3d_t));
	matrix3d_init(layer->output, output_rows, output_cols, input_depth);

	// if it's a max pooling layer, we need to keep track of the index of each element of the output matrix w.r.t. the input matrix
	// for instance, if the first slice of the input has the max element of the kernel at (1,0), the first element of indexes
	// will be a matrix3d with depth 2 (one for i, one for j)
	// TODO finish the explaination
	if(layer->type == POOLING_TYPE_MAX){
		layer->indexes = (matrix3d_t*)malloc(layer->output->depth * sizeof(matrix3d_t));
		for(int i=0;i<layer->output->depth;i++){
			matrix3d_init(&layer->indexes[i], output_rows, output_cols, 2);
		}
	}

}

// TODO move all the memory allocation here, except for the auxiliary structures used in the process/backpropagation
void conv_layer_init(
	conv_layer_t* layer,
	int input_height,
	int input_width,
	int input_depth,
	int kernel_size,
	int kernels_n,
	int stride,
	int padding,
	activation_type activation_type
)
{
	if(input_width == 0 || input_height == 0 || input_depth == 0 || kernel_size == 0 || stride == 0){
		return;
	}

	layer->kernels_n = kernels_n;
	layer->padding = padding;
	layer->stride = stride;
	layer->activation_type = activation_type;

	// allocate input and d_input
	layer->input = (matrix3d_t*)malloc(sizeof(matrix3d_t));
	matrix3d_init(layer->input, input_height, input_width, input_depth);
	layer->d_input = (matrix3d_t*)malloc(sizeof(matrix3d_t));
	matrix3d_init(layer->d_input, input_height, input_width, input_depth);

	// allocate kernels
	layer->kernels = (matrix3d_t*)malloc(layer->kernels_n * sizeof(matrix3d_t));

	for(int i=0;i<layer->kernels_n;i++){
		matrix3d_init(&layer->kernels[i], kernel_size, kernel_size, input_depth);
		matrix3d_randomize(&layer->kernels[i]);
	}

	int output_height = 0;
	int output_width = 0;

	compute_output_size(layer->input->rows_n, layer->input->cols_n, kernel_size, padding, stride, &output_height, &output_width);

	// allocate biases
	layer->biases = (matrix2d_t*)malloc(layer->kernels_n * sizeof(matrix2d_t));
	for(int i=0;i<kernels_n;i++){
		matrix2d_init(&layer->biases[i], output_height, output_width);
		matrix2d_randomize(&layer->biases[i]);
	}

	// allocate output and d_output
	layer->output = (matrix3d_t*)malloc(sizeof(matrix3d_t));
	matrix3d_init(layer->output, output_height, output_width, kernels_n);
	layer->output_activated = (matrix3d_t*)malloc(sizeof(matrix3d_t));
	matrix3d_init(layer->output_activated, output_height, output_width, kernels_n);
}

void dense_layer_init(dense_layer_t* layer, int input_n, int output_n, activation_type activation_type){
	matrix3d_init(&layer->inputs, 1, input_n, 1);
	matrix3d_init(&layer->d_inputs, 1, input_n, 1);
	matrix2d_init(&layer->weights, input_n, output_n);
	matrix2d_randomize(&layer->weights);
	matrix2d_init(&layer->biases, 1, output_n);
	matrix2d_randomize(&layer->biases);
	matrix3d_init(&layer->output, 1, output_n, 1);
	matrix3d_init(&layer->output_activated, 1, output_n, 1);
	layer->activation_type = activation_type;
}

void softmax_layer_init(softmax_layer_t* layer, int input_n){
	matrix3d_init(&layer->input, 1, input_n, 1);
	matrix3d_init(&layer->output, 1, input_n, 1);
	matrix3d_init(&layer->d_input, 1, input_n, 1);
}

void conv_layer_load_params(
	conv_layer_t* layer, 
	matrix3d_t* kernels, 
	int kernels_n, 
	matrix2d_t* biases,
	matrix3d_t* output,
	matrix3d_t* output_activated,
	matrix3d_t* d_input
){
	layer->kernels = kernels;
	layer->kernels_n = kernels_n;
	layer->biases = biases;
	layer->output = output;
	layer->output_activated = output_activated;
	layer->d_input = d_input;
}

// ------------------------------ FEED ------------------------------

void dense_layer_feed(dense_layer_t* layer, const matrix3d_t* const input){
	matrix3d_copy_inplace(input, &layer->inputs);
}

void pool_layer_feed(pool_layer_t* layer, const matrix3d_t* const input){
	matrix3d_copy_inplace(input, layer->input);
}

void conv_layer_feed(conv_layer_t* layer, matrix3d_t* input){
	matrix3d_copy_inplace(input, layer->input);
}

void softmax_layer_feed(softmax_layer_t* layer, const matrix3d_t* const input){
	matrix3d_copy_inplace(input, &layer->input);
}

// ------------------------------ PROCESS ------------------------------

void dense_layer_forwarding(dense_layer_t* layer){
	matrix2d_t output = {0};
	matrix2d_t output_activated = {0};
	matrix2d_t input = {0};

	matrix3d_get_slice_as_mut_ref(&layer->output, &output, 0);
	matrix3d_get_slice_as_mut_ref(&layer->output_activated, &output_activated, 0);
	matrix3d_get_slice_as_mut_ref(&layer->inputs, &input, 0);

	for(int i=0;i<output.cols_n;i++){
		float* out_val = matrix2d_get_elem_as_mut_ref(&output, 0, i);
		*out_val = matrix2d_get_elem(&layer->biases, 0, i);
		for(int j=0;j<layer->weights.rows_n;j++){
			*out_val += (matrix2d_get_elem(&input, 0, j) * matrix2d_get_elem(&layer->weights, j, i));
		}
	}
	matrix2d_copy_inplace(&output, &output_activated);
	switch(layer->activation_type){
		case ACTIVATION_TYPE_RELU:
			matrix2d_relu_inplace(&output_activated);
			break;
		case ACTIVATION_TYPE_SIGMOID:
			matrix2d_sigmoid_inplace(&output_activated);
			break;
		case ACTIVATION_TYPE_TANH:
			matrix2d_tanh_inplace(&output_activated);
			break;
		case ACTIVATION_TYPE_IDENTITY:
			break;
		default:
			break;
	}
}

void softmax_layer_forwarding(softmax_layer_t* layer){
	matrix2d_t in_slice = {0};
	matrix2d_t out_slice = {0};
	matrix3d_get_slice_as_mut_ref(&layer->input, &in_slice, 0);
	matrix3d_get_slice_as_mut_ref(&layer->output, &out_slice, 0);
	matrix2d_copy_inplace(&in_slice, &out_slice);
	matrix2d_softmax_inplace(&out_slice);
}

void pool_layer_forwarding(pool_layer_t* layer){
	matrix2d_t in_slice = {0};
	matrix2d_t out_slice = {0};
	for(int i=0;i<layer->input->depth;i++){
		matrix3d_get_slice_as_mut_ref(layer->input, &in_slice, i);
		matrix3d_get_slice_as_mut_ref(layer->output, &out_slice, i);
		switch(layer->type){
			case POOLING_TYPE_AVERAGE:
				avg_pooling(&in_slice, &out_slice, layer->kernel_size, layer->padding, layer->stride);
				break;
			case POOLING_TYPE_MAX:
				max_pooling(&in_slice, &out_slice, &layer->indexes[i], layer->kernel_size, layer->padding, layer->stride);
				break;
		}
	}
}

// TODO check if depth of each kernel is equal to the number of channels of the input
void conv_layer_forwarding(conv_layer_t* layer){
	matrix2d_t result = {0};
	matrix2d_t out_slice = {0};
	matrix2d_t out_act_slice = {0};
	matrix2d_t in_slice = {0};
	matrix2d_t kernel_slice = {0};

	matrix2d_init(&result, layer->output->rows_n, layer->output->cols_n);

	for(int i=0;i<layer->kernels_n;i++){
		matrix3d_get_slice_as_mut_ref(layer->output, &out_slice, i);
		matrix3d_get_slice_as_mut_ref(layer->output_activated, &out_act_slice, i);

		for(int j=0;j<layer->kernels[i].depth;j++){
			matrix3d_get_slice_as_mut_ref(&layer->kernels[i], &kernel_slice, j);
			matrix3d_get_slice_as_mut_ref(layer->input, &in_slice, j);
			// compute the cross correlation between a channel of the input and its corresponding kernel
			full_cross_correlation(&in_slice, &kernel_slice, &result, layer->padding, layer->stride);
			if(j == 0){
				// perform an early sum of the biases to the final output layer
				matrix2d_sum_inplace(&layer->biases[i], &out_slice);
			}
			// then we sum the resulting matrix to the output
			matrix2d_sum_inplace(&result, &out_slice);
		}
		matrix2d_copy_inplace(&out_slice, &out_act_slice);
		switch(layer->activation_type){
			case ACTIVATION_TYPE_RELU:
				matrix2d_relu_inplace(&out_act_slice);
				break;
			case ACTIVATION_TYPE_SIGMOID:
				matrix2d_sigmoid_inplace(&out_act_slice);
				break;
			case ACTIVATION_TYPE_TANH:
				matrix2d_tanh_inplace(&out_act_slice);
				break;
			default:
				break;
		}
	}
	matrix2d_destroy(&result);
}

// ------------------------------ BACK-PROPAGATE ------------------------------


// the input is the derivative of the cost w.r.t the output, coming from the next layer
// the output if the derivative of the input, that has to be passed to the previous layer
// https://www.youtube.com/watch?v=AbLvJVwySEo
void dense_layer_backpropagation(dense_layer_t* layer, const matrix3d_t* const input, float learning_rate)
{
	for(int i=0;i<layer->weights.rows_n;i++){
		float d_input = 0.f;
		for(int j=0;j<layer->weights.cols_n;j++){
			// d_actf(z_j) * d_act
			float common = d_activate(matrix3d_get_elem(&layer->output, 0, j, 0), layer->activation_type) * matrix3d_get_elem(input, 0, j, 0);
			// compute d_weight
			// dC/dw_ij = x_i * d_actf(z_j) * d_act
			float d_weight = matrix3d_get_elem(&layer->inputs, 0, i, 0) * common;

			// compute d_input
			d_input += (matrix2d_get_elem(&layer->weights, i, j) * common);

			float weight_new = gradient_descent(matrix2d_get_elem(&layer->weights, i, j), learning_rate, d_weight);
			matrix2d_set_elem(&layer->weights, i, j, weight_new);

			float bias_new = gradient_descent(matrix2d_get_elem(&layer->biases, 0, j), learning_rate, common);
			matrix2d_set_elem(&layer->biases, 0, j, bias_new);
		}
		matrix3d_set_elem(&layer->d_inputs, 0, i, 0, d_input);
	}
}

void softmax_layer_backpropagation(softmax_layer_t* layer, const matrix3d_t* const input){
	matrix2d_t aux = {0};
	matrix2d_t sample = {0};
	matrix3d_get_slice_as_mut_ref(&layer->output, &sample, 0);

	matrix2d_init(&aux, layer->output.cols_n, layer->output.cols_n);

	for(int i=0;i<sample.cols_n;i++){
		for(int j=0;j<sample.cols_n;j++){
			float aux_val = 0.f;
			if(i == j){
				aux_val = matrix2d_get_elem(&sample, 0, i) * (1 - matrix2d_get_elem(&sample, 0, i));
			}
			else{
				aux_val = -matrix2d_get_elem(&sample, 0, i) * matrix2d_get_elem(&sample, 0, j);
			}
			matrix2d_set_elem(&aux, i, j, aux_val);
		}
	}

	for (int i = 0; i < aux.rows_n; i++) {
		for (int j = 0; j < aux.cols_n; j++) {
			*matrix3d_get_elem_as_mut_ref(&layer->d_input, 0, i, 0) += (matrix2d_get_elem(&aux, i, j) * matrix3d_get_elem(input, 0, j, 0));
		}
    }

	matrix2d_destroy(&aux);
}

// https://lanstonchu.wordpress.com/2018/09/01/convolutional-neural-network-cnn-backward-propagation-of-the-pooling-layers/
// TODO add avg_pooling handling, this is correct just for max_pooling
void pool_layer_backpropagation(pool_layer_t* layer, const matrix3d_t* const input){
	matrix2d_t i_slice = {0};
	matrix2d_t j_slice = {0};

	switch(layer->type){
		case POOLING_TYPE_MAX: {
			for(int i=0;i<layer->d_input->depth;i++){
				matrix3d_get_slice_as_mut_ref(&layer->indexes[i], &i_slice, 0);
				matrix3d_get_slice_as_mut_ref(&layer->indexes[i], &j_slice, 1);

				for(int m=0;m<i_slice.rows_n;m++){
					for(int n=0;n<i_slice.cols_n;n++){
						int row_i = matrix2d_get_elem(&i_slice, m, n);
						int col_i = matrix2d_get_elem(&j_slice, m, n);
						*matrix3d_get_elem_as_mut_ref(layer->d_input, row_i, col_i, i) += matrix3d_get_elem(input, m, n, i);
					}
				}
			}
			break;
		}

		case POOLING_TYPE_AVERAGE: {
			for(int l=0;l<layer->output->depth;l++){
				for (int h = 0; h < layer->output->rows_n; h++) {
					for (int w = 0; w < layer->output->cols_n; w++) {
						// The gradient from the output for this pooled region
						float gradient = matrix3d_get_elem(input, h, w, l);
						
						// Distribute the gradient to each input element in the pooling region
						for (int i = 0; i < layer->kernel_size; i++) {
							for (int j = 0; j < layer->kernel_size; j++) {
								int input_h = h * layer->stride + i;
								int input_w = w * layer->stride + j;
								
								// Ensure we're within bounds (important for cases where pooling window goes out of input bounds)
								if (input_h < layer->input->rows_n && input_w < layer->input->cols_n) {
									*matrix3d_get_elem_as_mut_ref(layer->d_input, input_h, input_w, l) += (gradient / (float)(layer->kernel_size * layer->kernel_size));
								}
							}
						}
					}
				}
			}
			break;
		}
	}
}

void conv_layer_backpropagation(conv_layer_t* layer, const matrix3d_t* const input, float learning_rate){
	// matrix used to store the product (element x element) between the input and the derivative of the activation function of each
	// output of the layer
	matrix2d_t d_output = {0};
	// allocate memory for d_kernel, that is the matrix that contains the correction that has to be applied to the weights of the kernels
	// after the whole computation
	matrix3d_t* d_kernel = (matrix3d_t*)malloc(layer->kernels_n * sizeof(matrix3d_t));
	// matrix used to store the result of each convolution between the input (from the next layer) and the kernel
	matrix2d_t d_input_aux = {0};

	matrix2d_t in_slice = {0};
	matrix2d_t d_kernel_slice = {0};
	matrix2d_t kernel_slice = {0};
	matrix2d_t d_input_slice = {0};

	matrix2d_init(&d_output, layer->output->rows_n, layer->output->cols_n);

	for(int i=0;i<layer->kernels_n;i++){
		matrix3d_init(&d_kernel[i], layer->kernels[i].rows_n, layer->kernels[i].cols_n, layer->kernels[i].depth);
	}

	matrix2d_init(&d_input_aux, layer->d_input->rows_n, layer->d_input->cols_n);

	// for each kernel of the layer
	for(int i=0;i<layer->kernels_n;i++){
		for(int m=0;m<d_output.rows_n;m++){
			for(int n=0;n<d_output.cols_n;n++){
				*matrix2d_get_elem_as_mut_ref(&d_output, m, n) = d_activate(matrix3d_get_elem(layer->output, m, n, i), layer->activation_type);
			}
		}

		matrix3d_get_slice_as_mut_ref(input, &in_slice, i);

		matrix2d_element_wise_product_inplace(&d_output, &in_slice);
		
		// for each layer of the current kernel compute the derivative
		// using the cross correlation between the j-th input and the i-th output (rotated)
		matrix3d_t* kernel = &layer->kernels[i];
		for(int j=0;j<kernel->depth;j++){
			matrix3d_get_slice_as_mut_ref(input, &in_slice, j);
			matrix3d_get_slice_as_mut_ref(&d_kernel[i], &d_kernel_slice, j);
			matrix3d_get_slice_as_mut_ref(kernel, &kernel_slice, j);
			matrix3d_get_slice_as_mut_ref(layer->d_input, &d_input_slice, j);

			// compute the derivative for the correction of the kernel
			// TODO correct also we the stride
			full_cross_correlation(&in_slice, &d_output, &d_kernel_slice, layer->padding, 1);
			// compute the derivative for the correction of the input
			convolution(&d_output, &kernel_slice, &d_input_aux, kernel->rows_n - input->rows_n + 1);
			matrix2d_sum_inplace(&d_input_aux, &d_input_slice);
		}

		// update biases
		for(int m=0;m<d_output.rows_n;m++){
			for(int n=0;n<d_output.rows_n;n++){
				*matrix2d_get_elem_as_mut_ref(&layer->biases[i], m, n) = gradient_descent(matrix2d_get_elem(&layer->biases[i], m, n), matrix2d_get_elem(&d_output, m, n), learning_rate);
			}
		}
	}

	// update weights
	for(int i=0;i<layer->kernels_n;i++){
		for(int j=0;j<layer->kernels[i].depth;j++){
			for(int m=0;m<layer->kernels[i].rows_n;m++){
				for(int n=0;n<layer->kernels[i].cols_n;n++){
					*matrix3d_get_elem_as_mut_ref(&layer->kernels[i], m, n, j) = gradient_descent(matrix3d_get_elem(&layer->kernels[i], m, n, j), learning_rate, matrix3d_get_elem(&d_kernel[i], m, n, j));
				}
			}
		}
	}

	for(int i=0;i<layer->kernels_n;i++){
		matrix3d_destroy(&d_kernel[i]);
	}
	free(d_kernel);

	matrix2d_destroy(&d_output);
	matrix2d_destroy(&d_input_aux);
}

// ------------------------------ DESTROY ------------------------------

void dense_layer_destroy(dense_layer_t* layer){
	matrix3d_destroy(&layer->inputs);
	matrix3d_destroy(&layer->d_inputs);
	matrix2d_destroy(&layer->weights);
	matrix2d_destroy(&layer->biases);
	matrix3d_destroy(&layer->output);
	matrix3d_destroy(&layer->output_activated);
}

void conv_layer_destroy(conv_layer_t* layer){
	matrix3d_destroy(layer->input);
	matrix3d_destroy(layer->d_input);
	free(layer->input);
	free(layer->d_input);

	for(int i=0;i<layer->kernels_n;i++){
		matrix3d_destroy(&layer->kernels[i]);
		matrix2d_destroy(&layer->biases[i]);
	}
	free(layer->kernels);
	free(layer->biases);
	matrix3d_destroy(layer->output);
	matrix3d_destroy(layer->output_activated);
	free(layer->output);
	free(layer->output_activated);
}

void pool_layer_destroy(pool_layer_t* layer){
	if(layer->type == POOLING_TYPE_MAX){
		for(int i=0;i<layer->input->depth;i++){
			matrix3d_destroy(&layer->indexes[i]);
		}
	}
	free(layer->indexes);
	
	matrix3d_destroy(layer->output);
	free(layer->output);

	matrix3d_destroy(layer->input);
	free(layer->input);
	
	matrix3d_destroy(layer->d_input);
	free(layer->d_input);
}

void softmax_layer_destroy(softmax_layer_t* layer){
	matrix3d_destroy(&layer->input);
	matrix3d_destroy(&layer->output);
	matrix3d_destroy(&layer->d_input);
}
