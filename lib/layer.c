#include "layer.h"
#include <stdlib.h>
#include "utils.h"
#include "common.h"
#include "stdbool.h"

layer_t_old create_layer(int neurons_num)
{
	layer_t_old lay;
	lay.neurons_num = neurons_num;
	lay.neurons = (neuron_t*) malloc(neurons_num * sizeof(neuron_t));
	return lay;
}

// ------------------------------ INIT ------------------------------
void init_pool_layer(pool_layer_t* layer, int input_height, int input_width, int input_depth, int kernel_size, int padding, int stride, pooling_type type){
	if(input_width == 0 || input_height == 0 || input_depth == 0 || kernel_size == 0 || stride == 0){
		return;
	}

	layer->kernel_size = kernel_size;
	layer->padding = padding;
	layer->stride = stride;
	layer->type = type;
	
	create_matrix3d(&layer->input, input_height, input_width , input_depth);
	create_matrix3d(&layer->d_input, input_height, input_width, input_depth);
	
	matrix2d_t* sample = &layer->input.layers[0];
	
	int output_rows = 0;
    int output_cols = 0;
	compute_output_size(sample->rows_n, sample->cols_n, kernel_size, padding, stride, &output_rows, &output_cols);

	create_matrix3d(&layer->output, output_rows, output_cols, input_depth);

	// if it's a max pooling layer, we need to keep track of the index of each element of the output matrix w.r.t. the input matrix
	// for instance, if the first slice of the input has the max element of the kernel at (1,0), the first element of indexes
	// will be a matrix3d with depth 2 (one for i, one for j)
	// TODO finish the explaination
	if(layer->type == POOLING_TYPE_MAX){
		layer->indexes = (matrix3d_t*)malloc(layer->output.depth * sizeof(matrix3d_t));
		for(int i=0;i<layer->output.depth;i++){
			create_matrix3d(&layer->indexes[i], output_rows, output_cols, 2);
		}
	}

}

// TODO move all the memory allocation here, except for the auxiliary structures used in the process/backpropagation
void init_conv_layer(
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
	create_matrix3d(&layer->input, input_height, input_width, input_depth);
	create_matrix3d(&layer->d_input, input_height, input_width, input_depth);

	// allocate kernels
	layer->kernels = (matrix3d_t*)malloc(layer->kernels_n * sizeof(matrix3d_t));

	for(int i=0;i<layer->kernels_n;i++){
		create_matrix3d(&layer->kernels[i], kernel_size, kernel_size, input_depth);
		matrix3d_randomize(&layer->kernels[i]);
	}

	int output_height = 0;
	int output_width = 0;
	matrix2d_t* sample = &layer->input.layers[0];
	compute_output_size(sample->rows_n, sample->cols_n, kernel_size, padding, stride, &output_height, &output_width);

	// allocate biases
	layer->biases = (matrix2d_t*)malloc(layer->kernels_n * sizeof(matrix2d_t));
	for(int i=0;i<kernels_n;i++){
		create_matrix2d(&layer->biases[i], output_height, output_width);
		matrix2d_randomize(&layer->biases[i]);
	}

	// allocate output and d_output
	create_matrix3d(&layer->output, output_height, output_width, kernels_n);
	create_matrix3d(&layer->output_activated, output_height, output_width, kernels_n);
}

void init_dense_layer(dense_layer_t* layer, int input_n, int output_n, activation_type activation_type){
	create_matrix2d(&layer->inputs, 1, input_n);
	create_matrix2d(&layer->d_inputs, 1, input_n);
	create_matrix2d(&layer->weights, input_n, output_n);
	matrix2d_randomize(&layer->weights);
	create_matrix2d(&layer->biases, 1, output_n);
	matrix2d_randomize(&layer->biases);
	create_matrix2d(&layer->output, 1, output_n);
	create_matrix2d(&layer->output_activated, 1, output_n);
	layer->activation_type = activation_type;
}

// ------------------------------ FEED ------------------------------

void feed_dense_layer(dense_layer_t* layer, const matrix2d_t* const input){
	matrix2d_copy_inplace(input, &layer->inputs);
}

void feed_pool_layer(pool_layer_t* layer, const matrix3d_t* const input){
	matrix3d_copy_inplace(input, &layer->input);
}

void feed_conv_layer(conv_layer_t* layer, const matrix3d_t* const input){
	matrix3d_copy_inplace(input, &layer->input);
}

// ------------------------------ PROCESS ------------------------------

void process_dense_layer(dense_layer_t* layer){
	for(int i=0;i<layer->output.cols_n;i++){
		layer->output.values[0][i] = layer->biases.values[0][i];
		for(int j=0;j<layer->weights.rows_n;j++){
			layer->output.values[0][i] += (layer->inputs.values[0][j] * layer->weights.values[j][i]);
		}
	}
	matrix2d_copy_inplace(&layer->output, &layer->output_activated);
	switch(layer->activation_type){
		case ACTIVATION_TYPE_RELU:
			matrix2d_relu_inplace(&layer->output_activated);
			break;
		case ACTIVATION_TYPE_SIGMOID:
			matrix2d_sigmoid_inplace(&layer->output_activated);
			break;
		default:
			break;
	}
}

void process_pool_layer(pool_layer_t* layer){
	for(int i=0;i<layer->input.depth;i++){
		switch(layer->type){
			case POOLING_TYPE_AVERAGE:
				avg_pooling(&layer->input.layers[i], &layer->output.layers[i], layer->kernel_size, layer->padding, layer->stride);
				break;
			case POOLING_TYPE_MAX:
				max_pooling(&layer->input.layers[i], &layer->output.layers[i], &layer->indexes[i], layer->kernel_size, layer->padding, layer->stride);
				break;
		}
	}
}

// TODO check if depth of each kernel is equal to the number of channels of the input
void process_conv_layer(conv_layer_t* layer){
	matrix2d_t result = {0};
	create_matrix2d(&result, layer->output.layers[0].rows_n, layer->output.layers[0].cols_n);
	for(int i=0;i<layer->kernels_n;i++){
		for(int j=0;j<layer->kernels[i].depth;j++){
			// compute the cross correlation between a channel of the input and its corresponding kernel
			full_cross_correlation(&layer->input.layers[j], &layer->kernels[i].layers[j], &result, layer->padding, layer->stride);
			if(j == 0){
				// perform an early sum of the biases to the final output layer
				matrix2d_sum_inplace(&layer->biases[i], &layer->output.layers[i]);
			}
			// then we sum the resulting matrix to the output
			matrix2d_sum_inplace(&result, &layer->output.layers[i]);
		}
		matrix2d_copy_inplace(&layer->output.layers[i], &layer->output_activated.layers[i]);
		switch(layer->activation_type){
			case ACTIVATION_TYPE_RELU:
				matrix2d_relu_inplace(&layer->output_activated.layers[i]);
				break;
			case ACTIVATION_TYPE_SIGMOID:
				matrix2d_sigmoid_inplace(&layer->output_activated.layers[i]);
				break;
			default:
				break;
		}
	}
	destroy_matrix2d(&result);
}

// ------------------------------ BACK-PROPAGATE ------------------------------


// the input is the derivative of the cost w.r.t the output, coming from the next layer
// the output if the derivative of the input, that has to be passed to the previous layer
void backpropagation_dense_layer(dense_layer_t* layer, const matrix2d_t* const input, float learning_rate)
{
	for(int i=0;i<layer->weights.rows_n;i++){
		float d_input = 0.f;
		for(int j=0;j<layer->weights.cols_n;j++){
			// d_actf(z_j) * d_act
			float common = d_activate(layer->output.values[0][j], layer->activation_type) * input->values[0][j];
			// compute d_weight
			// dC/dw_ij = x_i * d_actf(z_j) * d_act
			float d_weight = layer->inputs.values[0][i] * common;

			// compute d_input
			d_input += (layer->weights.values[i][j] * common);

			layer->weights.values[i][j] = gradient_descent(layer->weights.values[i][j], learning_rate, d_weight);
			// TODO we don't need to update the bias
			layer->biases.values[0][j] = gradient_descent(layer->biases.values[0][j], learning_rate, common);
		}
		layer->d_inputs.values[0][i] = d_input;
	}
}

// https://lanstonchu.wordpress.com/2018/09/01/convolutional-neural-network-cnn-backward-propagation-of-the-pooling-layers/
// TODO add avg_pooling handling, this is correct just for max_pooling
void backpropagation_pool_layer(pool_layer_t* layer, const matrix3d_t* const input){
	switch(layer->type){
		case POOLING_TYPE_MAX: {
			for(int i=0;i<layer->d_input.depth;i++){
				for(int m=0;m<layer->indexes[i].layers[0].rows_n;m++){
					for(int n=0;n<layer->indexes[i].layers[0].cols_n;n++){
						int row_i = layer->indexes[i].layers[0].values[m][n];
						int col_i = layer->indexes[i].layers[1].values[m][n];
						layer->d_input.layers[i].values[row_i][col_i] += input->layers[i].values[m][n];
					}
				}
			}
			break;
		}

		case POOLING_TYPE_AVERAGE: {
			for(int l=0;l<layer->output.depth;l++){
				matrix2d_t* output_layer = &layer->output.layers[l];
				for (int h = 0; h < output_layer->rows_n; h++) {
					for (int w = 0; w < output_layer->cols_n; w++) {
						// The gradient from the output for this pooled region
						float gradient = input->layers[l].values[h][w];
						
						// Distribute the gradient to each input element in the pooling region
						for (int i = 0; i < layer->kernel_size; i++) {
							for (int j = 0; j < layer->kernel_size; j++) {
								int input_h = h * layer->stride + i;
								int input_w = w * layer->stride + j;
								
								// Ensure we're within bounds (important for cases where pooling window goes out of input bounds)
								if (input_h < layer->input.layers[l].rows_n && input_w < layer->input.layers[l].cols_n) {
									layer->d_input.layers[l].values[input_h][input_w] += gradient / (float)(layer->kernel_size * layer->kernel_size);
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

void backpropagation_conv_layer(conv_layer_t* layer, const matrix3d_t* const input, float learning_rate){
	// matrix used to store the product (element x element) between the input and the derivative of the activation function of each
	// output of the layer
	matrix2d_t d_output = {0};
	create_matrix2d(&d_output, layer->output.layers[0].rows_n, layer->output.layers[0].cols_n);

	// allocate memory for d_kernel, that is the matrix that contains the correction that has to be applied to the weights of the kernels
	// after the whole computation
	matrix3d_t* d_kernel = (matrix3d_t*)malloc(layer->kernels_n * sizeof(matrix3d_t));
	for(int i=0;i<layer->kernels_n;i++){
		create_matrix3d(&d_kernel[i], layer->kernels[i].layers[0].rows_n, layer->kernels[i].layers[0].cols_n, layer->kernels[i].depth);
	}

	// matrix used to store the result of each convolution between the input (from the next layer) and the kernel
	matrix2d_t d_input_aux = {0};
	create_matrix2d(&d_input_aux, layer->d_input.layers[0].rows_n, layer->d_input.layers[0].cols_n);

	// for each kernel of the layer
	for(int i=0;i<layer->kernels_n;i++){
		for(int m=0;m<d_output.rows_n;m++){
			for(int n=0;n<d_output.cols_n;n++){
				d_output.values[m][n] = d_activate(layer->output.layers[i].values[m][n], layer->activation_type);
			}
		}
		matrix2d_mul_inplace(&d_output, &input->layers[i]);
		
		// for each layer of the current kernel compute the derivative
		// using the cross correlation between the j-th input and the i-th output (rotated)
		matrix3d_t* kernel = &layer->kernels[i];
		for(int j=0;j<kernel->depth;j++){
			// compute the derivative for the correction of the kernel
			// TODO correct also we the stride
			full_cross_correlation(&layer->input.layers[j], &d_output, &d_kernel[i].layers[j], layer->padding, 1);
			// compute the derivative for the correction of the input
			convolution(&d_output, &kernel->layers[j], &d_input_aux, kernel->layers[j].rows_n - input->layers[j].rows_n + 1);
			matrix2d_sum_inplace(&d_input_aux, &layer->d_input.layers[j]);
		}

		// update biases
		for(int m=0;m<d_output.rows_n;m++){
			for(int n=0;n<d_output.rows_n;n++){
				layer->biases[i].values[m][n] = gradient_descent(layer->biases[i].values[m][n], d_output.values[m][n], learning_rate);
			}
		}
	}

	// update weights
	for(int i=0;i<layer->kernels_n;i++){
		for(int j=0;j<layer->kernels[i].depth;j++){
			for(int m=0;m<layer->kernels[i].layers[j].rows_n;m++){
				for(int n=0;n<layer->kernels[i].layers[j].cols_n;n++){
					layer->kernels[i].layers[j].values[m][n] = gradient_descent(layer->kernels[i].layers[j].values[m][n], learning_rate, d_kernel[i].layers[j].values[m][n]);
				}
			}
		}
	}

	for(int i=0;i<layer->kernels_n;i++){
		destroy_matrix3d(&d_kernel[i]);
	}
	free(d_kernel);

	destroy_matrix2d(&d_output);
	destroy_matrix2d(&d_input_aux);
}

// ------------------------------ DESTROY ------------------------------

void destroy_dense_layer(dense_layer_t* layer){
	destroy_matrix2d(&layer->inputs);
	destroy_matrix2d(&layer->d_inputs);
	destroy_matrix2d(&layer->weights);
	destroy_matrix2d(&layer->biases);
	destroy_matrix2d(&layer->output);
	destroy_matrix2d(&layer->output_activated);
}

void destroy_conv_layer(conv_layer_t* layer){
	destroy_matrix3d(&layer->input);
	destroy_matrix3d(&layer->d_input);
	for(int i=0;i<layer->kernels_n;i++){
		destroy_matrix3d(&layer->kernels[i]);
		destroy_matrix2d(&layer->biases[i]);
	}
	free(layer->kernels);
	free(layer->biases);
	destroy_matrix3d(&layer->output);
	destroy_matrix3d(&layer->output_activated);
}

void destroy_pool_layer(pool_layer_t* layer){
	destroy_matrix3d(&layer->output);
	destroy_matrix3d(&layer->input);
	destroy_matrix3d(&layer->d_input);
	if(layer->type == POOLING_TYPE_MAX){
		for(int i=0;i<layer->input.depth;i++){
			destroy_matrix3d(&layer->indexes[i]);
		}
	}
}

void destroy_layer(layer_t_old* layer){
	for(int i=0;i<layer->neurons_num;i++){
		destroy_neuron(&layer->neurons[i]);
	}
	free(layer->neurons);
}
