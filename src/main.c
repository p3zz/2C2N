#include "utils.h"
#include "common.h"
#include "layer.h"
#include "time.h"
#include "stdlib.h"
#include "stdio.h"

int main(void){
    srand(time(NULL));
    const float learning_rate = 0.001;
    const int iterations = 10;
    conv_layer_t layer0 = {0};
    pool_layer_t layer1 = {0};
    conv_layer_t layer2 = {0};
    pool_layer_t layer3 = {0};
    dense_layer_t layer4 = {0};
    dense_layer_t layer5 = {0};
    dense_layer_t layer6 = {0};
    softmax_layer_t layer7 = {0};
    
    matrix3d_t input = {0};
    matrix3d_t d_input = {0};
    matrix3d_t output_target = {0};
    matrix3d_init(&input, 32, 32, 1);
    matrix3d_init(&output_target, 1, 10, 1);
    matrix3d_init(&d_input, 1, 10, 1);

    conv_layer_init(&layer0, 32, 32, 1, 5, 6, 1, 0, ACTIVATION_TYPE_TANH);
    pool_layer_init(&layer1, 28, 28, 6, 2, 0, 2, POOLING_TYPE_AVERAGE);
    conv_layer_init(&layer2, 14, 14, 6, 5, 16, 1, 0, ACTIVATION_TYPE_TANH);
    pool_layer_init(&layer3, 10, 10, 16, 2, 0, 2, POOLING_TYPE_AVERAGE);
    dense_layer_init(&layer4, 400, 120, ACTIVATION_TYPE_TANH);
    dense_layer_init(&layer5, 120, 84, ACTIVATION_TYPE_TANH);
    dense_layer_init(&layer6, 84, 10, ACTIVATION_TYPE_IDENTITY);
    softmax_layer_init(&layer7, 10);

    // draw the digit 1
    for(int i=0;i<input.layers[0].rows_n;i++){
        for(int j=0;j<input.layers[0].cols_n;j++){
            if(j == input.layers[0].cols_n / 2){
                input.layers[0].values[i][j] = 255;
            }
        }
    }

    output_target.layers[0].values[0][1] = 1.0;

    // printf("Height: %d\tWidth: %d\tDepth: %d\n", layer1.output.layers[0].rows_n, layer1.output.layers[0].cols_n, layer1.output.depth);

    // printf("Input");
    // matrix3d_print(&input);

    matrix3d_t aux = {0};
    matrix3d_init(&aux, 5, 5, 16);

    for(int i=0;i<iterations;i++){
        conv_layer_feed(&layer0, &input);
        conv_layer_forwarding(&layer0);
        // printf("Conv layer 0-----------------------------\n");
        // matrix3d_print(&layer0.output);
        // matrix3d_print(&layer0.output_activated);

        pool_layer_feed(&layer1, &layer0.output_activated);
        pool_layer_forwarding(&layer1);
        // printf("Pool layer 1-----------------------------\n");
        // matrix3d_print(&layer1.output);

        conv_layer_feed(&layer2, &layer1.output);
        conv_layer_forwarding(&layer2);
        // printf("Conv layer 2-----------------------------\n");
        // matrix3d_print(&layer2.output);
        // matrix3d_print(&layer2.output_activated);

        pool_layer_feed(&layer3, &layer2.output_activated);
        pool_layer_forwarding(&layer3);
        // printf("Pool layer 3-----------------------------\n");
        // matrix3d_print(&layer3.output);

        matrix3d_reshape(&layer3.output, &layer4.inputs);
        dense_layer_forwarding(&layer4);
        // printf("Dense layer 4-----------------------------\n");
        // matrix3d_print(&layer4.output);
        // matrix3d_print(&layer4.output_activated);
        
        dense_layer_feed(&layer5, &layer4.output_activated);
        dense_layer_forwarding(&layer5);
        // printf("Dense layer 5-----------------------------\n");
        // matrix3d_print(&layer5.output);
        // matrix3d_print(&layer5.output_activated);

        dense_layer_feed(&layer6, &layer5.output_activated);
        dense_layer_forwarding(&layer6);
        // printf("Dense layer 6-----------------------------\n");
        // matrix3d_print(&layer6.output);
        // matrix3d_print(&layer6.output_activated);

        // printf("Softmax layer 7-----------------------------\n");
        softmax_layer_feed(&layer7, &layer6.output_activated);
        softmax_layer_forwarding(&layer7);
        // matrix3d_print(&layer7.output);

        cross_entropy_loss_derivative(&layer7.output.layers[0], &output_target.layers[0], &d_input.layers[0]);
        float cost = cross_entropy_loss(&layer7.output.layers[0], &output_target.layers[0]);
        printf("Cost: %f\n", cost);

        // matrix2d_print(&layer7.output.layers[0]);
        // matrix2d_print(&output_target.layers[0]);
        // matrix2d_print(&d_input.layers[0]);

        // matrix3d_print(&d_input);

        // printf("[BACKPROP] Softmax layer 7-----------------------------\n");
        softmax_layer_backpropagation(&layer7, &d_input);
        // matrix3d_print(&layer7.d_input);

        // printf("[BACKPROP] Dense layer 6-----------------------------\n");
        dense_layer_backpropagation(&layer6, &layer7.d_input, learning_rate);
        // matrix3d_print(&layer6.d_inputs);

        // printf("[BACKPROP] Dense layer 5-----------------------------\n");
        dense_layer_backpropagation(&layer5, &layer6.d_inputs, learning_rate);
        // matrix3d_print(&layer5.d_inputs);
        
        // printf("[BACKPROP] Dense layer 4-----------------------------\n");
        dense_layer_backpropagation(&layer4, &layer5.d_inputs, learning_rate);
        // matrix3d_print(&layer4.d_inputs);

        matrix3d_reshape(&layer4.d_inputs, &aux);
        
        pool_layer_backpropagation(&layer3, &aux);
        // matrix3d_print(&layer3.d_input);
        
        conv_layer_backpropagation(&layer2, &layer3.d_input, learning_rate);
        // matrix3d_print(&layer2.d_input);
        pool_layer_backpropagation(&layer1, &layer2.d_input);
        // matrix3d_print(&layer1.d_input);
        conv_layer_backpropagation(&layer0, &layer1.d_input, learning_rate);
    }

    matrix3d_destroy(&aux);

    conv_layer_destroy(&layer0);
    pool_layer_destroy(&layer1);
    conv_layer_destroy(&layer2);
    pool_layer_destroy(&layer3);
    dense_layer_destroy(&layer4);
    dense_layer_destroy(&layer5);
    dense_layer_destroy(&layer6);
    softmax_layer_destroy(&layer7);
    matrix3d_destroy(&input);
    matrix3d_destroy(&d_input);
    matrix3d_destroy(&output_target);

    // TEST_ASSERT_TRUE(false);
    return SUCCESS;
}
