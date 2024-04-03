#include "CBin-NN.h"
#include "main.h"
#include<stdio.h>
extern UART_HandleTypeDef huart3;
int buffer1[CONV1_OUT_CH*CONV1_OUT_DIM*CONV1_OUT_DIM/32];
int buffer2[CONV1_OUT_CH*CONV1_OUT_DIM*CONV1_OUT_DIM/32];
float classification[10];
#int t1, t2;
#int time;
int bnn_main()
{
	#t1 = HAL_GetTick();

	QQConv2D_Optimized_PReLU(buffer1, CONV1_OUT_CH, CONV1_OUT_DIM, CONV1_IN_CH, CONV1_IN_DIM, CONV1_KER_SIZE, CONV1_STRIDE, CONV1_PADDING, NULL, conv1_wt, bn1_alpha1, bn1_alpha2, shift1, input_image);
	BMaxPool2D_Optimized(buffer2, POOL1_CH, POOL1_IN_DIM, POOL1_OUT_DIM, POOL1_KER_SIZE, POOL1_STRIDE, POOL1_PADDING, buffer1);
	BBConv2D_Optimized_PReLU(buffer1, CONV2_OUT_CH, CONV2_OUT_DIM, CONV2_IN_CH, CONV2_IN_DIM, CONV2_KER_SIZE, CONV2_STRIDE, CONV2_PADDING, NULL, conv2_wt, bn2_wt, shift2, buffer2);
	BMaxPool2D_Optimized(buffer2, POOL2_CH, POOL2_IN_DIM, POOL2_OUT_DIM, POOL2_KER_SIZE, POOL2_STRIDE, POOL2_PADDING, buffer1);
	BBConv2D_Optimized_PReLU(buffer1, CONV3_OUT_CH, CONV3_OUT_DIM, CONV3_IN_CH, CONV3_IN_DIM, CONV3_KER_SIZE, CONV3_STRIDE, CONV3_PADDING, NULL, conv3_wt, bn3_wt, shift3, buffer2);
	BMaxPool2D_Optimized(buffer2, POOL3_CH, POOL3_IN_DIM, POOL3_OUT_DIM, POOL3_KER_SIZE, POOL3_STRIDE, POOL3_PADDING, buffer1);
	BBQFC_Optimized_PReLU(classification, FC1_OUT_DIM, FC1_IN_DIM, NULL, fc1_wt, bn4_wt, shift4, buffer2);
	#t2 = HAL_GetTick();
	#time = t2 - t1;
}
