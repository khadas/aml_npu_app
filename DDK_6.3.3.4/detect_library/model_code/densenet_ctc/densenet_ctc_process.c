#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "densenet_ctc_process.h"
#include "densenet_ctc.h"

#define NN_TENSOR_MAX_DIMENSION_NUMBER 4

/*Preprocess*/
void yolov3_preprocess(input_image_t imageData, uint8_t *ptr)
{
    int nn_width, nn_height, channels, tmpdata;
    int offset=0, i=0, j=0;
    uint8_t *src = (uint8_t *)imageData.data;

    model_getsize(&nn_width, &nn_height, &channels);
    memset(ptr, 0, nn_width * nn_height * channels * sizeof(uint8_t));

    if (imageData.pixel_format == PIX_FMT_NV21) {
        offset = nn_width * nn_height;

        for (j = 0; j < nn_width * nn_height; j++) {
            tmpdata = (src[j]>>1);
            tmpdata = (uint8_t)((tmpdata >  127) ? 127 : (tmpdata < -128) ? -128 : tmpdata);
            ptr[j] = tmpdata;
            ptr[j + offset*1] = tmpdata;
            ptr[j + offset*2] = tmpdata;
        }
        return;
    }

    for (i = 0; i < channels; i++) {
        offset = nn_width * nn_height *( channels -1 - i);  // prapare BGR input data
        for (j = 0; j < nn_width * nn_height; j++) {
        	tmpdata = (src[j * channels + i]>>1);
        	ptr[j + offset] = (uint8_t)((tmpdata >  127) ? 127 : (tmpdata < -128) ? -128 : tmpdata);
        }
    }
    return;
}

/* Postprocess */

static char *names[] = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "+", "-", "*", "/", ",", ".", "[", "]", "{", "}", "|", "~", "@", "#", "$", "%", "^", "&", "(", ")", "<", ">", "?", ":", ";", "a", "b", "c", "d", "e", "f", "g", "h", "i", "g", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};


static vx_float32 Float16ToFloat32(const vx_int16* src, float* dst, int length)
{
	vx_int32 t1;
	vx_int32 t2;
	vx_int32 t3;
	vx_float32 out;
	
	for (int i = 0; i < length; i++)
	{
		t1 = src[i] & 0x7fff;
		t2 = src[i] & 0x8000;
		t3 = src[i] & 0x7c00;
		
		t1 <<= 13;
		t2 <<= 16;
		
		t1 += 0x38000000;
		t1 = (t3 == 0 ? 0 : t1);
		t1 |= t2;
		*((uint32_t *)&out) = t1;
		dst[i] = out;
	}
	return out;
}

void yolov3_postprocess(vsi_nn_graph_t *graph, char* result, int* result_len)
{
    vsi_nn_tensor_t *tensor = NULL;
    tensor = vsi_nn_GetTensor(graph, graph->input.tensors[0]);
    float *predictions = NULL;
    int output_len = 0;
    int i, j, max_index, stride;
    int box = 35;
    int class_num = 88;
    float threshold = 0.25;
    int sz[10];
    uint8_t *tensor_data = NULL;
    float max_conf, conf;
    vsi_status status = VSI_FAILURE;
    
    for (i = 0; i < graph->output.num; i++) {
        tensor = vsi_nn_GetTensor(graph, graph->output.tensors[i]);
        sz[i] = 1;
        for (j = 0; j < tensor->attr.dim_num; j++) {
            sz[i] *= tensor->attr.size[j];
        }
        output_len += sz[i];
    }
    predictions = (float *)malloc(sizeof(float) * output_len);
    
    for (i = 0; i < graph->output.num; i++) {
        tensor = vsi_nn_GetTensor(graph, graph->output.tensors[i]);

        stride = vsi_nn_TypeGetBytes(tensor->attr.dtype.vx_type);
        tensor_data = (uint8_t *)vsi_nn_ConvertTensorToData(graph, tensor);
        vx_int16* data_ptr_32 = (vx_int16 *) tensor_data;
        Float16ToFloat32(data_ptr_32, predictions, output_len);

        vsi_nn_Free(tensor_data);
    }
    
    int last_index = class_num - 1;
    for (i = 0; i < box; ++i)
    {
    	max_conf = 0;
    	max_index = class_num - 1;
    	for (j = 0; j < class_num; ++j)
    	{
    		conf = predictions[i * class_num + j];
    		if (conf > threshold && conf > max_conf)
    		{
    			max_conf = conf;
    			max_index = j;
    		}
    	}
    	if (max_index != class_num - 1 && max_index != last_index)
    	{
    		result[*result_len] = *names[max_index];
    		(*result_len)++;
    	}
    	last_index = max_index;
    }
    
    if (predictions) free(predictions);
    return;
}
