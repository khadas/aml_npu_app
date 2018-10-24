/*******************************************************************************
 * Name        : vxc_nn_inceptionV1_node.c
 * Author      : Your name
 * Copyright   : Your copyright notice
 * Description : Neural network node source file generated by VivanteIDE
 *******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include "vxc_nn_inceptionV1.h"
#include "vxc_nn_inceptionV1_priv.h"

#define CONV_WEIGHT_DIMENSION 4
#define CONV_BIAS_DIMENSION 1
#define FC_WEIGHT_DIMENSION 4
#define FC_BIAS_DIMENSION 1
#define NN_CONV_MAX_GROUP 2

/* Get the size of specified format */
vx_uint32 vxcGetTypeSize(vx_enum format)
{
    switch (format)
    {
    case VX_TYPE_INT8:
    case VX_TYPE_UINT8:
        return 1;
    case VX_TYPE_INT16:
    case VX_TYPE_UINT16:
        return 2;
    case VX_TYPE_INT32:
    case VX_TYPE_UINT32:
        return 4;
    case VX_TYPE_INT64:
    case VX_TYPE_UINT64:
        return 8;
    case VX_TYPE_FLOAT32:
        return 4;
    case VX_TYPE_FLOAT64:
        return 8;
    case VX_TYPE_ENUM:
        return 4;
    case VX_TYPE_FLOAT16:
        return 2;
    }

    return 4;
}

/* Load a tensor from the resource file */
vx_tensor vxcLoadTensor(vx_context context, vx_uint32 num_of_dim, vx_uint32 x, vx_uint32 y, vx_uint32 z, vx_uint32 w,
		tensor_format_info_t* tfi, FILE* fp, vx_uint32 blob_offset, vx_uint32 blob_size, vx_uint32 blob_uint_size)
{
    vx_uint32      size[NN_TENSOR_MAX_DIMENSION];
    vx_uint32      stride_size[NN_TENSOR_MAX_DIMENSION];
    vx_tensor_addressing      tensor_addressing = NULL;
    vx_tensor_create_params_t tensor_create_params;
    vx_tensor      tensor = NULL;
    void*          buffer = NULL;
    vx_status      status;
    int            ret;

    size[0]       = x;
    size[1]       = y;
    size[2]       = z;
    size[3]       = w;
    if (tfi->quant_format == VX_QUANT_AFFINE_SCALE)
    {
        tensor_create_params.num_of_dims = num_of_dim;
        tensor_create_params.sizes = (vx_int32 *)size;
        tensor_create_params.data_format = tfi->data_format;
        tensor_create_params.quant_format = tfi->quant_format;
        tensor_create_params.quant_data.affine.scale = tfi->quant_data.affine.scale;
        tensor_create_params.quant_data.affine.zeroPoint = tfi->quant_data.affine.zero_point;
        tensor         = vxCreateTensor2(context, (const vx_tensor_create_params_t*)&tensor_create_params, sizeof(tensor_create_params));
    } else {
        tensor         = vxCreateTensor(context, num_of_dim, (vx_uint32*)size, tfi->data_format, tfi->quant_data.dfp.fixed_point_pos);
    }
    if (tensor == NULL)
    {
        printf("vxCreateTensor2 failure! at line %d\n", __LINE__);
        return NULL;
    }

    stride_size[0]     = blob_uint_size;
    stride_size[1]     = size[0] * stride_size[0];
    stride_size[2]     = size[1] * stride_size[1];
    stride_size[3]     = size[2] * stride_size[2];
    tensor_addressing   = vxCreateTensorAddressing(context, (vx_uint32*)size, stride_size, num_of_dim);
    if (tensor_addressing == NULL)
    {
        printf("vxCreateTensorAddressing failure! at line %d\n", __LINE__);
        goto error;
    }

    /* copy weight data to weight tensor  */
    ret = fseek(fp, blob_offset, SEEK_SET);
    if (ret != 0)
    {
        printf("blob read failure! at line %d\n", __LINE__);
        goto error;
    }

    buffer = malloc(blob_size);
    if (buffer == NULL)
    {
        printf("buffer malloc failure! at line %d\n", __LINE__);
        goto error;
    }

    ret = fread(buffer, 1, blob_size, fp);
    if ((vx_uint32)ret != blob_size) {
        printf("blob reading failure! at line %d\n", __LINE__);
        goto error;
    }

    status = vxCopyTensorPatch(tensor, NULL, tensor_addressing, buffer, VX_WRITE_ONLY, 0);
    if (status != VX_SUCCESS)
    {
        printf("vxCopyTensorPatch failure! at line %d\n", __LINE__);
        goto error;
    }

    free(buffer);
    vxReleaseTensorAddressing(&tensor_addressing);
    return tensor;

error:
    if (tensor)
        vxReleaseTensor(&tensor);

    if (buffer)
        free(buffer);

    if (tensor_addressing)
        vxReleaseTensorAddressing(&tensor_addressing);

    return NULL;
}

/* Create a tensor from the specified data */
vx_tensor vxcCreateTensorFromData(vx_context context, int num_of_dim, vx_uint32 extend[], tensor_format_info_t* tfi, void *data)
{
    vx_uint32                   image_stride_size[NN_TENSOR_MAX_DIMENSION] = {0, 0, 0, 0};
    vx_tensor_addressing        tensor_addressing = NULL;
    vx_tensor_create_params_t   tensor_create_params;
    vx_tensor                   tensor;
    vx_status                   status;
    int                         i;

    if (tfi->quant_format == VX_QUANT_AFFINE_SCALE)
    {
        tensor_create_params.num_of_dims = num_of_dim;
        tensor_create_params.sizes = (vx_int32 *)extend;
        tensor_create_params.data_format = tfi->data_format;
        tensor_create_params.quant_format = tfi->quant_format;
        tensor_create_params.quant_data.affine.scale = tfi->quant_data.affine.scale;
        tensor_create_params.quant_data.affine.zeroPoint = tfi->quant_data.affine.zero_point;
        tensor         = vxCreateTensor2(context, (const vx_tensor_create_params_t*)&tensor_create_params, sizeof(tensor_create_params));
    } else {
        tensor         = vxCreateTensor(context, num_of_dim, (vx_uint32*)extend, tfi->data_format, tfi->quant_data.dfp.fixed_point_pos);
    }
    if (tensor == NULL)
    {
        printf("vxCreateTensor2 failure! at line %d\n", __LINE__);
        goto error;
    }

    image_stride_size[0] = vxcGetTypeSize(tfi->data_format);
    for (i = 1; i < num_of_dim; i++)
    {
        image_stride_size[i] = image_stride_size[i-1] * extend[i-1];
    }

    tensor_addressing = vxCreateTensorAddressing(context, (vx_uint32*)extend, image_stride_size, num_of_dim);
    if (tensor_addressing == NULL)
    {
        printf("vxCreateTensorAddressing failure! at line %d\n", __LINE__);
        goto error;
    }

    status = vxCopyTensorPatch(tensor, NULL, tensor_addressing, data, VX_WRITE_ONLY, 0);
    if (status != VX_SUCCESS)
    {
        printf("vxCopyTensorPatch failure! at line %d\n", __LINE__);
        goto error;
    }

    vxReleaseTensorAddressing(&tensor_addressing);

    return tensor;

error:
    if (tensor_addressing) vxReleaseTensorAddressing(&tensor_addressing);

    return NULL;
}

/* Load weights_biases_parameter from the resource file */
vx_weights_biases_parameter vxcLoadWeightsBiasesParameter(vx_context context, FILE* fp, vx_uint32 blob_offset, vx_uint32 blob_size)
{
    vx_weights_biases_parameter    weight_bias = NULL;
    void*                          buffer = NULL;
    int                            ret;

    ret = fseek(fp, blob_offset, SEEK_SET);
    if (ret != 0)
    {
        printf("blob read failure! at line %d\n", __LINE__);
        goto error;
    }

    buffer = malloc(blob_size);
    if (buffer == NULL)
    {
        printf("buffer malloc failure! at line %d\n", __LINE__);
        goto error;
    }

    ret = fread(buffer, 1, blob_size, fp);
    if ((vx_uint32)ret != blob_size) {
        printf("date reading failure! at line %d\n", __LINE__);
        goto error;
    }

    weight_bias = vxCreateWeightsBiasesParameterFromStream(context, (vx_uint32*)buffer);
    if (weight_bias == NULL) {
        printf("weight bias loading failure! at line %d\n", __LINE__);
        goto error;
    }

    free(buffer);
    return weight_bias;

error:
    if (buffer)
        free(buffer);

    return NULL;
}

/* Create a sub-tensor from a specified tensor */
vx_tensor vxcCreateSubtensor(vx_context context, tensor_split_info_t *tsi)
{
    vx_status status;
    vx_int32 num_of_dim;
    vx_uint32 size[NN_TENSOR_MAX_DIMENSION];
    vx_uint32 start[NN_TENSOR_MAX_DIMENSION];
    vx_uint32 end[NN_TENSOR_MAX_DIMENSION];
    vx_tensor_view tensor_view = NULL;
    vx_tensor subtensor;
    int index1, index2;
    int i;

//    printf("Enter %s\n", __FUNCTION__);
    status = vxQueryTensor(tsi->tensor, VX_TENSOR_NUM_OF_DIMS, &num_of_dim, sizeof(num_of_dim));
    if (status != VX_SUCCESS)
    {
        printf("vxQueryTensor failure! at line %d\n", __LINE__);
        goto error;
    }
//    printf("num_of_dim = %d\n", num_of_dim);

    status = vxQueryTensor(tsi->tensor, VX_TENSOR_DIMS, size, sizeof(size));
    if (status != VX_SUCCESS)
    {
        printf("vxQueryTensor failure! at line %d\n", __LINE__);
        goto error;
    }

    index1 = tsi->axis >= 0 ? tsi->axis : num_of_dim + tsi->axis;
    index2 = tsi->axis2 >= 0 ? tsi->axis2 : num_of_dim + tsi->axis2;
//    printf("index1 = %d, index2 = %d\n", index1, index2);
//    printf("axis1 = %d, axis2 = %d\n", tsi->axis, tsi->axis2);

    for (i = 0; i < num_of_dim; i++)
    {
        if (i == index1)
        {
            start[i] = tsi->axis_start;;
            end[i] = tsi->axis_end;
//            printf("start[%d] = %d, end[%d] = %d\n", i, start[i], i, end[i]);
        }
        else if (i == index2)
        {
            start[i] = tsi->axis2_start;;
            end[i] = tsi->axis2_end;
//            printf("start[%d] = %d, end[%d] = %d\n", i, start[i], i, end[i]);
        }
        else
        {
            start[i] = 0;
            end[i] = size[i];
//            printf("start[%d] = %d, end[%d] = %d\n", i, start[i], i, end[i]);
        }
    }

    tensor_view = vxCreateTensorView(context, start, end, num_of_dim);
    if (tensor_view == NULL)
    {
        printf("vxCreateTensorView failure! at line %d\n", __LINE__);
        goto error;
    }

    subtensor = vxCreateTensorFromView(tsi->tensor, tensor_view);
    if (subtensor == NULL)
    {
        printf("vxCreateTensorFromView failure! at line %d\n", __LINE__);
        goto error;
    }

    vxReleaseTensorView(&tensor_view);
    return subtensor;

error:
    if (tensor_view)
    {
        vxReleaseTensorView(&tensor_view);
    }
    return NULL;
}

/* Create an output tensor */
vx_tensor vxcCreateOutputTensor(vx_context context, vx_graph graph, vx_uint32 num_of_dim,
        vx_uint32 *size, tensor_format_info_t* tfi,
        tensor_split_info_t *tensor_split_info, vx_bool output_layer)
{
    vx_tensor_create_params_t   tensor_create_params;
    vx_tensor output = NULL;

    if (tensor_split_info->tensor)
    {
        output = vxcCreateSubtensor(context, tensor_split_info);
    }
    else
    {
        if (tfi->quant_format == VX_QUANT_AFFINE_SCALE)
        {
            tensor_create_params.num_of_dims = num_of_dim;
            tensor_create_params.sizes = (vx_int32 *)size;
            tensor_create_params.data_format = tfi->data_format;
            tensor_create_params.quant_format = tfi->quant_format;
            tensor_create_params.quant_data.affine.scale = tfi->quant_data.affine.scale;
            tensor_create_params.quant_data.affine.zeroPoint = tfi->quant_data.affine.zero_point;
            output = vxCreateTensor2(context, (const vx_tensor_create_params_t*)&tensor_create_params, sizeof(tensor_create_params));
        } else {
            output = vxCreateTensor(context, num_of_dim, (vx_uint32*)size, tfi->data_format, tfi->quant_data.dfp.fixed_point_pos);
        }
    }

    return output;
}

/* Create a tensor by quant */
vx_tensor vxcCreateTensor(vx_context context, int num_of_dim, void *size, vx_enum data_format, vx_enum quant_format, vx_float32 scale, vx_int32 zeroPoint, vx_int32 fixed_point_pos)
{
	vx_tensor_create_params_t   tensor_create_params;
	vx_tensor output = NULL;
	if (quant_format == VX_QUANT_AFFINE_SCALE) {
		tensor_create_params.num_of_dims = num_of_dim;
		tensor_create_params.sizes = (vx_int32 *)size;
		tensor_create_params.data_format = data_format;
		tensor_create_params.quant_format = quant_format;
		tensor_create_params.quant_data.affine.scale = scale;
		tensor_create_params.quant_data.affine.zeroPoint = zeroPoint;
		output = vxCreateTensor2(context, (const vx_tensor_create_params_t*)&tensor_create_params, sizeof(tensor_create_params));
	} else
		output = vxCreateTensor(context, num_of_dim, (vx_uint32*)size, data_format, fixed_point_pos);
    return output;
}

static FILE *data_file;

/* initialize a network environment */
vx_status vxcNetworkInit(char *data_file_name)
{
    data_file = fopen(data_file_name, "rb");
    if (data_file == NULL)
    {
        return VX_FAILURE;
    }

    return VX_SUCCESS;
}

/* release a network environment */
void vxcNetworkExit(void)
{
    if (data_file != NULL)
    {
        fclose(data_file);
        data_file = NULL;
    }
}

/* Alloc a node_info */
void *vxcAllocNodeInfo(int size)
{
    return calloc(1, size);
}

/* Free the specified node_info */
void vxcFreeNodeInfo(void** node_info)
{
    free(*node_info);
    *node_info = NULL;
}

/* Broadcasting the source shapes into the destiny shape */
void vxcBroadcastShape(vx_int32 src1[], int size1, vx_int32 src2[], int size2, vx_int32 dest[])
{
    int low, high, *src, i;

    if (size1 >= size2)
    {
        low = size2;
        high = size1;
        src = src1;
    }
    else
    {
        low = size1;
        high = size2;
        src = src2;
    }

    for (i = 0; i < low; i++)
    {
        dest[i] = src1[i] >= src2[i] ? src1[i] : src2[i];
    }

    for (; i < high; i++)
    {
        dest[i] = src[i];
    }
}

/* Create and initialize a ConvolutionReluPooling layer node */
vx_status vxcCreateConvolutionReluPoolingLayer(void* node_info)
{
    convolution_relu_pooling_info_t         *cinfo = (convolution_relu_pooling_info_t *)node_info;
    tensor_split_info_t                     *tensor_split_info = &cinfo->tensor_split_info;
    vx_graph                                graph = cinfo->graph;
    vx_tensor                               input = cinfo->input;

    vx_uint32                               kernel_x = cinfo->kernel_x;
    vx_uint32                               kernel_y = cinfo->kernel_y;
    vx_uint32                               ofm = cinfo->ofm;
    vx_uint32                               weight_blob_offset = cinfo->weight_blob_offset;
    vx_uint32                               weight_blob_size = cinfo->weight_blob_size;
    vx_uint32                               bias_blob_offset = cinfo->bias_blob_offset;
    vx_uint32                               bias_blob_size = cinfo->bias_blob_size;
    vx_uint32                               stride_x = cinfo->stride_x;
    vx_uint32                               stride_y = cinfo->stride_y;
    vx_uint32                               padding_x = cinfo->padding_x;
    vx_uint32                               padding_x_right = cinfo->padding_x_right;
    vx_uint32                               padding_y = cinfo->padding_y;
    vx_uint32                               padding_y_bottom = cinfo->padding_y_bottom;
    vx_uint32                               dilation_x = cinfo->dilation_x;
    vx_uint32                               dilation_y = cinfo->dilation_y;
    vx_bool                                 has_relu = cinfo->has_relu;
    vx_enum                                 pool_type = cinfo->pool_type;
    vx_uint32                               pool_size_x = cinfo->pool_size_x;
    vx_uint32                               pool_size_y = cinfo->pool_size_y;
    vx_uint32                               pool_stride_x = cinfo->pool_stride_x;
    vx_uint32                               pool_stride_y = cinfo->pool_stride_y;
    vx_uint32                               pool_padding_x = cinfo->pool_padding_x;
    vx_uint32                               pool_padding_y = cinfo->pool_padding_y;
    vx_uint32                               group = cinfo->group == 0 ? 1 : cinfo->group;
    vx_enum                                 overflow_policy = cinfo->overflow_policy;
    vx_enum                                 rounding_policy = cinfo->rounding_policy;
    vx_enum                                 down_scale_size_rounding = cinfo->down_scale_size_rounding;
    FILE                                    *fp = data_file;
    tensor_format_info_t                    *output_format = &cinfo->output_format;
    vx_bool                                 output_layer = cinfo->output_layer;

    vx_context                              context;
    vx_uint32                               num_of_dim;
    vx_uint32                               input_size[NN_TENSOR_MAX_DIMENSION] = {0, 0, 0, 0};
    vx_uint32                               output_size[NN_TENSOR_MAX_DIMENSION] = {0, 0, 0, 0};
    vx_uint32                               input_start[NN_TENSOR_MAX_DIMENSION];
    vx_uint32                               input_end[NN_TENSOR_MAX_DIMENSION];
    vx_uint32                               output_start[NN_TENSOR_MAX_DIMENSION];
    vx_uint32                               output_end[NN_TENSOR_MAX_DIMENSION];
    vx_tensor_view                          input_view[NN_CONV_MAX_GROUP];
    vx_tensor_view                          output_view[NN_CONV_MAX_GROUP];
    vx_tensor                               inputs[NN_CONV_MAX_GROUP];
    vx_tensor                               outputs[NN_CONV_MAX_GROUP];
    vx_uint32                               weight_bias_blob_offset[NN_CONV_MAX_GROUP] = {weight_blob_offset, bias_blob_offset};
    vx_uint32                               weight_bias_blob_size[NN_CONV_MAX_GROUP] = {weight_blob_size, bias_blob_size};
    vx_weights_biases_parameter             weight_bias[NN_CONV_MAX_GROUP];
    vx_nn_convolution_relu_pooling_params_t convolution_params;
    vx_node                                 node[NN_CONV_MAX_GROUP];
    vx_uint32                               pool_size[NN_TENSOR_MAX_DIMENSION] = {0, 0, 0, 0};
    vx_bool                                 has_pool = (pool_type == 0 && pool_size_x == 0) ? vx_false_e : vx_true_e;
    vx_int32                                pad_const_val = 0;
    vx_scalar                               pad_const = NULL;
    vx_tensor                               output = NULL;
    vx_status                               status;
    int                                     i;

    context = vxGetContext((vx_reference)graph);
    if (context == NULL)
    {
        printf("vxGetContext failure! at line %d\n", __LINE__);
        goto error;
    }

    status = vxQueryTensor(input, VX_TENSOR_NUM_OF_DIMS, &num_of_dim, sizeof(num_of_dim));
    if (status != VX_SUCCESS)
    {
        printf("vxQueryTensor failure! at line %d\n", __LINE__);
        goto error;
    }

    status = vxQueryTensor(input, VX_TENSOR_DIMS, input_size, sizeof(input_size));
    if (status != VX_SUCCESS)
    {
        printf("vxQueryTensor failure! at line %d\n", __LINE__);
        goto error;
    }

    pad_const = vxCreateScalar(context, VX_TYPE_INT32, &pad_const_val);
    if (!pad_const)
    {
        printf("vxCreateScalar failure! at line %d\n", __LINE__);
        goto error;
    }

    output_size[0] = NN_DIV_DS_SIZE_ROUNDING((input_size[0] + padding_x + padding_x_right - kernel_x - (kernel_x - 1) * dilation_x), stride_x, down_scale_size_rounding) + 1;
    output_size[1] = NN_DIV_DS_SIZE_ROUNDING((input_size[1] + padding_y + padding_y_bottom - kernel_y - (kernel_y - 1) * dilation_y), stride_y, down_scale_size_rounding) + 1;
    output_size[2] = ofm;

    if (num_of_dim <= 3)
    {
        output_size[3] = 1;
    }
    else
    {
        output_size[3] = input_size[3];
    }

    if (has_pool)
    {
        pool_size[0] =  NN_DIV_DS_SIZE_ROUNDING(output_size[0] + 2 * pool_padding_x - pool_size_x, pool_stride_x, VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_CEILING) + 1;
        pool_size[1] =  NN_DIV_DS_SIZE_ROUNDING(output_size[1] + 2 * pool_padding_y - pool_size_y, pool_stride_y, VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_CEILING) + 1;
    }
    else
    {
        pool_size[0] =  output_size[0];
        pool_size[1] =  output_size[1];
    }
    pool_size[2] =  output_size[2];
    pool_size[3] =  output_size[3];

    output = vxcCreateOutputTensor(context, graph, num_of_dim, pool_size, output_format,
                tensor_split_info, output_layer);

    if (output == NULL)
    {
        printf("vxcCreateOutputTensor failure! at line %d\n", __LINE__);
        goto error;
    }

    output_size[2] /= group;
    pool_size[2] = output_size[2];

    if (group > 1)
    {
        for (i = 0; i < (int)group; i++)
        {
            input_view[i] = NULL;
            output_view[i] = NULL;

            inputs[i] = NULL;
            outputs[i] = NULL;
            weight_bias[i] = NULL;
            node[i] = NULL;
        }

        for (i = 0; i < (int)group; i++)
        {
            input_start[0] = 0;
            input_end[0] = input_size[0];
            input_start[1] = 0;
            input_end[1] = input_size[1];
            input_start[2] = (input_size[2] / group) * (i) ;
            input_end[2] = (input_size[2] / group) * (i + 1);
            input_start[3] = 0;
            input_end[3] = input_size[3];

            output_start[0] = 0;
            output_end[0] = pool_size[0];
            output_start[1] = 0;
            output_end[1] = pool_size[1];
            output_start[2] = pool_size[2] * (i) ;
            output_end[2] = pool_size[2] * (i + 1);
            output_start[3] = 0;
            output_end[3] = pool_size[3];

            input_view[i] = vxCreateTensorView(context, input_start, input_end, num_of_dim);
            if (input_view[i] == NULL)
            {
                printf("vxCreateTensorView failure! at line %d\n", __LINE__);
                goto error;
            }

            output_view[i] = vxCreateTensorView(context, output_start, output_end, num_of_dim);
            if (output_view[i] == NULL)
            {
                printf("vxCreateTensorView failure! at line %d\n", __LINE__);
                goto error;
            }

            weight_bias[i] = vxcLoadWeightsBiasesParameter(context, fp, weight_bias_blob_offset[i], weight_bias_blob_size[i]);
            if (weight_bias[i] == NULL)
            {
                printf("vxcLoadWeightsBiasesParameter failure! at line %d\n", __LINE__);
                goto error;
            }

            inputs[i] = vxCreateTensorFromView(input, input_view[i]);
            if (inputs[i] == NULL)
            {
                printf("vxCreateTensorFromView failure! at line %d\n", __LINE__);
                goto error;
            }

            outputs[i] = vxCreateTensorFromView(output, output_view[i]);
            if (outputs[i] == NULL)
            {
                printf("vxCreateTensorFromView failure! at line %d\n", __LINE__);
                goto error;
            }
        }
    }
    else
    {
        inputs[0] = input;
        outputs[0] = output;

        weight_bias[0] = vxcLoadWeightsBiasesParameter(context, fp, weight_bias_blob_offset[0], weight_bias_blob_size[0]);
        if (weight_bias[0] == NULL)
        {
            printf("vxcLoadWeightsBiasesParameter failure! at line %d\n", __LINE__);
            goto error;
        }
    }

    convolution_params.dilation_x = dilation_x;
    convolution_params.dilation_y = dilation_y;
    convolution_params.pad_x_left = padding_x;
    convolution_params.pad_x_right = (vx_int32)padding_x_right < 0 ? padding_x : padding_x_right;
    convolution_params.pad_y_top = padding_y;
    convolution_params.pad_y_bottom = (vx_int32)padding_y_bottom < 0 ? padding_y : padding_y_bottom;
    convolution_params.accumulator_bits = 0;
    convolution_params.overflow_policy = overflow_policy;
    convolution_params.rounding_policy = rounding_policy;
    convolution_params.down_scale_size_rounding = down_scale_size_rounding;
    convolution_params.enable_relu = has_relu;
    convolution_params.pool_type = pool_type;
    convolution_params.pool_size_x = pool_size_x;
    convolution_params.pool_size_y = pool_size_y;
    convolution_params.pad_mode = VX_PAD_CONSTANT;
    convolution_params.pad_const = pad_const;

    cinfo->pad_const = pad_const;
    cinfo->output = output;

    for (i = 0; i < (int)group; i++)
    {
        node[i] = vxConvolutionReluPoolingLayer2(graph,
                inputs[i],
                weight_bias[i],
                &convolution_params,
                sizeof(convolution_params),
                outputs[i]);

        if (node[i] == NULL)
        {
            printf("vxConvolutionReluPoolingLayer failure! at line %d\n", __LINE__);
            goto error;
        }

        cinfo->inputs[i] = inputs[i];
        cinfo->outputs[i] = outputs[i];
        cinfo->weight_bias[i] = weight_bias[i];
        cinfo->node[i] = node[i];
    }

    if (group > 1)
    {
        for (i = 0; i < (int)group; i++)
        {
            vxReleaseTensorView(&input_view[i]);
            vxReleaseTensorView(&output_view[i]);
        }
    }

    return VX_SUCCESS;

error:
    if (output)
        vxReleaseTensor(&output);

    if (pad_const)
        vxReleaseScalar(&pad_const);

    if (group > 1)
    {
        for (i = 0; i < (int)group; i++)
        {
            if (node[i])
                vxReleaseNode(&node[i]);

            if (inputs[i] != NULL)
                vxReleaseTensor(&inputs[i]);

            if (outputs[i] != NULL)
                vxReleaseTensor(&outputs[i]);

            if (weight_bias[i])
                vxReleaseWeightsBiasesParameter(&weight_bias[i]);

            if (input_view[i] != NULL)
                vxReleaseTensorView(&input_view[i]);

            if (output_view[i] != NULL)
                vxReleaseTensorView(&output_view[i]);
        }
    }
    else
    {
        if (weight_bias[0] != NULL)
            vxReleaseWeightsBiasesParameter(&weight_bias[0]);
    }
    return VX_FAILURE;
}

/* Release a Convolution layer */
void vxcReleaseConvolutionReluPoolingLayer(void** node_info)
{
    convolution_relu_pooling_info_t    *cinfo = (convolution_relu_pooling_info_t *)*node_info;
    int                                i;

    if (cinfo == NULL)
        return;

    if (cinfo->output != NULL)
    {
        vxReleaseTensor(&cinfo->output);
    }

    if (cinfo->pad_const != NULL)
    {
        vxReleaseScalar(&cinfo->pad_const);
    }

    if (cinfo->group > 1)
    {
        for (i = 0; i < (int)cinfo->group; i++)
        {
            if (cinfo->node[i] != NULL)
                vxReleaseNode(&cinfo->node[i]);

            if (cinfo->inputs[i] != NULL)
                vxReleaseTensor(&cinfo->inputs[i]);

            if (cinfo->outputs[i] != NULL)
                vxReleaseTensor(&cinfo->outputs[i]);

            if (cinfo->weight_bias[i])
                vxReleaseWeightsBiasesParameter(&cinfo->weight_bias[i]);
        }
    }
    else
    {
        if (cinfo->node[0] != NULL)
            vxReleaseNode(&cinfo->node[0]);

        if (cinfo->weight_bias[0])
            vxReleaseWeightsBiasesParameter(&cinfo->weight_bias[0]);
    }

    vxcFreeNodeInfo(node_info);
}

/* Create and initialize a Pooling layer node */
vx_status vxcCreatePoolingLayer(void* node_info)
{
    pooling_info_t              *pinfo = (pooling_info_t *)node_info;
    tensor_split_info_t         *tensor_split_info = &pinfo->tensor_split_info;
    vx_graph                    graph = pinfo->graph;
    vx_tensor                   input = pinfo->input;
    vx_enum                     pool_type = pinfo->pool_type;
    vx_uint32                   pool_size_x = pinfo->pool_size_x;
    vx_uint32                   pool_size_y = pinfo->pool_size_y;
    vx_uint32                   stride_x = pinfo->stride_x;
    vx_uint32                   stride_y = pinfo->stride_y;
    vx_uint32                   padding_x = pinfo->padding_x;
    vx_uint32                   padding_x_right = pinfo->padding_x_right;
    vx_uint32                   padding_y = pinfo->padding_y;
    vx_uint32                   padding_y_bottom = pinfo->padding_y_bottom;
    vx_enum                     downscale_size_rounding = pinfo->downscale_size_rounding;
    vx_nn_pooling_params_t      pooling_params;
    tensor_format_info_t        *output_format = &pinfo->output_format;
    vx_bool                     output_layer = pinfo->output_layer;

    vx_context                  context;
    vx_uint32                   num_of_dim;
    vx_uint32                   size[NN_TENSOR_MAX_DIMENSION] = {0, 0, 0, 0};
    vx_tensor                   output;
    vx_node                     node;
    vx_status                   status;

    context = vxGetContext((vx_reference)graph);
    if (context == NULL)
    {
        printf("vxGetContext failure! at line %d\n", __LINE__);
        goto error;
    }

    status = vxQueryTensor(input, VX_TENSOR_NUM_OF_DIMS, &num_of_dim, sizeof(num_of_dim));
    if (status != VX_SUCCESS)
    {
        printf("vxQueryTensor failure! at line %d\n", __LINE__);
        goto error;
    }

    status = vxQueryTensor(input, VX_TENSOR_DIMS, size, sizeof(size));
    if (status != VX_SUCCESS)
    {
        printf("vxQueryTensor failure! at line %d\n", __LINE__);
        goto error;
    }

    size[0] = NN_DIV_DS_SIZE_ROUNDING((size[0] + padding_x + padding_x_right - pool_size_x), stride_x, downscale_size_rounding) + 1;
    size[1] = NN_DIV_DS_SIZE_ROUNDING((size[1] + padding_y + padding_y_bottom - pool_size_y), stride_y, downscale_size_rounding) + 1;

    output = vxcCreateOutputTensor(context, graph, num_of_dim, size, output_format,
                                    tensor_split_info, output_layer);

    if (output == NULL)
    {
        printf("vxcCreateOutputTensor failure! at line %d\n", __LINE__);
        goto error;
    }

    pooling_params.pool_type = pool_type;
    pooling_params.pool_size_x = pool_size_x;
    pooling_params.pool_size_y = pool_size_y;
    pooling_params.pool_pad_x_left = padding_x;
    pooling_params.pool_pad_x_right = padding_x_right;
    pooling_params.pool_pad_y_top = padding_y;
    pooling_params.pool_pad_y_bottom = padding_y_bottom;
    pooling_params.rounding = downscale_size_rounding;

    node = vxPoolingLayer2(graph, input, (const vx_nn_pooling_params_t*)&pooling_params, sizeof(pooling_params), output);
    if (node == NULL)
    {
        printf("vxPoolingLayer failure! at line %d\n", __LINE__);
        goto error;
    }

    pinfo->output = output;
    pinfo->node = node;

    return VX_SUCCESS;

error:
    if (output)
        vxReleaseTensor(&output);

    return VX_FAILURE;
}

/* Release a Pooling layer node */
void vxcReleasePoolingLayer(void** node_info)
{
    pooling_info_t           *pinfo = (pooling_info_t *)*node_info;

    if (pinfo == NULL)
        return;

    if (pinfo->node)
        vxReleaseNode(&pinfo->node);

    if (pinfo->output)
        vxReleaseTensor(&pinfo->output);

    vxcFreeNodeInfo(node_info);
}

/* Create and initialize a ConcatN node */
vx_status vxcCreateConcatNNode(void* node_info)
{
    tensor_concat_info_t        *tinfo = (tensor_concat_info_t *)node_info;
    tensor_split_info_t         *tensor_split_info = &tinfo->tensor_split_info;
    vx_tensor                   *input = tinfo->input;
    vx_graph                    graph = tinfo->graph;

    vx_context                  context;
    vx_tensor                   output;

    context = vxGetContext((vx_reference)graph);
    if (context == NULL)
    {
        printf("vxGetContext failure! at line %d\n", __LINE__);
        goto error;
    }

    if (tensor_split_info->tensor)
    {
        output = vxcCreateSubtensor(context, tensor_split_info);
    }
    else
    {
        output = input[0];
        vxRetainReference((vx_reference)output);
    }

    if (output == NULL)
    {
        printf("Input tensor is a NULL pointer! at line %d\n", __LINE__);
        goto error;
    }

    tinfo->output = output;
    tinfo->node = NULL;

    return VX_SUCCESS;

error:
    if (input)
        free(input);

    return VX_FAILURE;
}

/* Release a ConcatN node */
void vxcReleaseConcatNNode(void** node_info)
{
    tensor_concat_info_t    *tinfo = (tensor_concat_info_t *)*node_info;

    if (tinfo == NULL)
        return;

    if (tinfo->node)
        vxReleaseNode(&tinfo->node);

    if (tinfo->output)
        vxReleaseTensor(&tinfo->output);

    if (tinfo->input)
        free(tinfo->input);

    vxcFreeNodeInfo(node_info);
}

/* Create and initialize a Permute node */
vx_status vxcCreatePermuteNode(void* node_info)
{
    tensor_permute_info_t       *tinfo = (tensor_permute_info_t *)node_info;
    tensor_split_info_t         *tensor_split_info = &tinfo->tensor_split_info;
    vx_graph                    graph = tinfo->graph;
    vx_tensor                   input = tinfo->input;
    vx_uint32                   *perm = tinfo->perm;
    tensor_format_info_t        *output_format = &tinfo->output_format;
    vx_bool                     output_layer = tinfo->output_layer;

    vx_context                  context;
    vx_int32                    num_of_dim;
    vx_uint32                   size[NN_TENSOR_MAX_DIMENSION] = {0, 0, 0, 0};
    vx_uint32                   output_size[NN_TENSOR_MAX_DIMENSION] = {0, 0, 0, 0};
    vx_tensor                   output;
    vx_node                     node;
    vx_status                   status;
    int                         i;

    context = vxGetContext((vx_reference)graph);
    if (context == NULL)
    {
        printf("vxGetContext failure! at line %d\n", __LINE__);
        goto error;
    }

    status = vxQueryTensor(input, VX_TENSOR_NUM_OF_DIMS, &num_of_dim, sizeof(num_of_dim));
    if (status != VX_SUCCESS)
    {
        printf("vxQueryTensor failure! at line %d\n", __LINE__);
        goto error;
    }

    status = vxQueryTensor(input, VX_TENSOR_DIMS, size, sizeof(size));
    if (status != VX_SUCCESS)
    {
        printf("vxQueryTensor failure! at line %d\n", __LINE__);
        goto error;
    }

    for (i = 0; i < num_of_dim; i++)
    {
        output_size[i] = size[perm[i]];
    }

    output = vxcCreateOutputTensor(context, graph, num_of_dim, output_size, output_format,
                                tensor_split_info, output_layer);

    if (output == NULL)
    {
        printf("vxcCreateOutputTensor failure! at line %d\n", __LINE__);
        goto error;
    }

    node = vxTensorPermuteNode(graph, input, output, perm, num_of_dim);
    if (node == NULL)
    {
        printf("vxTensorPermuteNode failure! at line %d\n", __LINE__);
        goto error;
    }

    tinfo->output = output;
    tinfo->node = node;

    return VX_SUCCESS;

error:
    if (output)
        vxReleaseTensor(&output);

    return VX_FAILURE;
}

/* Release a Permute node */
void vxcReleasePermuteNode(void** node_info)
{
    tensor_permute_info_t       *tinfo = (tensor_permute_info_t *)*node_info;

    if (tinfo == NULL)
        return;

    if (tinfo->node)
        vxReleaseNode(&tinfo->node);

    if (tinfo->output)
        vxReleaseTensor(&tinfo->output);

    vxcFreeNodeInfo(node_info);
}

/* Create and initialize a Reshape node */
vx_status vxcCreateReshapeNode(void* node_info)
{
    tensor_reshape_info_t       *tinfo = (tensor_reshape_info_t *)node_info;
    tensor_split_info_t         *tensor_split_info = &tinfo->tensor_split_info;
    vx_graph                    graph = tinfo->graph;
    vx_tensor                   input = tinfo->input;
    vx_int32                    *shape = tinfo->shape;

    vx_context                  context;
    vx_uint32                   num_of_dim;
    vx_uint32                   size[NN_TENSOR_MAX_DIMENSION] = {0, 0, 0, 0};
    vx_tensor                   output = NULL;
    vx_node                     node = NULL;
    vx_status                   status;

    context = vxGetContext((vx_reference)graph);
    if (context == NULL)
    {
        printf("vxGetContext failure! at line %d\n", __LINE__);
        goto error;
    }

    status = vxQueryTensor(input, VX_TENSOR_NUM_OF_DIMS, &num_of_dim, sizeof(num_of_dim));
    if (status != VX_SUCCESS)
    {
        printf("vxQueryTensor failure! at line %d\n", __LINE__);
        goto error;
    }

    status = vxQueryTensor(input, VX_TENSOR_DIMS, size, sizeof(size));
    if (status != VX_SUCCESS)
    {
        printf("vxQueryTensor failure! at line %d\n", __LINE__);
        goto error;
    }

    if (tensor_split_info->tensor)
    {
        output = vxcCreateSubtensor(context, tensor_split_info);
        if (output == NULL)
        {
            printf("vxcCreateSubtensor failure! at line %d\n", __LINE__);
            goto error;
        }

        node = vxTensorCopyNode(graph, input, output);
        if (node == NULL)
        {
            printf("vxTensorCopyNode failure! at line %d\n", __LINE__);
            goto error;
        }
    }
    else
    {
        output = vxReshapeTensor(input, shape, tinfo->num_of_dim);
        if (output == NULL)
        {
            printf("vxReshapeTensor failure! at line %d\n", __LINE__);
            goto error;
        }
    }

    tinfo->output = output;
    tinfo->node = node;

    return VX_SUCCESS;

error:
    if (node)
        vxReleaseNode(&node);

    if (output)
        vxReleaseTensor(&output);

    return VX_FAILURE;
}

/* Release a Reshape node */
void vxcReleaseReshapeNode(void** node_info)
{
    tensor_reshape_info_t       *tinfo = (tensor_reshape_info_t *)*node_info;

    if (tinfo == NULL)
        return;

    if (tinfo->node)
        vxReleaseNode(&tinfo->node);

    if (tinfo->output)
        vxReleaseTensor(&tinfo->output);

    vxcFreeNodeInfo(node_info);
}

/* Create and initialize a Softmax node */
vx_status vxcCreateSoftmaxLayer(void* node_info)
{
    softmax_info_t              *sinfo = (softmax_info_t *)node_info;
    tensor_split_info_t         *tensor_split_info = &sinfo->tensor_split_info;
    vx_graph                    graph = sinfo->graph;
    vx_tensor                   input = sinfo->input;
    tensor_format_info_t        *output_format = &sinfo->output_format;
    vx_bool                     output_layer = sinfo->output_layer;

    vx_context                  context;
    vx_uint32                   num_of_dim;
    vx_uint32                   size[NN_TENSOR_MAX_DIMENSION] = {0, 0, 0, 0};
    vx_tensor                   output = NULL;
    vx_node                     node;
    vx_status                   status;

    context = vxGetContext((vx_reference)graph);
    if (context == NULL)
    {
        printf("vxGetContext failure! at line %d\n", __LINE__);
        goto error;
    }

    status = vxQueryTensor(input, VX_TENSOR_NUM_OF_DIMS, &num_of_dim, sizeof(num_of_dim));
    if (status != VX_SUCCESS)
    {
        printf("vxQueryTensor failure! at line %d\n", __LINE__);
        goto error;
    }

    status = vxQueryTensor(input, VX_TENSOR_DIMS, size, sizeof(size));
    if (status != VX_SUCCESS)
    {
        printf("vxQueryTensor failure! at line %d\n", __LINE__);
        goto error;
    }

    output = vxcCreateOutputTensor(context, graph, num_of_dim, size, output_format,
            tensor_split_info, output_layer);

    if (output == NULL)
    {
        printf("vxcCreateOutputTensor failure! at line %d\n", __LINE__);
        goto error;
    }

    node = vxSoftmaxLayer(graph, input, output);
    if (node == NULL)
    {
        printf("vxSoftmaxLayer failure! at line %d\n", __LINE__);
        goto error;
    }

    sinfo->output = output;
    sinfo->node = node;

    return VX_SUCCESS;

error:
    if (output)
        vxReleaseTensor(&output);

    return VX_FAILURE;
}

/* Release a Softmax layer node */
void vxcReleaseSoftmaxLayer(void** node_info)
{
    softmax_info_t           *sinfo = (softmax_info_t *)*node_info;

    if (sinfo == NULL)
        return;

    if (sinfo->node)
        vxReleaseNode(&sinfo->node);

    if (sinfo->output)
        vxReleaseTensor(&sinfo->output);

    vxcFreeNodeInfo(node_info);
}
