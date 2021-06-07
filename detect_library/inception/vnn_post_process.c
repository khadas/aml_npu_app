/****************************************************************************
*   Generated by ACUITY 5.16.3_0205
*   Match ovxlib 1.1.26
*
*   Neural Network appliction post-process source file
****************************************************************************/
/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vsi_nn_pub.h"

#include "vnn_global.h"
#include "vnn_post_process.h"

#define _BASETSD_H

/*-------------------------------------------
                  Variable definitions
-------------------------------------------*/

/*{graph_output_idx, postprocess}*/
const static vsi_nn_postprocess_map_element_t* postprocess_map = NULL;


/*-------------------------------------------
                  Functions
-------------------------------------------*/
static void save_output_data(vsi_nn_graph_t *graph)
{
    uint32_t i;
#define _DUMP_FILE_LENGTH 1028
#define _DUMP_SHAPE_LENGTH 128
    char filename[_DUMP_FILE_LENGTH] = {0}, shape[_DUMP_SHAPE_LENGTH] = {0};
    vsi_nn_tensor_t *tensor;

    for(i = 0; i < graph->output.num; i++)
    {
        tensor = vsi_nn_GetTensor(graph, graph->output.tensors[i]);
        vsi_nn_ShapeToString( tensor->attr.size, tensor->attr.dim_num,
            shape, _DUMP_SHAPE_LENGTH, FALSE );
        snprintf(filename, _DUMP_FILE_LENGTH, "output%u_%s.dat", i, shape);
        vsi_nn_SaveTensorToBinary(graph, tensor, filename);

    }
}

static vsi_bool get_top
    (
    float *pfProb,
    float *pfMaxProb,
    uint32_t *pMaxClass,
    uint32_t outputCount,
    uint32_t topNum
    )
{
    uint32_t i, j, k;

    #define MAX_TOP_NUM 20
    if (topNum > MAX_TOP_NUM) return FALSE;

    memset(pfMaxProb, 0xfe, sizeof(float) * topNum);
    memset(pMaxClass, 0xff, sizeof(float) * topNum);

    for (j = 0; j < topNum; j++)
    {
        for (i=0; i<outputCount; i++)
        {
            for (k=0; k < topNum; k ++)
            {
                if(i == pMaxClass[k])
                    break;
            }

            if (k != topNum)
                continue;

            if (pfProb[i] > *(pfMaxProb+j))
            {
                *(pfMaxProb+j) = pfProb[i];
                *(pMaxClass+j) = i;
            }
        }
    }

    return TRUE;
}

static vsi_status show_top5
    (
    vsi_nn_graph_t *graph,
    vsi_nn_tensor_t *tensor
    )
{
    vsi_status status = VSI_FAILURE;
    uint32_t i,sz,stride;
    float *buffer = NULL;
    uint8_t *tensor_data = NULL;
    uint32_t MaxClass[5];
    float fMaxProb[5];
    uint32_t topk = 5;

    sz = 1;
    for(i = 0; i < tensor->attr.dim_num; i++)
    {
        sz *= tensor->attr.size[i];
    }

    if(topk > sz)
        topk = sz;

    stride = vsi_nn_TypeGetBytes(tensor->attr.dtype.vx_type);
    tensor_data = (uint8_t *)vsi_nn_ConvertTensorToData(graph, tensor);
    buffer = (float *)malloc(sizeof(float) * sz);

    for(i = 0; i < sz; i++)
    {
        status = vsi_nn_DtypeToFloat32(&tensor_data[stride * i], &buffer[i], &tensor->attr.dtype);
    }

    if (!get_top(buffer, fMaxProb, MaxClass, sz, topk))
    {
        printf("Fail to show result.\n");
        goto final;
    }

    printf(" --- Top%d ---\n", topk);
    for(i = 0; i< topk; i++)
    {
        printf("%3d: %8.6f\n", MaxClass[i], fMaxProb[i]);
    }
    status = VSI_SUCCESS;

final:
    if(tensor_data)vsi_nn_Free(tensor_data);
    if(buffer)free(buffer);
    return status;
}

vsi_status vnn_PostProcessInceptionV3(vsi_nn_graph_t *graph)
{
    vsi_status status = VSI_FAILURE;

    /* Show the top5 result */
    status = show_top5(graph, vsi_nn_GetTensor(graph, graph->output.tensors[0]));
    TEST_CHECK_STATUS(status, final);

    /* Save all output tensor data to txt file */
    save_output_data(graph);

final:
    return VSI_SUCCESS;
}

const vsi_nn_postprocess_map_element_t * vnn_GetPostPorcessMap()
{
    return postprocess_map;
}

uint32_t vnn_GetPostPorcessMapCount()
{
    if (postprocess_map == NULL)
       return 0;
    else
        return sizeof(postprocess_map) / sizeof(vsi_nn_postprocess_map_element_t);
}