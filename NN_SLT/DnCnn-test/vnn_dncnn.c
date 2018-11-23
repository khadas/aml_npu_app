/****************************************************************************
*   Generated by ACUITY 3.8.2
*   Match ovxlib 1.0.3
*
*   Neural Network appliction network definition source file
****************************************************************************/
/*-------------------------------------------
                   Includes
 -------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>

#include "vsi_nn_pub.h"

#include "vnn_global.h"
#include "vnn_dncnn.h"

/*-------------------------------------------
                   Macros
 -------------------------------------------*/

#define NEW_VXNODE(_node, _type, _uid) do {\
        _node = vsi_nn_AppendNode( graph, _type, NULL );\
        _node->uid = (uint32_t)_uid; \
        if ( NULL == _node ) {\
            goto error;\
        }\
    } while(0)

#define NEW_VIRTUAL_TENSOR(_id, _attr, _dtype) do {\
        memset( _attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));\
        _attr.dim_num = VSI_NN_DIM_AUTO;\
        _attr.vtl = !VNN_APP_DEBUG;\
        _attr.is_const = FALSE;\
        _attr.dtype.vx_type = _dtype;\
        _id = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO,\
                & _attr, NULL );\
        if ( VSI_NN_TENSOR_ID_NA == _id ) {\
            goto error;\
        }\
    } while(0)

// Set const tensor dims out of this macro.
#define NEW_CONST_TENSOR(_id, _attr, _dtype, _ofst, _size) do {\
        data = load_data( fp, _ofst, _size  );\
        _attr.vtl = FALSE;\
        _attr.is_const = TRUE;\
        _attr.dtype.vx_type = _dtype;\
        _id = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO,\
                & _attr, data );\
        free( data );\
        if ( VSI_NN_TENSOR_ID_NA == _id ) {\
            goto error;\
        }\
    } while(0)

// Set generic tensor dims out of this macro.
#define NEW_NORM_TENSOR(_id, _attr, _dtype) do {\
        _attr.vtl = FALSE;\
        _attr.is_const = FALSE;\
        _attr.dtype.vx_type = _dtype;\
        _id = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO,\
                & _attr, NULL );\
        if ( VSI_NN_TENSOR_ID_NA == _id ) {\
            goto error;\
        }\
    } while(0)

#define NET_NODE_NUM            (6)
#define NET_NORM_TENSOR_NUM     (2)
#define NET_CONST_TENSOR_NUM    (12)
#define NET_VIRTUAL_TENSOR_NUM  (6)
#define NET_TOTAL_TENSOR_NUM    (NET_NORM_TENSOR_NUM + NET_CONST_TENSOR_NUM + NET_VIRTUAL_TENSOR_NUM)

/*-------------------------------------------
               Local Variables
 -------------------------------------------*/

/*-------------------------------------------
                  Functions
 -------------------------------------------*/

static uint8_t* load_data
    (
    FILE  * fp,
    size_t  ofst,
    size_t  sz
    )
{
    uint8_t* data;
    int32_t ret;
    data = NULL;
    if ( NULL == fp )
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        VSILOGE("blob seek failure.");
        return NULL;
    }

    data = (uint8_t*)malloc(sz);
    if (data == NULL)
    {
        VSILOGE("buffer malloc failure.");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    VSILOGI("Read %d data.", ret);
    return data;
} /* load_data() */

vsi_nn_graph_t * vnn_CreateDncnn
    (
    const char * data_file_name,
    vsi_nn_context_t in_ctx
    )
{
    vsi_status              status;
    vsi_bool                release_ctx;
    vsi_nn_context_t        ctx;
    vsi_nn_graph_t *        graph;
    vsi_nn_node_t *         node[NET_NODE_NUM];
    vsi_nn_tensor_id_t      norm_tensor[NET_NORM_TENSOR_NUM];
    vsi_nn_tensor_id_t      const_tensor[NET_CONST_TENSOR_NUM];
    vsi_nn_tensor_attr_t    attr;
    FILE *                  fp;
    uint8_t *               data;



    ctx = NULL;
    graph = NULL;
    status = VSI_FAILURE;

    fp = fopen( data_file_name, "rb" );
    if ( NULL == fp )
    {
        VSILOGE( "Open file %s failed.", data_file_name );
        goto error;
    }

    if ( NULL == in_ctx )
    {
        ctx = vsi_nn_CreateContext();
    }
    else
    {
        ctx = in_ctx;
    }

    graph = vsi_nn_CreateGraph( ctx, NET_TOTAL_TENSOR_NUM, NET_NODE_NUM );
    if ( NULL == graph )
    {
        VSILOGE( "Create graph fail." );
        goto error;
    }
    vsi_nn_SetGraphInputs( graph, NULL, 1 );
    vsi_nn_SetGraphOutputs( graph, NULL, 1 );

/*-----------------------------------------
  Register client ops
 -----------------------------------------*/


/*-----------------------------------------
  Node definitions
 -----------------------------------------*/

    /*-----------------------------------------
      lid       - trans_con_2
      var       - node[0]
      name      - convolutionrelu
      operation - convolutionrelu
      in_shape  - [[640, 640, 3]]
      out_shape - [[640, 640, 64]]
    -----------------------------------------*/
    NEW_VXNODE(node[0], VSI_NN_OP_CONV_RELU, 2);
    node[0]->nn_param.conv2d.ksize[0] = 3;
    node[0]->nn_param.conv2d.ksize[1] = 3;
    node[0]->nn_param.conv2d.weights = 64;
    node[0]->nn_param.conv2d.stride[0] = 1;
    node[0]->nn_param.conv2d.stride[1] = 1;
    node[0]->nn_param.conv2d.pad[0] = 1;
    node[0]->nn_param.conv2d.pad[1] = 1;
    node[0]->nn_param.conv2d.pad[2] = 1;
    node[0]->nn_param.conv2d.pad[3] = 1;
    node[0]->nn_param.conv2d.group = 1;
    node[0]->nn_param.conv2d.dilation[0] = 1;
    node[0]->nn_param.conv2d.dilation[1] = 1;
    node[0]->vx_param.has_relu = FALSE;
    node[0]->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
    node[0]->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
    node[0]->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

    /*-----------------------------------------
      lid       - trans_con_3
      var       - node[1]
      name      - convolutionrelu
      operation - convolutionrelu
      in_shape  - [[640, 640, 64]]
      out_shape - [[640, 640, 64]]
    -----------------------------------------*/
    NEW_VXNODE(node[1], VSI_NN_OP_CONV_RELU, 3);
    node[1]->nn_param.conv2d.ksize[0] = 3;
    node[1]->nn_param.conv2d.ksize[1] = 3;
    node[1]->nn_param.conv2d.weights = 64;
    node[1]->nn_param.conv2d.stride[0] = 1;
    node[1]->nn_param.conv2d.stride[1] = 1;
    node[1]->nn_param.conv2d.pad[0] = 1;
    node[1]->nn_param.conv2d.pad[1] = 1;
    node[1]->nn_param.conv2d.pad[2] = 1;
    node[1]->nn_param.conv2d.pad[3] = 1;
    node[1]->nn_param.conv2d.group = 1;
    node[1]->nn_param.conv2d.dilation[0] = 1;
    node[1]->nn_param.conv2d.dilation[1] = 1;
    node[1]->vx_param.has_relu = FALSE;
    node[1]->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
    node[1]->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
    node[1]->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

    /*-----------------------------------------
      lid       - trans_con_4
      var       - node[2]
      name      - convolutionrelu
      operation - convolutionrelu
      in_shape  - [[640, 640, 64]]
      out_shape - [[640, 640, 64]]
    -----------------------------------------*/
    NEW_VXNODE(node[2], VSI_NN_OP_CONV_RELU, 4);
    node[2]->nn_param.conv2d.ksize[0] = 3;
    node[2]->nn_param.conv2d.ksize[1] = 3;
    node[2]->nn_param.conv2d.weights = 64;
    node[2]->nn_param.conv2d.stride[0] = 1;
    node[2]->nn_param.conv2d.stride[1] = 1;
    node[2]->nn_param.conv2d.pad[0] = 1;
    node[2]->nn_param.conv2d.pad[1] = 1;
    node[2]->nn_param.conv2d.pad[2] = 1;
    node[2]->nn_param.conv2d.pad[3] = 1;
    node[2]->nn_param.conv2d.group = 1;
    node[2]->nn_param.conv2d.dilation[0] = 1;
    node[2]->nn_param.conv2d.dilation[1] = 1;
    node[2]->vx_param.has_relu = FALSE;
    node[2]->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
    node[2]->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
    node[2]->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

    /*-----------------------------------------
      lid       - trans_con_5
      var       - node[3]
      name      - convolutionrelu
      operation - convolutionrelu
      in_shape  - [[640, 640, 64]]
      out_shape - [[640, 640, 64]]
    -----------------------------------------*/
    NEW_VXNODE(node[3], VSI_NN_OP_CONV_RELU, 5);
    node[3]->nn_param.conv2d.ksize[0] = 3;
    node[3]->nn_param.conv2d.ksize[1] = 3;
    node[3]->nn_param.conv2d.weights = 64;
    node[3]->nn_param.conv2d.stride[0] = 1;
    node[3]->nn_param.conv2d.stride[1] = 1;
    node[3]->nn_param.conv2d.pad[0] = 1;
    node[3]->nn_param.conv2d.pad[1] = 1;
    node[3]->nn_param.conv2d.pad[2] = 1;
    node[3]->nn_param.conv2d.pad[3] = 1;
    node[3]->nn_param.conv2d.group = 1;
    node[3]->nn_param.conv2d.dilation[0] = 1;
    node[3]->nn_param.conv2d.dilation[1] = 1;
    node[3]->vx_param.has_relu = FALSE;
    node[3]->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
    node[3]->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
    node[3]->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

    /*-----------------------------------------
      lid       - trans_con_6
      var       - node[4]
      name      - convolutionrelu
      operation - convolutionrelu
      in_shape  - [[640, 640, 64]]
      out_shape - [[640, 640, 64]]
    -----------------------------------------*/
    NEW_VXNODE(node[4], VSI_NN_OP_CONV_RELU, 6);
    node[4]->nn_param.conv2d.ksize[0] = 3;
    node[4]->nn_param.conv2d.ksize[1] = 3;
    node[4]->nn_param.conv2d.weights = 64;
    node[4]->nn_param.conv2d.stride[0] = 1;
    node[4]->nn_param.conv2d.stride[1] = 1;
    node[4]->nn_param.conv2d.pad[0] = 1;
    node[4]->nn_param.conv2d.pad[1] = 1;
    node[4]->nn_param.conv2d.pad[2] = 1;
    node[4]->nn_param.conv2d.pad[3] = 1;
    node[4]->nn_param.conv2d.group = 1;
    node[4]->nn_param.conv2d.dilation[0] = 1;
    node[4]->nn_param.conv2d.dilation[1] = 1;
    node[4]->vx_param.has_relu = FALSE;
    node[4]->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
    node[4]->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
    node[4]->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;

    /*-----------------------------------------
      lid       - trans_con_7
      var       - node[5]
      name      - convolutionrelu
      operation - convolutionrelu
      in_shape  - [[640, 640, 64]]
      out_shape - [[640, 640, 3]]
    -----------------------------------------*/
    NEW_VXNODE(node[5], VSI_NN_OP_CONV_RELU, 7);
    node[5]->nn_param.conv2d.ksize[0] = 3;
    node[5]->nn_param.conv2d.ksize[1] = 3;
    node[5]->nn_param.conv2d.weights = 3;
    node[5]->nn_param.conv2d.stride[0] = 1;
    node[5]->nn_param.conv2d.stride[1] = 1;
    node[5]->nn_param.conv2d.pad[0] = 1;
    node[5]->nn_param.conv2d.pad[1] = 1;
    node[5]->nn_param.conv2d.pad[2] = 1;
    node[5]->nn_param.conv2d.pad[3] = 1;
    node[5]->nn_param.conv2d.group = 1;
    node[5]->nn_param.conv2d.dilation[0] = 1;
    node[5]->nn_param.conv2d.dilation[1] = 1;
    node[5]->vx_param.has_relu = FALSE;
    node[5]->vx_param.overflow_policy = VX_CONVERT_POLICY_WRAP;
    node[5]->vx_param.rounding_policy = VX_ROUND_POLICY_TO_ZERO;
    node[5]->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR;


/*-----------------------------------------
  Tensor initialize
 -----------------------------------------*/
    attr.dtype.fmt = VSI_NN_DIM_FMT_NCHW;
    /* @inp_1:out0 */
    attr.size[0] = 640;
    attr.size[1] = 640;
    attr.size[2] = 3;
    attr.dim_num = 3;
    attr.dtype.fl = 7;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
    NEW_NORM_TENSOR(norm_tensor[0], attr, VSI_NN_TYPE_INT8);

    /* @out_8:out0 */
    attr.size[0] = 640;
    attr.size[1] = 640;
    attr.size[2] = 3;
    attr.dim_num = 3;
    attr.dtype.fl = 7;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
    NEW_NORM_TENSOR(norm_tensor[1], attr, VSI_NN_TYPE_INT8);



    /* @trans_con_2:weight */
    attr.size[0] = 3;
    attr.size[1] = 3;
    attr.size[2] = 3;
    attr.size[3] = 64;
    attr.dim_num = 4;
    attr.dtype.fl = 10;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
    NEW_CONST_TENSOR(const_tensor[0], attr, VSI_NN_TYPE_INT8, 256, 1728);

    /* @trans_con_2:bias */
    attr.size[0] = 64;
    attr.dim_num = 1;
    attr.dtype.fl = 17;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
    NEW_CONST_TENSOR(const_tensor[1], attr, VSI_NN_TYPE_INT32, 0, 256);

    /* @trans_con_3:weight */
    attr.size[0] = 3;
    attr.size[1] = 3;
    attr.size[2] = 64;
    attr.size[3] = 64;
    attr.dim_num = 4;
    attr.dtype.fl = 10;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
    NEW_CONST_TENSOR(const_tensor[2], attr, VSI_NN_TYPE_INT8, 2240, 36864);

    /* @trans_con_3:bias */
    attr.size[0] = 64;
    attr.dim_num = 1;
    attr.dtype.fl = 17;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
    NEW_CONST_TENSOR(const_tensor[3], attr, VSI_NN_TYPE_INT32, 1984, 256);

    /* @trans_con_4:weight */
    attr.size[0] = 3;
    attr.size[1] = 3;
    attr.size[2] = 64;
    attr.size[3] = 64;
    attr.dim_num = 4;
    attr.dtype.fl = 10;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
    NEW_CONST_TENSOR(const_tensor[4], attr, VSI_NN_TYPE_INT8, 39360, 36864);

    /* @trans_con_4:bias */
    attr.size[0] = 64;
    attr.dim_num = 1;
    attr.dtype.fl = 17;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
    NEW_CONST_TENSOR(const_tensor[5], attr, VSI_NN_TYPE_INT32, 39104, 256);

    /* @trans_con_5:weight */
    attr.size[0] = 3;
    attr.size[1] = 3;
    attr.size[2] = 64;
    attr.size[3] = 64;
    attr.dim_num = 4;
    attr.dtype.fl = 10;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
    NEW_CONST_TENSOR(const_tensor[6], attr, VSI_NN_TYPE_INT8, 76480, 36864);

    /* @trans_con_5:bias */
    attr.size[0] = 64;
    attr.dim_num = 1;
    attr.dtype.fl = 17;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
    NEW_CONST_TENSOR(const_tensor[7], attr, VSI_NN_TYPE_INT32, 76224, 256);

    /* @trans_con_6:weight */
    attr.size[0] = 3;
    attr.size[1] = 3;
    attr.size[2] = 64;
    attr.size[3] = 64;
    attr.dim_num = 4;
    attr.dtype.fl = 10;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
    NEW_CONST_TENSOR(const_tensor[8], attr, VSI_NN_TYPE_INT8, 113600, 36864);

    /* @trans_con_6:bias */
    attr.size[0] = 64;
    attr.dim_num = 1;
    attr.dtype.fl = 17;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
    NEW_CONST_TENSOR(const_tensor[9], attr, VSI_NN_TYPE_INT32, 113344, 256);

    /* @trans_con_7:weight */
    attr.size[0] = 3;
    attr.size[1] = 3;
    attr.size[2] = 64;
    attr.size[3] = 3;
    attr.dim_num = 4;
    attr.dtype.fl = 10;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
    NEW_CONST_TENSOR(const_tensor[10], attr, VSI_NN_TYPE_INT8, 150476, 1728);

    /* @trans_con_7:bias */
    attr.size[0] = 3;
    attr.dim_num = 1;
    attr.dtype.fl = 17;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
    NEW_CONST_TENSOR(const_tensor[11], attr, VSI_NN_TYPE_INT32, 150464, 12);



    /* @trans_con_2:out0 */
    attr.dtype.fl = 7;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
    NEW_VIRTUAL_TENSOR(node[0]->output.tensors[0], attr, VSI_NN_TYPE_INT8);

    /* @trans_con_3:out0 */
    attr.dtype.fl = 7;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
    NEW_VIRTUAL_TENSOR(node[1]->output.tensors[0], attr, VSI_NN_TYPE_INT8);

    /* @trans_con_4:out0 */
    attr.dtype.fl = 7;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
    NEW_VIRTUAL_TENSOR(node[2]->output.tensors[0], attr, VSI_NN_TYPE_INT8);

    /* @trans_con_5:out0 */
    attr.dtype.fl = 7;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
    NEW_VIRTUAL_TENSOR(node[3]->output.tensors[0], attr, VSI_NN_TYPE_INT8);

    /* @trans_con_6:out0 */
    attr.dtype.fl = 7;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
    NEW_VIRTUAL_TENSOR(node[4]->output.tensors[0], attr, VSI_NN_TYPE_INT8);



/*-----------------------------------------
  Connection initialize
 -----------------------------------------*/
    node[0]->input.tensors[0] = norm_tensor[0];
    node[5]->output.tensors[0] = norm_tensor[1];

    /* trans_con_2 */
    node[0]->input.tensors[1] = const_tensor[0]; /* data_weight */
    node[0]->input.tensors[2] = const_tensor[1]; /* data_bias */

    /* trans_con_3 */
    node[1]->input.tensors[0] = node[0]->output.tensors[0];
    node[1]->input.tensors[1] = const_tensor[2]; /* data_weight */
    node[1]->input.tensors[2] = const_tensor[3]; /* data_bias */

    /* trans_con_4 */
    node[2]->input.tensors[0] = node[1]->output.tensors[0];
    node[2]->input.tensors[1] = const_tensor[4]; /* data_weight */
    node[2]->input.tensors[2] = const_tensor[5]; /* data_bias */

    /* trans_con_5 */
    node[3]->input.tensors[0] = node[2]->output.tensors[0];
    node[3]->input.tensors[1] = const_tensor[6]; /* data_weight */
    node[3]->input.tensors[2] = const_tensor[7]; /* data_bias */

    /* trans_con_6 */
    node[4]->input.tensors[0] = node[3]->output.tensors[0];
    node[4]->input.tensors[1] = const_tensor[8]; /* data_weight */
    node[4]->input.tensors[2] = const_tensor[9]; /* data_bias */

    /* trans_con_7 */
    node[5]->input.tensors[0] = node[4]->output.tensors[0];
    node[5]->input.tensors[1] = const_tensor[10]; /* data_weight */
    node[5]->input.tensors[2] = const_tensor[11]; /* data_bias */



    graph->input.tensors[0] = norm_tensor[0];
    graph->output.tensors[0] = norm_tensor[1];


    status = vsi_nn_SetupGraph( graph, FALSE );
    if ( VSI_FAILURE == status )
    {
        goto error;
    }

    fclose( fp );

    return graph;

error:
    if ( NULL != fp )
    {
        fclose( fp );
    }

    release_ctx = ( NULL == in_ctx );
    vnn_ReleaseDncnn( graph, release_ctx );

    return NULL;
} /* vsi_nn_CreateDncnn() */

void vnn_ReleaseDncnn
    (
    vsi_nn_graph_t * graph,
    vsi_bool release_ctx
    )
{
    vsi_nn_context_t ctx;
    if ( NULL != graph )
    {
        ctx = graph->ctx;
        vsi_nn_ReleaseGraph( &graph );

        /*-----------------------------------------
        Unregister client ops
        -----------------------------------------*/


        if ( release_ctx )
        {
            vsi_nn_ReleaseContext( &ctx );
        }
    }
} /* vsi_nn_ReleaseDncnn() */
