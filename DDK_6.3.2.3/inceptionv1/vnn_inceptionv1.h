/****************************************************************************
*   Generated by ACUITY 3.12.0
*   Match ovxlib 1.0.9
*
*   Neural Network appliction network definition header file
****************************************************************************/

#ifndef _VSI_NN_INCEPTIONV1_H
#define _VSI_NN_INCEPTIONV1_H

#include "vsi_nn_pub.h"

void vnn_ReleaseInceptionv1
    (
    vsi_nn_graph_t * graph,
    vsi_bool release_ctx
    );

vsi_nn_graph_t * vnn_CreateInceptionv1
    (
    const char * data_file_name,
    vsi_nn_context_t in_ctx
    );

#endif
