/****************************************************************************
*   Generated by ACUITY 5.0.0
*   Match ovxlib 1.1.1
*
*   Neural Network appliction network definition header file
****************************************************************************/

#ifndef _VNN_FACENET_H
#define _VNN_FACENET_H

#include "vsi_nn_pub.h"

#define VNN_APP_DEBUG (FALSE)
#define VNN_VERSION_MAJOR 1
#define VNN_VERSION_MINOR 1
#define VNN_VERSION_PATCH 1
#define VNN_RUNTIME_VERSION \
    (VNN_VERSION_MAJOR * 10000 + VNN_VERSION_MINOR * 100 + VNN_VERSION_PATCH)

_version_assert(VNN_RUNTIME_VERSION <= VSI_NN_VERSION,
                CASE_VERSION_is_higher_than_OVXLIB_VERSION)

void vnn_ReleaseFacenet
    (
    vsi_nn_graph_t * graph,
    vsi_bool release_ctx
    );

vsi_nn_graph_t * vnn_CreateFacenet
    (
    const char * data_file_name,
    vsi_nn_context_t in_ctx
    );

#endif
