/****************************************************************************
*   Generated by ACUITY 5.21.1_0702
*   Match ovxlib 1.1.30
*
*   Neural Network global header file
****************************************************************************/
#ifndef _VNN_GLOBAL_H_
#define _VNN_GLOBAL_H_

typedef struct {
    uint32_t graph_input_idx;
    vsi_nn_preprocess_base_t *preprocesses;
    uint32_t preprocess_count;
} vsi_nn_preprocess_map_element_t;


typedef struct {
    uint32_t graph_output_idx;
    vsi_nn_postprocess_base_t *postprocesses;
    uint32_t postprocess_count;
} vsi_nn_postprocess_map_element_t;
/*
 * This file will be deprecated in the future
 */

#endif
