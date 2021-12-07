#include "yoloface.h"

#include "vnn_yoloface.h"
#include "yoloface_process.h"

vsi_nn_graph_t * g_graph = NULL;

const static vsi_nn_postprocess_map_element_t* postprocess_map = NULL;
const static vsi_nn_preprocess_map_element_t* preprocess_map = NULL;

const vsi_nn_preprocess_map_element_t * vnn_GetPrePorcessMap()
{
    return preprocess_map;
}

uint32_t vnn_GetPrePorcessMapCount()
{
    if (preprocess_map == NULL)
        return 0;
    else
        return sizeof(preprocess_map) / sizeof(vsi_nn_preprocess_map_element_t);
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

det_status_t model_create(const char * data_file_path, dev_type type)
{
	det_status_t ret = DET_STATUS_ERROR;
	vsi_status status = VSI_SUCCESS;
	char model_path[48];

	switch (type) {
		case DEV_REVA:
			sprintf(model_path, "%s%s", data_file_path, "/yolo_face_7d.nb");
			break;
		case DEV_REVB:
			sprintf(model_path, "%s%s", data_file_path, "/yolo_face_88.nb");
			break;
		case DEV_MS1:
			sprintf(model_path, "%s%s", data_file_path, "/yolo_face_99.nb");
			break;
		default:
			break;
	}
	g_graph = vnn_CreateYoloface(model_path, NULL,
			vnn_GetPrePorcessMap(), vnn_GetPrePorcessMapCount(),
			vnn_GetPostPorcessMap(), vnn_GetPostPorcessMapCount());

	TEST_CHECK_PTR(g_graph, exit);

	status = vsi_nn_VerifyGraph(g_graph);
	TEST_CHECK_STATUS(status, exit);

	ret = DET_STATUS_OK;
exit:
	return ret;
}

void model_getsize(int *width, int *height, int *channel)
{
	if (g_graph) {
		vsi_nn_tensor_t *tensor = NULL;
		tensor = vsi_nn_GetTensor(g_graph, g_graph->input.tensors[0] );

		*width = tensor->attr.size[0];
		*height = tensor->attr.size[1];
		*channel = tensor->attr.size[2];
	}
}

void model_setinput(input_image_t imageData, uint8_t* data)
{
	yolo_face_preprocess(imageData, data);
}

det_status_t model_getresult(pDetResult resultData, uint8_t* data)
{
	vsi_status status = VSI_FAILURE;
	det_status_t ret = DET_STATUS_ERROR;
	vsi_nn_tensor_t *tensor = vsi_nn_GetTensor(g_graph, g_graph->input.tensors[0]);

	status = vsi_nn_CopyDataToTensor(g_graph, tensor, data);
	TEST_CHECK_STATUS(status, exit);

	status = vsi_nn_RunGraph(g_graph);
	TEST_CHECK_STATUS(status, exit);

	yolo_face_postprocess(g_graph, resultData);

	ret = DET_STATUS_OK;
exit:
	return ret;
}

void model_release(dev_type type)
{
	vnn_ReleaseYoloface(g_graph, TRUE);
	g_graph = NULL;
}
