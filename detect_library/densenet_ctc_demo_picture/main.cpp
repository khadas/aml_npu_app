#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>

#include <unistd.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <dirent.h>
#include <queue>
#include <sched.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <linux/fb.h>
#include <linux/kd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <pthread.h>
#include <VX/vx.h>
#include <VX/vxu.h>
#include <VX/vx_api.h>
#include <VX/vx_khr_cnn.h>
#include <semaphore.h>
#include <sys/time.h>
#include <sched.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <semaphore.h>
#include <getopt.h>

#include "nn_detect.h"
#include "nn_detect_utils.h"

using namespace std;
using namespace cv;

#define MODEL_WIDTH 280
#define MODEL_HEIGHT 32
#define DEFAULT_DEVICE "/dev/video0"
#define MESON_BUFFER_SIZE 4
#define DEFAULT_OUTPUT "default.h264"
#define ION_DEVICE_NODE "/dev/ion"
#define FB_DEVICE_NODE "/dev/fb0"


struct option longopts[] = {
	{ "picture",        required_argument,  NULL,   'p' },
	{ "width",          required_argument,  NULL,   'w' },
	{ "height",         required_argument,  NULL,   'h' },
	{ "help",           no_argument,        NULL,   'H' },
	{ 0, 0, 0, 0 }
};

const char *picture;

#define MAX_HEIGHT 32
#define MAX_WIDTH 280


int width = MAX_WIDTH;
int height = MAX_HEIGHT;

#define DEFAULT_FRAME_RATE  30

struct  Frame
{   
	size_t length;
	int height;
	int width;
	unsigned char data[MAX_HEIGHT * MAX_WIDTH * 3];
} frame;


pthread_mutex_t mutex4q;

int g_nn_height, g_nn_width, g_nn_channel;
det_model_type g_model_type;

#define _CHECK_STATUS_(status, stat, lbl) do {\
	if (status != stat) \
	{ \
		cout << "_CHECK_STATUS_ File" << __FUNCTION__ << __LINE__ <<endl; \
	}\
	goto lbl; \
}while(0)


int minmax(int min, int v, int max)
{
	return (v < min) ? min : (max < v) ? max : v;
}

uint8_t* yuyv2rgb(uint8_t* yuyv, uint32_t width, uint32_t height)
{
  	uint8_t* rgb = (uint8_t *)calloc(width * height * 3, sizeof (uint8_t));
  	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < width; j += 2) {
	  		size_t index = i * width + j;
	  		int y0 = yuyv[index * 2 + 0] << 8;
	  		int u = yuyv[index * 2 + 1] - 128;
	  		int y1 = yuyv[index * 2 + 2] << 8;
	  		int v = yuyv[index * 2 + 3] - 128;
	  		rgb[index * 3 + 0] = minmax(0, (y0 + 359 * v) >> 8, 255);
	  		rgb[index * 3 + 1] = minmax(0, (y0 + 88 * v - 183 * u) >> 8, 255);
	  		rgb[index * 3 + 2] = minmax(0, (y0 + 454 * u) >> 8, 255);
	  		rgb[index * 3 + 3] = minmax(0, (y1 + 359 * v) >> 8, 255);
	  		rgb[index * 3 + 4] = minmax(0, (y1 + 88 * v - 183 * u) >> 8, 255);
	  		rgb[index * 3 + 5] = minmax(0, (y1 + 454 * u) >> 8, 255);
		}
  	}
  	return rgb;
}

int run_detect_model(det_model_type type){

	int ret = 0;
	int nn_height, nn_width, nn_channel;
	det_set_log_config(DET_DEBUG_LEVEL_WARN,DET_LOG_TERMINAL);
	cout << "det_set_log_config Debug" <<endl;

	//prepare model
	ret = det_set_model(type);
	if (ret) {
		cout << "det_set_model fail. ret=" << ret <<endl;
		return ret;
	}
	cout << "det_set_model success!!" << endl;

	ret = det_get_model_size(type, &nn_width, &nn_height, &nn_channel);
	if (ret) {
		cout << "det_get_model_size fail" <<endl;
		return ret;
	}

	cout << "\nmodel.width:" << nn_width <<endl;
	cout << "model.height:" << nn_height <<endl;
	cout << "model.channel:" << nn_channel << "\n" <<endl;

	g_nn_width = nn_width;
	g_nn_height = nn_height;
	g_nn_channel = nn_channel;

	return ret;
}

static void *thread_func(void *x){

    int ret = 0;
	char result[35] = {0};
	int result_len = 0;
	cv::Mat yolo_v2Image(g_nn_height, g_nn_width, CV_8UC1);
	struct timeval time_start, time_end;
	float total_time = 0;
	uint32_t isize;

	setpriority(PRIO_PROCESS, pthread_self(), -15);

	cv::Mat img = cv::imread(picture, 0);
	cv::resize(img, yolo_v2Image, yolo_v2Image.size());

		input_image_t image;
		image.data      = yolo_v2Image.data;
		image.width     = yolo_v2Image.cols;
		image.height    = yolo_v2Image.rows;
		image.channel   = yolo_v2Image.channels();
		image.pixel_format = PIX_FMT_RGB888;

		gettimeofday(&time_start, 0);
		ret = det_set_input(image, g_model_type);
		if (ret) {
			cout << "det_set_input fail. ret=" << ret << endl;
			det_release_model(g_model_type);
		}

		ret = densenet_ctc_get_result(result, &result_len, g_model_type);
		if (ret) {
			cout << "det_get_result fail. ret=" << ret << endl;
	 		det_release_model(g_model_type);
		}
		
		printf("%s\n", result);

		gettimeofday(&time_end, 0);
		total_time += (float)((time_end.tv_sec - time_start.tv_sec) + (time_end.tv_usec - time_start.tv_usec) / 1000.0f / 1000.0f);
		printf("time: %f\n", total_time);
    
}

int main(int argc, char** argv){

	int c;
	int i=0;
	pthread_t tid[2];

	while ((c = getopt_long(argc, argv, "p:w:h:m:H", longopts, NULL)) != -1) {
		switch (c) {
			case 'p':
				picture = optarg;
				break;

			case 'w':
				width = atoi(optarg);
				break;

			case 'h':
				height = atoi(optarg);
				break;

			case 'm':
				g_model_type  = (det_model_type)atoi(optarg);
				break;

			default:
				printf("%s [-p picture] [-w width] [-h height] [-m model type] [-H]\n", argv[0]);
				exit(1);
		}
	}


	run_detect_model(g_model_type);

	pthread_mutex_init(&mutex4q,NULL);

	if (0 != pthread_create(&tid[0], NULL, thread_func, NULL)) {
		fprintf(stderr, "Couldn't create thread func\n");
		return -1;
	}


	pthread_join(tid[i], NULL);
	
	return 0;
}
