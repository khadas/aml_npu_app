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
#include <fcntl.h>
#include <string.h>
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
#include <sys/resource.h>
#include <getopt.h>

#include "nn_detect.h"
#include "nn_detect_utils.h"


using namespace std;
using namespace cv;


#define MODEL_WIDTH 416
#define MODEL_HEIGHT 416
#define BUFFER_COUNT 4
#define MAX_HEIGHT  1080
#define MAX_WIDTH   1920
#define DEFAULT_DEVICE "/dev/video0"

int height = MAX_HEIGHT;
int width = MAX_WIDTH;

struct option longopts[] = {
	{ "device",         required_argument,  NULL,   'd' },
	{ "width",          required_argument,  NULL,   'w' },
	{ "height",         required_argument,  NULL,   'h' },
	{ "help",           no_argument,        NULL,   'H' },
	{ 0, 0, 0, 0 }
};


struct  Frame{   
	size_t length;
	int height;
	int width;
	unsigned char data[MAX_HEIGHT * MAX_WIDTH * 3];
} frame;

pthread_mutex_t mutex4q;

const char *device = DEFAULT_DEVICE;


unsigned char *displaybuf;
int g_nn_height, g_nn_width, g_nn_channel;
det_model_type g_model_type = (det_model_type)2;

#define _CHECK_STATUS_(status, stat, lbl) do {\
	if (status != stat) \
	{ \
		cout << "_CHECK_STATUS_ File" << __FUNCTION__ << __LINE__ <<endl; \
	}\
	goto lbl; \
}while(0)



static cv::Scalar obj_id_to_color(int obj_id) {

	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
	int const offset = obj_id * 123457 % 6;
	int const color_scale = 150 + (obj_id * 123457) % 100;
	cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
	color *= color_scale;
	return color;
}


static void draw_results(cv::Mat& frame, DetectResult resultData, int img_width, int img_height, det_model_type type){

	int i = 0;
	float left, right, top, bottom;

	for (i = 0; i < resultData.detect_num; i++) {
		left =  resultData.point[i].point.rectPoint.left*img_width;
        right = resultData.point[i].point.rectPoint.right*img_width;
        top = resultData.point[i].point.rectPoint.top*img_height;
        bottom = resultData.point[i].point.rectPoint.bottom*img_height;
		
//		cout << "i:" <<resultData.detect_num <<" left:" << left <<" right:" << right << " top:" << top << " bottom:" << bottom <<endl;
		CvPoint pt1;
		CvPoint pt2;
		pt1=cvPoint(left,top);
		pt2=cvPoint(right, bottom);

		cv::Rect rect(left, top, right-left, bottom-top);
		cv::rectangle(frame,rect,obj_id_to_color(resultData.result_name[i].lable_id),1,8,0);
		switch (type) {
			case DET_YOLOFACE_V2:
			break;
			case DET_YOLO_V2:
			case DET_YOLO_V3:
			case DET_YOLO_V4:
			case DET_YOLO_TINY:
			{
				if (top < 50) {
					top = 50;
					left +=10;
				}
				int baseline;
				cv::Size text_size = cv::getTextSize(resultData.result_name[i].lable_name, cv::FONT_HERSHEY_COMPLEX,0.5,1,&baseline);
				cv::Rect rect1(left, top-20, text_size.width+10, 20);
				cv::rectangle(frame,rect1,obj_id_to_color(resultData.result_name[i].lable_id),-1);
				cv::putText(frame,resultData.result_name[i].lable_name,cvPoint(left+5,top-5),cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(0,0,0),1);
				break;
			}
			default:
			break;
		}
	}
	cv::imshow("Image Window",frame);
	cv::waitKey(1);
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

    int ret = 0,frames = 0;
	float total_time = 0;
	struct timeval time_start, time_end;
	DetectResult resultData;

	cv::Mat yolo_v2Image(g_nn_width, g_nn_height, CV_8UC1);
	cv::Mat img(height,width,CV_8UC3,cv::Scalar(0,0,0));

    string str = device;

	int video_width,video_height;

    string res=str.substr(10);
	cv::VideoCapture cap(stoi(res));
    cap.set(CAP_PROP_FRAME_WIDTH, width);
    cap.set(CAP_PROP_FRAME_HEIGHT, height);

	if (!cap.isOpened()) {
		cout << "capture device failed to open!" << endl;

		goto out;
	}

	cout << "open video successfully!" << endl;

	video_width = cap.get(CAP_PROP_FRAME_WIDTH);
	video_height = cap.get(CAP_PROP_FRAME_HEIGHT);

	printf("video_width: %d, video_height: %d\n", video_width, video_height);

    setpriority(PRIO_PROCESS, pthread_self(), -15);

    cv::namedWindow("Image Window");

	gettimeofday(&time_start, 0);

   while (true) {
	   pthread_mutex_lock(&mutex4q);
	   if (!cap.read(img)) {
		   cout<<"Capture read error"<<std::endl;
		   break;
	   }
	   pthread_mutex_unlock(&mutex4q);

	   cv::resize(img, yolo_v2Image, yolo_v2Image.size());

		input_image_t image;
		image.data      = yolo_v2Image.data;
		image.width     = yolo_v2Image.cols;
		image.height    = yolo_v2Image.rows;
		image.channel   = yolo_v2Image.channels();
		image.pixel_format = PIX_FMT_RGB888;

		ret = det_set_input(image, g_model_type);
		if (ret) {
			cout << "det_set_input fail. ret=" << ret << endl;
			det_release_model(g_model_type);
			goto out;
		}

		ret = det_get_result(&resultData, g_model_type);
		if (ret) {
			cout << "det_get_result fail. ret=" << ret << endl;
			det_release_model(g_model_type);
			goto out;
		}

		draw_results(img, resultData, width, height, g_model_type);

		// Measure FPS
		++frames;

		gettimeofday(&time_end, 0);
		total_time += (float)((time_end.tv_sec - time_start.tv_sec) + (time_end.tv_usec - time_start.tv_usec) / 1000.0f / 1000.0f);
		gettimeofday(&time_start, 0);

		if (total_time >= 1.0f) {
			int fps = (int)(frames / total_time);
			fprintf(stderr, "FPS: %i\n", fps);
			frames = 0;
			total_time = 0;
		}
    }
out:
   cap.release();
	printf("thread_func exit\n");
	exit(-1);
}

int main(int argc, char** argv){

	int c;
	int i=0,ret=0;
	pthread_t tid[2];

	while ((c = getopt_long(argc, argv, "d:w:h:m:H", longopts, NULL)) != -1) {
		switch (c) {
			case 'd':
				device = optarg;
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
				printf("%s [-d device] [-w width] [-h height] [-m model type] [-H]\n", argv[0]);
				exit(1);
		}
	}

	pthread_mutex_init(&mutex4q,NULL);

	run_detect_model(g_model_type);


	if (0 != pthread_create(&tid[0], NULL, thread_func, NULL)) {
		fprintf(stderr, "Couldn't create thread func\n");
		return -1;
	}

	while(1)
	{
		for(i=0; i<(int)(sizeof(tid)/sizeof(tid[0])); i++)
		{
			pthread_join(tid[i], NULL);
		}
		sleep(1);
	}

	return 0;
}
