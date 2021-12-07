#include <iostream>
#include <fstream>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include <unistd.h>
#include <stdio.h>
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
#include <signal.h>
#include <stdlib.h> 

#include <getopt.h>


#include "nn_detect.h"
#include "nn_detect_utils.h"


using namespace std;
using namespace cv;

#define MODEL_WIDTH 416
#define MODEL_HEIGHT 416
#define DEFAULT_DEVICE "/dev/video0"
#define MESON_BUFFER_SIZE 4
#define FB_DEVICE_NODE "/dev/fb0"

struct option longopts[] = {
	{ "device",         required_argument,  NULL,   'd' },
	{ "width",          required_argument,  NULL,   'w' },
	{ "height",         required_argument,  NULL,   'h' },
	{ "help",           no_argument,        NULL,   'H' },
	{ 0, 0, 0, 0 }
};


const char *device = DEFAULT_DEVICE;

#define MAX_HEIGHT 1080
#define MAX_WIDTH 1920

#define BUFFER_COUNT	4

int width = MAX_WIDTH;
int height = MAX_HEIGHT;

#define DEFAULT_FRAME_RATE  30

const size_t EXTERNAL_PTS = 0x01;
const size_t SYNC_OUTSIDE = 0x02;
const size_t USE_IDR_FRAMERATE = 0x04;
const size_t UCODE_IP_ONLY_PARAM = 0x08;
const size_t MAX_REFER_BUF = 0x10;
const size_t ERROR_RECOVERY_MODE_IN = 0x20;

int fb_fd = -1;
int capture_fd = -1;
int fps = DEFAULT_FRAME_RATE;

struct fb_var_screeninfo var_info;


struct  Frame{
	size_t length;
	int height;
	int width;
	unsigned char data[MAX_HEIGHT * MAX_WIDTH * 3];
} frame;


const char *xcmd="echo 1080p60hz > /sys/class/display/mode;\
				  fbset -fb /dev/fb0 -g 1920 1080 1920 2160 32;\
				  echo 1 > /sys/class/graphics/fb0/freescale_mode;\
				  echo 0 0 1919 1079 >  /sys/class/graphics/fb0/window_axis;\
				  echo 0 0 1919 1079 > /sys/class/graphics/fb0/free_scale_axis;\
				  echo 0x10001 > /sys/class/graphics/fb0/free_scale;\
				  echo 0 > /sys/class/graphics/fb0/blank;";

static int fbfd = 0;
static struct fb_var_screeninfo vinfo;
static struct fb_fix_screeninfo finfo;
static long int screensize = 0;
char *fbp;

pthread_mutex_t mutex4q;

unsigned char *displaybuf;
int g_nn_height, g_nn_width, g_nn_channel;
det_model_type g_model_type;

#define _CHECK_STATUS_(status, stat, lbl) do {\
	if (status != stat) \
	{ \
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

	cvtColor(frame, frame, CV_BGR2RGB);
	for (i = 0; i < resultData.detect_num; i++) {
		left =  resultData.point[i].point.rectPoint.left*img_width;
        right = resultData.point[i].point.rectPoint.right*img_width;
        top = resultData.point[i].point.rectPoint.top*img_height;
        bottom = resultData.point[i].point.rectPoint.bottom*img_height;
		
	//	cout << "i:" << i <<" left:" << left <<" right:" << right << " top:" << top << " bottom:" << bottom << "class:" << resultData.result_name[i].lable_name <<endl;

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
	memcpy(fbp+MAX_HEIGHT*MAX_WIDTH*3,frame.data,MAX_HEIGHT*MAX_WIDTH*3);
	vinfo.activate = FB_ACTIVATE_NOW;
	vinfo.vmode &= ~FB_VMODE_YWRAP;
	vinfo.yoffset = 1080;
	ioctl(fbfd, FBIOPAN_DISPLAY, &vinfo);
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

static int init_fb(void){

	printf("init_fb...\n");

	// Open the file for reading and writing
	fbfd = open("/dev/fb0", O_RDWR);
	if (!fbfd){
		printf("Error: cannot open framebuffer device.\n");
		exit(1);
	}

	// Get fixed screen information
	if (ioctl(fbfd, FBIOGET_FSCREENINFO, &finfo)){
        printf("Error reading fixed information.\n");
        exit(2);
    }

	// Get variable screen information
	if (ioctl(fbfd, FBIOGET_VSCREENINFO, &vinfo)){
		printf("Error reading variable information.\n");
		exit(3);
	}
	printf("%dx%d, %dbpp\n", vinfo.xres, vinfo.yres, vinfo.bits_per_pixel );
/*============add for display BGR begin================,for imx290,reverse color*/
	vinfo.red.offset = 0;
	vinfo.red.length = 8;
	vinfo.red.msb_right = 0;

	vinfo.green.offset = 8;
	vinfo.green.length = 8;
	vinfo.green.msb_right = 0;

	vinfo.blue.offset = 16;
	vinfo.blue.length = 8;
	vinfo.blue.msb_right = 0;

	vinfo.transp.offset = 0;
	vinfo.transp.length = 0;
	vinfo.transp.msb_right = 0;
	vinfo.nonstd = 0;
	vinfo.bits_per_pixel = 24;

	//vinfo.activate = FB_ACTIVATE_NOW;   //zxw
	//vinfo.vmode &= ~FB_VMODE_YWRAP;
	if (ioctl(fbfd, FBIOPUT_VSCREENINFO, &vinfo) == -1) {
		printf("Error reading variable information\n");
	}
/*============add for display BGR end ================*/	
	// Figure out the size of the screen in bytes
	screensize = vinfo.xres * vinfo.yres * vinfo.bits_per_pixel / 4;  //8 to 4

	// Map the device to memory
	fbp = (char *)mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED,
			fbfd, 0);

	if (fbp == NULL){
		printf("Error: failed to map framebuffer device to memory.\n");
		exit(4);
	}
	return 0;
}


static void *thread_func(void *x){

	cv::Mat yolo_v2Image(g_nn_width, g_nn_height, CV_8UC1);
	cv::Mat img(height,width,CV_8UC3,cv::Scalar(0,0,0));
	DetectResult resultData;

	int ret;
	int frames = 0;
	struct timeval time_start, time_end;
	float total_time = 0;
	uint32_t isize;

   	string str = device;
	string res=str.substr(10);
   	cv::VideoCapture cap(stoi(res));
   	cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
   	cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

	system(xcmd);
	init_fb();

	gettimeofday(&time_start, 0);

 	while (true) {

		pthread_mutex_lock(&mutex4q);
	   	if (!cap.read(img)) {
	   		pthread_mutex_unlock(&mutex4q);
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

	   	gettimeofday(&time_start, 0);
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

		gettimeofday(&time_end, 0);
		draw_results(img, resultData, width, height, g_model_type);
		++frames;
		total_time += (float)((time_end.tv_sec - time_start.tv_sec) + (time_end.tv_usec - time_start.tv_usec) / 1000.0f / 1000.0f);

	   	if (total_time >= 1.0f) {
	   		int fps = (int)(frames / total_time);
	   		fprintf(stderr, "Inference FPS: %i\n", fps);
	   		frames = 0;
	   		total_time = 0;
	   	}
    }
out:
	printf("thread_func exit\n");
	exit(-1);
}

int main(int argc, char** argv){

	int i;
	int c;
	int ret;
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


	run_detect_model(g_model_type);

	pthread_mutex_init(&mutex4q,NULL);

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
