#include <iostream>
#include <fstream>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>

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
#include <vpcodec_1_0.h>
#include <linux/meson_ion.h>
#include <linux/ge2d.h>
#include <linux/fb.h>
#include <ge2d_port.h>
#include <aml_ge2d.h>

#include <ion/ion.h>
#include <ion/IONmem.h>
#include <linux/ion.h>
#include <Amsysfsutils.h>


#include "nn_detect.h"
#include "nn_detect_utils.h"

// The headers are not aware C++ exists
extern "C"  {
	#include <amvideo.h>
	#include <codec.h>
}


// ge2d
aml_ge2d_t amlge2d;


using namespace std;
using namespace cv;

#define MODEL_WIDTH 416
#define MODEL_HEIGHT 416
#define BUFFER_COUNT 4
#define MAX_HEIGHT  1080
#define MAX_WIDTH   1920

int height = MAX_HEIGHT;
int width = MAX_WIDTH;

#define DEFAULT_DEVICE "/dev/video0"

extern char *video_device;

struct option longopts[] = {
	{ "device",         required_argument,  NULL,   'd' },
	{ "width",          required_argument,  NULL,   'w' },
	{ "height",         required_argument,  NULL,   'h' },
	{ "model type",     required_argument,  NULL,   'm' },
	{ "ir-cut",         required_argument,  NULL,   'i' },
	{ "help",           no_argument,        NULL,   'H' },
	{ 0, 0, 0, 0 }
};

int capture_fd = -1;

typedef struct __video_buffer{
	void *start;
	size_t length;

}video_buf_t;

struct  Frame{
	size_t length;
	int height;
	int width;
	unsigned char data[MAX_HEIGHT * MAX_WIDTH * 3];
} frame;

extern "C" { void *camera_thread_func(void *arg); }

extern int ir_cut_state;

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
int opencv_ok = 0;
static int pingflag = 0;

pthread_mutex_t mutex4q;

unsigned char *displaybuf;
int g_nn_height, g_nn_width, g_nn_channel;
det_model_type g_model_type;

#define _CHECK_STATUS_(status, stat, lbl) do {\
	if (status != stat) \
	{ \
		cout << "_CHECK_STATUS_ File" << __FUNCTION__ << __LINE__ <<endl; \
	}\
	goto lbl; \
}while(0)

int ge2d_init(int width, int height){

	int ret;

	memset(&amlge2d, 0, sizeof(aml_ge2d_t));
	memset(&(amlge2d.ge2dinfo.src_info[0]), 0, sizeof(buffer_info_t));
	memset(&(amlge2d.ge2dinfo.src_info[1]), 0, sizeof(buffer_info_t));
	memset(&(amlge2d.ge2dinfo.dst_info), 0, sizeof(buffer_info_t));

	amlge2d.ge2dinfo.src_info[0].canvas_w = width;
	amlge2d.ge2dinfo.src_info[0].canvas_h = height;
	amlge2d.ge2dinfo.src_info[0].format = PIXEL_FORMAT_RGB_888;
	amlge2d.ge2dinfo.src_info[0].plane_number = 1;

	amlge2d.ge2dinfo.dst_info.canvas_w = MODEL_WIDTH;
	amlge2d.ge2dinfo.dst_info.canvas_h = MODEL_HEIGHT;
	amlge2d.ge2dinfo.dst_info.format = PIXEL_FORMAT_RGB_888;
	amlge2d.ge2dinfo.dst_info.plane_number = 1;
	amlge2d.ge2dinfo.dst_info.rotation = GE2D_ROTATION_0;
	amlge2d.ge2dinfo.offset = 0;
	amlge2d.ge2dinfo.ge2d_op = AML_GE2D_STRETCHBLIT;
	amlge2d.ge2dinfo.blend_mode = BLEND_MODE_PREMULTIPLIED;

	amlge2d.ge2dinfo.src_info[0].memtype = GE2D_CANVAS_ALLOC;
	amlge2d.ge2dinfo.src_info[1].memtype = GE2D_CANVAS_TYPE_INVALID;
	amlge2d.ge2dinfo.dst_info.memtype = GE2D_CANVAS_ALLOC;
	amlge2d.ge2dinfo.src_info[0].mem_alloc_type = AML_GE2D_MEM_ION;//AML_GE2D_MEM_DMABUF
	amlge2d.ge2dinfo.src_info[1].mem_alloc_type = AML_GE2D_MEM_INVALID;//AML_GE2D_MEM_ION;
	amlge2d.ge2dinfo.dst_info.mem_alloc_type = AML_GE2D_MEM_ION;

	ret = aml_ge2d_init(&amlge2d);
	if (ret < 0) {
		printf("aml_ge2d_init failed!\n");
		return -1;
	}

	ret = aml_ge2d_mem_alloc(&amlge2d);
	if (ret < 0) {
		printf("aml_ge2d_mem_alloc failed!\n");
		return -1;
	}

	return 0;
}

int ge2d_destroy(void){

	int i;

	if (amlge2d.ge2dinfo.dst_info.mem_alloc_type == AML_GE2D_MEM_ION)
		aml_ge2d_invalid_cache(&amlge2d.ge2dinfo);

	for (i = 0; i < amlge2d.ge2dinfo.src_info[0].plane_number; i++) {
		if (amlge2d.src_data[i]) {
			free(amlge2d.src_data[i]);
			amlge2d.src_data[i] = NULL;
		}
	}

	for (i = 0; i < amlge2d.ge2dinfo.src_info[1].plane_number; i++) {
		if (amlge2d.src2_data[i]) {
			free(amlge2d.src2_data[i]);
			amlge2d.src2_data[i] = NULL;
		}
	}

	for (i = 0; i < amlge2d.ge2dinfo.dst_info.plane_number; i++) {
		if (amlge2d.dst_data[i]) {
			free(amlge2d.dst_data[i]);
			amlge2d.dst_data[i] = NULL;
		}
	}

	aml_ge2d_mem_free(&amlge2d);
	aml_ge2d_exit(&amlge2d);

	return 0;
}

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
//					cout << "left:" << left << " top-10:" << top-10 <<endl;
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

	if(pingflag == 0)	{
		memcpy(fbp+1920*1080*3,frame.data,1920*1080*3);
		vinfo.activate = FB_ACTIVATE_NOW;
		vinfo.vmode &= ~FB_VMODE_YWRAP;
		vinfo.yoffset = 1080;
		pingflag = 1;
		ioctl(fbfd, FBIOPAN_DISPLAY, &vinfo);
	}
	else
	{
		memcpy(fbp,frame.data,1920*1080*3);
		vinfo.activate = FB_ACTIVATE_NOW;
		vinfo.vmode &= ~FB_VMODE_YWRAP;
		vinfo.yoffset = 0;
		pingflag = 0;
		ioctl(fbfd, FBIOPAN_DISPLAY, &vinfo);
	}
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
	if (!fbfd)
	{
		printf("Error: cannot open framebuffer device.\n");
		exit(1);
	}

	// Get fixed screen information
	if (ioctl(fbfd, FBIOGET_FSCREENINFO, &finfo)){
		printf("Error reading fixed information.\n");
		exit(2);
	}

	// Get variable screen information
	if (ioctl(fbfd, FBIOGET_VSCREENINFO, &vinfo))
	{
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

	cv::Mat frame(MAX_HEIGHT, MAX_WIDTH, CV_8UC3);
    int ret = 0,frames=0;
	DetectResult resultData;
	struct timeval time_start, time_end;
	float total_time = 0;

	system(xcmd);
	init_fb();
 
	gettimeofday(&time_start, 0);

	while (true) {
		memcpy(frame.data,displaybuf,1920*1080*3);

		memcpy(amlge2d.ge2dinfo.src_info[0].vaddr[0],frame.data,amlge2d.src_size[0]);

		amlge2d.ge2dinfo.src_info[0].rect.x = 0;

		amlge2d.ge2dinfo.src_info[0].rect.y = 0;
		amlge2d.ge2dinfo.src_info[0].rect.w = amlge2d.ge2dinfo.src_info[0].canvas_w;
		amlge2d.ge2dinfo.src_info[0].rect.h = amlge2d.ge2dinfo.src_info[0].canvas_h;

		amlge2d.ge2dinfo.dst_info.rect.x = 0;
		amlge2d.ge2dinfo.dst_info.rect.y = 0;
		amlge2d.ge2dinfo.dst_info.rect.w = MODEL_WIDTH;
		amlge2d.ge2dinfo.dst_info.rect.h = MODEL_HEIGHT;
		amlge2d.ge2dinfo.dst_info.rotation = GE2D_ROTATION_0;
		amlge2d.ge2dinfo.src_info[0].layer_mode = 0;
		amlge2d.ge2dinfo.src_info[0].plane_alpha = 0xff;


		ret = aml_ge2d_process(&amlge2d.ge2dinfo);
		if (ret < 0) {
			printf("aml_ge2d_process failed!\n");
			return NULL;
		}



		input_image_t image;
		image.data      = (unsigned char*)amlge2d.ge2dinfo.dst_info.vaddr[0];
		image.width     = MODEL_WIDTH;
		image.height    = MODEL_HEIGHT;
		image.channel   = 3;
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

		draw_results(frame, resultData, width, height, g_model_type);

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
	printf("thread_func exit\n");
	exit(-1);
}

int main(int argc, char** argv){

	video_device = (char *)DEFAULT_DEVICE;
	int c = 0;	
	int i=0,ret=0;
	pthread_t tid[2];

	while ((c = getopt_long(argc, argv, "d:w:h:m:i:H", longopts, NULL)) != -1) {
		switch (c) {
			case 'd':
				video_device = optarg;
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

			case 'i':
				if (0 == strcmp(optarg, "on")) {
					ir_cut_state = 1;
				} else if (0 == strcmp(optarg, "off")) {
					ir_cut_state = 0;
				} else {
					printf("Error: IR-CUT set error!\n");
					return -1;
				}
				break;

			default:
				printf("%s [-d device] [-w width] [-h height] [-m model type] [-i ir-cut stat] [-H]\n", argv[0]);
				exit(1);
		}
	}

	run_detect_model(g_model_type);

	ret = ge2d_init(width, height);
	if (ret < 0) {
		printf("ge2d_init failed!\n");
	}


	pthread_mutex_init(&mutex4q,NULL);

	if (0 != pthread_create(&tid[0], NULL, thread_func, NULL)) {
		fprintf(stderr, "Couldn't create thread func\n");
		return -1;
	}

	if (0 != pthread_create(&tid[1], NULL, camera_thread_func, NULL)) {
		fprintf(stderr, "Couldn't create camera_thread_func thread func\n");
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
	ge2d_destroy();

	return 0;
}
