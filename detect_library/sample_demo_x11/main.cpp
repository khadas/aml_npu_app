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

struct option longopts[] = {
	{ "path",           required_argument,  NULL,   'p' },
	{ "width",          required_argument,  NULL,   'w' },
	{ "height",         required_argument,  NULL,   'h' },
	{ "facenet_f",      required_argument,  NULL,   'f' },
	{ "help",           no_argument,        NULL,   'H' },
	{ 0, 0, 0, 0 }
};


// ge2d
aml_ge2d_t amlge2d;

#define MAX_HEIGHT  1080
#define MAX_WIDTH   1920

int height = MAX_HEIGHT;
int width = MAX_WIDTH;

int facenet_falge = 0;

using namespace std;
using namespace cv;

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
char* picture_path = NULL;
det_model_type g_model_type;

#define _CHECK_STATUS_(status, stat, lbl) do {\
	if (status != stat) \
	{ \
		cout << "_CHECK_STATUS_ File" << __FUNCTION__ << __LINE__ <<endl; \
	}\
	goto lbl; \
}while(0)

int ge2d_init(int width, int height)
{
	int ret;

	memset(&amlge2d, 0, sizeof(aml_ge2d_t));
	memset(&(amlge2d.ge2dinfo.src_info[0]), 0, sizeof(buffer_info_t));
	memset(&(amlge2d.ge2dinfo.src_info[1]), 0, sizeof(buffer_info_t));
	memset(&(amlge2d.ge2dinfo.dst_info), 0, sizeof(buffer_info_t));

	amlge2d.ge2dinfo.src_info[0].canvas_w = width;
	amlge2d.ge2dinfo.src_info[0].canvas_h = height;
	amlge2d.ge2dinfo.src_info[0].format = PIXEL_FORMAT_RGB_888;
	amlge2d.ge2dinfo.src_info[0].plane_number = 1;

	amlge2d.ge2dinfo.dst_info.canvas_w = 416;
	amlge2d.ge2dinfo.dst_info.canvas_h = 416;
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

int ge2d_destroy(void)
{
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


static void draw_results(cv::Mat& frame, DetectResult resultData, int img_width, int img_height, det_model_type type)
{
	int i = 0;
	float left, right, top, bottom;

	cout << "\nresultData.detect_num=" << resultData.detect_num <<endl;
	cout << "result type is " << resultData.point[i].type << endl;
	for (i = 0; i < resultData.detect_num; i++) {
		left =  resultData.point[i].point.rectPoint.left*img_width;
        right = resultData.point[i].point.rectPoint.right*img_width;
        top = resultData.point[i].point.rectPoint.top*img_height;
        bottom = resultData.point[i].point.rectPoint.bottom*img_height;
		cout << "i:" <<resultData.detect_num <<" left:" << left <<" right:" << right << " top:" << top << " bottom:" << bottom <<endl;
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
					cout << "left:" << left << " top-10:" << top-10 <<endl;
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

		cv::imwrite("output.bmp", frame);
        cv::namedWindow("Image Window");
        cv::imshow("Image Window",frame);
	}
}

static void crop_face(cv::Mat sourceFrame, cv::Mat& imageROI, DetectResult resultData, int img_height, int img_width) {
	float left, right, top, bottom;
	int tempw,temph;

	left =  resultData.point[0].point.rectPoint.left*img_width;
    right = resultData.point[0].point.rectPoint.right*img_width;
    top = resultData.point[0].point.rectPoint.top*img_height;
    bottom = resultData.point[0].point.rectPoint.bottom*img_height;

    tempw = abs((int)(left - right));
	temph = abs((int)(top - bottom));
	if (tempw > 1) tempw = tempw -1;
	if (temph > 1) temph = temph -1;

	if (left + tempw > img_width)
		tempw = img_width - left;
	if (bottom + temph > img_height)
		temph = img_height - top;

	imageROI = sourceFrame(cv::Rect(left, top, tempw, temph));
	cv::imwrite("face.bmp", imageROI);
	return;
}


int run_detect_model(int argc, char** argv)
{
	int ret = 0;
	int nn_height, nn_width, nn_channel, img_width, img_height;
	det_model_type type = DET_YOLOFACE_V2;
	DetectResult resultData;

	type = g_model_type;

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


	cv::Mat frame = cv::imread(picture_path);
	if (frame.empty()) {
		cout << "Picture : "<< picture_path << " load fail" <<endl;
		det_release_model(type);
		return -1;
	}

	memcpy(amlge2d.ge2dinfo.src_info[0].vaddr[0],frame.data,amlge2d.src_size[0]);

	amlge2d.ge2dinfo.src_info[0].rect.x = 0;

	amlge2d.ge2dinfo.src_info[0].rect.y = 0;
	amlge2d.ge2dinfo.src_info[0].rect.w = amlge2d.ge2dinfo.src_info[0].canvas_w;
	amlge2d.ge2dinfo.src_info[0].rect.h = amlge2d.ge2dinfo.src_info[0].canvas_h;

	amlge2d.ge2dinfo.dst_info.rect.x = 0;
	amlge2d.ge2dinfo.dst_info.rect.y = 0;
	amlge2d.ge2dinfo.dst_info.rect.w = nn_width;
	amlge2d.ge2dinfo.dst_info.rect.h = nn_height;
	amlge2d.ge2dinfo.dst_info.rotation = GE2D_ROTATION_0;
	amlge2d.ge2dinfo.src_info[0].layer_mode = 0;
	amlge2d.ge2dinfo.src_info[0].plane_alpha = 0xff;

	ret = aml_ge2d_process(&amlge2d.ge2dinfo);
	if (ret < 0) {
		printf("aml_ge2d_process failed!\n");
		return NULL;
	}


	input_image_t image;
	image.data      = (unsigned char*)amlge2d.ge2dinfo.dst_info.vaddr[0];;
	image.width     = nn_width;
	image.height    = nn_height;
	image.channel   = 3;
	image.pixel_format = PIX_FMT_RGB888;

	cout << "Det_set_input START" << endl;
	ret = det_set_input(image, type);
	if (ret) {
		cout << "det_set_input fail. ret=" << ret << endl;
		det_release_model(type);
		return ret;
	}
	cout << "Det_set_input END" << endl;

	cout << "Det_get_result START" << endl;
	ret = det_get_result(&resultData, type);
	if (ret) {
		cout << "det_get_result fail. ret=" << ret << endl;
		det_release_model(type);
		return ret;
	}
	cout << "Det_get_result END" << endl;

	draw_results(frame, resultData, width, height, type);

	det_release_model(type);
	return ret;
}

int run_detect_facent(int argc, char** argv)
{

	int ret = 0;
	int nn_height, nn_width, nn_channel, img_width, img_height;
	det_model_type type = DET_YOLOFACE_V2;
	DetectResult resultData;

	det_set_log_config(DET_DEBUG_LEVEL_WARN,DET_LOG_TERMINAL);
	//prepare model
	ret = det_set_model(type);
	if (ret) {
		cout << "det_set_model fail. ret=" << ret <<endl;
		return ret;
	}

	ret = det_get_model_size(type, &nn_width, &nn_height, &nn_channel);
	if (ret) {
		cout << "det_get_model_size fail" <<endl;
		return ret;
	}

	cv::Mat frame = cv::imread(picture_path);
	if (frame.empty()) {
		cout << "Picture : "<< picture_path << " load fail" <<endl;
		det_release_model(type);
		return -1;
	}

	memcpy(amlge2d.ge2dinfo.src_info[0].vaddr[0],frame.data,amlge2d.src_size[0]);

	amlge2d.ge2dinfo.src_info[0].rect.x = 0;

	amlge2d.ge2dinfo.src_info[0].rect.y = 0;
	amlge2d.ge2dinfo.src_info[0].rect.w = amlge2d.ge2dinfo.src_info[0].canvas_w;
	amlge2d.ge2dinfo.src_info[0].rect.h = amlge2d.ge2dinfo.src_info[0].canvas_h;

	amlge2d.ge2dinfo.dst_info.rect.x = 0;
	amlge2d.ge2dinfo.dst_info.rect.y = 0;
	amlge2d.ge2dinfo.dst_info.rect.w = nn_width;
	amlge2d.ge2dinfo.dst_info.rect.h = nn_height;
	amlge2d.ge2dinfo.dst_info.rotation = GE2D_ROTATION_0;
	amlge2d.ge2dinfo.src_info[0].layer_mode = 0;
	amlge2d.ge2dinfo.src_info[0].plane_alpha = 0xff;

	ret = aml_ge2d_process(&amlge2d.ge2dinfo);
	if (ret < 0) {
		printf("aml_ge2d_process failed!\n");
		return NULL;
	}

	input_image_t image;
	image.data      = (unsigned char*)amlge2d.ge2dinfo.dst_info.vaddr[0];;
	image.width     = nn_width;
	image.height    = nn_height;
	image.channel   = 3;
	image.pixel_format = PIX_FMT_RGB888;

	ret = det_get_result(&resultData, type);
	if (ret) {
		cout << "det_get_result fail. ret=" << ret << endl;
		det_release_model(type);
		return ret;
	}
	det_release_model(type);

	if (resultData.detect_num == 0) {
		cout << "No face detected " <<endl;
		return -1;
	}

	cv::Mat imageROI;
	cv::Size ResImgSiz = cv::Size(160, 160);
	cv::Mat  ResImg160 = cv::Mat(ResImgSiz, imageROI.type());

	crop_face(frame, imageROI, resultData, height, width);
	cv::resize(imageROI, ResImg160, ResImgSiz, CV_INTER_NN);
	image.data		= ResImg160.data;
	image.width 	= ResImg160.cols;
	image.height 	= ResImg160.rows;
	image.channel 	= ResImg160.channels();
	image.pixel_format = PIX_FMT_RGB888;
	cv::imwrite("face_160.bmp", ResImg160);

	type = DET_FACENET;
	ret = det_set_model(type);
	if (ret) {
		cout << "det_set_model fail. ret=" << ret <<endl;
		return ret;
	}

	ret = det_get_model_size(type, &nn_width, &nn_height, &nn_channel);
	if (ret) {
		cout << "det_get_model_size fail" <<endl;
		return ret;
	}

	ret = det_set_input(image, type);
	if (ret) {
		cout << "det_set_input fail. ret=" << ret << endl;
		det_release_model(type);
		return ret;
	}

	ret = det_get_result(&resultData, type);
	if (ret) {
		cout << "det_get_result fail. ret=" << ret << endl;
		det_release_model(type);
		return ret;
	}

	FILE *fp;
	if (!facenet_falge) {
		fp = fopen("emb.db","ab+");
		if (fp == NULL)	{
			cout << "open fp out_emb fail" << endl;
			return -1;
		}
		fwrite((void *)resultData.facenet_result,128,sizeof(float),fp);
		fclose(fp);
	} else {
		#define NAME_NUM 3
		string name[NAME_NUM]={"deng.liu","jian.cao", "xxxx.xie"};

		fp = fopen("emb.db","rb");
		if (fp == NULL)	{
			cout << "open fp out_emb fail" << endl;
			return -1;
		}

		float sum = 0,temp=0, mindis = -1;
		float threshold = 0.6;
		int index =0, i, j;
		float tempbuff[128];
		float* buffer = resultData.facenet_result;

		for (i=0; i < NAME_NUM; i++)
		{
			memset(tempbuff,0,128*sizeof(float));
			fread(tempbuff,128,sizeof(float),fp);

			sum = 0;
			for (j = 0;j< 128;j++) {
				temp = tempbuff[j]-buffer[j];
				sum = sum + temp*temp;
			}
			temp = sqrt(sum);
			cout <<"i=" << i << " temp=" << temp <<endl;

			//first result
			if (i == 0) {
				mindis = temp;
				index = i;
			} else {
				if (temp < mindis ) {
					mindis = temp;
					index = i;
				}
			}
		}

		cout << "mindis:" << mindis <<", index=" << index <<endl;
		if (mindis < threshold) {
			cout <<"face detected,your id is "<< index << ", name is "<< name[index].c_str() <<endl;
		}
		fclose(fp);
	}

	det_release_model(type);
	return ret;
}

static int init_fb(void)
{
	long int i;

	printf("init_fb...\n");

    // Open the file for reading and writing
    fbfd = open("/dev/fb0", O_RDWR);
    if (!fbfd)
    {
        printf("Error: cannot open framebuffer device.\n");
        exit(1);
    }

    // Get fixed screen information
    if (ioctl(fbfd, FBIOGET_FSCREENINFO, &finfo))
    {
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
	vinfo.red.length = 0;
	vinfo.red.msb_right = 0;

	vinfo.green.offset = 8;
	vinfo.green.length = 0;
	vinfo.green.msb_right = 0;

	vinfo.blue.offset = 16;
	vinfo.blue.length = 0;
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
					   
    if (fbp == NULL)
    {
        printf("Error: failed to map framebuffer device to memory.\n");
        exit(4);
    }
	return 0;
}

int main(int argc, char** argv)
{
	int c;
	int ret;

	while ((c = getopt_long(argc, argv, "p:w:h:m:f:H", longopts, NULL)) != -1) {
		switch (c) {
			case 'p':
				picture_path = optarg;
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

			case 'f':
				facenet_falge = atoi(optarg);
				break;

			default:
				printf("%s [-p picture path] [-w width] [-h height] [-m model type] [-f facenet flag] [-H]\n", argv[0]);
				exit(1);
		}
	}


	ret = ge2d_init(width, height);
	if (ret < 0) {
		printf("ge2d_init failed!\n");
	}

	switch (g_model_type) {
		case DET_YOLOFACE_V2:
		case DET_YOLO_V2:
		case DET_YOLO_TINY:
		case DET_YOLO_V3:
		case DET_YOLO_V4:
		case DET_MTCNN_V1:
			run_detect_model(argc, argv);
			break;
		case DET_FACENET:
			run_detect_facent(argc, argv);
			break;
		default:
			cerr << "not support type=" << g_model_type <<endl;
			break;
	}
    cv::waitKey(0);
	return 0;
}
