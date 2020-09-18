#include <iostream>
#include <fstream>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

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

#include "nn_detect.h"
#include "nn_detect_utils.h"



using namespace std;
using namespace cv;

#define BUFFER_COUNT 6
#define MAX_HEIGHT  1920
#define MAX_WIDTH   1080

typedef struct __video_buffer
{
	void *start;
	size_t length;

}video_buf_t;

struct  Frame
{   
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
extern char *video_device;

pthread_mutex_t mutex4q;

unsigned char *displaybuf;
int g_nn_height, g_nn_width, g_nn_channel;
det_model_type g_model_type;
static unsigned int tmpVal;

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

static void draw_results(IplImage *pImg, DetectResult resultData, int img_width, int img_height, det_model_type type)
{
	int i = 0;
	float left, right, top, bottom;
	CvFont font;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1, 1,0,3,8);

//	cout << "\nresultData.detect_num=" << resultData.detect_num <<endl;
//	cout << "result type is " << resultData.point[i].type << endl;
	for (i = 0; i < resultData.detect_num; i++) {
		left =  resultData.point[i].point.rectPoint.left*img_width;
        right = resultData.point[i].point.rectPoint.right*img_width;
        top = resultData.point[i].point.rectPoint.top*img_height;
        bottom = resultData.point[i].point.rectPoint.bottom*img_height;
#if 0
		#define XRATE 3.0
		#define YRATE 2.25
		cout << "i:" <<resultData.detect_num <<" left:" << resultData.point[i].point.rectPoint.left <<" right:" << resultData.point[i].point.rectPoint.right ;
        cout << " top:" << resultData.point[i].point.rectPoint.top << " bottom:" << resultData.point[i].point.rectPoint.bottom <<endl;
        left =  resultData.point[i].point.rectPoint.left*XRATE;
        right = resultData.point[i].point.rectPoint.right*XRATE;
        top = resultData.point[i].point.rectPoint.top*YRATE;
        bottom = resultData.point[i].point.rectPoint.bottom*YRATE;
#endif
//		cout << "i:" <<resultData.detect_num <<" left:" << left <<" right:" << right << " top:" << top << " bottom:" << bottom <<endl;
		CvPoint pt1;
		CvPoint pt2;
		pt1=cvPoint(left,top);
		pt2=cvPoint(right, bottom);

		cvRectangle(pImg,pt1,pt2,CV_RGB(10,10,250),3,4,0);
		switch (type) {
			case DET_YOLOFACE_V2:
			break;
			case DET_MTCNN_V1:
			{
				int j = 0;
				cv::Mat testImage;
				testImage = cv::cvarrToMat(pImg);
				for (j = 0; j < 5; j ++) {
					cv::circle(testImage, cv::Point(resultData.point[i].tpts.floatX[j]*img_width, resultData.point[i].tpts.floatY[j]*img_height), 2, cv::Scalar(0, 255, 255), 2);
				}
				break;
			}
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
				cvPutText(pImg, resultData.result_name[i].lable_name, cvPoint(left,top-10), &font, CV_RGB(0,255,0));
				break;
			}
			default:
			break;
		}
	}

	if(pingflag == 0)
	{
		memcpy(fbp+1920*1080*3,pImg->imageData,1920*1080*3);
		vinfo.activate = FB_ACTIVATE_NOW;
		vinfo.vmode &= ~FB_VMODE_YWRAP;
		vinfo.yoffset = 1080;
		pingflag = 1;
		ioctl(fbfd, FBIOPAN_DISPLAY, &vinfo);
	}
	else
	{
		memcpy(fbp,pImg->imageData,1920*1080*3);
		vinfo.activate = FB_ACTIVATE_NOW;
		vinfo.vmode &= ~FB_VMODE_YWRAP;
		vinfo.yoffset = 0;
		pingflag = 0;
		ioctl(fbfd, FBIOPAN_DISPLAY, &vinfo);
	}
}

static void crop_face(cv::Mat sourceFrame, cv::Mat& imageROI, DetectResult resultData, int img_height, int img_width) {
	float left, right, top, bottom;
	int tempw,temph;
	CvFont font;

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


int run_detect_model(det_model_type type)
{
	int ret = 0;
	int nn_height, nn_width, nn_channel, img_width, img_height;
	DetectResult resultData;

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

#if 0
	//cv::Mat testImage;
	IplImage* frame2process = cvLoadImage(picture_path,CV_LOAD_IMAGE_COLOR);
	if (!frame2process) {
		cout << "Picture : "<< picture_path << " load fail" <<endl;
		det_release_model(type);
		return -1;
	}


	cv::Mat testImage(nn_height,nn_width,CV_8UC1);
	cv::Mat sourceFrame = cvarrToMat(frame2process);
	cv::resize(sourceFrame, testImage, testImage.size());
	img_width = sourceFrame.cols;
	img_height = sourceFrame.rows;

	printf("img_width: %d, img_height: %d\n", img_width, img_height);

	input_image_t image;
	image.data		= testImage.data;
	image.width 	= testImage.cols;
	image.height 	= testImage.rows;
	image.channel 	= testImage.channels();
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

	draw_results(frame2process, resultData, img_width, img_height, type);
	cv::imwrite("output.bmp", sourceFrame);

	det_release_model(type);
	cvReleaseImage(&frame2process);
#endif
	return ret;
}

int run_detect_facent(int argc, char** argv)
{
	if (argc != 4) {
		cout << "input param error" <<endl;
		cout << "Usage: " << argv[0] << " type  picture_path facenet_falge"<<endl;
		cout << "facenet_falge: 0-->write to emb.db, 1--> facenet inference" <<endl;
		return -1;
	}

	int ret = 0;
	int nn_height, nn_width, nn_channel, img_width, img_height;
	det_model_type type = DET_YOLOFACE_V2;
	DetectResult resultData;

	char* picture_path = argv[2];
	int facenet_falge =(int)atoi(argv[3]);

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

	IplImage* frame2process = cvLoadImage(picture_path,CV_LOAD_IMAGE_COLOR);
	if (!frame2process) {
		cout << "Picture : "<< picture_path << " load fail" <<endl;
		det_release_model(type);
		return -1;
	}

	cv::Mat testImage(nn_height,nn_width,CV_8UC1);
	cv::Mat sourceFrame = cvarrToMat(frame2process);
	cv::resize(sourceFrame, testImage, testImage.size());
	img_width = sourceFrame.cols;
	img_height = sourceFrame.rows;

	input_image_t image;
	image.data		= testImage.data;
	image.width 	= testImage.cols;
	image.height 	= testImage.rows;
	image.channel 	= testImage.channels();
	image.pixel_format = PIX_FMT_RGB888;
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
	det_release_model(type);

	if (resultData.detect_num == 0) {
		cout << "No face detected " <<endl;
		return -1;
	}

	cv::Mat imageROI;
	cv::Size ResImgSiz = cv::Size(160, 160);
	cv::Mat  ResImg160 = cv::Mat(ResImgSiz, imageROI.type());

	crop_face(sourceFrame, imageROI, resultData, img_height, img_width);
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
	cvReleaseImage(&frame2process);
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

    if (fbp == NULL)
    {
        printf("Error: failed to map framebuffer device to memory.\n");
        exit(4);
    }
	return 0;
}


static void *thread_func(void *x)
{
    IplImage *frame2process = NULL,*frameclone = NULL;
    int width , height; 
    bool bFrameReady = false;
    int i = 0,ret = 0;
    FILE *tfd;
    struct timeval tmsStart, tmsEnd;
	DetectResult resultData;

	cv::Mat yolo_v2Image(g_nn_width, g_nn_width, CV_8UC3);

    setpriority(PRIO_PROCESS, pthread_self(), -15);

   while (true) {
        pthread_mutex_lock(&mutex4q);
        if(opencv_ok == 1)
        {
            memcpy(frame2process->imageData,displaybuf,1920*1080*3);
        }
        else
        {
            if(frame2process == NULL)
                frame2process = cvCreateImage(cvSize(1920, 1080), IPL_DEPTH_8U, 3);
            if (frame2process == NULL)
            {
                pthread_mutex_unlock(&mutex4q);
                usleep(100000);
                //printf("can't load temp bmp in thread to parse\n");
                continue;
                }
            if(frame2process->width != 1920)
            {
                printf("read image not 1920 width\n");
                pthread_mutex_unlock(&mutex4q);
                continue;
            }
            printf("prepare 1080p image ok\n");
            opencv_ok = 1;   //zxw
        }
        pthread_mutex_unlock(&mutex4q);


        gettimeofday(&tmsStart, 0);

		cv::Mat sourceFrame = cvarrToMat(frame2process);
//		cv::Mat sourceFrame = frame2process;
        cv::resize(sourceFrame, yolo_v2Image, yolo_v2Image.size());
		int img_width = sourceFrame.cols;
		int img_height = sourceFrame.rows;

		gettimeofday(&tmsEnd, 0);
		tmpVal = 1000 * (tmsEnd.tv_sec - tmsStart.tv_sec) + (tmsEnd.tv_usec - tmsStart.tv_usec) / 1000;

		gettimeofday(&tmsStart, 0);
		input_image_t image;
		image.data      = yolo_v2Image.data;
		image.width     = yolo_v2Image.cols;
		image.height    = yolo_v2Image.rows;
		image.channel   = yolo_v2Image.channels();
		image.pixel_format = PIX_FMT_RGB888;

//		cout << "Det_set_input START" << endl;
		ret = det_set_input(image, g_model_type);
		if (ret) {
			cout << "det_set_input fail. ret=" << ret << endl;
			det_release_model(g_model_type);
			goto out;
		}
//		cout << "Det_set_input END" << endl;

//		cout << "Det_get_result START" << endl;
		ret = det_get_result(&resultData, g_model_type);
		if (ret) {
			cout << "det_get_result fail. ret=" << ret << endl;
			det_release_model(g_model_type);
			goto out;
		}
//		cout << "Det_get_result END" << endl;

		draw_results(frame2process, resultData, img_width, img_height, g_model_type);
        gettimeofday(&tmsEnd, 0);
        tmpVal = 1000 * (tmsEnd.tv_sec - tmsStart.tv_sec) + (tmsEnd.tv_usec - tmsStart.tv_usec) / 1000;
        if(tmpVal < 56)
        printf("FPS:%d\n",1000/(tmpVal+8));
        sourceFrame.release();
    }
out:
	printf("thread_func exit\n");
}

static void usage(char **argv)
{
	cout << "input param error" <<endl;
	cout << "Usage: " << argv[0] << " <video device> <type> <ir-cut>"<<endl;
	cout << "       video device:"<<endl;
	cout << "       /dev/videoX\n"<<endl;
	cout << "       type: " <<endl;
	cout << "       0 - Yoloface"<<endl;
	cout << "       1 - YoloV2"<<endl;
	cout << "       2 - YoloV3"<<endl;
	cout << "       3 - YoloV3" << endl;
	cout << "       ir-cut: " <<endl;
	cout << "       on - enable IR-CUT"<<endl;
	cout << "       off - disable IR-CUT"<<endl;
}

int main(int argc, char** argv)
{	
	int i;
	pthread_t tid[2];
	det_model_type type;
	if (argc < 4) {
		usage(argv);
		return -1;
	}

	system(xcmd);
	init_fb();

	video_device = argv[1];
	type = (det_model_type)atoi(argv[2]);
	g_model_type = type;
	if (0 == strcmp(argv[3], "on")) {
		ir_cut_state = 1;
	} else if (0 == strcmp(argv[3], "off")) {
		ir_cut_state = 0;
	} else {
		printf("Error: IR-CUT set error!\n");
		usage(argv);
		return -1;
	}
	run_detect_model(type);

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
		for(i=0; i<sizeof(tid)/sizeof(tid[0]); i++)
		{
			pthread_join(tid[i], NULL);
		}
		sleep(1);
	}


	return 0;
}