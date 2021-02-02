#include <iostream>
#include <fstream>
#include <string>

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


using namespace std;
using namespace cv;

#define MODEL_WIDTH 416
#define MODEL_HEIGHT 416
#define DEFAULT_DEVICE "/dev/video0"
#define MESON_BUFFER_SIZE 4
#define DEFAULT_OUTPUT "default.h264"
#define ION_DEVICE_NODE "/dev/ion"
#define FB_DEVICE_NODE "/dev/fb0"
#define DEFAULT_BITRATE (1000000 * 5)

codec_para_t codec_context;

struct option longopts[] = {
	{ "device",         required_argument,  NULL,   'd' },
	{ "width",          required_argument,  NULL,   'w' },
	{ "height",         required_argument,  NULL,   'h' },
	{ "help",           no_argument,        NULL,   'H' },
	{ 0, 0, 0, 0 }
};

struct buffer_mapping{
	void* start;
	size_t length;
};

struct out_buffer_t {
	int index;
	int size;
	bool own_by_v4l;
	void *ptr;
	IONMEM_AllocParams buffer;
	unsigned long phy_addr;
} vbuffer[MESON_BUFFER_SIZE];

const char *device = DEFAULT_DEVICE;

#define MAX_HEIGHT 1080
#define MAX_WIDTH 1920

#define BUFFER_COUNT    4

int width = MAX_WIDTH;
int height = MAX_HEIGHT;

#define DEFAULT_FRAME_RATE  30

const size_t EXTERNAL_PTS = 0x01;
const size_t SYNC_OUTSIDE = 0x02;
const size_t USE_IDR_FRAMERATE = 0x04;
const size_t UCODE_IP_ONLY_PARAM = 0x08;
const size_t MAX_REFER_BUF = 0x10;
const size_t ERROR_RECOVERY_MODE_IN = 0x20;

int ion_fd = -1;
int fb_fd = -1;
int capture_fd = -1;
int fps = DEFAULT_FRAME_RATE;
const char *output = DEFAULT_OUTPUT;
int bitrate = DEFAULT_BITRATE;


// ge2d
aml_ge2d_t amlge2d;
aml_ge2d_t amlge2d1;

struct buffer_mapping buffer_mappings[BUFFER_COUNT];

struct amvideo_dev *amvideo;

struct fb_var_screeninfo var_info;


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


int opencv_ok = 0;

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

int open_codec(int width, int height, int fps){

	int ret;
	// Initialize the codec
	memset(&codec_context, 0, sizeof(codec_context));

	codec_context.stream_type = STREAM_TYPE_ES_VIDEO;
	codec_context.video_type = VFORMAT_MJPEG;
	codec_context.has_video = 1;
	codec_context.noblock = 0;
	codec_context.am_sysinfo.format = VIDEO_DEC_FORMAT_MJPEG;
	codec_context.am_sysinfo.width = width;
	codec_context.am_sysinfo.height = height;
	codec_context.am_sysinfo.rate = (96000.0 / fps);
	codec_context.am_sysinfo.param = (void*)(EXTERNAL_PTS | SYNC_OUTSIDE);
    
	ret = codec_init(&codec_context);
	if (ret != 0) {
		printf("codec_init failed.\n");
	}
    
	return ret;
}

void reset_codec(void){

	codec_reset(&codec_context);
}

void close_codec(void){

	codec_close(&codec_context);
}


int open_device_node(const char *path, int *pfd){

	if (NULL == path || NULL == pfd)
		return -1;

	int fd = open(path, O_RDWR);
	if (fd < 0) {
		printf("open %s failed.\n", path);
		return fd;
	}

	*pfd = fd;

	printf("open %s, fd: %d\n", path, *pfd);

	return 0;
}


void close_device_node(int fd){

	if (fd > 0)
		close(fd);
}

void set_vfm_state(void){

	amsysfs_set_sysfs_str("/sys/class/vfm/map", "rm default");
	amsysfs_set_sysfs_str("/sys/class/vfm/map", "add default decoder ionvideo");
}

void reset_vfm_state(void){

	amsysfs_set_sysfs_str("/sys/class/vfm/map", "rm default");
	amsysfs_set_sysfs_str("/sys/class/vfm/map", "add default decoder ppmgr deinterlace amvideo");
}

void free_buffers(void){

	int i;
	for (i = 0; i < MESON_BUFFER_SIZE; i++) {
		if (vbuffer[i].ptr) {
			munmap(vbuffer[i].ptr, vbuffer[i].size);
			ion_free(ion_fd, vbuffer[i].buffer.mIonHnd);
			close(vbuffer[i].buffer.mImageFd);
			close(ion_fd);
		}
	}
}


int alloc_buffers(int width, int height){

	int i = 0;
	int size = 0;
	int ret = -1;
	struct meson_phys_data phy_data;
	struct ion_custom_data custom_data;

	ion_fd = ion_mem_init();
	if (ion_fd < 0) {
		printf("ion_open failed!\n");
		goto fail;
	}
	printf("ion_fd: %d\n", ion_fd);

   	size = width * height * 3;

	for (i=0; i<MESON_BUFFER_SIZE; i++) {
		ret = ion_mem_alloc(ion_fd, size, &vbuffer[i].buffer, true);
		if (ret < 0) {
			printf("ion_mem_alloc failed\n");
			free_buffers();
			goto fail;
		}
		vbuffer[i].index = i;
		vbuffer[i].size = size;
		vbuffer[i].ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, vbuffer[i].buffer.mImageFd, 0);

		phy_data.handle = vbuffer[i].buffer.mImageFd;
		phy_data.phys_addr = 0;
		phy_data.size = 0;
		custom_data.cmd = ION_IOC_MESON_PHYS_ADDR;
		custom_data.arg = (unsigned long)&phy_data;
		ret = ioctl(ion_fd, ION_IOC_CUSTOM, (unsigned long)&custom_data);
		if (ret < 0) {
			vbuffer[i].phy_addr = 0;
			free_buffers();
			goto fail;
		} else {
			vbuffer[i].phy_addr = phy_data.phys_addr;
		}
	}

fail:
	return ret;
}

static int ionvideo_init(int width, int height){

	int i, ret;

	alloc_buffers(width, height);

	amvideo = new_amvideo(FLAGS_V4L_MODE);
	if (!amvideo) {
		printf("amvideo create failed\n");
		ret = -ENODEV;
		goto fail;
	}
	amvideo->display_mode = 0;
	amvideo->use_frame_mode = 0;

	ret = amvideo_init(amvideo, 0, width, height,
			V4L2_PIX_FMT_RGB24, MESON_BUFFER_SIZE);
	if (ret < 0) {
		printf("amvideo_init failed\n");
		amvideo_release(amvideo);
		goto fail;
	}
	ret = amvideo_start(amvideo);
	if (ret < 0) {
		amvideo_release(amvideo);
		goto fail;
	}
	for (i = 0; i < MESON_BUFFER_SIZE; i++) {
		vframebuf_t vf;
		vf.fd = vbuffer[i].buffer.mImageFd;
		vf.length = vbuffer[i].buffer.size;
		vf.index = vbuffer[i].index;
		ret = amlv4l_queuebuf(amvideo, &vf);
	}
fail:
	return ret;
}

void ionvideo_close(){

	amvideo_stop(amvideo);
	amvideo_release(amvideo);
}

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
	amlge2d.ge2dinfo.src_info[0].mem_alloc_type = AML_GE2D_MEM_INVALID;//AML_GE2D_MEM_DMABUF
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

	cvtColor(frame, frame, CV_BGR2RGB);
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

    int ret = 0;
	DetectResult resultData;
	cv::Mat img(height,width,CV_8UC3,cv::Scalar(0,0,0));

	int frames = 0;
	struct timeval time_start, time_end;
	float total_time = 0;
	vframebuf_t vf;
	uint32_t isize;

    cv::namedWindow("Image Window");

	// Create codec
	ret = open_codec(width, height, 60);
	if (ret < 0) {
		printf("open codec failed.\n");
		return NULL;
	}

	// Start streaming
	int buffer_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	ret = ioctl(capture_fd, VIDIOC_STREAMON, &buffer_type);

	if (ret < 0) {
		printf("VIDIOC_STREAMON failed.\n");
		close_codec();
		return NULL;
	}

	gettimeofday(&time_start, 0);


	while (true) {

 		struct v4l2_buffer buffer = { 0 };
 		buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
 		buffer.memory = V4L2_MEMORY_MMAP;

 		ret = ioctl(capture_fd, VIDIOC_DQBUF, &buffer);
 		if (ret < 0) {
 			printf("VIDIOC_DQBUF failed.\n");
 			return NULL;
 		}

		isize = 0;
 		do {
 			ret = amlv4l_dequeuebuf(amvideo, &vf);
 			if (ret >= 0) {
 				//printf("vf idx%d pts 0x%x\n", vf.index, vf.pts);
 				ret = amlv4l_queuebuf(amvideo, &vf);
 				if (ret < 0) {
 					//printf("amlv4l_queuebuf %d\n", ret);
				}
			} else {
 				//printf("amlv4l_dequeuebuf %d\n", ret);
 				//continue;
 			}

 			ret = codec_write(&codec_context, (unsigned char*)buffer_mappings[buffer.index].start + isize, buffer.bytesused);
 			if (ret < 0) {
 				if (errno != EAGAIN) {
 					printf("write data failed, errno %d\n", errno);
 					return NULL;
				} else {
 					continue;
 				}
 			} else {
 				isize += ret;
 			}
 		} while (isize < buffer.bytesused);


 		amlge2d.ge2dinfo.src_info[0].offset[0] = vbuffer[vf.index].phy_addr;

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

 		// return buffer
 		ret = ioctl(capture_fd, VIDIOC_QBUF, &buffer);
 		if (ret < 0) {
 			printf("VIDIOC_QBUF failed.\n");
 			close_codec();
 			return NULL;
 		}

		// Syncronize the destination data
		ret = ion_sync_fd(ion_fd, vbuffer[vf.index].buffer.mImageFd);
		if (ret != 0) {
 			printf("ion_sync_fd failed.\n");
			return NULL;
 		}

 		img.data = (unsigned char *)vbuffer[vf.index].ptr;

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

		draw_results(img, resultData, width, height, g_model_type);
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


	run_detect_model(g_model_type);

	ret = open_device_node(device, &capture_fd);
	if (ret < 0) {
		printf("capture device open failed.\n");
		exit(1);
	}

	// Apply capture settings
	struct v4l2_format format = { 0 };
	format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	format.fmt.pix.width = width;
	format.fmt.pix.height = height;
	format.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
	format.fmt.pix.field = V4L2_FIELD_ANY;

	ret = ioctl(capture_fd, VIDIOC_S_FMT, &format);
	if (ret < 0) {
		printf("VIDIOC_S_FMT failed.\n");
		close_device_node(capture_fd);
		exit(1);
	}

	printf("v4l2_format: width=%d, height=%d, pixelformat=0x%x\n",
			format.fmt.pix.width, format.fmt.pix.height, format.fmt.pix.pixelformat);

	// Readback device selected settings
	width = format.fmt.pix.width;
	height = format.fmt.pix.height;

	struct v4l2_streamparm stream_parm = { 0 };
	stream_parm.type = format.type;
	stream_parm.parm.capture.timeperframe.numerator = 1;
	stream_parm.parm.capture.timeperframe.denominator = fps;

	ret = ioctl(capture_fd, VIDIOC_S_PARM, &stream_parm);
	if (ret < 0) {
		printf("VIDIOC_S_PARM failed.\n");
		close_device_node(capture_fd);
		exit(1);
	}

	printf("capture.timeperframe: numerator=%d, denominator=%d\n",
			stream_parm.parm.capture.timeperframe.numerator,
			stream_parm.parm.capture.timeperframe.denominator);

	// Request buffers
	struct v4l2_requestbuffers request_buffers = { 0 };
	request_buffers.count = BUFFER_COUNT;
	request_buffers.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	request_buffers.memory = V4L2_MEMORY_MMAP;

	ret = ioctl(capture_fd, VIDIOC_REQBUFS, &request_buffers);
	if (ret < 0) {
		printf("VIDIOC_REQBUFS failed.\n");
		close_device_node(capture_fd);
		exit(1);
	}

	for (int i = 0; i < (int)request_buffers.count; ++i) {
		struct v4l2_buffer buffer = { 0 };
		buffer.type = request_buffers.type;
		buffer.memory = V4L2_MEMORY_MMAP;
		buffer.index = i;

		ret = ioctl(capture_fd, VIDIOC_QUERYBUF, &buffer);
		if (ret < 0) {
			printf("VIDIOC_QUERYBUF failed.\n");
			close_device_node(capture_fd);
			exit(1);
		}

		buffer_mappings[i].length = buffer.length;
		buffer_mappings[i].start = mmap(NULL, buffer.length,
				PROT_READ | PROT_WRITE, /* recommended */
				MAP_SHARED,             /* recommended */
				capture_fd, buffer.m.offset);
	}

	// Queue buffers
	for (int i = 0; i < (int)request_buffers.count; ++i) {
		struct v4l2_buffer buffer = { 0 };
		buffer.index = i;
		buffer.type = request_buffers.type;
		buffer.memory = request_buffers.memory;

		ret = ioctl(capture_fd, VIDIOC_QBUF, &buffer);
		if (ret < 0) {
			printf("VIDIOC_QBUF failed.\n");
			close_device_node(capture_fd);
			exit(1);
		}
	}

	set_vfm_state();

	ret = ionvideo_init(width, height);
	if (ret < 0) {
		printf("ionvideo_init failed!\n");
		close_device_node(capture_fd);
		exit(1);
	}

	ret = ge2d_init(width, height);
	if (ret < 0) {
		printf("ge2d_init failed!\n");
		goto close_ionvideo;
	}


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


	close_device_node(fb_fd);

	ge2d_destroy();

close_ionvideo:
	ionvideo_close();
	free_buffers();

	close_device_node(capture_fd);

	reset_vfm_state();

	printf("main exit ..\n");

	return 0;
}
