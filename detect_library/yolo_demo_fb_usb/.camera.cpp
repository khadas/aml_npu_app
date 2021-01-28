#include <sys/stat.h>         
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <getopt.h>
#include <linux/videodev2.h>
#include <sys/mman.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <pthread.h>

#include <codec.h>
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
#include <amvideo.h>

#define MESON_BUFFER_SIZE 4

int open_device_node(const char *path, int *pfd)
{
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


void close_device_node(int fd)
{
    if (fd > 0)
        close(fd);
}

void set_vfm_state(void)                                                                          
{
	amsysfs_set_sysfs_str("/sys/class/vfm/map", "rm default");
	amsysfs_set_sysfs_str("/sys/class/vfm/map", "add default decoder ionvideo");
}

void reset_vfm_state(void)
{
	amsysfs_set_sysfs_str("/sys/class/vfm/map", "rm default");
	amsysfs_set_sysfs_str("/sys/class/vfm/map", "add default decoder ppmgr deinterlace amvideo");
}

void free_buffers(void)
{
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


int alloc_buffers(int width, int height)
{
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


static int ionvideo_init(int width, int height)
{
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



