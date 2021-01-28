#ifndef __CAMERA_H__
#define __CAMERA_H__


int open_device_node(const char *path, int *pfd);
void close_device_node(int fd);
void set_vfm_state(void);
void reset_vfm_state(void);
int alloc_buffers(int width, int height);
static int ionvideo_init(int width, int height);

#endif
