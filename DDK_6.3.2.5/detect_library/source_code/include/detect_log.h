#ifndef __DETECT_LOG__
#define __DETECT_LOG__

#include "nn_detect_common.h"

#define DEBUG_BUFFER_LEN 1024
#define LOGE( fmt, ... ) \
    detect_nn_LogMsg(DET_DEBUG_LEVEL_ERROR, "E [%s:%d]" fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define LOGW( fmt, ... ) \
    detect_nn_LogMsg(DET_DEBUG_LEVEL_WARN,  "W [%s:%d]" fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define LOGI( fmt, ... ) \
    detect_nn_LogMsg(DET_DEBUG_LEVEL_INFO,  "I [%s:%d]" fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define LOGP( fmt, ... ) \
    detect_nn_LogMsg(DET_DEBUG_LEVEL_PROCESS, "D [%s:%d]" fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define LOGD( fmt, ... ) \
    detect_nn_LogMsg(DET_DEBUG_LEVEL_DEBUG, "D [%s:%d]" fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define __LOG__( fmt, ... ) \
    detect_nn_LogMsg(DET_DEBUG_LEVEL_DEBUG, "[%s:%d]" fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)

void detect_nn_LogMsg(det_debug_level_t level, const char *fmt, ...);
void det_set_log_level(det_debug_level_t level,det_log_format_t output_format);

#endif