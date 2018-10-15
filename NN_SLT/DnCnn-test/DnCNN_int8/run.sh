#! /bin/sh

export VIVANTE_SDK_DIR=`pwd`
export VNN_TIME=1000
export VNN_LOOP_TIME=10000
./dncnn DnCNN.export.data inp_1_out0_1_640_640_3_nchw.tensor
