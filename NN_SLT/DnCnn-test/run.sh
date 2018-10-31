#! /bin/sh
path=$(cd `dirname $0`;pwd)
cd $path
export VSI_NN_LOG_LEVEL=0
./dncnn DnCNN.export.data inp_1_out0_1_640_640_3_nchw.tensor 1000 10
