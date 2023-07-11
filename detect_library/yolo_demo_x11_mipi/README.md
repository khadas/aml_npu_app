## Before running this demo, please finish model_code/detect_yolo_xxx and source_code compilation first.

x11 means to run on the linux Graphical User Interfacee and mipi means to use mipi camera.
This demo can run yoloface, yolov2, yolov3, yolov3_tiny, yolov4.

## Compile demo

```sh
$ bash build_vx.sh
```

## Put model

```sh
$ cd bin_r_cv4
$ mkdir nn_data
# yoloface
$ cp -r {path}/yolo_face_88.nb ./nn_data
# yolov2
$ cp -r {path}/yolov2_88.nb ./nn_data
# yolov3
$ cp -r {path}/yolov3_88.nb ./nn_data
# yolov3_tiny
$ cp -r {path}/yolotiny_88.nb ./nn_data
# yolov4
$ cp -r {path}/yolov4_88.nb ./nn_data
```

## Inference demo

```sh
# yoloface
$ ./detect_demo_x11_usb -m 0 -d /dev/video0
# yolov2
$ ./detect_demo_x11_usb -m 1 -d /dev/video0
# yolov3
$ ./detect_demo_x11_usb -m 2 -d /dev/video0
# yolov3_tiny
$ ./detect_demo_x11_usb -m 3 -d /dev/video0
# yolov4
$ ./detect_demo_x11_usb -m 4 -d /dev/video0
```