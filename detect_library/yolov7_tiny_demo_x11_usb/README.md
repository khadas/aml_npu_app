## Before running this demo, please finish model_code/detect_yolo_v7_tiny and source_code compilation first.

## Compile demo

```sh
$ bash build_vx.sh
```

## Put model

```sh
$ cd bin_r_cv4
$ mkdir nn_data
$ cp -r {path}/yolov7_tiny.nb ./nn_data
```

## Inference demo

```sh
$ ./detect_demo_x11_usb -m 13 -d /dev/video1
```