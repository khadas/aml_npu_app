## Before running this demo, please finish model_code/detect_yolov8n and source_code compilation first.

## Compile demo

```sh
$ bash build_vx.sh
```

## Put model

```sh
$ cd bin_r_cv4
$ mkdir nn_data
$ cp -r {path}/yolov8n.nb ./nn_data
```

## Inference demo

```sh
$ ./detect_demo_x11_usb -m 14 -d /dev/video1
```