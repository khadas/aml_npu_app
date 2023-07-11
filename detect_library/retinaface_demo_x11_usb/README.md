## Before running this demo, please finish model_code/detect_retinaface and source_code compilation first.

## Compile demo

```sh
$ bash build_vx.sh
```

## Put model

```sh
$ cd bin_r_cv4
$ mkdir nn_data
$ cp -r {path}/retinaface.nb ./nn_data
```

## Inference demo

```sh
$ ./retinaface -m 16 -d /dev/video1
```