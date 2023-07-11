## Before running this demo, please finish model_code/densenet_ctc and source_code compilation first.

## Compile demo

```sh
$ bash build_vx.sh
```

## Put model

```sh
$ cd bin_r_cv4
$ mkdir nn_data
$ cp -r {path}/densenet_ctc.nb ./nn_data
```

## Inference demo

```sh
$ ./densenet_ctc_picture -m 15 -p ../KhadasTeam.png
```