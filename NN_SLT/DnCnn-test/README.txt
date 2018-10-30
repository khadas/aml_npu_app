×××××××××××××××编译DNCNN×××××××××××××××××
1. 编译
   运行build_vx.sh脚本：./build_vx.sh {bildroot的根目录} {工程名称}
   
   buildroot的根目录:代表与bootloader buildroot hardware output等文件夹所在的目录
   工程名称：代表所编译的工程，目前所用的工程为mesong12b_skt_release

2. 注意事项
   编译成功后会在当前目录下生成一个bin_r文件夹，文件夹里面包含dncnn执行文件和若干对象文件（*.o）,需把dncnn拷贝到和run.sh同一目录下
×××××××××××××××执行DNCNN×××××××××××××××××
0.程序必须保证dncnn、DnCNN.export.data、inp_1_out0_1_640_640_3_nchw.tensor、output.txt、run.sh五个文件在同一目录下才能运行

1.运行run.sh
  命令：./run.sh  

2. run.sh 中运行的是dncnn程序，需要输入四个参数：./dncnn DnCNN.export.data inp_1_out0_1_640_640_3_nchw.tensor 1000 10
  dncnn：代表可执行程序
  DnCNN.export.data：程序的第一个参数，表示网络的参数
  inp_1_out0_1_640_640_3_nchw.tensor：程序的第二个参数，是网络的输入
  1000 ： 程序的第三个参数，代表网络执行时间的阈值，默认值为1000，代表当模型的processGraph的时间超过1000ms时，认为是运行异常
  10 ： 程序的第四个参数，代表网络的循环次数，默认值为10，（循环10次，程序运行总时间约为16s）
  tips：第三、四个参数可根据需要自主选择

3. 如果DNCnn完整的运行返回值为0，若未能完全运行，则返回值为1

4. 若运行失败，失败信息会保存在error_log.txt中

×××××××××××××××执行时间×××××××××××××××××××
1.网络的启动时间包括createNetwork（3000ms）、verify（500ms）和processGraph 以及判断时间，四个部分。其中createNetwork、verify只在网络运行开始执行一次；processGraph和判断时间与VNN_LOOP_TIME有关

2.DnCnn的processGraph平均在110ms左右，判断时间大约1140ms
  其中：判断时间（1140ms）= 输出output.txt(1080ms) + md5（60ms）

3. 以VNN_LOOP_TIME=10为例，网络运行总时间为
  createNetwork（3000ms）+ verify（500ms）+【processGraph（110ms）+判断时间（1140ms）】×10 = 16000ms （至少需16s）



 


