
×××××××××××××××执行DNCNN×××××××××××××××××
1.切换到DnCnn-test的目录
 命令：cd /.../DnCnn-test/DnCNN_int8

2.运行run.sh
 命令：./run.sh

3.编译DnCnn
 命令： ./build_vx.sh {drivers_dir} {toolchain} 
 (drivers_dir和toolchain两个参数根据自己的drivers和toolchain目录决定)
*************说明×××××××××××××××××××××××××
tip:

1. run.sh脚本中“VNN_TIME”对应的timeout时间，默认值为1000，代表当模型的processGraph的时间超过1000ms时，认为是运行异常，则停止运行。

2. run.sh脚本中“VNN_LOOP_TIME”对应模型循环次数，默认值为10000

3. 如果DNCnn完整的运行返回值为0，若未能完全运行，则返回值为1


×××××××××××××××执行时间×××××××××××××××××××
1.网络的启动时间包括createNetwork（3000ms）、verify（500ms）和processGraph 以及判断时间，四个部分。其中createNetwork、verify只在网络运行开始执行一次；processGraph和判断时间与VNN_LOOP_TIME有关

2.DnCnn的processGraph平均在110ms左右，判断时间大约1140ms
  其中：判断时间（1140ms）= 输出output.txt(1080ms) + md5（60ms）

3. 以VNN_LOOP_TIME=10为例，网络运行总时间为
  createNetwork（3000ms）+ verify（500ms）+【processGraph（110ms）+判断时间（1140ms）】×10 = 16000ms （至少需16s）



 


