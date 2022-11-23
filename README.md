# night_vision_aid

ROS package for night vision aid (thermal -> rgb)

## 安装
 - 建立ROS工作空间并拷贝这个库
   ```Shell
   mkdir -p ros_ws/src
   cd ros_ws/src
   git clone git@github.com:shangjie-li/night_vision_aid.git --recursive
   cd ..
   catkin_make
   ```
 - 使用Anaconda设置环境依赖
   ```Shell
   conda create -n yolact-env python=3.6.9
   conda activate yolact-env
   conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
   pip install cython
   pip install opencv-python pillow pycocotools matplotlib
   pip install -U scikit-learn
   pip install catkin_tools
   pip install rospkg
   ```
 - 准备模型文件，并保存至目录`night_vision_aid/modules/yolact-test/weights/seumm_lwir/`

## 运行
 - 启动算法（检测结果图像可由`rqt_image_view /result`查看）
   ```
   python3 demo.py
   
   # If you want print infos and save videos, run
   python3 demo.py --print --display
   ```
