# YOLOv5 ROS
This is a ROS interface for using YOLOv5 for real time object detection on a ROS image topic. It supports inference on multiple deep learning frameworks used in the [official YOLOv5 repository](https://github.com/ultralytics/yolov5).

## Installation

### Dependencies
This package is built and tested on Ubuntu 20.04 LTS and ROS Noetic with Python 3.8.

* Clone the packages to ROS workspace and install requirement for YOLOv5 submodule:
```bash
cd <ros_workspace>/src
git clone https://github.com/mats-robotics/detection_msgs.git
git clone https://github.com/avadhmp/rsyv5.git
* Build the ROS package:
```bash
cd <ros_workspace>
catkin build rsyv5 # build the ROS package
or 
catkin_make then catkin_make install
```

* Launch the node:
```bash
roslaunch rsyv5 yolov5.launch
```
