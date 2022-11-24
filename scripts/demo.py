# -*- coding: UTF-8 -*-

import os
import sys
import rospy
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import threading
import argparse

from std_msgs.msg import Header
from sensor_msgs.msg import Image

try:
    import cv2
except ImportError:
    import sys

    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from yolact_detector import YolactDetector, draw_contour, draw_mask
from functions import get_stamp, publish_image
from functions import display, print_info


parser = argparse.ArgumentParser(
    description='Demo script for night vision aid')
parser.add_argument('--print', action='store_true',
                    help='Whether to print and record infos.')
parser.add_argument('--sub_image1', default='/pub_rgb', type=str,
                    help='The image topic to subscribe.')
parser.add_argument('--sub_image2', default='/pub_t', type=str,
                    help='The image topic to subscribe.')
parser.add_argument('--pub_image', default='/result', type=str,
                    help='The image topic to publish.')
parser.add_argument('--frame_rate', default=10, type=int,
                    help='Working frequency.')
parser.add_argument('--display', action='store_true',
                    help='Whether to display and save all videos.')
parser.add_argument('--draw_mask', action='store_true',
                    help='Whether to draw the mask.')
args = parser.parse_args()


image1_lock = threading.Lock()
image2_lock = threading.Lock()


def image1_callback(image):
    global image1_stamp, image1_frame
    image1_lock.acquire()
    image1_stamp = get_stamp(image.header)
    image1_frame = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)  # BGR image
    image1_lock.release()


def image2_callback(image):
    global image2_stamp, image2_frame
    image2_lock.acquire()
    image2_stamp = get_stamp(image.header)
    image2_frame = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)  # BGR image
    image2_lock.release()


def timer_callback(event):
    global image1_stamp, image1_frame
    image1_lock.acquire()
    cur_stamp1 = image1_stamp
    cur_frame1 = image1_frame.copy()
    # cur_frame1 = cur_frame1[:, :, ::-1].copy()  # to RGB
    image1_lock.release()

    global image2_stamp, image2_frame
    image2_lock.acquire()
    cur_stamp2 = image2_stamp
    cur_frame2 = image2_frame.copy()
    # cur_frame2 = cur_frame2[:, :, ::-1].copy()  # to RGB
    image2_lock.release()

    global frame
    frame += 1
    start = time.time()
    masks, labels, scores, boxes = detector.run(
        cur_frame2, score_threshold=0.1, top_k=20, class_filter=[0, 1, 2, 3, 4]
    )  # pedestrian, cyclist, car, bus, truck
    labels_temp = labels.copy()
    labels = []
    for i in labels_temp:
        labels.append(i if i not in ['pedestrian', 'cyclist'] else 'person')

    cur_frame1 = cur_frame1[:, :, ::-1].copy()  # to BGR
    for i in np.argsort(scores):
        color = colors[labels[i]][::-1]  # to BGR
        cur_frame1 = draw_contour(cur_frame1, masks[i], color)
        if args.draw_mask:
            cur_frame1 = draw_mask(cur_frame1, masks[i], color)
    result_frame = cur_frame1

    if args.display:
        if not display(result_frame, v_writer, win_name='result'):
            print("\nReceived the shutdown signal.\n")
            rospy.signal_shutdown("Everything is over now.")
    result_frame = result_frame[:, :, ::-1]  # to RGB
    publish_image(pub, result_frame)
    delay = round(time.time() - start, 3)

    if args.print:
        print_info(frame, cur_stamp1, delay, labels, scores, boxes, locs=None, file_name=file_name)


if __name__ == '__main__':
    # 初始化节点
    rospy.init_node("night_vision_aid", anonymous=True, disable_signals=True)
    frame = 0

    # 定义颜色
    colors = {
        'person': (255, 255, 255),
        'car': (244, 67, 54),
        'bus': (0, 188, 212),
        'truck': (255, 235, 59),
    }  # RGB colors

    # 记录时间戳和检测结果
    if args.print:
        file_name = 'result.txt'
        with open(file_name, 'w') as fob:
            fob.seek(0)
            fob.truncate()

    # 初始化YolactDetector
    detector = YolactDetector(
        trained_model='weights/seumm_lwir_15200/yolact_resnet50_157_200000.pth',
        dataset='seumm_lwir_15200_dataset'
    )

    # 准备图像序列
    image1_stamp, image1_frame = None, None
    image2_stamp, image2_frame = None, None
    rospy.Subscriber(args.sub_image1, Image, image1_callback, queue_size=1,
                     buff_size=52428800)
    rospy.Subscriber(args.sub_image2, Image, image2_callback, queue_size=1,
                     buff_size=52428800)
    while image1_frame is None or image2_frame is None:
        time.sleep(0.1)
        print('Waiting for topic %s and %s...' % (args.sub_image1, args.sub_image2))
    print('  Done.\n')

    # 保存视频
    if args.display:
        assert image1_frame.shape == image2_frame.shape, \
            'image1_frame.shape must be equal to image2_frame.shape.'
        win_h, win_w = image1_frame.shape[0], image1_frame.shape[1]
        v_path = 'result.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_writer = cv2.VideoWriter(v_path, v_format, args.frame_rate, (win_w, win_h), True)

    # 启动定时检测线程
    pub = rospy.Publisher(args.pub_image, Image, queue_size=1)
    rospy.Timer(rospy.Duration(1 / args.frame_rate), timer_callback)

    # 与C++的spin不同，rospy.spin()的作用是当节点停止时让python程序退出
    rospy.spin()
