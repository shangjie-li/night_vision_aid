import cv2
import rospy
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import Image


def get_stamp(header):
    return header.stamp.secs + 0.000000001 * header.stamp.nsecs


def publish_image(pub, data, frame_id='base_link'):
    assert len(data.shape) == 3, 'len(data.shape) must be equal to 3.'
    header = Header(stamp=rospy.Time.now())
    header.frame_id = frame_id

    msg = Image()
    msg.height = data.shape[0]
    msg.width = data.shape[1]
    msg.encoding = 'rgb8'
    msg.data = np.array(data).tostring()
    msg.header = header
    msg.step = msg.width * 1 * 3

    pub.publish(msg)


def display(img, v_writer, win_name='result'):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    v_writer.write(img)
    key = cv2.waitKey(1)
    if key == 27:
        v_writer.release()
        return False
    else:
        return True


def print_info(frame, stamp, delay, labels, scores, boxes, locs=None, file_name='result.txt'):
    time_str = 'frame:%d  stamp:%.3f  delay:%.3f' % (frame, stamp, delay)
    print(time_str)
    with open(file_name, 'a') as fob:
        fob.write(time_str + '\n')
    for i in range(len(labels)):
        info_str = 'box:%d %d %d %d  score:%.2f  label:%s' % (
            boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i], labels[i]
        )
        if locs is not None:
            info_str += '  loc:(%.2f, %.2f)' % (locs[i][0], locs[i][1])
        print(info_str)
        with open(file_name, 'a') as fob:
            fob.write(info_str + '\n')
    print()
    with open(file_name, 'a') as fob:
        fob.write('\n')
