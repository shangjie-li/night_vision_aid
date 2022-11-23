# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt  # For WARNING: QApplication was not created in the main() thread.

import os
import sys

cwd = os.getcwd().rstrip('scripts')
sys.path.append(os.path.join(cwd, 'modules/yolact-test'))

try:
    import cv2
except ImportError:
    import sys

    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

import time
import numpy as np
from numpy import random

import torch
import torch.backends.cudnn as cudnn

from yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess
from data import cfg, set_cfg, set_dataset


def create_random_color():
    r = np.random.randint(0, 255)
    g = np.random.randint(0, 255)
    b = np.random.randint(0, 255)
    color = (r, g, b)

    return color


def draw_mask(img, mask, color):
    """
    Args:
        img: numpy.ndarray, (h, w, 3), BGR format
        mask: torch.Tensor, (h, w)
        color: tuple
    Returns:
        img_numpy: numpy.ndarray, (h, w, 3), BGR format
    """
    img_gpu = torch.from_numpy(img).cuda().float()
    img_gpu = img_gpu / 255.0

    mask = mask[:, :, None]
    color_tensor = torch.Tensor(color).to(img_gpu.device.index).float() / 255.
    alpha = 0.45
    mask_color = mask.repeat(1, 1, 3) * color_tensor * alpha
    inv_alph_mask = mask * (- alpha) + 1
    img_gpu = img_gpu * inv_alph_mask + mask_color
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    return img_numpy


def draw_segmentation_result(img, mask, label, score, box, color):
    """
    Args:
        img: numpy.ndarray, (h, w, 3), BGR format
        mask: torch.Tensor, (h, w)
        label: str
        score: float
        box: numpy.ndarray, (4,), xyxy format
        color: tuple
    Returns:
        img: numpy.ndarray, (h, w, 3), BGR format
    """
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5
    font_thickness, line_thickness = 1, 2

    x1, y1, x2, y2 = box.squeeze()
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness)
    img = draw_mask(img, mask, color)

    u, v = x1, y1
    text_str = '%s: %.2f' % (label, score)
    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
    cv2.rectangle(img, (u, v), (u + text_w, v - text_h - 4), color, -1)
    cv2.putText(img, text_str, (u, v - 3), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return img


class YolactDetector():
    def __init__(self, trained_model, dataset=None):
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        trained_model = os.path.join(cwd, 'modules/yolact-test', trained_model)
        pth = SavePath.from_str(trained_model)
        config = pth.model_name + '_config'
        set_cfg(config)

        if dataset is not None:
            set_dataset(dataset)

        self.net = Yolact()
        self.net.load_weights(trained_model)
        self.net.eval()
        self.net = self.net.cuda()
        self.net.detect.use_fast_nms = True
        self.net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False

    def run(self, img, score_threshold=0.3, top_k=20, class_filter=None):
        """
        Args:
            img: (h, w, 3), RGB format
            score_threshold: float, object confidence threshold
            top_k: int, number of objects for detection
            class_filter: list(int), filter by class, for instance [0, 2, 3]
        Returns:
            masks: torch.Tensor, (n, h, w)
            labels: list(str), names of objects
            scores: list(float)
            boxes: numpy.ndarray, (n, 4), xyxy format
        """
        frame = torch.from_numpy(img).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = self.net(batch)

        with torch.no_grad():
            h, w, _ = frame.shape
            with timer.env('Postprocess'):
                save = cfg.rescore_bbox
                cfg.rescore_bbox = True
                t = postprocess(preds, w, h, visualize_lincomb=False, crop_masks=True,
                                score_threshold=score_threshold)
                cfg.rescore_bbox = save

            with timer.env('Copy'):
                idx = t[1].argsort(0, descending=True)[:top_k]
                masks = t[3][idx]
                labels, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

        if class_filter is not None:
            selected_indices = [i for i in range(labels.shape[0]) if labels[i] in class_filter]
            labels = labels[selected_indices]
            scores = scores[selected_indices]
            boxes = boxes[selected_indices]
            masks = masks[selected_indices]

        labels = [cfg.dataset.class_names[label] for label in labels]
        scores = [score for score in scores]
        return masks, labels, scores, boxes


if __name__ == '__main__':
    file_name = '/home/lishangjie/data/SEUMM/seumm_lwir_15200/images/001000.jpg'
    assert os.path.exists(file_name), '%s Not Found' % file_name
    img = cv2.imread(file_name)
    color = (244, 67, 54)  # RGB

    detector = YolactDetector(
        trained_model='weights/seumm_lwir_15200/yolact_resnet50_157_200000.pth',
        dataset='seumm_lwir_15200_dataset'
    )

    t1 = time.time()
    img = img[:, :, ::-1].copy()  # to RGB
    masks, labels, scores, boxes = detector.run(
        img, score_threshold=0.1, top_k=20, class_filter=[0, 1, 2, 3, 4]
    )  # pedestrian, cyclist, car, bus, truck
    t2 = time.time()
    print('time cost:', t2 - t1, '\n')

    print('masks', type(masks), masks.shape)
    print('labels', type(labels), len(labels))
    print('scores', type(scores), len(scores))
    print('boxes', type(boxes), boxes.shape)

    img = img[:, :, ::-1].copy()  # to BGR
    color = color[::-1]  # to BGR
    for i in range(len(labels)):
        # img = draw_segmentation_result(
        #     img, masks[i], str(labels[i]), float(scores[i]), boxes[i], color
        # )
        img = draw_mask(img, mask=masks[i], color=color)

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)

    if cv2.waitKey(0) == 27:
        cv2.destroyWindow("img")
