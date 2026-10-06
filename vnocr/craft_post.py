"""CRAFT score maps -> word boxes. Adapted from EasyOCR's craft_utils.getDetBoxes
(Copyright (c) 2019-present NAVER Corp., MIT License), using cv_compat instead of OpenCV."""

import math

import numpy as np

from . import cv_compat as cv2


def get_det_boxes(textmap, linkmap, text_threshold, link_threshold, low_text):
    img_h, img_w = textmap.shape
    _, text_score = cv2.threshold(textmap, low_text, 1, 0)
    _, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)
    text_score_comb = np.clip(text_score + link_score, 0, 1)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    det = []
    for k in range(1, n_labels):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        sl = (slice(y, y + h), slice(x, x + w))
        comp = labels[sl] == k
        if np.max(textmap[sl][comp]) < text_threshold:
            continue
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0   # remove link area
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        sx, sy = max(sx, 0), max(sy, 0)
        ex, ey = min(ex, img_w), min(ey, img_h)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        box = cv2.boxPoints(cv2.minAreaRect(np_contours))
        bw, bh = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        if abs(1 - max(bw, bh) / (min(bw, bh) + 1e-5)) <= 0.1:    # align diamond shapes
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
        startidx = box.sum(axis=1).argmin()                     # clockwise from top-left
        det.append(np.roll(box, 4 - startidx, 0))
    return det


def adjust_coordinates(polys, ratio_w, ratio_h, ratio_net=2):
    polys = np.array(polys)
    for k in range(len(polys)):
        polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys
