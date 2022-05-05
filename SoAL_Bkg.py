# -*- coding: utf-8 -*-
"""
Background processing
Author: Jing Ning @ SunLab
"""

import numpy as np
import cv2

from SoAL_Constants import FIX_VIDEO_SIZE

BG_REF_FRAME_COUNT = 100
NORMAL_BLACK = 180


def sub_img(img, roi):
    return img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]

def calc_bg(cap, start_frame=0, end_frame=0):
    from tqdm import tqdm
    print("calc bg...")
    if not end_frame:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # remove_frame = int((end_frame - start_frame) / 3)
    # start_frame += remove_frame
    # end_frame -= remove_frame
    step = max(1, int((end_frame - start_frame) / BG_REF_FRAME_COUNT))
    img_a = []
    for seq in tqdm(range(start_frame, end_frame, step)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, seq)
        ret, img = cap.read()
        if not ret:
            break
        img_gray = img[:, :, 1]
        if FIX_VIDEO_SIZE:
            img_gray = cv2.resize(img_gray, FIX_VIDEO_SIZE)
        img_a.append(img_gray)
        # img_bg = np.maximum(img_bg, img_gray)

    # img_bg = np.median(img_a, 0)
    img_bg = np.max(img_a, 0)
    return img_bg.astype(float)

def remove_bg3(img_gray, img_bg):  # NOTE: dark food make fly body brighter, fixed by img_bg_center_uniform
    return norm_subbg(np.fabs(img_bg - img_gray))

def norm_subbg(r_subbg):# 0.008s
    r_subbg = np.clip(r_subbg, 0, 255)
    black = np.max(r_subbg)
    r_norm = r_subbg * (NORMAL_BLACK / black)
    return 255.0 - r_norm

def norm_img(img):
    black = np.max(img)
    r_norm = img * (255.0 / black)
    return r_norm

def imshow_test(img, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import NoNorm
    plt.figure("test")
    plt.imshow(img, cmap=plt.cm.gray, norm=NoNorm())
    if path:
        plt.savefig(path)
    else:
        plt.show()
