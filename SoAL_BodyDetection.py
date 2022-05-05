# -*- coding: utf-8 -*-
"""
Step 2: Body detection
Author: Jing Ning @ SunLab
"""

import os
import cv2
import time
from skimage import measure
from scipy import ndimage
import numpy as np

from SoAL_Utils import load_dict
from SoAL_Constants import FLY_NUM, FLY_AVG_AREA, MODEL_SHAPE, MODEL_SHAPE_EXTEND, SCALE_NORMAL, DURATION_LIMIT, \
    START_FRAME, FRAME_STEP
from SoAL_Bkg import norm_subbg, remove_bg3, calc_bg

FIX_SHAPE = MODEL_SHAPE#[58, 58]  # test: need change according to fly size in video
CENTER_UNIFORM_RANGE = 0#0.2  # NOTE: fill center range (food) by median (0=close)
CENTER_PAD = (1 - CENTER_UNIFORM_RANGE) / 2
# cv2.setNumThreads(1)

class RegionSeg(object):
    def __init__(self, video, task_id, pred_q):
        # redirect_output("log/rs%d_%s.log" % (task_id, os.path.basename(video)))
        self._task_id = task_id
        self._pred_q = pred_q
        self._video = video
        self._meta = load_dict(video.replace(".avi", "_config.json"))
        self._total_frame = min(DURATION_LIMIT * self._meta["FPS"], self._meta["total_frame"]) or self._meta["total_frame"]
        self._bg_filename = video.replace(".avi", ".bmp")

        self._gray_thresh = self._meta.get("GRAY_THRESHOLD")
        self._scale_factor = self._meta.get("SCALE")
        self._roi = [r["roi"] for r in self._meta.get("ROI")]
        self._avg_area = self._scale_factor ** 2 * FLY_AVG_AREA
        self._max_area = self._avg_area * FLY_NUM * 2
        self._min_area = self._avg_area / 3
        self._model_shape = MODEL_SHAPE
        self._model_shape_extend = MODEL_SHAPE_EXTEND
        self.shape_error = False
        self._ts = 0
        self._start_ts = 0
        self.init_model_shape()
        self.init_edge_mask()

    def init_model_shape(self):
        if abs(self._scale_factor - SCALE_NORMAL) > 2:
            """ video scale != train video scale
                just change input size, scale video is not necessary
            """
            print("shape not equal !!! %f %s need fix shape:" % (self._scale_factor, self._video))
            # self.shape_error = True
            s = SCALE_NORMAL / self._scale_factor
            self._model_shape = FIX_SHAPE or [int(MODEL_SHAPE[0] / s), int(MODEL_SHAPE[1] / s)] #[48,48]#
            self._model_shape_extend = (int(self._model_shape[0] * 1.8), int(self._model_shape[1] * 1.8))
            if FIX_SHAPE:
                print("!!!fix_shape=", FIX_SHAPE)
            else:
                print("!!!auto_shape=", self._model_shape)

    def init_edge_mask(self):
        h, w = self._model_shape
        edge_mask = np.zeros([w-2, h-2])
        edge_mask = np.vstack([np.ones([1, h-2]), edge_mask, np.ones([1, h-2])])
        self._edge_mask = np.hstack([np.ones([w, 1]), edge_mask, np.ones([w, 1])])
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    @staticmethod
    def process(video, task_id, pred_q):
        inst = RegionSeg(video, task_id, pred_q)
        if inst.shape_error:
            state_file = os.path.join(os.path.dirname(video), ".state")
            f = open(state_file, "w")
            f.write("error\nshape\n")
            f.close()
        else:
            inst.proc_video()

    @staticmethod
    def process_sub(video, task_id, pred_q, input_q):
        inst = RegionSeg(video, task_id, pred_q)
        inst.init_bg()
        while True:
            pack = input_q.get()
            if pack is None:
                break
            inst.proc_img2(*pack)

    # @profile
    def proc_video(self):
        # from queue import Queue
        # from threading import Thread
        # input_q_l = []
        # for i in range(IMG_PROCESS_NUM):  # NOTE: sub processes call proc_img2
        #     input_q = Queue(maxsize=IMG_PROCESS_CAPACITY)
        #     input_q_l.append(input_q)
        #     rs_p = Thread(target=RegionSeg.process_sub, args=(self._video, self._task_id, self._pred_q, input_q))
        #     rs_p.start()

        self.init_bg()

        self._cap = cv2.VideoCapture(self._video)
        if not os.path.exists(self._bg_filename):
            img_bg = calc_bg(self._cap)
            cv2.imwrite(self._bg_filename, img_bg)
        self._ts = time.time()
        self._start_ts = self._ts
        print("[BodyDetection]#%d ---start: %d frames" % (self._task_id, self._total_frame))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
        if FRAME_STEP:
            seq_range = range(START_FRAME, self._total_frame, FRAME_STEP)
        else:
            seq_range = range(START_FRAME, self._total_frame)

        for seq in seq_range:
            ret, img = self._cap.read()
            if not ret:
                print("read video error: " + self._video)
                self._pred_q.put([self._task_id, None, seq, 0, None, 0])
                break
            self.proc_img2(img[:, :, 1], seq)
            # input_q_l[seq % IMG_PROCESS_NUM].put([img[:, :, 1], seq])

        # for i in range(IMG_PROCESS_NUM):  # NOTE: end subprocess
        #     input_q_l[i].put(None)

        d = time.time() - self._start_ts
        print("[BodyDetection]#%d ---finish: (%.2f/%.2f=%.2fframe/s)" % (self._task_id, self._total_frame, d, self._total_frame / d))

    def measure_regionprops(self, sub_bin, roi):
        label_img = measure.label(sub_bin, connectivity=2)
        props = measure.regionprops(label_img, coordinates='rc')
        ret = []
        for i, p in enumerate(props):
            if self._min_area < p.area < self._max_area:
                center = p.centroid[1], p.centroid[0]
                center_global = (center[0] + roi[0][0], center[1] + roi[0][1])
                major, minor = p.major_axis_length, p.minor_axis_length
                orient = np.rad2deg(p.orientation)
                ret.append([p.area, center, -orient + 90, major, minor, center_global])
        return ret

    def init_bg(self):
        self._img_bg = cv2.imread(self._bg_filename, cv2.IMREAD_GRAYSCALE)
        # self._l_img_bg = []
        self._l_img_bg_f = []
        self._l_img_bg_cu = []
        for i, roi in enumerate(self._roi):
            img_bg = self._img_bg[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
            sh = img_bg.shape
            img_bg_f = img_bg.astype(float)
            i_median = np.full(sh, np.median(img_bg_f))
            p_sh = min(int(sh[0] * CENTER_PAD), int(sh[1] * CENTER_PAD))
            c_sh = sh[0] - 2 * p_sh, sh[1] - 2 * p_sh
            i_mask = np.pad(np.ones(c_sh), (p_sh, p_sh)).astype(bool)
            img_bg_center_uniform = img_bg_f * (~i_mask) + (i_median * i_mask)
            cv2.imwrite("temp/img_bg_center_uniform_%d_%d.png" % (self._task_id, i), img_bg_center_uniform)
            # self._l_img_bg.append(img_bg)
            self._l_img_bg_f.append(img_bg_f)
            self._l_img_bg_cu.append(img_bg_center_uniform)

        from sklearn.mixture import GaussianMixture
        self.gmm = GaussianMixture(n_components=FLY_NUM, covariance_type='full', random_state=42)

    # @profile
    def proc_img2(self, img, frame):
        # if FIX_VIDEO_SIZE:
        #     img = cv2.resize(img, FIX_VIDEO_SIZE)
        for i, roi in enumerate(self._roi):
            img_roi = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
            img_roi_n = remove_bg3(img_roi, self._l_img_bg_cu[i])
            img_no_bg = remove_bg3(img_roi, self._l_img_bg_f[i])
            sh = img_roi_n.shape
            sub_bin = (img_roi_n < self._gray_thresh).astype(np.uint8)
            contours, hierarchy = cv2.findContours(sub_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            roi_regions = []
            for c in contours:
                if len(c) > 2:  # at least 2 edges
                    area = cv2.contourArea(c)
                    if self._min_area < area < self._max_area:
                        center, axes, orient = cv2.fitEllipse(c)
                        if abs(center[1]) > sh[0] or abs(center[0]) > sh[1]:  # NOTE: fitEllipse result abnormal
                            print("[BodyDetection]#%d center out of roi !!! frame:%d roi:%d" % (self._task_id, i, frame))
                            cv2.imwrite("temp/error%d.png" % frame, img_no_bg)
                            roi_regions = self.measure_regionprops(sub_bin, roi)
                            break
                        # print("center:",frame, center)
                        center_global = (center[0] + roi[0][0], center[1] + roi[0][1])
                        major, minor = axes[0], axes[1]
                        if major < minor:
                            major, minor = minor, major
                        region = (area, center, orient - 90, major, minor, center_global)
                        roi_regions.append(region)
            region_n = len(roi_regions)
            if region_n < FLY_NUM:  # need gmm
                roi_regions = self.gmm_seg(sub_bin, roi[0])  # PROFILE
                if roi_regions is None:
                    print("[BodyDetection]#%d gmm_seg found nothing !!! (video %s, roi %d, frame %d)" % (self._task_id, self._video, i, frame))
                    cv2.imwrite("temp/error%d.png" % frame, img_roi_n)
                    roi_regions = []
                    for fly in range(FLY_NUM):
                        roi_regions.append((0, (0, 0), 0, 0, 0, (0, 0)))
            else:  # need reduce
                roi_regions.sort(key=lambda t: abs(self._avg_area - t[0]))
                roi_regions = roi_regions[:FLY_NUM]
            for reg in roi_regions:
                img_center = center_img(img_no_bg, reg[1], reg[2], self._model_shape, self._model_shape_extend)  # PROFILE
                # img_center = self.preproc_img_center(img_center, self._gray_thresh)
                # img_center = self._trans.transform(img_center.astype(np.uint8))
                img_center = img_center.astype(np.uint8)
                self._pred_q.put([self._task_id, img_center, frame, i, reg, region_n])
            # cv2.imwrite("temp/img_center-%d_%d_%d.png" % (self._task_id, i, frame), img_center)  # test output bg
            # cv2.imwrite("temp/sub_bin-%d_%d_%d.png" % (self._task_id, i, frame), sub_bin*255)
        # plot_regions(img, roi_regions)

    def is_on_edge(self, img_bin):
        return (img_bin * self._edge_mask > 0).any()

    def preproc_img_center(self, img_ego, gray_thresh):
        for threshold in range(int(gray_thresh + 20), int(gray_thresh * 0.6), -10):
            img_bin = img_ego < threshold
            if not self.is_on_edge(img_bin):  # img is clean(only one fly)
                break
            img_bin = img_bin.astype(np.uint8)
            img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, self._kernel)
            if not self.is_on_edge(img_bin):  # img is clean(only one fly)
                break
            contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(img_no_bg, contours, -1, (0, 0, 255), 3)
            count = len(contours)
            if count > 1:
                area_max_l = [0]
                idx_max_l = [0]
                for idx, c in enumerate(contours):
                    area = cv2.contourArea(c)
                    if area > area_max_l[-1]:
                        area_max_l.append(area)
                        idx_max_l.append(idx)
                area_min = area_max_l[-1] * 0.6
                main_idx = idx_max_l[-1]
                main_idx0 = main_idx
                max_is_on_edge = True
                for i in range(len(area_max_l)):
                    area = area_max_l[-i-1]
                    if area > area_min:
                        idx = idx_max_l[-i-1]
                        label = np.zeros(img_bin.shape, dtype=np.uint8)
                        cv2.drawContours(label, contours, idx, color=255, thickness=-1)
                        if not self.is_on_edge(label):
                            main_idx = idx
                            max_is_on_edge = False
                            break
                if main_idx == main_idx0 and max_is_on_edge:
                    # max is on edge
                    pass
                else:
                    cv2.drawContours(img_bin, contours, main_idx, color=0, thickness=-1)  # only true fly remains
                    img_ego[img_bin.astype(bool)] = 255
                break
        img_ego = norm_subbg(255-img_ego)  # NOTE: for different light condition
        r2 = np.clip(img_ego ** 2 / 255, 0, 255)
        return r2

    def gmm_seg(self, sub_img, global_offset):
        ret = []
        fg = sub_img.nonzero()
        if len(fg[0]) < 5:
            return None
        X = np.array(fg).T
        self.gmm.fit(X)  #.predict(X) # no need to predict !!!

        # from matplotlib.patches import Ellipse
        # fig, ax = plt.subplots()
        # ax.imshow(sub_img, cmap="Greys_r")

        for i in range(FLY_NUM):
            center = self.gmm.means_[i]
            orient, major, minor = get_gmm_ellipse_info(self.gmm.covariances_[i])
            area = np.pi * major * minor / 4  # NOTE: ellipse area bigger than contour area
            center_global = (center[1] + global_offset[0], center[0] + global_offset[1])
            ret.append((area, (center[1], center[0]), orient, major, minor, center_global))
            # ax.add_patch(Ellipse((center[1], center[0]), major, minor, orient, alpha=0.5))
        # plt.show()
        return ret

    def proc_one_img(self, img, roi, i, proc_center=False, no_center_img=False):
        # NOTE: need self.init_bg()
        img_roi = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
        img_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        img_roi_n = remove_bg3(img_roi, self._l_img_bg_cu[i])
        img_no_bg = remove_bg3(img_roi, self._l_img_bg_f[i])
        sh = img_roi_n.shape
        sub_bin = (img_roi_n < self._gray_thresh).astype(np.uint8)
        contours, hierarchy = cv2.findContours(sub_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_regions = []
        for c in contours:
            if len(c) > 2:  # at least 2 edges
                area = cv2.contourArea(c)
                if self._min_area < area < self._max_area:
                    center, axes, orient = cv2.fitEllipse(c)
                    if center[0] > sh[0] or center[1] > sh[1]:  # NOTE: fitEllipse result abnormal
                        print("[BodyDetection] center out of roi !!!")
                        roi_regions = self.measure_regionprops(sub_bin, roi)
                        break
                    # print("center:",frame, center)
                    center_global = (center[0] + roi[0][0], center[1] + roi[0][1])
                    major, minor = axes[0], axes[1]
                    if major < minor:
                        major, minor = minor, major
                    region = (area, center, orient - 90, major, minor, center_global)
                    roi_regions.append(region)
        region_n = len(roi_regions)
        if region_n < FLY_NUM:  # need gmm
            roi_regions = self.gmm_seg(sub_bin, roi[0])
            if roi_regions is None:
                return None, None, None
        else:  # need reduce
            roi_regions.sort(key=lambda t: abs(self._avg_area - t[0]))
            roi_regions = roi_regions[:FLY_NUM]
        if no_center_img:
            return img_roi, [(reg[1], reg[2]) for reg in roi_regions]
        img_center_l = []
        pos_l = []
        for reg in roi_regions:
            if proc_center:
                img_center = center_img(img_no_bg, reg[1], reg[2], self._model_shape, self._model_shape_extend)
                img_center = self.preproc_img_center(img_center, self._gray_thresh)#self.correct_dir_by_wing()
            else:
                img_center = center_img(img_no_bg, reg[1], reg[2], self._model_shape, self._model_shape_extend)
                img_center = norm_subbg(255-img_center)  # NOTE: for different light condition
                img_center = np.clip(img_center ** 2 / 255, 0, 255)
            img_center_l.append(img_center)
            pos_l.append([reg[1][0]/sh[1], reg[1][1]/sh[0]])
        return img_center_l, region_n, pos_l

def plot_regions(img, regions):
    from matplotlib.patches import Ellipse
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="Greys_r")

    for regs in regions:
        area, center, orient, major, minor, center_global = regs
        ax.add_patch(Ellipse((center_global[0], center_global[1]), major, minor, orient, linewidth=1, fill=False, color="b"))
    plt.show()

def get_gmm_ellipse_info(covariance):
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[0, 0], U[1, 0]))
        width, height = 4 * np.sqrt(s)
    else:
        angle = 0
        width, height = 4 * np.sqrt(covariance)
    return angle, width, height

def center_img(img_gray, center, theta, out_shape, out_shape_w, fill_gray=255):
    # shift = -center[0] + in_shape[0] / 2, -center[1] + in_shape[1] / 2
    sub0 = int(center[1] - out_shape_w[1]/2)
    sub1 = int(center[0] - out_shape_w[0]/2)
    end0 = sub0 + out_shape_w[1]#img_gray.shape[0]+5
    end1 = sub1 + out_shape_w[0]#img_gray.shape[1]+5
    img_shift = img_gray[max(sub0, 0):end0, max(sub1, 0):end1]#ndimage.shift(img_gray, shift)
    s0, s1 = img_shift.shape
    if sub0 < 0:
        img_shift = np.vstack((np.full((-sub0, s1), fill_gray), img_shift))
    if end0 > img_gray.shape[0]:
        img_shift = np.vstack((img_shift, np.full((end0 - img_gray.shape[0], s1), fill_gray)))
    if sub1 < 0:
        img_shift = np.hstack((np.full((img_shift.shape[0], -sub1), fill_gray), img_shift))
    if end1 > img_gray.shape[1]:
        img_shift = np.hstack((img_shift, np.full((img_shift.shape[0], end1 - img_gray.shape[1]), fill_gray)))
    img_rotate = ndimage.rotate(img_shift, theta+90)
    shape_r = img_rotate.shape
    sub0 = int((shape_r[0] - out_shape[1]) / 2)
    sub1 = int((shape_r[1] - out_shape[0]) / 2)
    return img_rotate[sub0:(sub0 + out_shape[1]), sub1:(sub1 + out_shape[0])]

def center_img3(img_gray, center, theta, out_shape, out_shape_w, fill_gray=0):
    # shift = -center[0] + in_shape[0] / 2, -center[1] + in_shape[1] / 2
    sub0 = int(center[1] - out_shape_w[1]/2)
    sub1 = int(center[0] - out_shape_w[0]/2)
    end0 = sub0 + out_shape_w[1]#img_gray.shape[0]+5
    end1 = sub1 + out_shape_w[0]#img_gray.shape[1]+5
    img_shift = img_gray[max(sub0, 0):end0, max(sub1, 0):end1]#ndimage.shift(img_gray, shift)
    s0, s1, s2 = img_shift.shape
    if sub0 < 0:
        img_shift = np.vstack((np.full((-sub0, s1, s2), fill_gray), img_shift))
    if end0 > img_gray.shape[0]:
        img_shift = np.vstack((img_shift, np.full((end0 - img_gray.shape[0], s1, s2), fill_gray)))
    if sub1 < 0:
        img_shift = np.hstack((np.full((img_shift.shape[0], -sub1, s2), fill_gray), img_shift))
    if end1 > img_gray.shape[1]:
        img_shift = np.hstack((img_shift, np.full((img_shift.shape[0], end1 - img_gray.shape[1], s2), fill_gray)))
    img_rotate = ndimage.rotate(img_shift, theta+90)
    shape_r = img_rotate.shape
    sub0 = int((shape_r[0] - out_shape[1]) / 2)
    sub1 = int((shape_r[1] - out_shape[0]) / 2)
    return img_rotate[sub0:(sub0 + out_shape[1]), sub1:(sub1 + out_shape[0])]

def main():
    import os, sys
    # import profile
    # profile.run("RegionSeg(sys.argv[1]).proc_video()")
    RegionSeg(sys.argv[1], None, None).proc_video()
    # os.system("pause")


# if __name__ == '__main__':
#     main()
# kernprof.exe -l .\SoAL_BodyDetection.py "D:\exp\video_todo\20190902_150503_A\20190902_150503_A.avi"
# python -m line_profiler .\SoAL_BodyDetection.py.lprof
