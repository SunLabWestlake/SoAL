# -*- coding: utf-8 -*-
"""
Stage 3: Key points detection
Author: Jing Ning @ SunLab
"""

import time
import numpy as np
from SoAL_Constants import MODEL_FOLDER, PRINT_FRAME

"""
Queue pack:
[task_id, img, frame, roi_i, {reg}]
reg:
[area, center, orient + 90, major, minor, center_global] 
"""
PACK_TASK_ID, PACK_IMG, PACK_FRAME, PACK_ROI_I, PACK_REG, PACK_REG_N = 0, 1, 2, 3, 4, 5
REG_AREA, REG_CENTER, REG_ORIENT90, REG_MAJOR, REG_MINOR, REG_CENTER_GLOBAL = 0, 1, 2, 3, 4, 5
class PredictParts(object):
    def __init__(self, q, i):
        # redirect_output("log/pred%d.log" % i)
        self._pred_q = q
        self._idx = i
        self.stop = False
        self._task_q_l = {}
        self.init_pred_model() #TEST:NEED_PRED

    def init_pred_model(self):
        from hrnet import predict
        self._model = predict.PredictModel(MODEL_FOLDER)

    def append_new_task(self, pack):
        from multiprocessing import Process, Queue
        task_id, video = pack
        q = Queue()
        self._task_q_l[task_id] = q
        from SoAL_IDAssign import IdAssign  # NOTE: diff
        print("[predict_parts] new task %d %s" % (task_id, video))
        rs_p = Process(target=IdAssign.process, args=(video, task_id, q))  # copy instance
        rs_p.start()

    def end_task(self, task_id):
        self._task_q_l[task_id].put(None)

    def predict_loop(self):
        print("[predict_parts%d] loop..."%self._idx)
        count, n = 0, 0
        last_ts = time.time()
        while not self.stop:
            l = self._pred_q.qsize()
            if not l:
                l = 1
                # print("q empty")
            pack_l = []
            img_l = []
            for i in range(l):
                pack = self._pred_q.get()
                task_id = pack[PACK_TASK_ID]
                if task_id not in self._task_q_l:
                    self.append_new_task(pack)
                else:
                    img = pack[PACK_IMG]
                    if img is None:
                        self.end_task(task_id)
                    else:
                        # plot_ego("img/img%06d%08d.jpg" % (pack[PACK_FRAME], i), pack[PACK_IMG], None)
                        # continue
                        img_l.append(img)
                        pack_l.append(pack)
            if not img_l:
                continue
            # for i, img in enumerate(img_l):
            #     print("%d %s"%(i,img.shape))
            ypk, scmap = self._model.pred_img_batch(img_l)
            # ypk = np.zeros((3, 5, len(img_l))) #TEST:NEED_PRED
            # ypk: (3,5,n)  scmap: (n,8,8,5)
            for i, img in enumerate(img_l):
                # print("task", task_id, "batch", len(img_l))
                pack = pack_l[i]
                task_id = pack[PACK_TASK_ID]
                reg = pack[PACK_REG]
                # if pack[PACK_FRAME] >=0 and pack[PACK_ROI_I]==0:
                #     plot_ego("img/img%06d%08d.jpg" % (pack[PACK_FRAME], i), pack[PACK_IMG], np.array(ypk[:, :, i]))  #TEST:PLOT_EGO
                points = rotate_back_points(ypk[:, :, i], reg[REG_CENTER], reg[REG_ORIENT90], img.shape)
                # points: (3,5)
                self._task_q_l[task_id].put([pack[PACK_ROI_I], pack[PACK_FRAME], reg, pack[PACK_REG_N], np.array(points).flatten()])
            count += l
            n += 1
            if count >= PRINT_FRAME*5:
                # print("[predict_parts] pred_img_batch: %f/%d" % (prof_t, prof_c))
                ts = time.time()
                d_ts = ts - last_ts
                last_ts = ts
                print("[predict_parts%d] (%d/%d)fly/q %.2fs %.2ffly/s" % (self._idx, count, n, d_ts, count/d_ts))
                # profile: 1800fly/s = tasks(4)*avg_fps(15)*rois(15)*2
                count, n = 0, 0

    @staticmethod
    def process(q, i=0):
        #import os
        #os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (i%GPU_N)
        try:
            #     with tf.device("/gpu:%d" % (i%GPU_N)):
            PredictParts.inst = PredictParts(q, i)
            PredictParts.inst.predict_loop()
        except Exception as e:
            import traceback
            traceback.print_exc()

    @staticmethod
    def stop():
        PredictParts.inst.stop = True

def rotate_back_points(ypk0, pos, flydir, img_shape):
    flyx, flyy = pos[0], pos[1]
    fd = np.deg2rad(flydir - 90)  # FIXME version dependent (flydir)
    xy1 = np.vstack((ypk0[0], ypk0[1], np.ones((1, 5))))
    xy1center = np.matrix([
        [-1, 0, img_shape[1] / 2],
        [0, -1, img_shape[0] / 2],
        [0, 0, 1],
    ]) * xy1
    rot_m = np.matrix([
        [np.cos(fd), -np.sin(fd), 0],
        [np.sin(fd), np.cos(fd), 0],
        [0, 0, 1],
    ])
    trans_m = np.matrix([
        [1, 0, flyx],
        [0, 1, flyy],
        [0, 0, 1],
    ])
    m = trans_m * rot_m
    xy1_trans = m * xy1center
    return xy1_trans

def plot_ego(name, img, points):
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.colors import NoNorm
    cv2.imwrite(name.replace("jpg", "png"), img)
    # return
    xs, ys = points[0], points[1]
    plt.cla()
    plt.imshow(img.astype(int), cmap=plt.cm.gray, norm=NoNorm())
    plt.plot(xs[:3], ys[:3], 0.5, "r")
    plt.plot(xs[1:4:2], ys[1:4:2], 0.5, "g")#1, 3
    plt.plot(xs[1:5:3], ys[1:5:3], 0.5, "b")#1,4
    plt.scatter(xs, ys, c="rgbkw")
    s = img.shape
    plt.xlim((0, s[0]))
    plt.ylim((s[1], 0))
    plt.savefig(name)
