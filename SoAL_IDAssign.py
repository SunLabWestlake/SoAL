# -*- coding: utf-8 -*-
"""
Stage 4: Write results (used to implement ID assignment)
Author: Jing Ning @ SunLab
"""

import os
import time
from datetime import datetime
from SoAL_Constants import DURATION_LIMIT, PRINT_FRAME, PRINT_FRAME1, MODEL_FOLDER
from SoAL_Utils import load_dict, save_dict, to_feat_s, get_feat_header


class IdAssign(object):
    def __init__(self, video, task_id, task_q):
        # redirect_output("log/ia%d_%s.log" % (task_id, os.path.basename(video)))
        self._video = video
        self._task_id = task_id
        self._task_q = task_q
        self._meta = load_dict(video.replace(".avi", "_meta.txt"))
        self._total_frame = min(DURATION_LIMIT * self._meta["FPS"], self._meta["total_frame"]) or self._meta["total_frame"]
        self._max_frame = self._total_frame - 1
        roi_l = self._meta.get("ROI")
        self._max_fly = self._max_frame * len(roi_l) * 2
        self._roi = [r["roi"] for r in roi_l]
        self._feat_l = []
        self._feat_f_l = []
        for roi_i in range(len(roi_l)):
            self._feat_f_l.append(self.init_mata_feat(roi_i))
            self._feat_l.append([])
        self._feat_header = get_feat_header()

    def init_mata_feat(self, roi_i):
        parent = os.path.dirname(self._video)
        base = os.path.basename(self._video)
        parent = os.path.join(parent, str(roi_i))
        not os.path.exists(parent) and os.mkdir(parent)
        feat = os.path.join(parent, base.replace(".avi", "_%d_feat.csv" % roi_i))
        meta = os.path.join(parent, base.replace(".avi", "_%d_meta.txt" % roi_i))
        meta_d = self._meta.copy()
        meta_d["ROI"] = meta_d["ROI"][roi_i]
        meta_d["MODEL_FOLDER"] = MODEL_FOLDER
        save_dict(meta, meta_d)
        return feat

    @staticmethod
    def process(video, task_id, q):
        inst = IdAssign(video, task_id, q)
        inst.proc_video()

    def proc_video(self):
        roi_num = len(self._roi)
        print("[id_assign]#%d ---start: %dframes %drois %s" % (self._task_id, self._total_frame, roi_num, datetime.strftime(datetime.now(), "%Y%m%d %H:%M:%S")))
        fly_info = []
        last_fly_info = []
        for roi_i in range(roi_num):
            last_fly_info.append([])
            fly_info.append([])
        start_ts = time.time()
        last_ts = start_ts
        frame = 0
        i = 0
        fs = [open(f, "w") for f in self._feat_f_l]
        for f in fs:
            f.write(",".join(self._feat_header) + "\n")
        while True:
            pack = self._task_q.get()
            if pack is None:
                for f in fs:
                    f.close()
                break
            roi_i, frame, reg, reg_n, points = pack
            # fly_info[roi_i].append((frame, reg, reg_n, points))  # NOTE: diff
            if True: #len(fly_info[roi_i]) >= FLY_NUM:
                # id_assign(fly_info[roi_i], last_fly_info[roi_i])

                # last_fly_info[roi_i] = fly_info_c
                # fly_info[roi_i] = []
                fs[roi_i].write(to_feat_s([(frame, reg, reg_n, points)]) + "\n")
                i += 1
                # if frame >= self._max_frame:  # FIXME: closed before finish!!!
                #     fs[roi_i].close()
                #     break
                if i >= self._max_fly:  # NOTE: cant stop if FRAME_STEP > 1
                    for f in fs:
                        f.close()
                    break

                if frame % PRINT_FRAME == PRINT_FRAME1 and roi_i == 0:
                    ts = time.time()
                    # d_ts = ts - last_ts # NOTE: not correct
                    # last_ts = ts
                    print("[id_assign]#%d (%d/%d) %drois a%.2fframe/s %d%%" % (self._task_id, frame, self._total_frame, roi_num, frame/(ts-start_ts), frame*100.0/self._total_frame))

        end_ts = time.time()
        d = end_ts - start_ts
        finish_s = "#%d (%d(%d)/%.2fs=%.2fframe/s)\n" % (self._task_id, frame, self._total_frame, d, frame / d) + datetime.strftime(datetime.now(), "%Y%m%d %H:%M:%S")
        print("[id_assign]---finish: %s %s" % (finish_s, self._video))

        state_file = os.path.join(os.path.dirname(self._video), ".state")
        f = open(state_file, "w")
        f.write("finish\n" + finish_s)
        f.close()

