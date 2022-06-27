# -*- coding: utf-8 -*-
"""
Step 4: Write results
Author: Jing Ning @ SunLab
"""

import os
import time
from datetime import datetime
from SoAL_Constants import DURATION_LIMIT, PRINT_FRAME_WRITE, MODEL_CONFIG, START_FRAME
from SoAL_Utils import load_dict, save_dict, to_kpt_s, KPT_HEADER


class WriteKpt(object):
    def __init__(self, video, task_id, task_q):
        # redirect_output("log/ia%d_%s.log" % (task_id, os.path.basename(video)))
        self._video = video
        self._task_id = task_id
        self._task_q = task_q
        self._meta = load_dict(video.replace(".avi", "_config.json"))
        self._total_frame = min(DURATION_LIMIT * self._meta["FPS"], self._meta["total_frame"]) or self._meta["total_frame"]
        self._max_frame = self._total_frame - START_FRAME - 1
        roi_l = self._meta.get("ROI")
        self._max_fly = self._max_frame * len(roi_l) * 2
        self._roi = [r["roi"] for r in roi_l]
        self._kpt_l = []
        self._kpt_f_l = []
        for roi_i in range(len(roi_l)):
            self._kpt_f_l.append(self.init_mata_kpt(roi_i))
            self._kpt_l.append([])
        self._kpt_header = KPT_HEADER

    def init_mata_kpt(self, roi_i):
        parent = os.path.dirname(self._video)
        base = os.path.basename(self._video)
        parent = os.path.join(parent, str(roi_i))
        not os.path.exists(parent) and os.mkdir(parent)
        kpt = os.path.join(parent, base.replace(".avi", "_%d_kpt.csv" % roi_i))
        meta = os.path.join(parent, base.replace(".avi", "_%d_config.json" % roi_i))
        meta_d = self._meta.copy()
        meta_d["ROI"] = meta_d["ROI"][roi_i]
        meta_d["MODEL_FOLDER"] = MODEL_CONFIG
        save_dict(meta, meta_d)
        return kpt

    @staticmethod
    def process(video, task_id, q):
        inst = WriteKpt(video, task_id, q)
        inst.proc_video()

    def proc_video(self):
        roi_num = len(self._roi)
        print("[WriteKpt]#%d ---start: %d frames, %d rois %s" % (self._task_id, self._total_frame, roi_num, datetime.strftime(datetime.now(), "%Y%m%d %H:%M:%S")))
        fly_info = []
        last_fly_info = []
        for roi_i in range(roi_num):
            last_fly_info.append([])
            fly_info.append([])
        start_ts = time.time()
        frame = 0
        i = 0
        fs = [open(f, "w") for f in self._kpt_f_l]
        for f in fs:
            f.write(",".join(self._kpt_header) + "\n")
        while True:
            pack = self._task_q.get()
            if pack is None:
                for f in fs:
                    f.close()
                break
            roi_i, frame, reg, reg_n, points = pack
            fs[roi_i].write(to_kpt_s([(frame, reg, reg_n, points)]) + "\n")
            i += 1
            if i >= self._max_fly:  # NOTE: can't stop if FRAME_STEP > 1
                for f in fs:
                    f.close()
                break

            if frame % PRINT_FRAME_WRITE == 1 and roi_i == 0:
                ts = time.time()
                speed = frame/(ts-start_ts)
                left_time = (self._total_frame - frame) / speed
                print("[WriteKpt]#%d %d rois, frame_rate(%.2f fps), remaining(%d s), progress(%d/%d %d%%)" % (self._task_id, roi_num, speed, left_time, frame, self._total_frame, frame*100.0/self._total_frame))

        end_ts = time.time()
        d = end_ts - start_ts
        finish_s = "#%d (%d(%d)/%.2fs=%.2fframe/s)\n" % (self._task_id, frame-START_FRAME, self._total_frame, d, (frame-START_FRAME) / d) + datetime.strftime(datetime.now(), "%Y%m%d %H:%M:%S")
        print("[WriteKpt]---finish: %s %s" % (finish_s, self._video))

        state_file = os.path.join(os.path.dirname(self._video), ".state")
        f = open(state_file, "w")
        f.write("finish\n" + finish_s)
        f.close()

