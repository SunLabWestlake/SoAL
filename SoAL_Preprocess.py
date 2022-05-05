# -*- coding: utf-8 -*-
"""
Step 1: Pre-processing
    scale(px/mm) and config information
    ROI segmentation
    calc background
    binarization threshold
Author: Jing Ning @ SunLab
"""

import sys
import os
import cv2
from SoAL_UI import InputExpInfoUI, InputRoiInfoUI, InputBgInfoUI
from SoAL_Utils import load_dict, save_dict, find_file
from SoAL_Constants import *

def rename_video_t(video):
    if video.endswith("_t.avi"):
        video_o = video.replace("_t.avi", ".avi")
        if os.path.exists(video):
            # if os.path.exists(video_o):
            #     os.rename(video_o, video + ".old")
            os.rename(video, video_o)
        return video_o
    return video

def get_video_in_dir(video_dir):
    f = os.path.basename(video_dir)
    video = os.path.join(video_dir, f + "_t.avi")
    if os.path.exists(video):
        return video
    return os.path.join(video_dir, f + ".avi")

def preprocess(video, replace):
    print("[1.1] frame_align...", video)
    if os.path.isdir(video):
        video = get_video_in_dir(video)
    if video.endswith("_t.avi"):
        video = rename_video_t(video)
        cap = cv2.VideoCapture(video)
        info = update_meta_info(video, cap)  # modify "total_frame"
        # input_meta_info(video, cap, info)  # modify "ROI:info", "log"
        cap.release()
    else:
        cap = cv2.VideoCapture(video)
        info = {} if replace else update_meta_info(video, cap)
        input_meta_info(video, cap, info)
        cap.release()
    return video

def update_meta_info(video, cap):
    print("[1.3] update_meta_info...", video)
    meta = video.replace(".avi", "_config.json")
    if not os.path.exists(meta):
        return {}
    info = load_dict(meta)
    # info["total_frame"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # save_dict(meta, info)
    return info

def update_male_day(video):
    meta = video.replace(".avi", "_config.json")
    if not os.path.exists(meta):
        return
    info = load_dict(meta)
    parent = os.path.dirname(meta)
    for i, roi in enumerate(info["ROI"]):
        # roi["male_date"] = day_add_s(EXP_MALE_DAY1, int(roi["info"][1:]))
        # roi["male_days"] = day_str_diff(info["exp_date"], roi["male_date"])
        # roi["male_days"] = roi["male_days"]+5
        # roi["male_date"] = day_add_s(info["exp_date"], -roi["male_days"])
        print(roi["male_date"], roi["male_days"])
        # di = os.path.join(parent, str(i))
        # for f in find_file(di, "config.json"):
        #     print(f)
        #     finfo = load_dict(os.path.join(di, f))
        #     finfo["ROI"]["male_date"] = roi["male_date"]
        #     finfo["ROI"]["male_days"] = roi["male_days"]
        #     print(roi["male_date"])
            # save_dict(os.path.join(di, f), finfo)
    save_dict(meta, info)

def input_meta_info(video, cap, info=None):
    print("[1.2] input_meta_info...", video)
    meta = video.replace(".avi", "_config.json")
    # if os.path.exists(meta) and os.path.getsize(meta) > 0:
    #     return False

    # camera, size, start, end, duration, FPS
    # VERSION, ROUND_ARENA, MODEL_FOLDER
    static_info = info or get_video_static_info(video, cap)
    if FIX_VIDEO_SIZE:
        static_info["width"], static_info["height"] = FIX_VIDEO_SIZE
    print(static_info)

    print("input_exp_info...", video)
    # SCALE, AREA_RANGE, temperature, female_date, female_days
    ui_exp = InputExpInfoUI("input_exp_info", (9, 8), cap, static_info)
    ui_exp.show()

    print("input_roi_info...", video)
    # idx, fly_id, {roi(2x2)}, {male_geno}, {male_date}, male_days
    ui_roi = InputRoiInfoUI("input_roi_info", (9, 8), cap, ui_exp.exp_info)
    ui_roi.show()

    print("input_bg_info...", video)
    # bg, GRAY_THRESHOLD
    bg_filename = video.replace(".avi", ".bmp")
    ui_bg = InputBgInfoUI("input_bg_info", (9, 8), cap, ui_roi.roi_info, bg_filename)
    ui_bg.show()

    save_dict(meta, ui_bg.bg_info)

def get_video_static_info(video, cap):
    # file, camera, size, start, end, duration, FPS
    # VERSION, ROUND_ARENA, MODEL_FOLDER
    info = {}
    info["file"] = os.path.basename(video)
    info["total_frame"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info["orig_fps"] = int(cap.get(cv2.CAP_PROP_FPS))
    info["width"], info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    info["VERSION"] = VERSION
    info["FPS"] = FPS
    info["ROUND_ARENA"] = ROUND_ARENA
    info["MODEL_FOLDER"] = MODEL_CONFIG

    log = video.replace(".avi", ".log")
    if not os.path.exists(video):
        logs = find_file(os.path.dirname(video), ".log")
        if not logs:
            return
        log = logs[0]
    if os.path.exists(log):
        logf = open(log, "r")
        first_frame = None
        for i in range(20):
            line = str(logf.readline())
            if line.startswith("camera:"):
                info["camera"] = line.split()[-1]
            elif line.startswith("start:") or line.startswith("begin"):
                info["start"] = line.split()[-1]
            elif line.find(":") < 0 and first_frame is None:
                first_frame = int(line.split()[0])
        logf.close()
        logf = open(log, "rb")
        logf.seek(-50, 2)
        lines = logf.readlines()
        last_line = str(lines[-1], encoding="utf-8").split()
        last_ts = last_line[-1]
        if first_frame is not None:
            last_frame = int(last_line[0])
            info["FPS"] = (last_frame - first_frame) / float(last_ts)
        if last_ts.find(":") > 0:
            info["end"] = last_ts
            info["duration"] = tt_to_second(info["end"].split(":")) - tt_to_second(info["start"].split(":"))
        else:
            info["duration"] = float(last_ts)
            info["end"] = "%02d:%02d:%02d" % (second_to_tt(tt_to_second(info["start"].split(":")) + int(info["duration"] + 0.5)))
    else:
        info["camera"] = log.split("_")[-1][:-4]
        start_str = log.split("_")[-2]
        info["start"] = start_str[:2] + ":" + start_str[2:4] + ":" + start_str[4:6]
        info["duration"] = info["total_frame"] / info["orig_fps"]
        info["end"] = "%02d:%02d:%02d" % (second_to_tt(tt_to_second(info["start"].split(":")) + int(info["duration"] + 0.5)))
    return info

if __name__ == '__main__':
    print("[0] preprocess...")
    preprocess(sys.argv[1], replace=True)
    # update_male_day(sys.argv[1])
    # from functools import cmp_to_key
    # args = sys.argv[1:]
    # args.sort(key=cmp_to_key(cmp_file))
    # for a in args:
    #     preprocess(a)
    # os.system("pause")

