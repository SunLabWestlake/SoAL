# -*- coding: utf-8 -*-
"""
Constants
Author: Jing Ning @ SunLab
"""

from datetime import datetime, timedelta

START_FRAME = 0  # test
FRAME_STEP = 0  # test
DURATION_LIMIT = 60 * 60  # 1 hour
FIX_VIDEO_SIZE = None#1920, 1080  # test
ARENA_ROWS = 4#2  # test
PRINT_FRAME = 600#30
PRINT_FRAME_WRITE = 60

VIDEO_TODO_DIR = "video/"

MAX_TASK = 2
BATCH_SIZE = 60  #NOTE: basic(60) 150 fly/s; advanced(200) 400 fly/s
IMG_QUEUE_CAPACITY = BATCH_SIZE*5
PRED_PROCESS_NUM = MAX_TASK

IMG_PROCESS_NUM = 2  # NOTE: not used
IMG_PROCESS_CAPACITY = IMG_QUEUE_CAPACITY
POOL_SIZE = 0

REL_POLAR_T_OFFSET = 0

MODEL_CONFIG = "hrnet/fly_w32.yaml"
MODEL_SHAPE = 64, 48
MODEL_SHAPE_EXTEND = (int(MODEL_SHAPE[0] * 1.8 + 2), int(MODEL_SHAPE[0] * 1.8 + 2))
SCALE_NORMAL = 12

FLY_NUM = 2
FLY_AVG_WID = 1
FLY_AVG_LEN = 2.5
FLY_AVG_LEN_MALE = 2.2
BODY_LEN = [2.4, 2.53, 2.2]
BODY_WID = [0.8, 0.87, 0.78]
FLY_AVG_LEN_L = 1.6
FLY_AVG_LEN_H = 3.0
FLY_AVG_AREA = 1.6
DIST_TO_FLY_FAR = 2.7
DIST_TO_FLY_INNER_T = 2.5
DIST_TO_FLY_INNER_H = 3

EXP_MALE_GENO_MAP = {
    "C": ("CS", "20201109"),
    "I": ("IR", "20201109"),
}
EFFECTOR_MAP = {
    "c": "CS",
    "s": "Shi",
    "t": "Trp",
	"S": "",
}
def code_to_geno(code):
    g = code[0]
    geno, day1 = EXP_MALE_GENO_MAP.get(g, (None, None))
    if not geno:
        return "", 0
    if len(code) > 1:
        eff = code[1]
        return geno + EFFECTOR_MAP[eff], day1
    else:
        return geno, day1

# NOTE: config for UI
DEFAULT_TWO_POINT_LEN = 20
DEFAULT_GRAY_THRESHOLD = 150
FPS = 66
VERSION = 20191026
ROUND_ARENA = True
POINT_NUM = 5

DIST_TO_CENTER_THRESHOLD = 7
DIST_TO_CENTER_THRESHOLD_FEMALE = 5.5
DIST_TO_CENTER_THRESHOLD_MALE = 6.5

TURN_BINS = 9
FINE_CIR_MALE_MIN_V = 1
FINE_CIR_FEMALE_MAX_V = 5
FINE_CIR_FEMALE_MAX_AV = 120

FA_CIR_FEMALE_MIN_V = 5#1
FA_CIR_FEMALE_MIN_AV = 120


def str2day(s):
    return datetime.strptime(s, "%Y%m%d")

def day_diff(d2, d1):
    return (d2 - d1).days

def day_str_diff(s2, s1):
    return day_diff(str2day(s2), str2day(s1))

def day_add_s(s1, i):
    return day2str(str2day(s1) + timedelta(days=i))

def day2str(d):
    return datetime.strftime(d, "%Y%m%d")

def time_now_str():
    return datetime.strftime(datetime.now(), "%Y%m%d %H:%M:%S")

def tt_to_second(tt):
    return int(tt[0])*3600+int(tt[1])*60+int(tt[2])

def second_to_tt(s):
    return int(s/3600), int(s%3600/60), int(s%60)


