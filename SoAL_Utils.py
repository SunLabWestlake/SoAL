# -*- coding: utf-8 -*-
"""
Utils
Author: Jing Ning @ SunLab
"""

import json
import os
import cv2
import numpy as np
import pandas as pd
from SoAL_Constants import FLY_NUM, POINT_NUM, DURATION_LIMIT


def load_dict(filename):
    if not os.path.exists(filename):
        return None
    # print("load_dict %s" % filename)
    f = open(filename, "r")
    j = json.load(f)
    f.close()
    return j

def save_dict(filename, obj):
    f = open(filename, "w")
    json.dump(obj, f, indent=4)
    f.close()
    print("save_dict %s" % filename)

def array_to_str(a):
    return " ".join(["%.2f" % p for p in a])

def distance(p1, p2):
    return np.sqrt(distance2(p1, p2))

def distance2(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def find_file(folder, postfix):
    ret = []
    for name in os.listdir(folder):
        if name.endswith(postfix):
            ret.append(os.path.join(folder, name))
    return ret

KPT_HEADER = ["frame", "reg_n", "area", "pos:x", "pos:y", "ori", "e_maj", "e_min", "point:xs", "point:ys"]

def to_kpt_s(info_n):
    # "frame", "reg_n", ":area", ":pos:x", ":pos:y", ":ori", ":e_maj", ":e_min", ":point:xs", ":point:ys"
    ret = ""
    frame = 0
    reg_n = 0
    for info in info_n:
        frame, reg, reg_n, points = info
        area, center, orient90, major, minor, center_global = reg
        ret += ",%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%s,%s" % (area, center[0], center[1],
                                                        orient90, major, minor,
                                                        array_to_str(points[:POINT_NUM]),
                                                        array_to_str(points[POINT_NUM:POINT_NUM + POINT_NUM]))
    return "%d,%d" % (frame, reg_n) + ret

def parse_float_list(xs):
    return [float(x) for x in xs.split()]

def id_assign_by_last_frame(fly_info, last_fly_info):
    if not last_fly_info or FLY_NUM == 1:
        return fly_info
    ret = [None] * FLY_NUM
    flag = [False] * FLY_NUM
    for i in range(FLY_NUM):
        posi = fly_info[i][1][1]
        dl = []
        for j in range(FLY_NUM):
            if flag[j]:
                dl.append(np.inf)
            else:
                posj = last_fly_info[j][1][1]
                dl.append(distance2(posi, posj))
        min_j = np.argmin(dl)
        ret[min_j] = fly_info[i]
        flag[min_j] = True
    return ret

def id_assign_dict(d): # NOTE: id assign by distance
    fly_info = []
    last_fly_info = None
    ret = []
    for t in d:
        pos = (t["pos:x"], t["pos:y"])
        fly_info.append([t, (0, pos)])
        if len(fly_info) >= FLY_NUM:
            fly_info = id_assign_by_last_frame(fly_info, last_fly_info)
            t_all = {"frame": t["frame"], "reg_n": t["reg_n"]}
            for i, info in enumerate(fly_info):
                k1 = "%d:" % (i+1)
                for k in KPT_HEADER:
                    t_all[k1 + k] = info[0][k]
            ret.append(t_all)
            last_fly_info = fly_info
            fly_info = []
    return ret

def load_kpt_csv(kpt_file, calib=False):
    end_csv = kpt_file.split("_")[-1]
    meta = load_dict(kpt_file.replace(end_csv, "config.json"))
    kptc_df = pd.read_csv(kpt_file, nrows=DURATION_LIMIT * int(meta["FPS"] + 0.5) * 2)
    if FLY_NUM > 1:
        kptc_df.sort_values("frame", inplace=True)
        kptc = kptc_df.to_dict(orient="records")
        kptc = id_assign_dict(kptc)
        kptc_df = pd.DataFrame(kptc)
    if calib:
        meta, kptc_df = calib_kpt(meta, kptc_df, kpt_file)
    return meta, kptc_df

def load_kpt(kpt_file, need_extra=True, calib=False):
    meta, df = load_kpt_csv(kpt_file, calib)
    if not need_extra:
        return df, meta
    for i in range(1, 1 + FLY_NUM):
        df["%d:point:xs" % i] = df["%d:point:xs" % i].map(parse_float_list)
        df["%d:point:ys" % i] = df["%d:point:ys" % i].map(parse_float_list)

        headx = df["%d:point:xs" % i].map(lambda t: t[0])
        heady = df["%d:point:ys" % i].map(lambda t: t[0])
        tailx = df["%d:point:xs" % i].map(lambda t: t[2])
        taily = df["%d:point:ys" % i].map(lambda t: t[2])
        body_dir_v = headx - tailx, heady - taily
        df["%d:dir" % i] = np.rad2deg(np.arctan2(body_dir_v[1], body_dir_v[0]))
    return df, meta

def save_kpt(filename, kpt):
    df = pd.DataFrame(kpt)
    # keys = get_kpt_header()
    for i in range(1, 1 + FLY_NUM):
        df["%d:point:xs" % i] = df["%d:point:xs" % i].map(lambda t: " ".join([str(tt) for tt in t]))
        df["%d:point:ys" % i] = df["%d:point:ys" % i].map(lambda t: " ".join([str(tt) for tt in t]))
    return df.to_csv(filename, index=False)#, columns=keys)

def save_dataframe(df, filename):
    print("save_df: ", filename)
    if filename.endswith(".pickle"):
        df.to_pickle(filename, compression="gzip")  # "infer", "gzip"
    else:
        df.to_csv(filename, index=False)

def load_dataframe(filename):
    pickle = filename.replace(".csv", ".pickle")
    if os.path.exists(pickle):
        print("load_df pickle: ", pickle)
        return pd.read_pickle(pickle, compression='gzip')
    if os.path.exists(filename):
        print("load_df: ", filename)
        return pd.read_csv(filename)
    return None

def load_dfs(kpt_file):  # NOTE: mot_para0.pickle|pair_folder...
    if os.path.isdir(kpt_file):
        b = os.path.basename(kpt_file)
        if len(b) <= 2:
            pp = os.path.basename(os.path.dirname(kpt_file))
            prefix = os.path.join(kpt_file, "%s_%s" % (pp, b))
        else:
            prefix = os.path.join(kpt_file, os.path.basename(kpt_file))
    else:
        prefix = kpt_file[:kpt_file.rfind("_mot")]

    mot_para0 = load_dataframe(prefix + "_mot_para0.pickle")
    mot_para1 = load_dataframe(prefix + "_mot_para1.pickle")
    mot_para2 = load_dataframe(prefix + "_mot_para2.pickle")
    dfs = [mot_para0, mot_para1, mot_para2]
    return dfs

def save_dfs(dfs, prefix):
    save_dataframe(dfs[0], prefix + "_mot_para0.pickle")
    save_dataframe(dfs[1], prefix + "_mot_para1.pickle")
    save_dataframe(dfs[2], prefix + "_mot_para2.pickle")

def angle_points(p1, c, p2):
    return dir_diff((p2[0] - c[0], p2[1] - c[1]), (p1[0] - c[0], p1[1] - c[1]))

def lim_dir_a(a):
    a[a > 180] -= 360
    a[a < -180] += 360
    return a

def dir_diff(body_dir, wing_dir):
    theta = np.rad2deg(np.arctan2(body_dir[1], body_dir[0]) - np.arctan2(wing_dir[1], wing_dir[0]))
    return lim_dir_a(theta)

def angle_diff_a(a1, a2, r=360):
    t = (a1 - a2) % r
    t[t >= r/2] -= r
    return t

def calib_kpt(meta, df, kpt_file):
    import pickle
    camera_name = kpt_file.split("_")[-3]
    calib_info = pickle.load(open("tools/calib/calib_%s_info.pickle" % camera_name, "rb"))
    mtx, dist, newcameramtx, sz = calib_info
    meta["calib"] = True
    roi = meta["ROI"]["roi"]
    x0, y0 = roi[0][0], roi[0][1]
    points = [roi[0], [roi[1][0], roi[0][1]], [roi[0][0], roi[1][1]], roi[1]]
    u_points = cv2.undistortPoints(np.array([points]).astype(float), mtx, dist, P=newcameramtx).squeeze()
    u_roi = [[min(u_points[0][0], u_points[2][0]), min(u_points[0][1], u_points[1][1])],
             [max(u_points[1][0], u_points[3][0]), max(u_points[2][1], u_points[3][1])]]
    ux0, uy0 = u_roi[0][0], u_roi[0][1]
    meta["ROI"]["roi"] = u_roi
    save_dict(kpt_file.replace("kpt.csv", "calib_config.json"), meta)

    def undistort_row(tc, fly):
        xs = [float(tt) + x0 for tt in tc["%d:point:xs" % fly].split()]
        ys = [float(tt) + y0 for tt in tc["%d:point:ys" % fly].split()]
        points = np.array(list(zip(xs, ys)))
        u_points = cv2.undistortPoints(np.array([points]), mtx, dist, P=newcameramtx).squeeze()
        u_xs = u_points[:, 0] - ux0
        u_ys = u_points[:, 1] - uy0
        r_xs = " ".join(["%.2f" % x for x in u_xs])
        r_ys = " ".join(["%.2f" % x for x in u_ys])
        return r_xs, r_ys

    for i in range(1, 1 + FLY_NUM):
        a_pos_x = df["%d:pos:x" % i] + x0
        a_pos_y = df["%d:pos:y" % i] + y0
        a_points = np.array(list(zip(a_pos_x, a_pos_y)))
        u_points = cv2.undistortPoints(a_points, mtx, dist, P=newcameramtx).squeeze()
        df["%d:pos:x" % i] = u_points[:, 0] - ux0
        df["%d:pos:y" % i] = u_points[:, 1] - uy0
        r_points = df.apply(undistort_row, axis=1, args=(i,))
        r_points = np.array(r_points.tolist())
        df["%d:point:xs" % i] = r_points[:, 0]
        df["%d:point:ys" % i] = r_points[:, 1]
    return meta, df

def get_video_in_dir(video_dir):
    f = os.path.basename(video_dir)
    return os.path.join(video_dir, f + ".avi")

def calib_video_frame(kpt_file, calib_info, u_roi, roi, frame=0):
    mtx, dist, newcameramtx, sz = calib_info
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, sz, 5)
    video_file = get_video_in_dir(os.path.dirname(os.path.dirname(kpt_file)))
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, img = cap.read()
    img_calib = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    img_calib = img_calib[int(u_roi[0][1]):int(u_roi[1][1]), int(u_roi[0][0]):int(u_roi[1][0])]
    img = img[int(roi[0][1]):int(roi[1][1]), int(roi[0][0]):int(roi[1][0])]
    cap.release()
    cv2.imwrite(kpt_file.replace("kpt.csv", "calib_on.png"), img_calib)
    cv2.imwrite(kpt_file.replace("kpt.csv", "calib_off.png"), img)
