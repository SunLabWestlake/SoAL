# -*- coding: utf-8 -*-
"""
Behavior annotation & ID assignment
Author: Jing Ning @ SunLab
"""
import os
import sys
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np

from SoAL_Constants import POOL_SIZE
from SoAL_Utils import load_kpt, distance, angle_points, lim_dir_a, angle_diff_a, save_dfs, save_dict, load_dfs, \
    load_dict

NEED_UNDISTORT = False
ANGLE_EDGE_MIN = 10
ANGLE_WE_MIN = 45  # wing extension
ANGLE_WE_MIN_30 = 30  # wing extension
ANGLE_WE_MAX = 120  # wing extension

SPEED_STATIC_MAX = 0.3  # (mm/s) for determine static(speed range 0-10)
ANGLE_SIDE = 30  # (degree) for determine backward and side walk

STATE_FORWARD = 0
STATE_SIDE_WALK = 1
STATE_STATIC = 2
STATE_BACKWARD = 3

COP_DURATION_MIN = 60  # seconds
CIR_DURATION_MIN = 0.4  # (s)(0.4s) min frames
CIR_SIDE_RATIO_MIN = 0.5  # (percent) for determine circling

WINGEXT_N_CONTINUE = 2  # smooth (NOTE: FPS dependent)
CIR_CONTINUE = 2  # (frame) determine circling start & end (NOTE: FPS dependent)
OVERLAP_CONTINUE_T = 0.12  # NOTE: unequal tail distribution
SIDE_WALK_CONTINUE_T = 0.08
WINGEXT_CONTINUE_T = 0.12

CIR_RECHECK_SPEED = 2
CIR_RECHECK_AV = 20
CIR_MIN_X_RANGE = 0.7  # mm
CIR_MIN_DIR_STD = 12  # (NOTE: FPS dependent)
CIR_RECHECK_DIST_MAX = 5  # (NOTE: body size dependent)

def calc_param(df, meta):
    scale = meta["SCALE"]
    roi = meta["ROI"]["roi"]
    fps = meta["FPS"]
    max_y = roi[1][1] - roi[0][1]
    center = (roi[1][0] - roi[0][0]) / scale / 2, max_y / scale / 2

    def parse_and_scale(l):
        return [float(tt) / scale for tt in l]

    def parse_and_scale_y(l):
        return [(max_y - float(tt)) / scale for tt in l]  # NOTE: flip y

    dfs = df[["frame", "reg_n"]], df[["frame"]], df[["frame"]]
    dfs[0]["time"] = df["frame"] / fps
    for fly in (1, 2):
        dd = dfs[fly]
        dd["pos:x"] = df["%d:pos:x" % fly] / scale
        dd["pos:y"] = (max_y - df["%d:pos:y" % fly]) / scale
        dd["e_maj"] = df["%d:e_maj" % fly] / scale
        dd["e_min"] = df["%d:e_min" % fly] / scale
        dd["area"] = dd["e_maj"] * dfs[fly]["e_min"] * np.pi / 4
        dd["point:xs"] = df["%d:point:xs" % fly].apply(parse_and_scale)
        dd["point:ys"] = df["%d:point:ys" % fly].apply(parse_and_scale_y)

        xsi, ysi = [], []
        for i in range(5):
            xsi.append(dfs[fly]["point:xs"].apply(lambda x: x[i]))
            ysi.append(dfs[fly]["point:ys"].apply(lambda x: x[i]))
        head = xsi[0], ysi[0]
        thorax = xsi[1], ysi[1]
        tail = xsi[2], ysi[2]
        wingl = xsi[3], ysi[3]
        wingr = xsi[4], ysi[4]
        dd["dist_c"] = distance(head, center)
        body_dir_v = head[0] - tail[0], head[1] - tail[1]
        body_dir = np.rad2deg(np.arctan2(body_dir_v[1], body_dir_v[0]))
        dd["dir"] = body_dir
        awl = angle_points(tail, thorax, wingl)
        awr = angle_points(tail, thorax, wingr)
        dd["wing_l"] = awl
        dd["wing_r"] = awr
        dd["on_edge"] = (((awl > ANGLE_EDGE_MIN) & (awr > ANGLE_EDGE_MIN)) | ((awl < -ANGLE_EDGE_MIN) & (awr < -ANGLE_EDGE_MIN)))
        phi_la = np.abs(awl)
        phi_ra = np.abs(awr)
        we_l = (~dd["on_edge"]) & (ANGLE_WE_MAX > phi_la) & (phi_la > ANGLE_WE_MIN)
        we_ls = (~dd["on_edge"]) & (ANGLE_WE_MAX > phi_la) & (phi_la > ANGLE_WE_MIN_30)
        we_r = (~dd["on_edge"]) & (ANGLE_WE_MAX > phi_ra) & (phi_ra > ANGLE_WE_MIN)
        we_rs = (~dd["on_edge"]) & (ANGLE_WE_MAX > phi_ra) & (phi_ra > ANGLE_WE_MIN_30)
        dd["we_l"] = we_l
        dd["we_r"] = we_r
        dd["we_lr"] = we_l & we_r
        dd["wing_m"] = np.maximum(phi_la, phi_ra)
        dd["we"] = we_l | we_r
        dd["we30"] = we_ls | we_rs
        dd["we_ipsi"] = 0
    calc_center_info(dfs)
    for fly in (1, 2):
        dd = dfs[fly]
        idx_wl = dd["we_l"] & (~dd["we_r"])
        idx_wr = dd["we_r"] & (~dd["we_l"])
        idx_l = dd["rel_pos_h:x"] < 0
        dd["we_ipsi"][idx_wl & idx_l] = 1
        dd["we_ipsi"][idx_wl & ~idx_l] = -1
        dd["we_ipsi"][idx_wr & idx_l] = -1
        dd["we_ipsi"][idx_wr & ~idx_l] = 1
    return dfs

def calc_center_info(dfs):
    print("calc_center_info ...")
    head = [None] * 3
    center = [None] * 3
    tail = [None] * 3
    body_dir = [None] * 3
    for fly in (1, 2):
        dd = dfs[fly]
        xs = dd["point:xs"]
        ys = dd["point:ys"]
        center[fly] = dd["pos:x"], dd["pos:y"]
        head[fly] = xs.apply(lambda x: x[0]), ys.apply(lambda x: x[0])
        tail[fly] = xs.apply(lambda x: x[2]), ys.apply(lambda x: x[2])
        body_dir[fly] = dd["dir"]
    for fly in (1, 2):
        # 1:rel_pos_h|c|t:x|y, 1:rel_polar_h|c|t:r|t, 1:rel_pos_hh|ht|th:x|y, 1:rel_polar_hh|ht|th:r|t
        rel_x, rel_y, rel_r, rel_phi = get_centered_info_l(center[fly], body_dir[fly], center[3-fly])
        rel_c_h = get_centered_info_l(center[fly], body_dir[fly], head[3-fly])  # c-h
        rel_c_t = get_centered_info_l(center[fly], body_dir[fly], tail[3-fly])  # c-t
        rel_h_h = get_centered_info_l(head[fly], body_dir[fly], head[3-fly])  # h-h
        rel_h_t = get_centered_info_l(head[fly], body_dir[fly], tail[3-fly])  # h-t
        dfs[fly]["rel_pos:x"], dfs[fly]["rel_pos:y"] = rel_x, rel_y
        dfs[fly]["rel_polar:r"], dfs[fly]["rel_polar:t"] = rel_r, rel_phi
        dfs[fly]["rel_pos_h:x"], dfs[fly]["rel_pos_h:y"] = rel_c_h[0], rel_c_h[1]
        dfs[fly]["rel_polar_h:r"], dfs[fly]["rel_polar_h:t"] = rel_c_h[2], rel_c_h[3]
        dfs[fly]["rel_pos_t:x"], dfs[fly]["rel_pos_t:y"] = rel_c_t[0], rel_c_t[1]
        dfs[fly]["rel_polar_t:r"], dfs[fly]["rel_polar_t:t"] = rel_c_t[2], rel_c_t[3]
        dfs[fly]["rel_polar_hh:r"], dfs[fly]["rel_polar_hh:t"] = rel_h_h[2], rel_h_h[3]
        dfs[fly]["rel_polar_ht:r"], dfs[fly]["rel_polar_ht:t"] = rel_h_t[2], rel_h_t[3]

def get_centered_info_l(center, body_dir, part):
    center_x, center_y = center
    part_x, part_y = part
    d = np.deg2rad(body_dir)
    dirv_x, dirv_y = np.cos(d), np.sin(d)
    vx, vy = part_x - center_x, part_y - center_y

    rotx = (vx * dirv_y) - (vy * dirv_x)
    roty = (vx * dirv_x) + (vy * dirv_y)

    lenv = np.sqrt((vx * vx) + (vy * vy))
    phi = lim_dir_a(np.rad2deg((np.arctan2(vy, vx) - d)))
    return rotx, roty, lenv, phi

def walk_state_a(speed, theta):
    sta = speed <= SPEED_STATIC_MAX
    state = sta * STATE_STATIC
    backward = theta > 180 - ANGLE_SIDE
    state[~sta & backward] = STATE_BACKWARD
    state[~sta & ~backward & (theta > ANGLE_SIDE)] = STATE_SIDE_WALK
    return state

def calc_v(dfs, fps, post_id_correct=False):
    print("fps=", fps)
    d = int(fps/30.0 + 0.5)
    fps_scale = fps / (d*2)
    for fly in (1, 2):
        df = dfs[fly]
        i = df.index
        dfb = df.reindex(i - d, method="nearest")
        dff = df.reindex(i + d, method="nearest")
        xb, yb = np.array(dfb["pos:x"]), np.array(dfb["pos:y"])
        xf, yf = np.array(dff["pos:x"]), np.array(dff["pos:y"])
        vx, vy = xf - xb, yf - yb

        v_len = np.sqrt(vx**2 + vy**2) * fps_scale
        v_dir = np.rad2deg(np.arctan2(vy, vx))
        theta = angle_diff_a(v_dir, np.array(df["dir"]))
        df["theta"] = theta
        df["av"] = angle_diff_a(np.array(dff["dir"]), np.array(dfb["dir"])) * fps_scale
        df["v_len"] = v_len
        df["v_dir"] = v_dir
        df["walk"] = walk_state_a(v_len, np.fabs(theta))

        if post_id_correct:
            theta_r = np.deg2rad(theta)
            df["vs"] = v_len * np.sin(theta_r)
            df["vf"] = v_len * np.cos(theta_r)
            accx, accy = (vx[d:] - vx[:-d]), (vy[d:] - vy[:-d])
            acc_len = np.sqrt(accx**2, accy**2) * fps_scale
            df["acc"] = np.hstack([[np.nan]*d, (v_len[d:] - v_len[:-d]) * fps_scale])
            df["acc_dir"] = np.hstack([[np.nan]*d, np.rad2deg(np.arctan2(accy, accx))])
            df["acc_len"] = np.hstack([[np.nan]*d, acc_len])
    return dfs

def correct_id(dfs, behs):
    frames = len(dfs[0])
    for last_male in behs[0]["male_idx"]:
        if last_male > 0:
            break
    i_l = []
    for i in tqdm(range(frames), "loop 4 correct_id"):
        male = behs[0]["male_idx"][i]
        if male != 0:
            last_male = male
        if last_male == 2:
            i_l.append(i)

    dfs[1].iloc[i_l], dfs[2].iloc[i_l] = dfs[2].iloc[i_l], dfs[1].iloc[i_l]
    for i in i_l:
        for k in behs[1].keys():
            behs[1][k][i], behs[2][k][i] = behs[2][k][i], behs[1][k][i]
    print("\ncorrect_id [%d] frames" % len(i_l))
    return dfs, behs

def infer_male(we_as_male, overlap, frames):
    # NOTE: infer male by extend we_as_male to non-overlap ranges
    # NOTE: overlap--no_wing_ext--overlap not corrected
    male_idx = np.zeros(frames, dtype=int)
    p = 0
    last_m = 0
    end = frames - 1
    for i, ov in enumerate(overlap):
        male = we_as_male[i]
        if male == 0:
            if ov:
                last_m = 0
                p = 0
            else:
                if last_m:
                    male_idx[i] = last_m
                else:
                    p += 1
        if male != 0 or i == end:
            last_m = male
            male_idx[i] = last_m
            if p > 0:
                for pp in range(p):
                    male_idx[i - pp - 1] = last_m
                p = 0
    return male_idx

def calc_bouts(a, find_v=1):
    s = -1
    ret = []
    end = len(a) - 1
    for i, v in enumerate(a):
        condition = v == find_v
        if condition:
            if s < 0:
                s = i
        if not condition or i == end:
            if i == end:
                i += 1
            if s >= 0:
                ret.append((s, i))
            s = -1
    return ret

def smooth_sequence(a, inter, remove_v=False, fill_v=True):
    # NOTE: replace continuous "remove_v" shorter than "inter" by "fill_v"
    c = 0
    ret = []
    for i, v in enumerate(a):
        condition = v == remove_v
        if condition:
            c += 1
        else:
            if c <= inter:
                ret.extend([fill_v] * c)
            else:
                ret.extend([remove_v] * c)
            c = 0
            ret.append(v)
    if c > 0:
        if c <= inter:
            ret.extend([fill_v] * c)
        else:
            ret.extend([remove_v] * c)
    return np.array(ret)

WRONG_ANGLE_CHANGE_MIN = 50
def correct_angle(v):
    # return np.array(v) - v[0]
    ret = []
    li = v[0]
    offset = 0
    for i in v:
        i += offset
        d = i - li
        if d > 150:
            offset -= 360
            i -= 360
        elif d < -150:
            offset += 360
            i += 360
        ret.append(i)
        li = i
    for j in range(1, len(ret) - 1):
        i1, i2, i3 = ret[j - 1], ret[j], ret[j + 1]
        if abs(i1 - i3) < WRONG_ANGLE_CHANGE_MIN and abs(i2 - i3) > WRONG_ANGLE_CHANGE_MIN and abs(i2 - i1) > WRONG_ANGLE_CHANGE_MIN:
            ret[j] = (i1 + i3) / 2
    return ret

def find_wing_extention(dfs, meta):
    fps = meta["FPS"]
    dfs0 = dfs[0]
    behs = [{}, {}, {}]
    frames = len(dfs0)

    we_s_l = []
    for fly in (1, 2):
        dfs1 = dfs[fly]
        we_s = smooth_sequence(dfs1["we"] == 1, WINGEXT_N_CONTINUE, True, False)  # shrink
        we_s = smooth_sequence(we_s, WINGEXT_CONTINUE_T * fps)
        we_s_l.append(we_s)
        behs[fly]["we_s"] = we_s

    we_as_male = np.zeros(frames, dtype=int)
    we_as_male[we_s_l[0]] = 1
    we_as_male[we_s_l[1]] = 2
    we_as_male[we_s_l[0] & we_s_l[1]] = 0
    overlap = dfs0["reg_n"] < 2
    behs[0]["we_as_male"] = we_as_male
    behs[0]["male_idx"] = infer_male(we_as_male, overlap, frames)
    return behs

def calc_beh(dfs, meta, behs):
    fps = meta["FPS"]
    dfs0 = dfs[0]
    # behs = [{}, {}, {}]
    frames = len(dfs0)
    calc_v(dfs, fps, True)

    cop_frame_min_frame = COP_DURATION_MIN * fps
    cir_frame_min_frame = CIR_DURATION_MIN * fps
    overlap = dfs0["reg_n"] < 2
    overlap_s = smooth_sequence(overlap, OVERLAP_CONTINUE_T*fps, 0, 1)
    copulation_s = smooth_sequence(overlap_s, cop_frame_min_frame, 1, 0)  # NOTE: keep overlap longer than 1min
    behs[0]["copulation"] = copulation_s

    we_s_l = []
    for fly in (1, 2):
        dfs1 = dfs[fly]
        walk = dfs1["walk"] == STATE_SIDE_WALK
        crabwalk_s = smooth_sequence(walk, SIDE_WALK_CONTINUE_T*fps)
        we_s = smooth_sequence(dfs1["we"] == 1, WINGEXT_N_CONTINUE, True, False)  # shrink
        we_s = smooth_sequence(we_s, WINGEXT_CONTINUE_T*fps)
        we_s_l.append(we_s)
        # NOTE: cir conditions:
        #   in side-walking
        #   in wing extension
        #   not overlap
        we30_s = smooth_sequence(dfs1["we30"] == 1, WINGEXT_N_CONTINUE, True, False)  # shrink
        we30_s = smooth_sequence(we30_s, WINGEXT_CONTINUE_T*fps)
        overlap_s = smooth_sequence(overlap, OVERLAP_CONTINUE_T*fps, 1, 0)  # shrink
        circle_s1 = smooth_sequence(crabwalk_s & we30_s & ~overlap_s, CIR_CONTINUE)  #
        circle_s = smooth_sequence(circle_s1, cir_frame_min_frame, True, False)  # NOTE: keep side walk longer than 0.4s

        behs[fly]["we_s"] = we_s
        behs[fly]["we30_s"] = we30_s
        behs[fly]["crabwalk_s"] = crabwalk_s
        behs[fly]["circle_s1"] = circle_s1
        behs[fly]["circling"] = circle_s

    for fly in (1, 2):
        # NOTE: recheck circle
        dfs1 = dfs[fly]
        circle_s = behs[fly]["circling"]
        cir_bouts = calc_bouts(circle_s, True)
        v_len = dfs1["v_len"]
        walk = dfs1["walk"]
        av = dfs1["av"]
        pos_x = dfs1["pos:x"]
        pos_y = dfs1["pos:y"]
        body_dir = dfs1["dir"]
        for s, e in cir_bouts:
            l = e - s
            if l < CIR_DURATION_MIN * fps:
                circle_s[s:e] = False
                # print("%d %d length=%.2f" % (s, e, l))
                continue
            side_r = np.count_nonzero(walk[s:e]) / l
            if side_r < CIR_SIDE_RATIO_MIN:
                circle_s[s:e] = False
                # print("%d %d side_r=%.2f" % (s, e, side_r))
                continue
            v1 = np.mean(v_len[s:e])
            if v1 < CIR_RECHECK_SPEED:
                # print("%d %d v1=%.2f" % (s, e, v1))
                circle_s[s:e] = False
                continue
            av1 = np.mean(np.abs(av[s:e]))
            if av1 < CIR_RECHECK_AV:
                circle_s[s:e] = False
                # print("%d %d av1=%.2f" % (s, e, av1))
                continue
            xs, ys = pos_x[s:e], pos_y[s:e],
            x_lim = np.max(xs) - np.min(xs)
            y_lim = np.max(ys) - np.min(ys)
            if x_lim < CIR_MIN_X_RANGE and y_lim < CIR_MIN_X_RANGE:
                circle_s[s:e] = False
                # print("%d %d x_lim=%.2f y_lim=%.2f" % (s, e, x_lim, y_lim))
                continue
            dir_std = np.std(correct_angle(np.array(body_dir[s:e])))
            if dir_std < CIR_MIN_DIR_STD:
                circle_s[s:e] = False
                # print("%d %d dir_std=%.2f" % (s, e, dir_std))
                continue
            dist1 = np.mean(dfs1["rel_polar:r"][s:e])
            if dist1 > CIR_RECHECK_DIST_MAX:
                circle_s[s:e] = False
                # print("%d %d dist1=%.2f" % (s, e, dist1))
                continue
        behs[fly]["circling"] = circle_s

    for i in range(3):
        for k in behs[i].keys():
            dfs[i][k] = behs[i][k]
    return dfs

def convert_pickle_to_csv(pickle_path):
    dfs = load_dfs(pickle_path)
    meta = load_dict(pickle_path.replace("mot_para0.pickle", "config_circl.json"))
    scale = meta["SCALE"]
    roi = meta["ROI"]["roi"]
    max_y = roi[1][1] - roi[0][1]
    keys = ['area', 'pos:x', 'pos:y', 'dir', 'e_maj', 'e_min', 'point:xs', 'point:ys']
    dfs12 = []
    for i in (1, 2):
        dfs1 = dfs[i][keys]
        dfs1['area'] = dfs1['area'] * scale * scale
        dfs1['e_maj'] = dfs1['e_maj'] * scale
        dfs1['e_min'] = dfs1['e_min'] * scale
        dfs1['pos:x'] = dfs1['pos:x'] * scale
        dfs1['pos:y'] = max_y - dfs1['pos:y'] * scale
        dfs1['point:xs'] = dfs1['point:xs'].apply(lambda x: " ".join([str(np.round(xx*scale, 2)) for xx in x]))
        dfs1['point:ys'] = dfs1['point:ys'].apply(lambda x: " ".join([str(np.round(max_y - xx*scale, 2)) for xx in x]))
        dfs1.columns = ["%d:" % i + key for key in keys]
        dfs12.append(dfs1)
    dfo = pd.concat([
        dfs[0][['frame', 'reg_n']],
        dfs12[0].round(2), dfs12[1].round(2)
    ], axis=1)
    dfo.to_csv(pickle_path.replace("mot_para0.pickle", "stat.csv"), index=False)

def main(kpt):
    print("[BehAnn] load", kpt)
    prefix = kpt.replace("_kpt.csv", "")
    df, meta = load_kpt(kpt, calib=NEED_UNDISTORT)
    print("[BehAnn] calc params ...")
    dfs = calc_param(df, meta)  # NOTE: calc parameters
    print("[BehAnn] assign id ...")
    behs = find_wing_extention(dfs, meta)  # NOTE: find overlap frames & wing extension frames
    dfs, behs = correct_id(dfs, behs)  # NOTE: identity assignment
    print("[BehAnn] annotate behavior ...")
    dfs = calc_beh(dfs, meta, behs)  # NOTE: behavior annotation (circling, copulation, wing extension)
    save_dfs(dfs, prefix)

    config_circl = meta
    config_circl["copulation"] = calc_bouts(dfs[0]["copulation"])
    config_circl["cir_bouts1"] = calc_bouts(dfs[1]["circling"])
    config_circl["cir_bouts2"] = calc_bouts(dfs[2]["circling"])
    save_dict(prefix + "_config_circl.json", config_circl)

    # convert_pickle_to_csv(prefix + "_mot_para0.pickle")

if __name__ == "__main__":
    kpt = sys.argv[1]
    if os.path.isdir(kpt):
        kpt_l = glob(os.path.join(kpt, "*", "*_kpt.csv"))
        if POOL_SIZE > 0:
            from multiprocessing import Pool
            p = Pool(POOL_SIZE)
            p.map(main, kpt_l)
            p.close()
        else:
            for ff in kpt_l:
                main(ff)
    else:
        main(kpt)
