# -*- coding: utf-8 -*-
"""
UI for viewing keypoint file (kpt.csv)
Author: Jing Ning @ SunLab
"""

import sys, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm

sys.path.append(".")
from SoAL_Constants import FIX_VIDEO_SIZE, FLY_NUM
from SoAL_Utils import load_kpt, save_kpt
from SoAL_BodyDetection import center_img, RegionSeg

PLOT_OFFSET = 0
CENTER_RANGE_R = 5

plot_centric = True

g_img = None
g_frame = 0
g_frame_step = 32
g_track_info = None
g_cap = None
g_total_frame = 0
track_file = None
video_file = None
g_slider = None
need_save = False
axes = None
mapx, mapy = None, None

def triangle_for_vector(x, y, px, py, r=4):
    return ((px+y/r)-x, (py-x/r)-y), ((px-y/r)-x, (py+x/r)-y), ((px+x), (py+y))

def triangle_for_angle(a, px, py, l):
    return triangle_for_vector(np.cos(np.deg2rad(a)) * l, np.sin(np.deg2rad(a)) * l, px, py)

def line_for_angle(a, px, py, l):
    return (np.cos(np.deg2rad(a)) * l + px, px), (np.sin(np.deg2rad(a)) * l + py, py)

def vlen(v):
    return np.sqrt((v**2).sum())

def swap_track_fly(track):
    for key in track.keys():
        if key.find("1") >= 0:
            key2 = key.replace("1", "2")
            if key2 in track:
                track[key], track[key2] = track[key2], track[key]

def angle_to_vector(angle):
    d = np.deg2rad(angle)
    return np.array([np.cos(d), np.sin(d)])

g_input_int = 0
def onkey(event):
    print(event.key)
    global g_frame, g_input_int, need_save
    if event.key == "left":
        g_frame -= 1
    elif event.key == "right":
        g_frame += 1
    elif event.key == "up":
        g_frame -= g_frame_step
    elif event.key == "down":
        g_frame += g_frame_step
    elif event.key == "a":
        g_frame = g_total_frame - 1
    elif event.key == "z":
        g_frame = 0
    elif event.key == "h":
        g_frame += 10
    elif event.key == "j":
        g_frame += 100
    elif event.key == "k":
        g_frame += 1000
    elif event.key == "l":
        g_frame += 10000
    elif event.key == "y":
        g_frame -= 10
    elif event.key == "u":
        g_frame -= 100
    elif event.key == "i":
        g_frame -= 1000
    elif event.key == "o":
        g_frame -= 10000
    elif event.key == "f1":
        save_error_frame(1)
        save_error_frame(2)
    elif event.key == "f12":
        save_all_frame()
    elif event.key == " ":
        for f in g_reg_n_1:
            if f > g_frame:
                if f == g_frame + 1:
                    g_frame = f
                    continue
                g_frame = f
                break
    elif event.key == "r":
        for i, f in enumerate(g_reg_n_1):
            if f >= g_frame:
                g_frame = g_reg_n_1[max(i-1, 0)]
                break
    elif event.key == "control":
        for track in g_track_info[g_frame:]:
            swap_track_fly(track)
        need_save = True
    elif event.key == "x":
        save_kpt_correct()
    elif event.key == "enter":
        g_frame = g_input_int
        g_input_int = 0
    elif event.key in list([*"1234567890"]):
        g_input_int = int(event.key) + g_input_int * 10
        print("input: %d" % g_input_int)
        return
    if g_frame >= g_total_frame:
        g_frame = g_total_frame - 1
    if g_frame < 0:
        g_frame = 0
    g_slider and g_slider.set_val(g_frame)
    event.canvas.draw()

def init_track_info(path):
    global g_track_info, g_df, g_total_frame, g_meta, g_roi, g_reg_n_1
    global g_cap, video_file, track_file, need_save
    track_file = path
    parent = os.path.dirname(path)
    pp = os.path.dirname(parent)
    g_df, g_meta = load_kpt(track_file)
    g_reg_n_1 = np.nonzero((g_df["reg_n"] < 2).tolist())[0]
    g_track_info = g_df.to_dict("record")
    g_total_frame = len(g_track_info)
    g_roi = g_meta["ROI"]["roi"]
    video_file = os.path.join(os.path.dirname(parent), os.path.basename(pp) + ".avi")

    if g_cap:
        g_cap.release()
    g_cap = cv2.VideoCapture(video_file)
    if not g_cap:
        print("open video failed!")
        exit(0)

def init_path(path):
    global track_file, video_file
    track_file = path
    video_file = track_file[:track_file.rfind(".")] + ".avi"
    if not os.path.isfile(video_file):
        video_file = track_file[:track_file.rfind("_")] + ".avi"
        if not os.path.isfile(video_file):
            video_file = track_file[:track_file.rfind("_")] + ".mts"

def on_slider(val):
    global g_frame
    g_frame = int(val)
    plot_one_frame()

def plot_one_frame():
    track = g_track_info[g_frame]
    real_frame = track.get("frame")
    plt.suptitle("%s %s" % (os.path.basename(track_file), real_frame))
    if FLY_NUM == 1:
        x, y = np.array(track["1:pos:x"]), np.array(track["1:pos:y"])
    else:
        x, y = np.array((track["1:pos:x"], track["2:pos:x"])), np.array((track["1:pos:y"], track["2:pos:y"]))

    img = plot_video_frame(axes[0], real_frame, track, x, y)
    plot_two_fly_info(axes[1], track, x, y)
    if plot_centric:
        axes[2].cla()
        plot_centric_fly(axes[2], 1, track, img, marker="o")
        if FLY_NUM > 1:
            axes[3].cla()
            plot_centric_fly(axes[3], 2, track, img, marker="x")

def save_all_frame():
    i = 0
    ax100 = None
    r, c = 10, 10
    fpi = r*c
    sh = int(g_roi[1][1]) - int(g_roi[0][1]), int(g_roi[1][0]) - int(g_roi[0][0])
    for frame, track in enumerate(g_track_info):
        reg_n = track["reg_n"]
        # px1, py1 = track["1:pos:x"] / sh[1], track["1:pos:y"] / sh[0]
        # px2, py2 = track["2:pos:x"] / sh[1], track["2:pos:y"] / sh[0]
        if reg_n == 1:# and frame % 10 == 0:
                # or (abs(px1 - 0.5) < 0.1 and abs(py1 - 0.5) < 0.1)\
                # or (abs(px2 - 0.5) < 0.1 and abs(py2 - 0.5) < 0.1):
        # if frame % 10 == 0:
            frame = track["frame"]
            g_cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = g_cap.read()
            if not ret:
                break
            img_gray = img[:, :, 1]
            if mapx is not None:
                img_gray = cv2.remap(img_gray, mapx, mapy, cv2.INTER_LINEAR)
            img_gray = img_gray[int(g_roi[0][1]):int(g_roi[1][1]), int(g_roi[0][0]):int(g_roi[1][0])]

            if i % fpi == 0:
                i = 0
                if ax100 is not None:
                    plt.savefig("temp1/%d.jpg" % frame)
                    plt.close()
                fig, ax100 = plt.subplots(r, c*2, figsize=(c*4, r*2))
                plt.subplots_adjust(left=0, right=1, top=0.99, bottom=0, wspace=0, hspace=0)
                ax100 = ax100.flatten()
            ax1, ax2 = ax100[i*2], ax100[i*2 + 1]
            ax1.text(0, 8, str(frame), color="b")
            i += 1

            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2))
            # plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

            ax1.axis("off")
            plot_centric_fly(ax1, 1, track, img_gray, lw=1, s=50)
            ax2.axis("off")
            plot_centric_fly(ax2, 2, track, img_gray, lw=1, s=50)
            # plt.savefig("temp1/%d.jpg" % frame)
            # plt.close()
    if ax100 is not None:
        plt.savefig("temp1/%d.jpg" % frame)
        plt.close()

def rotate_points(xs, ys):
    body_dir_v = xs[0] - xs[2], ys[0] - ys[2]
    fd = np.pi/2 - np.arctan2(body_dir_v[1], body_dir_v[0])
    xy1 = np.vstack((xs, ys, np.ones((1, 5))))
    xy1center = np.matrix([
        [1, 0, -xs[1]],
        [0, 1, -ys[1]],
        [0, 0, 1],
    ]) * xy1
    xy1_trans = np.matrix([
        [np.cos(fd), -np.sin(fd), 0],
        [np.sin(fd), np.cos(fd), 0],
        [0, 0, 1],
    ]) * xy1center
    return np.array(xy1_trans[0])[0], np.array(xy1_trans[1])[0]

def save_kpt_correct():
    global need_save
    if need_save:
        save_kpt(track_file.replace(".csv", "_correct.csv"), g_track_info)
        need_save = False

g_rs = None
def save_error_frame(fly):
    global g_rs
    if not g_rs:
        g_rs = RegionSeg(video_file, 0, None)
        g_rs.set_model_shape((64, 48), (94, 94))
        g_rs.init_edge_mask()
        g_rs.init_bg()
    roi_i = g_meta["ROI"]["idx"]
    img_p = g_rs.proc_one_img(g_img, g_roi, roi_i, False)[fly-1]
    f = "temp/%s_%d_%d_%d.jpg" % (os.path.basename(video_file)[:-4], roi_i, g_track_info[g_frame].get("frame", g_frame), fly)
    cv2.imwrite(f, img_p)
    print("saved", f)

def plot_video_frame(ax, frame, track, x, y):
    global g_img
    g_cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, img = g_cap.read()
    if ret:
        img_gray = img[:, :, 1]
        g_img = img_gray
        if FIX_VIDEO_SIZE:
            img_gray = cv2.resize(img_gray, FIX_VIDEO_SIZE)
        if mapx is not None:
            img_gray = cv2.remap(img_gray, mapx, mapy, cv2.INTER_LINEAR)
        img_gray = img_gray[int(g_roi[0][1]):int(g_roi[1][1]), int(g_roi[0][0]):int(g_roi[1][0])]
        ax.cla()
        ax.imshow(img_gray, cmap=plt.cm.gray, norm=NoNorm())
        ax.scatter(x + PLOT_OFFSET, y + PLOT_OFFSET, c="br" if FLY_NUM == 2 else "b", s=5, marker="o")
        for fly in range(1, FLY_NUM+1):
            if track.get("%d:part:p0:x" % fly):
                xs, ys = [], []
                for i in range(5):
                    xs.append(track.get("%d:part:p%d:x" % (fly, i), 0) + PLOT_OFFSET)
                    ys.append(track.get("%d:part:p%d:y" % (fly, i), 0) + PLOT_OFFSET)
                ax.scatter(xs, ys, s=15, c="ywmkg", marker="o" if fly == 1 else "x")

                ax.plot(xs[:3], ys[:3], linewidth=0.4, c="w")
                ax.plot(xs[1:4:2], ys[1:4:2], linewidth=0.4, c="w")
                ax.plot(xs[1:5:3], ys[1:5:3], linewidth=0.4, c="w")
            pre = ("%d:" % fly)
            c = "b" if fly == 1 else "r"
            plot_body_box(ax, track[pre + "pos:x"], track[pre + "pos:y"], track[pre + "ori"] if (pre + "ori") in track.keys() else track[pre + "dir"],
                       track[pre + "e_maj"] / 2, track[pre + "e_min"] / 2, track[pre + "area"], c)
    # ax.invert_yaxis()
    return img_gray

def plot_body_box(ax, x, y, d, h_maj, h_min, area, c):
    print(x, y, d, h_maj, h_min, area)
    t = np.deg2rad(d)
    dx1, dy1 = h_maj*np.cos(t), h_maj*np.sin(t)
    dx2, dy2 = -h_min*np.sin(t), h_min*np.cos(t)
    ax.plot([x+dx1, x+dx2, x-dx1, x-dx2, x+dx1], [y+dy1, y+dy2, y-dy1, y-dy2, y+dy1], linewidth=0.5, c=c)

def plot_fly_info(ax, track, fly, color):
    x, y = track["%d:pos:x" % fly], track["%d:pos:y" % fly]
    dir = track["%d:dir" % fly]
    # dir
    e_maj = track.get("%d:e_maj" % fly)
    rect = plt.Polygon(triangle_for_angle(dir, x, y, e_maj/2), color=color, alpha=0.3)
    ax.add_patch(rect)
    # move
    fly_move = track.get("%d:move_dir" % fly)
    if fly_move is not None:
        v_move = angle_to_vector(fly_move)
        line_xs, line_ys = (v_move[0] + x, x), (v_move[1] + y, y)
        ax.add_line(plt.Line2D(line_xs, line_ys, linewidth=0.2, color=color))

        i_move = track.get("%d:i_move_dir" % fly)
        v_move = angle_to_vector(i_move)
        line_xs, line_ys = (v_move[0] + x, x), (v_move[1] + y, y)
        ax.add_line(plt.Line2D(line_xs, line_ys, linewidth=0.2, color=color, alpha=0.5))
        # print("mv_dir %d, i_mv_dir %d" % (fly_move, i_move))
    # if e_maj:
    #     line_xs, line_ys = line_for_angle2(dir, x, y, e_maj/2)
    #     ax.add_line(plt.Line2D(line_xs, line_ys, marker=".", linewidth=0.5, color=color))
    phi_l = track.get("phi%d_l" % fly)
    fly_l = track.get("fly%d_l" % fly)
    if phi_l and fly_l:
        line_xs, line_ys = line_for_angle(dir + phi_l, x, y, -fly_l)
        ax.add_line(plt.Line2D(line_xs, line_ys, marker=".", linewidth=0.5, color=color))
    phi_r = track.get("phi%d_r" % fly)
    fly_r = track.get("fly%d_r" % fly)
    if phi_r and fly_r:
        line_xs, line_ys = line_for_angle(dir - phi_r, x, y, -fly_r)
        ax.add_line(plt.Line2D(line_xs, line_ys, marker=".", linewidth=0.5, color=color))
    if track.get("%d:point:xs" % fly):
        xs, ys = track["%d:point:xs" % fly], track["%d:point:ys" % fly]
        ax.scatter(xs, ys, s=15, c="ywmkg", marker="o" if fly == 1 else "x")

        # print(angle_diff_dir((xs[0] - xs[1], ys[0] - ys[1]), (xs[2] - xs[1], ys[2] - ys[1])))
        ax.plot(xs[:3], ys[:3], linewidth=0.4, c="k")
        ax.plot(xs[1:4:2], ys[1:4:2], linewidth=0.4, c="k")
        ax.plot(xs[1:5:3], ys[1:5:3], linewidth=0.4, c="k")

def plot_two_fly_info(ax, track, x, y):
    # pos, dir, move_vector...
    ax.cla()
    ax.axis("equal")
    ax.scatter(x, y, s=20, c="br" if FLY_NUM == 2 else "b", marker="o")
    plot_fly_info(ax, track, 1, "b")
    if FLY_NUM > 1:
        plot_fly_info(ax, track, 2, "r")
    ax.invert_yaxis()

def get_centered_info2(pos1, dir1, pos2):
    dirv = angle_to_vector(dir1)
    v12 = (pos2 - pos1)

    theta = np.arctan2(dirv[0], dirv[1])
    rotx = (v12[0] * np.cos(theta)) - (v12[1] * np.sin(theta))
    roty = (v12[0] * np.sin(theta)) + (v12[1] * np.cos(theta))

    lenv12 = vlen(v12)
    t = np.cross(v12, dirv)
    if t > 0:
        phi = np.pi / 2 - np.arccos(np.dot(v12, dirv) / lenv12)
    else:
        phi = np.pi / 2 + np.arccos(np.dot(v12, dirv) / lenv12)
    return rotx, roty, lenv12, np.rad2deg(phi)

MODEL_SHAPE = (64, 64)
MODEL_SHAPE_EXTEND = (int(MODEL_SHAPE[0] * 1.5), int(MODEL_SHAPE[1] * 1.5))
def plot_centric_fly(ax, fly, track, img, lw=4, s=95, marker="o"):
    ax.axis("equal")
    pos = (track["%d:pos:x" % fly], track["%d:pos:y" % fly])
    img_ego = center_img(img, pos, track["%d:dir" % fly], MODEL_SHAPE, MODEL_SHAPE_EXTEND)
    ax.imshow(img_ego, cmap=plt.cm.gray, norm=NoNorm())
    ax.scatter([MODEL_SHAPE[0]/2], [MODEL_SHAPE[1]/2], c="b" if fly==1 else "r")
    ax.add_patch(plt.Rectangle((0, 8), 64, 48, color="b" if fly==1 else "r", fill=False))
    if track.get("%d:point:xs" % fly):  # !!!!False
        xso, yso = track["%d:point:xs" % fly], track["%d:point:ys" % fly]
        xs, ys = [], []
        dir1 = track["%d:dir" % fly]
        for i in range(5):
            pos2 = np.array([xso[i], yso[i]])
            x, y, r, phi = get_centered_info2(pos, dir1, pos2)
            xs.append(-x + MODEL_SHAPE[0]/2)
            ys.append(-y + MODEL_SHAPE[1]/2)

        ax.plot(xs[:3], ys[:3], linewidth=lw, c="w", zorder=2)
        ax.plot(xs[1:4:2], ys[1:4:2], linewidth=lw, c="w", zorder=2)
        ax.plot(xs[1:5:3], ys[1:5:3], linewidth=lw, c="w", zorder=2)
        ax.scatter(xs, ys, s=s, c="ywmkg", marker=marker, zorder=10)
    # ax.invert_yaxis()

def main(path):
    global g_track_info, g_cap, g_frame, g_slider, axes
    g_frame = 0
    init_track_info(path)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.21)
    fig.canvas.mpl_connect('key_press_event', onkey)
    from matplotlib.widgets import Slider
    g_slider = Slider(plt.axes([0.05, 0.05, 0.9, 0.05]), "", valmin=0, valmax=g_total_frame-1, valfmt="%d", valinit=0)
    g_slider.on_changed(on_slider)

    plot_one_frame()
    plt.show()
    save_kpt_correct()

if __name__ == "__main__":
    if len(sys.argv) > 2:
        if sys.argv[2] == "console":
            from threading import Thread
            main_tread = Thread(target=main, args=(sys.argv[1],))
            main_tread.start()
            while True:
                cmd = input(">> ")
                try:
                    print(eval(cmd.replace("#", "g_track_info[g_frame]")))
                except Exception:
                    print("exception")
        elif sys.argv[2] == "correct":
            main(sys.argv[1])
    else:
        main(sys.argv[1])
