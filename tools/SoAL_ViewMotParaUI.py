# -*- coding: utf-8 -*-
"""
UI for viewing motion parameter file (mot_para0.pickle)
Author: Jing Ning @ SunLab
"""

import os
import sys
import cv2
from threading import Timer, Thread
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
from matplotlib.widgets import Slider
sys.path.append(".")
from SoAL_Utils import load_dict, load_dfs, load_dataframe

NEED_CONSOLE = False
PLOT_MOT_LENGTH = 100
PLOT_TRACK = True
# court we
# court_30 we30
# court_s we_s
# court_s_30 we30_s
# court_as_male we_as_male
# court_infer_male male_idx
# sidewalk crabwalk
# copulate copulation
# circle circling


mot_para0_keys = ["copulation", "reg_n"] #"we_as_male",
mot_para1_keys = ["circling", "we", "crabwalk_s", "walk", "we_ipsi", "on_edge",  #, "we30_s"
              "dir", "theta", "v_dir",
              "v_len", "vf", "vs", "av",  # "acc_dir", #"acc", "acc_len"
              "wing_m",  # "wing_l", "we_l", "wing_r", "we_r",
                  # "ht_span", "dist_ht",
              "dist_c", "rel_pos:x", "rel_pos:y", "rel_polar:r",# "rel_polar:t",
                  # "rel_pos_h:x", "rel_pos_h:y", "rel_polar_h:r", "rel_polar_h:t",
                  # "rel_pos_t:x", "rel_pos_t:y", "rel_polar_t:r", "rel_polar_t:t"
                  # "rel_polar_t:r", "rel_polar_t:t",
                  # "rel_polar_hh:r", "rel_polar_ht:r"
                  # "we_s_30", "we_s",
                  # "e_maj", "area", "pos:x", "acc"
                  ]
mot_keys = [mot_para0_keys, mot_para1_keys, mot_para1_keys]

class ViewMotPara(object):
    def __init__(self, mot_para0_file, meta_file, video_file, pair=""):
        self.frame = 0
        self.caption = []
        self.input_int = 0
        self.cur_play_frame = -1
        self.timer = None
        self.in_update = False
        self.last_x_range = None

        self.pair = pair
        self.meta = load_dict(meta_file)
        self.dfs = load_dfs(mot_para0_file)

        self.cap = cv2.VideoCapture(video_file)
        if not self.cap:
            print("open video failed! %s" % video_file)
        self.total_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap_size = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.scale_factor = self.meta["SCALE"]
        self.roi = np.array(self.meta["ROI"]["roi"]).astype(int)
        self.roi_size = (self.roi[1][0] - self.roi[0][0], self.roi[1][1] - self.roi[0][1])
        self.roi_size_scale = (self.roi_size[0] / self.scale_factor, self.roi_size[1] / self.scale_factor)

        fig0, ax0 = plt.subplots(len(mot_para0_keys) + len(mot_para1_keys), 1, sharex=True, figsize=(6, 8), num="motion parameters")
        plt.subplots_adjust(hspace=0.2, top=0.99, bottom=0.04, right=0.96)
        self.set_window_pos(0, 0)
        fig0.canvas.mpl_connect("button_press_event", self.on_click_mot)
        fig0.canvas.mpl_connect("key_press_event", self.onkey)
        fig0.canvas.mpl_connect("close_event", self.onclose)
        self.axes = [ax0]
        self.figs = [fig0]

        self.fig, self.cap_ax = plt.subplots(figsize=(6, 8), num="+/-: next/previous circling")
        plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.17)  # , hspace=0.06, wspace=0.06)
        self.slider_ax = plt.axes([0.1, 0.05, 0.78, 0.03])
        self.slider = Slider(self.slider_ax, "", valmin=0, valmax=self.total_frame - 1, valfmt="%d", valinit=0)
        self.slider.on_changed(self.on_slider)
        self.fig.canvas.mpl_connect("key_press_event", self.onkey)
        self.fig.canvas.mpl_connect("close_event", self.onclose)
        self.set_window_pos(600, 0)

    def set_window_pos(self, x, y):
        mngr = plt.get_current_fig_manager()
        mngr.window.wm_geometry("+%d+%d" % (x+500, y+200))

    def plot_one_frame(self):
        # video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame)
        ret, img = self.cap.read()
        if ret:
            img_gray = img[self.roi[0][1]:self.roi[1][1], self.roi[0][0]:self.roi[1][0], 1]
            self.cap_ax.cla()
            self.cap_ax.imshow(img_gray, cmap=plt.cm.gray, norm=NoNorm(), extent=(0, self.roi_size_scale[0], 0, self.roi_size_scale[1]))
            #self.cap_ax.invert_yaxis()
            track = None, self.dfs[1].iloc[self.frame], self.dfs[2].iloc[self.frame]
            for i in (1, 2):
                c = "b" if i == 1 else "r"
                xs, ys = track[i]["point:xs"], track[i]["point:ys"]
                self.cap_ax.scatter(xs, ys, s=35, c="ywmkg", marker="x" if i == 1 else "o")

                self.cap_ax.plot(xs[:3], ys[:3], linewidth=1, c="w")
                self.cap_ax.plot(xs[1:4:2], ys[1:4:2], linewidth=1, c="w")
                self.cap_ax.plot(xs[1:5:3], ys[1:5:3], linewidth=1, c="w")
                self.cap_ax.scatter([track[i]["pos:x"]], [track[i]["pos:y"]], c=c, marker="+")
                self.plot_body_box(track[i]["pos:x"], track[i]["pos:y"], track[i]["dir"], track[i]["e_maj"]/2, track[i]["e_min"]/2, c)

                if PLOT_TRACK:
                    frame_range = 200
                    start_frame = max(0, self.frame - frame_range)
                    frame_count = self.frame - start_frame
                    xs = self.dfs[i]["pos:x"].iloc[start_frame:self.frame]
                    ys = self.dfs[i]["pos:y"].iloc[start_frame:self.frame]
                    self.cap_ax.scatter(xs, ys, c=np.arange(0, 1, 1/(frame_count+1))[:frame_count], s=1, cmap="viridis" if i == 1 else "autumn")

        t_sec = self.frame / self.cap_fps
        self.slider_ax.set_title("%02d:%02.2f" % (t_sec / 60, t_sec % 60), fontsize=20)#set_xlabel(
        self.cap_ax.set_title(self.pair)
        print("#%d" % self.frame)

    def plot_body_box(self, x, y, d, h_maj, h_min, c):
        # return
        print(x, y, d, h_maj, h_min)
        t = np.deg2rad(d)
        dx1, dy1 = h_maj*np.cos(t), h_maj*np.sin(t)
        dx2, dy2 = -h_min*np.sin(t), h_min*np.cos(t)
        self.cap_ax.plot([x+dx1, x+dx2, x-dx1, x-dx2, x+dx1], [y+dy1, y+dy2, y-dy1, y-dy2, y+dy1], linewidth=0.5, c=c)

    def plot_mot_para(self):
        half = PLOT_MOT_LENGTH / 2
        center = np.clip(self.frame, half, self.total_frame - half)
        x_range = (int(center - half), int(center + half))
        i = 0
        for idx in [1, 0]:
            mot_para = self.dfs[idx]
            f = mot_para.iloc[self.frame]
            if idx == 1:
                f2 = self.dfs[2].iloc[self.frame]
            keys = mot_keys[idx]
            for k in keys:
                ax = self.axes[0][i]
                ax.cla()

                if mot_para.get(k) is not None:
                    ax.plot(mot_para[k][x_range[0]:x_range[1]], lw=0.6, label="%s %.2f" % (k, f[k]), color="b" if idx == 1 else "k")
                    # ax.scatter([self.frame], [f[k]], color="b", marker=".", s=1)
                    if idx == 1:
                        ax.plot(self.dfs[2][k][x_range[0]:x_range[1]], lw=0.4, label="%s %.2f" % (k, f2[k]), c="r")
                        # ax.scatter([self.frame], [f2[k]], color="r", marker=".", s=1)
                    ax.legend(loc="upper right", fontsize="x-small")
                    ax.axvline(self.frame, color="k", lw=0.4, alpha=0.6)
                ax.tick_params(axis='y', labelsize="x-small")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                i += 1
            self.figs[0].canvas.draw()

    def on_click_mot(self, event):
        if event.xdata:
            self.frame = int(event.xdata)
            self.update_plot_frame()
            self.plot_mot_para()

    def onkey(self, event):
        print(event.key)
        if event.key == "left":
            self.frame -= 1
        elif event.key == "right":
            self.frame += 1
        elif event.key == "up":
            self.frame = int(self.frame - self.cap_fps)
        elif event.key == "down":
            self.frame = int(self.frame + self.cap_fps)
        elif event.key == "a":
            self.frame = self.total_frame - 1
        elif event.key == "z":
            self.frame = 0
        elif event.key == "g":
            self.frame += 1
        elif event.key == "h":
            self.frame += 10
        elif event.key == "j":
            self.frame += 100
        elif event.key == "k":
            self.frame += 1000
        elif event.key == "l":
            self.frame += 10000
        elif event.key == "y":
            self.frame -= 10
        elif event.key == "u":
            self.frame -= 100
        elif event.key == "i":
            self.frame -= 1000
        elif event.key == "o":
            self.frame -= 10000
        elif event.key == "enter":
            self.frame = self.input_int
            self.input_int = 0
            self.plot_mot_para()
        elif event.key == "c":
            s = "#%s %s" % (self.frame, self.input_int)
            self.caption.append(s)
            print("caption:", s)
            self.input_int = 0
        elif event.key in list([*"1234567890"]):
            self.input_int = int(event.key) + self.input_int * 10
            print("input: %d" % self.input_int)
            return
        elif event.key == " ":
            self.plot_mot_para()
            return
        elif event.key == "+" or event.key == "-":
            self.frame = self.get_last_cir_range()[0] if event.key == "-" else self.get_next_cir_range()[1]
            self.update_plot_frame()
            # if self.cur_play_frame >= 0:
            #     self.cur_play_frame = -1
            #     self.timer.cancel()
            #     self.frame = self.cur_play_frame_range[0 if event.key == "-" else 1]
            #     self.update_plot_frame()
            # else:
            #     if event.key == "+":
            #         self.cur_play_frame_range = self.get_next_cir_range()
            #     else:
            #         self.cur_play_frame_range = self.get_last_cir_range()
            #     self.cur_play_frame = self.cur_play_frame_range[0]
            #     self.timer and self.timer.cancel()
            #     self.timer = Timer(0.1, self.play_frames)
            #     self.timer.start()
            return
        else:
            self.input_int = 0
            return
        if self.frame >= self.total_frame:
            self.frame = self.total_frame - 1
        if self.frame < 0:
            self.frame = 0
        self.update_plot_frame()

    def update_plot_frame(self):
        if self.in_update:
            print("update conflict!")
            return
        self.in_update = True
        self.plot_one_frame()
        self.slider.set_val(self.frame)
        self.fig.canvas.draw()
        self.in_update = False

    def play_frames(self):
        if self.cur_play_frame >= 0 and self.cur_play_frame <= self.cur_play_frame_range[1]:
            print("play_frames %d" % self.cur_play_frame)
            self.frame = self.cur_play_frame
            self.update_plot_frame()
            if self.cur_play_frame >= 0:
                self.timer.cancel()
                self.timer = Timer(0.1, self.play_frames)
                self.timer.start()
                self.cur_play_frame += 1
                return
        self.cur_play_frame = -1

    def get_next_cir_range(self):
        if self.frame >= self.total_frame - 1:
            return
        c = self.dfs[1]["circling"]
        for frame in range(self.frame, self.total_frame):
            if c[frame]:
                s = frame
                break
        for frame in range(s, self.total_frame):
            if c[frame] < 1:
                e = frame
                break
        return s, e

    def get_last_cir_range(self):
        f = self.frame
        if f <= 1:
            return
        c = self.dfs[1]["circling"]
        if c[self.frame] or c[self.frame - 1]:
            for frame in range(0, self.frame):
                if c[self.frame - frame] < 0:
                    f = self.frame - frame - 1
                    break
        for frame in range(0, f):
            if c[f - frame]:
                e = f - frame
                break
        for frame in range(0, e):
            if c[e - frame] < 1:
                s = e - frame
                break
        return s, e

    def onclose(self, val):
        global ex
        ex = True
        plt.close("all")

    def on_slider(self, val):
        self.frame = int(val)
        self.plot_one_frame()

    def show(self):
        self.plot_one_frame()
        self.plot_mot_para()
        plt.show()

vs = None
ex = False
vm = None
def main(path):
    global vs, vm
    if path.endswith("_mot_para0.pickle"):
        prefix = path[:path.rfind("_mot")]
        meta_path = prefix + "_config.json"
        if not os.path.exists(meta_path):
            meta_path = prefix + "_config_circl.json"
        video_parent = os.path.dirname(os.path.dirname(path))
        video_path = os.path.join(video_parent, os.path.basename(video_parent) + ".avi")
        if not os.path.exists(meta_path):
            print(meta_path, "not found")
            return
        if not os.path.exists(video_path):
            print(video_path, "not found")
            return
        vs = ViewMotPara(path, meta_path, video_path)
        vm = vs.dfs[1]
        vs.show()
    else:
        vm = load_dataframe(path)
        print("len:", len(vm))
        print(vm.keys())
        return

if __name__ == "__main__":
    if NEED_CONSOLE:
        main_tread = Thread(target=main, args=(sys.argv[1],))
        main_tread.start()
        print(">>100\n>> 1.1000.av\n>> vm[\"frame\"]")
        while not ex:
            cmd = input(">> ")
            try:
                if cmd[0].isdigit():
                    if cmd.isdigit():
                        print(vm.iloc[int(cmd)])
                    else:
                        cmd_t = cmd.split(".")
                        print(vs.dfs[int(cmd_t[0])].iloc[int(cmd_t[1])][cmd_t[2]])
                else:
                    print(eval(cmd))
            except Exception:
                print("trace")
    else:
        main(sys.argv[1])
