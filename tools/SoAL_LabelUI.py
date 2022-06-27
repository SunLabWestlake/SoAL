# -*- coding: utf-8 -*-
"""
Manual labeling UI for coco dataset
Author: Jing Ning @ SunLab
"""

import sys
import cv2
import numpy as np
from os.path import join as pjoin
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
sys.path.append(".")
from tools.SoAL_DatasetUtils import *
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = "6.0"

DRAW_KEYPOINT_TEXT = True
COLOR = [*"yrmgkyrmkgyrmkg"][:NUM_KEYPOINTS]
MARKER = [*"x+.,*"]
TOOLS = ["", "draw bbox", "draw keypoints"]
DEFAULT_TOOL = 2 if CENTERED else 1

class LabelUI(object):
    def __init__(self, ann_file, img_folder, res_file=None, res_file2=None):
        self.frame = 0
        self.fly_id = 1
        self.tool = 0  # NOTE: 1: bbox; 2: keypoints
        self.tool_step = 0  # NOTE: 1~5
        self.patch = None
        self.ann_file = ann_file
        self.res_file = res_file
        self.ann = load_dict(ann_file)
        if res_file2:
            self.res = load_dict(res_file2)
            self.ann["annotations"] = load_dict(res_file)
        elif res_file:
            self.res = load_dict(res_file)
        else:
            self.res = None
        self.cat_d = {cat["id"]: cat for cat in self.ann["categories"]}
        self.cur_ann_l = None
        self.cur_res_l = None
        self.ann_d = ann_l_to_d(self.ann["annotations"])
        self.res_d = ann_l_to_d(self.res) if self.res else None
        self.images = self.ann["images"]
        self.img_folder = img_folder
        self.total_frame = len(self.images)
        self.bbox_centered = [0, 0, self.images[1]["width"], self.images[1]["height"]]
        self.area = self.images[1]["width"] * self.images[1]["height"]

        self.fig, self.ax = plt.subplots(figsize=(1.9, 1.7), num=img_folder, dpi=300)
        plt.subplots_adjust(left=0.11, right=0.95, top=0.9, bottom=0.25)#, hspace=0.06, wspace=0.06)
        self.slider_ax = plt.axes([0.14, 0.08, 0.68, 0.05])
        self.slider = Slider(self.slider_ax, "", valmin=0, valmax=self.total_frame - 1, valfmt="%d", valinit=0)
        self.slider.on_changed(self.on_slider)
        self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        #self.set_window_pos(500, 300)
        self.change_tool(DEFAULT_TOOL)
        self.change_tool_step(1)

    def set_window_pos(self, x, y):
        mngr = plt.get_current_fig_manager()
        mngr.window.wm_geometry("+%d+%d" % (x, y))

    def plot_one_frame(self):
        self.img_gray = cv2.imread(pjoin(self.img_folder, self.images[self.frame]["file_name"]),cv2.IMREAD_COLOR)
        # b, g, r = cv2.split(img)
        # img_rgb = cv2.merge([r, g, b])
        # self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.refresh_fig()

    def print_img_norm_config(self):
        ima = []
        for img in self.images:
            im = cv2.imread(pjoin(self.img_folder, img["file_name"]))
            ima.extend(im.reshape(-1, 3))
        ima = np.array(ima)
        print(ima.mean(axis=0), ima.std(axis=0))

    def refresh_fig(self):
        self.ax.cla()
        self.ax.imshow(self.img_gray)#, cmap=plt.cm.gray, norm=NoNorm(), extent=(0, self.roi_size_scale[0], 0, self.roi_size_scale[1]))
        img_id = self.images[self.frame]["id"]
        # self.slider_ax.set_xlabel("%d:%s" % (self.frame, img_id))
        print("#%d" % self.frame)
        self.plot_fly_info()
        self.ax.set_xlim(self.bbox_centered[0], self.bbox_centered[2])
        self.ax.set_ylim(self.bbox_centered[1], self.bbox_centered[3])
        self.ax.invert_yaxis()
        self.refresh_title()

    def refresh_title(self):
        self.ax.set_title("fly%d:%s %d" % (self.fly_id, TOOLS[self.tool], self.tool_step))
        try:
            if self.patch:
                self.patch.remove()
                self.patch = None
        except:
            pass
        try:
            if self.tool == 1:
                bbox = self.cur_ann()["bbox"]
                if self.tool_step == 1:
                    self.patch = self.ax.add_patch(plt.Circle(bbox[:2], 3, edgecolor="w", fill=False))
                else:
                    self.patch = self.ax.add_patch(plt.Circle((bbox[0] + bbox[2], bbox[1] + bbox[3]), 3, edgecolor="w", fill=False))
            elif self.tool == 2:
                keypoints = self.cur_ann()["keypoints"]
                s = 3 * (self.tool_step - 1)
                self.patch = self.ax.add_patch(plt.Circle(keypoints[s:s + 2], 3, edgecolor="w", fill=False))
        except:
            pass
        plt.draw()

    def move_bbox(self, dx, dy):
        if self.tool == 1:
            bbox = self.cur_ann()["bbox"]
            x, y = bbox[:2] if self.tool_step == 1 else (bbox[0] + bbox[2], bbox[1] + bbox[3])
            self.update_bbox(self.tool_step, x + dx, y + dy)
        self.refresh_fig()

    def cur_ann(self):
        return self.cur_ann_l[self.fly_id - 1]

    def save_label(self):
        ann = self.cur_ann()
        l, b, w, h = ann["bbox"]
        plt.figure("t", figsize=(3, 2), dpi=300)
        ax = plt.gca()
        ax.cla()
        ax.axis("off")
        ax.imshow(self.img_gray)
        ax.add_patch(plt.Rectangle(xy=(l-1, b-1), width=max(1, w), height=max(1, h), edgecolor="k", fill=False, lw=1))
        keypoints = np.array(ann["keypoints"]).reshape([-1, 3])[:NUM_KEYPOINTS]
        ax.scatter(keypoints[:, 0], keypoints[:, 1], c="w", marker="o", s=100)
        ax.scatter(keypoints[:, 0], keypoints[:, 1], c=COLOR, marker="o", s=50)
        plt.savefig(r"label_example\%d.png" % self.frame)

    def plot_fly_info(self):
        self.plot_fly_keypoints(self.cur_ann(), self.fly_id - 1)
        ann_l = self.cur_ann_l
        for idx, ann in enumerate(ann_l):
            if idx == self.fly_id - 1:
                continue
            self.plot_fly_keypoints(ann, idx)

        if self.cur_res_l:
            for idx, ann in enumerate(self.cur_res_l):
                if ann.get("score", 1) > 0.1:
                    self.plot_fly_keypoints(ann, idx + 1000, color="w")#""C%d" % (idx % 9 + 1))

    def plot_fly_keypoints(self, ann, idx, color=None):
        if ann.get("bbox"):
            l, b, w, h = ann["bbox"]
            self.ax.add_patch(plt.Rectangle(xy=(l, b), width=max(1, w), height=max(1, h), edgecolor=color or COLOR[idx % 5], fill=False))
            cid = ann["category_id"]
            if cid == 1:
                t = "%d" % (idx + 1)
            else:
                t = self.cat_d.get(cid, {}).get("name", cid)
            # self.ax.text(l, b, t, color=color or COLOR[idx % 5])
        if ann.get("keypoints"):
            keypoints = np.array(ann["keypoints"]).reshape([-1, 3])[:NUM_KEYPOINTS]
            self.ax.scatter(keypoints[:, 0], keypoints[:, 1], c="w", marker=MARKER[idx % 5], s=60, lw=0.8)
            self.ax.scatter(keypoints[:, 0], keypoints[:, 1], c=color or COLOR, marker="+", s=40, lw=0.8)
            if DRAW_KEYPOINT_TEXT:
                for i, kpt in enumerate(keypoints):
                    self.ax.text(kpt[0]+2, kpt[1], str(i+1), color="w", fontsize=12)
                    self.ax.text(kpt[0]+2, kpt[1]+0.2, str(i+1), color=color or COLOR[i], fontsize=12)

    def copy_last_ann(self):
        img_id = self.images[self.frame - 1]["id"]
        ann_l = self.ann_d.get(img_id, [])
        if not ann_l:
            return
        last_ann = ann_l[0]
        ann = self.cur_ann()
        ann["keypoints"] = last_ann["keypoints"].copy()
        ann["bbox"] = last_ann["bbox"].copy()

    def change_fly(self, fly):
        if SINGLE_FLY:
            fly = 1
        self.fly_id = fly
        print("fly: %d" % self.fly_id)

        img_id = self.images[self.frame]["id"]
        ann_l = self.ann_d.get(img_id, [])
        if len(ann_l) < self.fly_id:
            self.fly_id = len(ann_l) + 1
            print("add fly:%d" % self.fly_id)
            ann = {
                "segmentation": [],
                "num_keypoints": NUM_KEYPOINTS,
                "area": self.area,
                "iscrowd": 0,
                "keypoints": [],
                "image_id": img_id,
                "bbox": self.bbox_centered if CENTERED else [],
                "category_id": 1,
                "id": img_id
            }
            ann_l.append(ann)
            self.ann_d[img_id] = ann_l
        self.cur_ann_l = ann_l
        if self.res_d:
            self.cur_res_l = self.res_d.get(img_id, [])

    def change_tool(self, tool):
        self.tool = tool
        self.change_tool_step(1)

    def change_tool_step(self, step):
        if self.tool == 1:
            if step > 2:
                self.tool = 2
                self.tool_step = 1
            elif step < 1:
                self.tool = 2
                self.tool_step = NUM_KEYPOINTS
                self.change_fly(2 if self.fly_id == 1 else 1)
            else:
                self.tool_step = step
        else:
            if CENTERED:
                self.tool_step = (step-1) % NUM_KEYPOINTS + 1
                return
            if step > NUM_KEYPOINTS:
                self.tool = 1
                self.tool_step = 1
                self.change_fly(2 if self.fly_id == 1 else 1)
            elif step < 1:
                self.tool = 1
                self.tool_step = 2
            else:
                self.tool_step = step


    def update_tool_data(self, x, y):
        if not self.tool:
            return
        if self.tool_step < 1 or self.tool_step > NUM_KEYPOINTS:
            self.change_tool_step(1)
        if self.tool == 1 and self.tool_step > 2:
            self.change_tool_step(1)
        if self.tool == 1:
            self.update_bbox(self.tool_step, x, y)
        else:
            self.update_keypoint(self.tool_step, x, y)

    def update_bbox(self, step, x, y):
        bbox = self.cur_ann()["bbox"]
        if not bbox:
            bbox = [0] * 4
            self.cur_ann()["bbox"] = bbox
        if step == 1:
            bbox[0] = x
            bbox[1] = y
        else:
            bbox[2] = x - bbox[0]
            bbox[3] = y - bbox[1]

    def update_keypoint(self, step, x, y):
        keypoint = self.cur_ann()["keypoints"]
        if not keypoint or len(keypoint) < 3 * NUM_KEYPOINTS:
            keypoint = [np.nan] * (3 * NUM_KEYPOINTS)
            self.cur_ann()["keypoints"] = keypoint
        s = 3 * (step - 1)
        keypoint[s] = x
        keypoint[s+1] = y
        keypoint[s+2] = 2

    def onclick(self, event):
        if not event.xdata or event.inaxes != self.ax:
            return
        self.update_tool_data(int(event.xdata + 0.5), int(event.ydata + 0.5))
        self.change_tool_step(self.tool_step + 1)
        self.refresh_fig()

    def is_frame_not_labeled(self, f):
        img_id = self.images[f]["id"]
        ann_l = self.ann_d.get(img_id, [])
        return not ann_l or not ann_l[0]["keypoints"] or np.isnan(ann_l[0]["keypoints"]).any()

    def is_wing_lr_error(self, f):
        if self.is_frame_not_labeled(f):
            return False
        img_id = self.images[f]["id"]
        ann_l = self.ann_d.get(img_id, [])
        wlr = np.reshape(ann_l[0]["keypoints"][-6:], (2, 3))
        wlx, wly = wlr[0][0], wlr[0][1]
        wrx, wry = wlr[1][0], wlr[1][1]
        if wly + wry > 48 and wlx < wrx:
            return True
        if wly + wry < 48 and wlx > wrx:
            return True

    def onkey(self, event):
        print(event.key)
        # if event.key == "left":
        #     return self.move_bbox(-1, 0)
        # elif event.key == "right":
        #     return self.move_bbox(1, 0)
        # elif event.key == "up":
        #     return self.move_bbox(0, -1)
        # elif event.key == "down":
        #     return self.move_bbox(0, 1)
        if event.key == "z":
            self.frame -= 1
        elif event.key == " " or event.key == "x":
            self.frame += 1
        elif event.key == "a":
            # self.frame = self.total_frame - 1
            b = self.is_frame_not_labeled(self.frame)
            for f in range(self.frame, self.total_frame):
                if self.is_frame_not_labeled(f) != b:
                    print("labeled: ", not b)
                    break
            self.frame = f
        elif event.key == "c":
            for f in range(self.frame+1, self.total_frame):
                if self.is_wing_lr_error(f):
                    break
            self.frame = f
        elif event.key == "d":
            # self.fig.savefig("img/pred_%s.png" % self.images[self.frame]["id"])
            self.save_ann(self.ann_file)
        elif event.key == "-":
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
        elif event.key in [*"1234567890"]:
            if event.key == 0:
                event.key = 10
            self.change_tool_step(int(event.key))
            self.refresh_title()
            return
        elif event.key == ",":
            # self.change_tool(2 if self.tool == 1 else 1)
            # self.refresh_title()
            self.save_label()
            return
        elif event.key == "`":
            self.change_tool_step(self.tool_step - 1)
            self.refresh_title()
            return
        elif event.key == "control":
            self.change_tool_step(self.tool_step + 1)
            self.refresh_title()
            return
        elif event.key == "r":
            self.copy_last_ann()
        elif event.key in ["f1", "f2", "f3", "f4", "f5"]:
            self.change_fly(int(event.key[1]))
            self.change_tool(DEFAULT_TOOL)
            self.refresh_fig()
            return
        else:
            return
        if self.frame >= self.total_frame:
            self.frame = self.total_frame - 1
        if self.frame < 0:
            self.frame = 0
        self.slider.set_val(self.frame)

    def on_slider(self, val):
        self.frame = int(val)
        self.change_fly(1)
        self.change_tool(DEFAULT_TOOL)
        self.plot_one_frame()

    def show(self):
        self.on_slider(0)
        plt.show()

    def save_ann(self, ann_file):
        self.ann["annotations"] = ann_d_to_l(self.ann_d)
        save_dict(ann_file, self.ann)

    def save_result(self, res_file):
        keypoints = sorted(self.ann["annotations"], key=lambda x: x["image_id"])
        res_l = []
        header = " ,Area,Mean,Min,Max,X,Y,Slice\n"
        for i, a in enumerate(keypoints):
            kp = a["keypoints"]
            if len(kp) > 13 and not np.isnan(np.max(kp)):
                res_l.append("%d,0,0,0,0,%d,%d,%d" % (i*5+1, kp[0], kp[1], i))
                res_l.append("%d,0,0,0,0,%d,%d,%d" % (i*5+2, kp[3], kp[4], i))
                res_l.append("%d,0,0,0,0,%d,%d,%d" % (i*5+3, kp[6], kp[7], i))
                res_l.append("%d,0,0,0,0,%d,%d,%d" % (i*5+4, kp[9], kp[10], i))
                res_l.append("%d,0,0,0,0,%d,%d,%d" % (i*5+5, kp[12], kp[13], i))
        open(res_file, "w").write(header + "\n".join(res_l))

def main(name, result_file_name=None, res_file2=None, save=True):
    # generate_dataset(parent, name)
    ann_file, ann_folder, img_folder = get_folder_names(DATASET_PATH, name)
    # scale_ann(ann_file)
    result, result2 = None, None
    if result_file_name:
        result = pjoin(ann_folder, result_file_name)
    if res_file2:
        result2 = pjoin(ann_folder, res_file2)
    # build_ann_by_img(img_folder)
    ui = LabelUI(ann_file, img_folder, result, result2)
    ui.show()
    if not result_file_name and save:
        ui.save_ann(ann_file)
        # ui.save_result(ann_file.replace(".json", ".csv"))

def main2(ann_file1, ann_file2, img_folder):
    ui = LabelUI(ann_file1, img_folder, ann_file2)
    ui.show()

if __name__ == "__main__":
    # USAGE: python SoAL_LabelUI.py a220407
    import sys
    main(sys.argv[1])

    # main2(r"E:\git\human_pose_data\fly_centered\annotations\person_keypoints_val1163.json",
    #       r"E:\git\human_pose_data\fly_centered\annotations\hrnet\hrnet_val1163_results_final.json",
    #       r"E:\git\human_pose_data\fly_centered\images\val1163")
