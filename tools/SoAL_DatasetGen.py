# -*- coding: utf-8 -*-
"""
Generating datasets
Author: Jing Ning @ SunLab
"""


import os, sys
import cv2
import shutil
import numpy as np
from os.path import join as pjoin
sys.path.append(".")
from SoAL_BodyDetection import RegionSeg
from tools.SoAL_DatasetUtils import *

MAX_FRAME = 400
PROC_CENTER = False

g_cache = {}
# DATASET_RAW_ROOT = r"D:/exp/video_test/test_trk"
DATASET_RAW_ROOT = r"G:\_video_hrnet_finish"
def get_frame(url, centered=True):
    video, roi_i, frame, fly = url.split(":")
    video = "%s/%s/%s.avi" % (DATASET_RAW_ROOT, video, video)
    if not os.path.exists(video):
        print(video, "not found!!!")
        return None
    roi_i, fly, frame = int(roi_i), int(fly), int(frame)

    if g_cache.get(video):
        cap, rs, rois = g_cache.get(video)
    else:
        ROI = load_dict(video.replace(".avi", "_config.json"))["ROI"]
        rois = [roi["roi"] for roi in ROI]
        rs = RegionSeg(video, 0, None)
        rs.init_bg()
        cap = cv2.VideoCapture(video)
        g_cache[video] = (cap, rs, rois)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, img = cap.read()

    if centered:
        img_centered_l, reg_n, pos_l = rs.proc_one_img(img, rois[roi_i], roi_i, proc_center=PROC_CENTER)
        return img_centered_l[0].astype(np.uint8), img_centered_l[1].astype(np.uint8)
    img_centered, center_angle = rs.proc_one_img(img, rois[roi_i], roi_i, no_center_img=True)
    return img_centered, center_angle

def generate_dataset_by_ann(name):
    ann_file, ann_folder, img_folder = get_folder_names(DATASET_PATH, name)
    ann = load_dict(ann_file)
    for img_info in ann["images"]:
        url = img_info.get("url")
        if not url:
            continue
        img_centered = get_frame(url)
        cv2.imwrite(pjoin(img_folder, img_info["file_name"]), img_centered)

def generate_dataset_by_img(name, orig_img_folder):
    ann_file, ann_folder, img_folder = get_folder_names(DATASET_PATH, name)
    ann = load_dict(TEMPLATE_PATH)
    sh = None
    for i, f in enumerate(os.listdir(orig_img_folder)):
        src = pjoin(orig_img_folder, f)
        if not sh:
            img = cv2.imread(src, cv2.IMREAD_COLOR)
            sh = img.shape
        img_name = "0000%08d.jpg" % i
        dst = pjoin(img_folder, img_name)
        shutil.copy(src, dst)
        ann["images"].append({
            "license": 1,
            "file_name": img_name,
            "url": f,
            "height": sh[0],
            "width": sh[1],
            "id": i
        })
    save_dict(ann_file, ann)
    return ann

def check_ann_file(ann_file):
    ann = load_dict(ann_file)
    ann_l = []
    img_s = set()
    idx = 0
    for a in ann["annotations"]:
        if a["keypoints"] and a["bbox"]:
            bbox = a["bbox"]
            a["segmentation"] = []
            a["area"] = bbox[2] * bbox[3]
            a["id"] = idx
            idx += 1
            ann_l.append(a)
            img_s.add(a["image_id"])
    img_l = []
    for img in ann["images"]:
        if img["id"] in img_s:
            img_l.append(img)
    ann["annotations"] = ann_l
    ann["images"] = img_l
    save_dict(ann_file.replace(".json", "_final.json"), ann)

def remove_unlabeled_image(ann_file, name):
    image_folder = os.path.dirname(ann_file).replace("annotations", "images")
    image_remove = pjoin(image_folder, "remove")
    os.makedirs(image_remove, exist_ok=True)
    ann = load_dict(ann_file)
    for img in ann["images"]:
        shutil.move(pjoin(image_folder, name, img["file_name"]), image_remove)

def keypoints_in_box(kp, bbox):
    x1, x2, y1, y2 = bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3]
    for i in range(0, len(kp), 3):
        if kp[i] < x1:
            kp[i] = x1
        elif kp[i] > x2:
            kp[i] = x2-1
        if kp[i+1] < y1:
            kp[i+1] = y1
        if kp[i+1] > y2:
            kp[i+1] = y2-1
    return kp

def build_ann_by_img(img_folder):
    fl = []
    sh = None
    for f in os.listdir(img_folder):
        if f.endswith("jpg"):
            idx = int(f.split(".")[0])
            fl.append((f, idx))
            if not sh:
                img = cv2.imread(pjoin(img_folder, f), cv2.IMREAD_COLOR)
                sh = img.shape
    fl.sort(key=lambda ff: ff[1])

    w, h = sh[1], sh[0]
    area = w * h
    bbox = [0, 0, w, h]
    imgs = []
    anns = []
    for i, (f, idx) in enumerate(fl):
        imgs.append({
            "license": 1,
            "file_name": f,
            "coco_url": "",
            "height": sh[0],
            "width": sh[1],
            "date_captured": "2020-09-02 00:00:00",
            "flickr_url": "",
            "id": i
        })
        anns.append({
            "segmentation": [],
            "num_keypoints": NUM_KEYPOINTS,
            "area": area,
            "iscrowd": 0,
            "keypoints": [],
            "image_id": i,
            "bbox": bbox,
            "category_id": 1,
            "id": i
        })

    ann = load_dict(TEMPLATE_PATH)
    ann["annotations"] = anns
    ann["images"] = imgs
    return ann

def remove_ann(ann_file):
    ann = load_dict(ann_file)
    ann["annotations"] = []
    save_dict(ann_file, ann)

def change_ann(name):
    ann_file, ann_folder, img_folder = get_folder_names(DATASET_PATH, name)
    res = load_dict(ann_file)
    for ann in res["annotations"]:
        kpt = ann["keypoints"]
        kpt[9:12], kpt[12:] = kpt[12:], kpt[9:12]
    save_dict(ann_file, res)

def merge_ann(name_l, name_out):
    ann_file_out, ann_folder_out, img_folder_out = get_folder_names(DATASET_PATH, name_out)
    anns, imgs = [], []
    res = {}
    for name in name_l:
        ann_file, ann_folder, img_folder = get_folder_names(DATASET_PATH, name)
        res = load_dict(ann_file)
        anns.extend(res["annotations"])
        imgs.extend(res["images"])
        for img in res["images"]:
            shutil.copy(pjoin(img_folder, img["file_name"]), img_folder_out)
    anns.sort(key=lambda x: x["id"])
    imgs.sort(key=lambda x: x["id"])
    res["annotations"] = anns
    res["images"] = imgs
    save_dict(ann_file_out, res)

def add_ann(name1, names):
    ann_file1, ann_folder1, img_folder1 = get_folder_names(DATASET_PATH, name1)
    ann1 = load_dict(ann_file1)
    print("%s: %d images, %d anns" % (name1, len(ann1["images"]), len(ann1["annotations"])))
    a1_d = ann_l_to_d(ann1["annotations"])
    url_to_img_ann = {}
    for img in ann1["images"]:
        url = img["url"]
        url_to_img_ann[url] = [url, img, a1_d[img["id"]][0], pjoin(img_folder1, img["file_name"])]  # NOTE: url, img, ann, img_path
    for name2 in names:
        ann_file2, ann_folder2, img_folder2 = get_folder_names(DATASET_PATH, name2)
        ann2 = load_dict(ann_file2)
        print("%s: %d images, %d anns" % (name2, len(ann2["images"]), len(ann2["annotations"])))
        a2_d = ann_l_to_d(ann2["annotations"])
        for img in ann2["images"]:
            url = img.get("url")
            if not url:
                continue
            if url[-2] != ":":
                print("invalid img:", url)
                continue
            if url_to_img_ann.get(url) is not None:
                print("duplicate img:", url)
                continue
            url_to_img_ann[url] = [url, img, a2_d[img["id"]][0], pjoin(img_folder2, img["file_name"])]
        print("cur:", len(url_to_img_ann))
    info_l = sorted(url_to_img_ann.values(), key=lambda x: x[0])
    ann_file3, ann_folder3, img_folder3 = get_folder_names(DATASET_PATH, "all")
    for idx, info in enumerate(info_l):
        url, img, a, path = info
        img["id"] = idx
        a["image_id"] = idx
        a["id"] = idx
        img["file_name"] = "0000%08d.jpg" % idx
        print(idx, url)
        # shutil.copy(path, pjoin(img_folder3, img["file_name"]))
    ann_all = ann1
    ann_all["images"] = [x[1] for x in info_l]
    ann_all["annotations"] = [x[2] for x in info_l]
    save_dict(ann_file3, ann_all)

def divide_ann(name1, val_r=0.2, val_n=0):
    ann_file1, ann_folder1, img_folder1 = get_folder_names(DATASET_PATH, name1)
    ann1 = load_dict(ann_file1)
    a1_d = ann_l_to_d(ann1["annotations"])
    url_to_img_ann = {}
    for img in ann1["images"]:
        url = img["url"]
        anno_l = a1_d.get(img["id"], [])
        if len(anno_l) == 0:
            anno = {}
        else:
            anno = anno_l[0]
        url_to_img_ann[url] = [url, img, anno, pjoin(img_folder1, img["file_name"])]
    info_l = list(url_to_img_ann.values())

    n = len(info_l)
    if val_n == 0:
        val_n = int(n * val_r)
    info_l = np.random.permutation(info_l)
    info_ll = [info_l[:val_n], info_l[val_n:]]

    for i, name2 in enumerate(("val", "train")):
        ann_file2, ann_folder2, img_folder2 = get_folder_names(DATASET_PATH, name2 + "_" + name1)
        ann2 = load_dict(TEMPLATE_PATH)
        ann2["images"] = [x[1] for x in info_ll[i]]
        ann2["annotations"] = [x[2] for x in info_ll[i]]
        for info in info_ll[i]:
            shutil.copy(info[-1], img_folder2)
        print(name2, len(ann2["images"]))
        save_dict(ann_file2, ann2)

def rmse(ann_file1, ann_file2):
    ann1 = load_dict(ann_file1)
    ann2 = load_dict(ann_file2)
    ann_l1 = ann_l_to_d(ann1["annotations"])
    ann_l2 = ann_l_to_d(ann2["annotations"])
    kpd_l = []
    for img_id, ann_l in ann_l1.items():
        if ann_l2.get(img_id):
            kp1 = np.reshape(ann_l[0]["keypoints"], (-1, 3))
            kp2 = np.reshape(ann_l2[img_id][0]["keypoints"], (-1, 3))
            kpd_l.append((kp1 - kp2)[:, :2])
            print(img_id)
    kpd = np.array(kpd_l)  # n*5*2
    x = kpd[:, :, 0]
    y = kpd[:, :, 1]
    ret = np.sqrt(x**2+y**2)
    m = ret.mean(axis=0)
    print(ret)
    print(m)
    print(np.sqrt(m**2/(64*48)))

def correct_wing_error(ann_file):
    ann = load_dict(ann_file)
    img_id_s = set()
    print(len(ann["annotations"]))
    ann_new = []
    for a in ann["annotations"]:
        bbox = a["bbox"]
        a["segmentation"] = []
        a["area"] = bbox[2] * bbox[3]

        img_id = a["image_id"]
        kpt = a["keypoints"]
        if not kpt or np.isnan(kpt).any():
            print("invalid label: ", img_id, a)
            continue
        if img_id in img_id_s:
            print("duplicate id: ", img_id)
            continue
        else:
            img_id_s.add(img_id)

        pt = np.reshape(kpt, (5, 3))
        wlx, wly = pt[3][0], pt[3][1]
        wrx, wry = pt[4][0], pt[4][1]
        if wly + wry > 48 and wlx > wrx:
            a["keypoints"] = kpt[:9] + kpt[12:15] + kpt[9:12]
        if wly + wry < 48 and wlx < wrx:
            a["keypoints"] = kpt[:9] + kpt[12:15] + kpt[9:12]
        ann_new.append(a)
    print(len(ann_new))
    ann["annotations"] = ann_new

    ann["images"] = sorted(ann["images"], key=lambda x: x["id"])
    ann["annotations"] = sorted(ann["annotations"], key=lambda x: x["image_id"])
    save_dict(ann_file, ann)

def rotate_back_points(kpts, pos, flydir, img_shape):
    flyx, flyy = pos[0], pos[1]
    fd = np.deg2rad(flydir - 90)
    kpts = np.reshape(kpts, (5, 3))
    kpts[:, 2] = 1
    xy1 = kpts.T
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

def convert_dataset_to_allocentric(coco_label_json, coco_label_json2):
    d = json.load(open(coco_label_json, "r"))
    anns, imgs = d["annotations"], d["images"]
    dataset_name = coco_label_json2[coco_label_json2.rfind("person_keypoints_")+len("person_keypoints_"):-5]
    img_folder1 = pjoin(os.path.dirname(coco_label_json), "..", "images", dataset_name)
    img_folder = pjoin(os.path.dirname(coco_label_json2), "..", "images", dataset_name)
    os.makedirs(img_folder, exist_ok=True)
    center_angle_d = {}
    url_d = {}
    img_id_d = {}
    imgs_new = []
    for img in imgs:
        url = img["url"]
        img_id = img["id"]
        im, center_angle = get_frame(url, False)
        img_o = url_d.get(url[:-1])

        # NOTE: determine which fly
        im1, im2 = get_frame(img["url"], True)
        im0 = cv2.imread(pjoin(img_folder1, img["file_name"]))[:, :, 0]
        d01 = np.sum(np.abs(im0.astype(float)-im1.astype(float)))
        d02 = np.sum(np.abs(im0.astype(float)-im2.astype(float)))
        fly = 0 if d01 < d02 else 1
        # cv2.imshow("0", im0)
        # cv2.imshow("1", im1)
        # cv2.imshow("2", im2)
        # cv2.waitKey(-1)

        center_angle_d[img_id] = center_angle[fly]
        if img_o is None:
            img["height"], img["width"] = im.shape
            img["center_info"] = "%s,%s" % center_angle[fly]
            img["url"] = url[:-1] + "1"
            cv2.imwrite(pjoin(img_folder, img["file_name"]), im)
            url_d[url[:-1]] = img
            imgs_new.append(img)
        else:
            img_o["center_info"] += "%s,%s" % center_angle[fly]
            img_o["url"] = url[:-1] + "2"
            img_id_d[img["id"]] = img_o["id"]

    for ann in anns:
        img_id = ann["image_id"]
        center, angle = center_angle_d[img_id]
        xy = rotate_back_points(ann["keypoints"], center, angle, (ann["bbox"][-1], ann["bbox"][-2]))
        kpt = xy.T.flatten()
        ann["keypoints"] = kpt.tolist()[0]
        ann["image_id"] = img_id_d.get(img_id, img_id)
    d["images"] = imgs_new
    json.dump(d, open(coco_label_json2, "w"), indent=4)

def generate_dataset_no_roi(video, name, start=0, step=1000):
    ann_file, ann_folder, img_folder = get_folder_names(DATASET_PATH, name)
    ann = load_dict(TEMPLATE_PATH)
    cap = cv2.VideoCapture(video)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    imgs = []
    for i, seq in enumerate(range(start, total_frame, step)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, seq)
        print(seq)
        ret, img = cap.read()
        if not ret:
            break
        img_roi = img
        img_name = "%012d.jpg" % seq
        img_id = int(img_name[:-4])
        cv2.imwrite(pjoin(img_folder, img_name), img_roi)
        imgs.append({
            "license": 1,
            "file_name": img_name,
            "coco_url": "",
            "width": img_roi.shape[1],
            "height": img_roi.shape[0],
            "date_captured": "2020-09-19 00:00:00",
            "id": img_id
        })
    if os.path.exists(ann_file):
        ann_o = load_dict(ann_file)
        ann["annotations"] = ann_o["annotations"]
    ann["images"].extend(imgs)
    save_dict(ann_file, ann)
    return ann

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def generate_dataset_by_videos(video_l, name, frames_per_roi):
    ann = None
    idx = 0
    for video in video_l:
        ann, idx = generate_dataset(video, name, frames_per_roi, idx=idx, merge_with_ann=ann)
    return ann

def generate_dataset(video, name, frames_per_roi=100, hard=False, idx=0, merge_with_ann=None):
    ann_file, ann_folder, img_folder = get_folder_names(DATASET_PATH, name)
    # if os.path.exists(ann_file):
    #     print("already exists:", ann_file)
    #     return
    ann = merge_with_ann or load_dict(TEMPLATE_PATH)
    ROI = load_dict(video.replace(".avi", "_config.json"))["ROI"]
    video_s = os.path.basename(video).split(".")[0]
    rois = [roi["roi"] for roi in ROI]
    cap = cv2.VideoCapture(video)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # step = int(total_frame / frames_per_roi)
    start = 0
    imgs = []
    rs, img_centered = None, None
    if CENTERED:
        rs = RegionSeg(video, 0, None)
        rs.init_bg()
    for seq in np.linspace(start, total_frame-1, 10).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, seq)
        ret, img = cap.read()
        if not ret or len(imgs) >= MAX_FRAME:
            break
        if CENTERED:
            if CENTERED_ROI is not None:
                roi_i_l = [CENTERED_ROI]
            else:
                roi_i_l = range(len(rois))
                # roi_i_l = np.random.choice(range(len(rois)), 1)
            if CENTERED_FLY:
                fly_l = [CENTERED_FLY]
            else:
                fly_l = [1, 2]
            for roi_i in roi_i_l:
                img_centered_l, reg_n, pos_l = rs.proc_one_img(img, rois[roi_i], roi_i, proc_center=PROC_CENTER)
                if reg_n == 0:
                    continue
                for fly in fly_l:
                    img_centered = img_centered_l[fly-1]
                    if hard:
                        px, py = pos_l[fly-1]
                        if reg_n == 1:  # hard 1
                            pr = 0.8
                        elif abs(px - 0.5) < 0.1 and abs(py - 0.5) < 0.1:  # hard 2
                            pr = 0.1
                        elif distance(pos_l[0], pos_l[1]) < 0.25:  # hard_3
                            pr = 0.5
                        else:
                            continue
                        if np.random.rand() > pr:
                            continue
                    print("roi%d, frame%d, fly%d" % (roi_i, seq, fly))
                    img_name = "0000%08d.jpg" % idx
                    idx += 1
                    cv2.imwrite(pjoin(img_folder, img_name), img_centered)
                    imgs.append({
                        "license": 1,
                        "file_name": img_name,
                        "url": "%s:%d:%d:%d" % (video_s, roi_i, seq, fly),
                        "height": img_centered.shape[0],
                        "width": img_centered.shape[1],
                        "id": int(img_name[:-4])
                    })
        else:
            for ri, roi in enumerate(rois):
                img_roi = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
                img_name = "00%08d%02d.jpg" % (seq, ri)
                img_id = int(img_name[:-4])
                cv2.imwrite(pjoin(img_folder, img_name), img_roi)
                imgs.append({
                    "license": 1,
                    "file_name": img_name,
                    "coco_url": "",
                    "height": img_roi.shape[0],
                    "width": img_roi.shape[1],
                    "id": img_id
                })
    if os.path.exists(ann_file):
        ann_o = load_dict(ann_file)
        ann["annotations"] = ann_o["annotations"]
    ann["images"].extend(imgs)
    save_dict(ann_file, ann)
    print("finish %s : %d" % (video, len(imgs)))
    return ann, idx


if __name__ == "__main__":
    # USAGE: python SoAL_DatasetGen.py a220407 20
    import sys
    from glob import glob

    dataset_name = sys.argv[1]
    if sys.argv[2] == "divide":
        divide_ann(dataset_name)
    else:
        frames_per_roi = int(sys.argv[2])
        generate_dataset_by_videos(glob(pjoin(VIDEO_PATH, "*", "*.avi")), dataset_name, frames_per_roi)

    # change_ann("val_SDPD-200")
    # merge_ann(["train_SDPD-200", "val_SDPD-200"], "SDPD-200")
    # rmse(r"E:\git\human_pose_data\fly_centered_processed2\annotations\person_keypoints_traina.json",
    #      r"E:\git\human_pose_data\fly_centered_processed\annotations\person_keypoints_train1.json")
