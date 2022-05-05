# -*- coding: utf-8 -*-
"""
Dataset utils
Author: Jing Ning @ SunLab
"""

import os
import json

NUM_KEYPOINTS = 5
VIDEO_PATH = "video/"
DATASET_PATH = "dataset/"

SINGLE_FLY = True
CENTERED = True
CENTERED_ROI = None
CENTERED_FLY = None
TEMPLATE_PATH = "tools/template.json"


def ann_l_to_d(ann_l, id_key="image_id"):
    if isinstance(ann_l, dict):
        ann_l = ann_l["annotations"]
    ann_d = {}
    for ann in ann_l:
        if ann.get(id_key) is None:
            continue
        img_id = ann[id_key]
        ann_d.setdefault(img_id, [])
        ann_d[img_id].append(ann)
    return ann_d

def ann_d_to_l(ann_d):
    ann_l = []
    for ann, al in ann_d.items():
        ann_l.extend(al[:2])
    return ann_l

def save_dict(filename, obj):
    f = open(filename, "w")
    json.dump(obj, f, indent=4)
    f.close()
    print("save_dict %s" % filename)

def load_dict(filename):
    if not os.path.exists(filename):
        print("not found:", filename)
        return None
    f = open(filename, "r")
    j = json.load(f)
    f.close()
    return j

def get_folder_names(root, name):
    ann_folder = os.path.join(root, "annotations")
    img_folder = os.path.join(root, "images", name)
    os.makedirs(ann_folder, exist_ok=True)
    os.makedirs(img_folder, exist_ok=True)
    ann_file = os.path.join(ann_folder, "person_keypoints_%s.json" % name)
    return ann_file, ann_folder, img_folder
