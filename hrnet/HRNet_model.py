# -*- coding: utf-8 -*-
"""
Wrap class for HRNet prediction
Author: Jing Ning @ SunLab
"""

import cv2
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

# NOTE: for workstation
import sys
sys.path.insert(0, "hrnet/lib")
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from config import cfg
from core.inference import get_final_preds
from utils.transforms import get_affine_transform
import dataset
import models
import numpy as np
from SoAL_Constants import MODEL_SHAPE, BATCH_SIZE


class HRNetModel(object):
    def __init__(self, cfg_file, model_file=None):
        cfg.defrost()
        cfg.merge_from_file(cfg_file)
        cfg.freeze()
        self.cfg = cfg

        cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
        model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False)
        model.load_state_dict(torch.load(model_file or cfg.TEST.MODEL_FILE), strict=False)

        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
        model.eval()
        self.model = model
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.ToTensor(), normalize])

        self.batch_size = BATCH_SIZE# cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS)
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.center, self.scale = self._box2cs([0, 0, MODEL_SHAPE[0], MODEL_SHAPE[1]])
        self.center_b, self.scale_b = np.tile(self.center, (self.batch_size, 1)), np.tile(self.scale, (self.batch_size, 1))
        self.trans = get_affine_transform(self.center, self.scale, 0, (self.image_width, self.image_height))
        # input = cv2.warpAffine(img, self.trans, (self.image_width, self.image_height), flags=cv2.INTER_LINEAR)

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def kpt_detect_batch(self, img_l):  # NOTE: grayscale
        with torch.no_grad():
            inputs = []
            for img in img_l:
                # cv2.imwrite("temp/1.jpg", img)
                input = cv2.warpAffine(img, self.trans, (self.image_width, self.image_height), flags=cv2.INTER_LINEAR)
                # input = img
                input = np.tile(np.expand_dims(input, 2), (1, 1, 3))
                inputs.append(self.transform(input))
            # NOTE: input: batch_size * 3 * 256 * 192
            inputs = torch.stack(inputs)
            # NOTE: output: batch_size * 5 * 64 * 48
            output = self.model(inputs)
            # NOTE: preds: batch_size * 5 * 2, maxvals: batch_size * 5 * 1
            preds, maxvals = get_final_preds(self.cfg, output.clone().cpu().numpy(), self.center_b, self.scale_b)
            return preds.T, maxvals

    def kpt_detect_batch_color(self, img_l):
        with torch.no_grad():
            # NOTE: input: batch_size * 3 * 256 * 192
            inputs = []
            for img in img_l:
                # cv2.imwrite("temp/1.jpg", img)
                input = cv2.warpAffine(img, self.trans, (self.image_width, self.image_height), flags=cv2.INTER_LINEAR)
                # input = img
                inputs.append(self.transform(input))
            inputs = torch.stack(inputs)
            # NOTE: output: batch_size * 5 * 64 * 48
            output = self.model(inputs)
            # NOTE: preds: batch_size * 5 * 2, maxvals: batch_size * 5 * 1
            preds, maxvals = get_final_preds(self.cfg, output.clone().cpu().numpy(), self.center_b, self.scale_b)
            return preds.T, maxvals

    def kpt_detect_dataset(self, folder, postfix=""):
        import cv2
        import json
        name = os.path.basename(folder)
        ann_file = os.path.join(folder, "../../annotations/hrnet/hrnet_%s_results%s.json" % (name, postfix))
        ann = []
        batch_size = BATCH_SIZE
        batch_img, batch = [], []
        fl = os.listdir(folder)
        num = len(fl)
        for i, f in enumerate(fl):
            img = os.path.join(folder, f)
            input = cv2.warpAffine(cv2.imread(img), self.trans, (self.image_width, self.image_height), flags=cv2.INTER_LINEAR)
            batch_img.append(input)
            batch.append(int(f[:-4]))
            if len(batch_img) >= batch_size or i == num - 1:
                ypks = self.kpt_detect_batch_color(batch_img)[0].T
                for j, ypk in enumerate(ypks):
                    kpt = np.hstack([ypk, np.ones((5, 1))])
                    ann.append({
                        "category_id": 1,
                        "image_id": batch[j],
                        "keypoints": kpt.flatten().tolist(), #ypk.T.flatten().tolist(),
                        "score": 1,
                    })
                batch_img.clear()
                batch.clear()

        f = open(ann_file, "w")
        json.dump(ann, f, indent=4)
        f.close()
        print("save_dict %s" % ann_file)
