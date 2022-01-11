import argparse
import os

import cv2
import mmcv
import torch
import numpy as np
import json

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from tools.fuse_conv_bn import fuse_module

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.datasets.cityscapes import PALETTE
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector, show_result
from mmdet.core import cityscapes_originalIds

from PIL import Image
from skimage.morphology import dilation
from skimage.segmentation import find_boundaries



def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input', help='input folder')
    parser.add_argument('out', help='output folder')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--show', type=bool, default=False, help='display option')
    parser.add_argument(
        '--wait', type=int, default=0, help='cv2 wait time')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    cfg = mmcv.Config.fromfile(args.config)

    model = init_detector(args.config, args.checkpoint, device=device)

    PALETTE.append([0,0,0])
    colors = np.array(PALETTE, dtype=np.uint8)

    for img_file in os.listdir(args.input):
        img = cv2.imread(os.path.join(args.input, img_file))
        img_shape = img.shape[:2][::-1]
        img_ = cv2.resize(img, cfg.test_pipeline[1]['img_scale'])

        result = inference_detector(model, img_, eval='panoptic')
        pan_pred, cat_pred, _ = result[0]

        sem = cat_pred[pan_pred].numpy()
        sem_tmp = sem.copy()
        sem_tmp[sem==255] = colors.shape[0] - 1
        sem_img = Image.fromarray(colors[sem_tmp])

        is_background = (sem < 11) | (sem == 255)
        pan_pred = pan_pred.numpy() 
        pan_pred[is_background] = 0

        contours = find_boundaries(pan_pred, mode="outer", background=0).astype(np.uint8) * 255
        contours = dilation(contours)

        contours = np.expand_dims(contours, -1).repeat(4, -1)
        contours_img = Image.fromarray(contours, mode="RGBA")

        out = Image.blend(Image.fromarray(img_[:,:,::-1]), sem_img, 0.5).convert(mode="RGBA")
        out = Image.alpha_composite(out, contours_img).convert(mode="RGB")
        out = cv2.resize(np.array(out)[:,:,::-1], img_shape)

        if args.show:
            cv2.imshow('img', img)
            cv2.imshow('panopitc', out)
            ch = cv2.waitKey(args.wait)

        cv2.imwrite(os.path.join(args.out, img_file), out)

if __name__ == '__main__':
    main()
