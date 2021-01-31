import argparse
import os

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
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input', help='input folder')
    parser.add_argument('out', help='output folder')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    model = init_detector(args.config, args.checkpoint, device='cuda:0')

    images = []
    annotations = []
    if not os.path.exists(args.out):
        os.mkdir(args.out)

    PALETTE.append([0,0,0])
    colors = np.array(PALETTE, dtype=np.uint8)

    for city in os.listdir(args.input):
        path = os.path.join(args.input, city)
        out_dir = os.path.join(args.out, city)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        prog_bar = mmcv.ProgressBar(len(os.listdir(path)))
        for imgName in os.listdir(path):
            result = inference_detector(model, os.path.join(path, imgName), eval='panoptic')
            pan_pred, cat_pred, _ = result[0]

            imageId = imgName.replace("_leftImg8bit.png", "")
            inputFileName = imgName
            outputFileName = imgName.replace("_leftImg8bit.png", "_panoptic.png")

            img = Image.open(os.path.join(path, imgName))
            out_path = os.path.join(out_dir, outputFileName)

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

            out = Image.blend(img, sem_img, 0.5).convert(mode="RGBA")
            out = Image.alpha_composite(out, contours_img)
            out.convert(mode="RGB").save(out_path)

            prog_bar.update()   

if __name__ == '__main__':
    main()
