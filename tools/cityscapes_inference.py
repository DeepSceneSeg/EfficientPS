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
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector, show_result
from mmdet.core import cityscapes_originalIds

from PIL import Image

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
    out_base = args.out.split('/')[-1]
    out_base_folder = os.path.join(args.out, out_base)
    out_base_json = out_base_folder + '.json'

    if not os.path.exists(out_base_folder):
        os.mkdir(out_base_folder)

    originalIds = cityscapes_originalIds()

    for city in os.listdir(args.input):
        path = os.path.join(args.input, city)
        prog_bar = mmcv.ProgressBar(len(os.listdir(path)))
        for imgName in os.listdir(path):
            result = inference_detector(model, os.path.join(path, imgName), eval='panoptic')
            pan_pred, cat_pred, _ = result[0]
            pan_pred, cat_pred = pan_pred.numpy(), cat_pred.numpy()

            imageId = imgName.replace("_leftImg8bit.png", "")
            inputFileName = imgName
            outputFileName = imgName.replace("_leftImg8bit.png", "_panoptic.png")
            # image entry, id for image is its filename without extension
            images.append({"id": imageId,
                           "width": int(pan_pred.shape[1]),
                           "height": int(pan_pred.shape[0]),
                           "file_name": inputFileName})

            pan_format = np.zeros(
                (pan_pred.shape[0], pan_pred.shape[1], 3), dtype=np.uint8
            )

            panPredIds = np.unique(pan_pred)
            segmInfo = []   
            for panPredId in panPredIds:
                if cat_pred[panPredId] == 255:
                    continue
                elif cat_pred[panPredId] <= 10:
                    semanticId = segmentId = originalIds[cat_pred[panPredId]] 
                else:
                    semanticId = originalIds[cat_pred[panPredId]]
                    segmentId = semanticId * 1000 + panPredId 
                
                isCrowd = 0
                categoryId = semanticId

                mask = pan_pred == panPredId
                color = [segmentId % 256, segmentId // 256, segmentId // 256 // 256]
                pan_format[mask] = color

                area = np.sum(mask) # segment area computation

                # bbox computation for a segment
                hor = np.sum(mask, axis=0)
                hor_idx = np.nonzero(hor)[0]
                x = hor_idx[0]
                width = hor_idx[-1] - x + 1
                vert = np.sum(mask, axis=1)
                vert_idx = np.nonzero(vert)[0]
                y = vert_idx[0]
                height = vert_idx[-1] - y + 1
                bbox = [int(x), int(y), int(width), int(height)]

                segmInfo.append({"id": int(segmentId),
                                 "category_id": int(categoryId),
                                 "area": int(area),
                                 "bbox": bbox,
                                 "iscrowd": isCrowd})

            annotations.append({'image_id': imageId,
                                'file_name': outputFileName,
                                "segments_info": segmInfo})

            Image.fromarray(pan_format).save(os.path.join(out_base_folder, outputFileName))
            prog_bar.update()

    print("\nSaving the json file {}".format(out_base_json))
    d = {'images': images,
         'annotations': annotations,
         'categories': {}}
    with open(out_base_json, 'w') as f:
        json.dump(d, f, sort_keys=True, indent=4)
   

if __name__ == '__main__':
    main()
