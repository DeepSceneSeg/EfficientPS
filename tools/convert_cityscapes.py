import argparse
import glob
import json
import shutil
from multiprocessing import Pool, Value, Lock
from os import path, mkdir, listdir
import sys 

import numpy as np
import tqdm
from PIL import Image
from cityscapesscripts.helpers.labels import labels as cs_labels
from pycococreatortools import pycococreatortools as pct

parser = argparse.ArgumentParser(description="Convert Cityscapes to coco format")
parser.add_argument("root_dir", metavar="ROOT_DIR", type=str, help="Root directory of Cityscapes")
parser.add_argument("out_dir", metavar="OUT_DIR", type=str, help="Output directory")

_SPLITS = {
    "train": ("leftImg8bit/train", "gtFine/train"),
    "val": ("leftImg8bit/val", "gtFine/val"),
}
_INSTANCE_EXT = "_instanceIds.png"
_IMAGE_EXT = "_leftImg8bit.png"


def main(args):
    print("Loading Cityscapes from", args.root_dir)
    num_stuff, num_thing = _get_meta()

    _ensure_dir(args.out_dir)
    ann_dir = path.join(args.out_dir, "annotations")
    _ensure_dir(ann_dir)
    stuff_dir = path.join(args.out_dir, "stuffthingmaps")
    _ensure_dir(stuff_dir)



    # COCO-style category list
    coco_categories = []
    for lbl in cs_labels:
        if lbl.trainId != 255 and lbl.trainId != -1 and lbl.hasInstances:
            coco_categories.append({
                "id": lbl.trainId,
                "name": lbl.name
            })

    # Process splits
    images = []
    for split, (split_img_subdir, split_msk_subdir) in _SPLITS.items():
        print("Converting", split, "...")

        img_base_dir = path.join(args.root_dir, split_img_subdir)
        msk_base_dir = path.join(args.root_dir, split_msk_subdir)
        img_list = _get_images(msk_base_dir)

        img_dir = path.join(args.out_dir, split)
        _ensure_dir(img_dir)

        split_dir = path.join(stuff_dir, split)
        _ensure_dir(split_dir)


        # Convert to COCO detection format
        coco_out = {
            "info": {"version": "1.0"},
            "images": [],
            "categories": coco_categories,
            "annotations": []
        }

        # Process images in parallel
        worker = _Worker(img_base_dir, msk_base_dir, img_dir, split_dir)
        with Pool(initializer=_init_counter, initargs=(_Counter(0),), processes=8) as pool:
            total = len(img_list)
            for coco_img, coco_ann in tqdm.tqdm(pool.imap(worker, img_list, 8), total=total):

                # COCO annotation
                coco_out["images"].append(coco_img)
                coco_out["annotations"] += coco_ann

        # Write COCO detection format annotation
        with open(path.join(ann_dir, split + ".json"), "w") as fid:
            json.dump(coco_out, fid)


def _get_images(base_dir):
    img_list = []
    for subdir in listdir(base_dir):
        subdir_abs = path.join(base_dir, subdir)
        if path.isdir(subdir_abs):
            for img in glob.glob(path.join(subdir_abs, "*" + _INSTANCE_EXT)):
                _, img = path.split(img)

                parts = img.split("_")
                img_id = "_".join(parts[:-2])
                lbl_cat = parts[-2]
                img_list.append((subdir, img_id, lbl_cat))

    return img_list


def _get_meta():
    num_stuff = sum(1 for lbl in cs_labels if 0 <= lbl.trainId < 255 and not lbl.hasInstances)
    num_thing = sum(1 for lbl in cs_labels if 0 <= lbl.trainId < 255 and lbl.hasInstances)
    return num_stuff, num_thing


def _ensure_dir(dir_path):
    try:
        mkdir(dir_path)
    except FileExistsError:
        pass


class _Worker:
    def __init__(self, img_base_dir, msk_base_dir, img_dir, msk_dir):
        self.img_base_dir = img_base_dir
        self.msk_base_dir = msk_base_dir
        self.img_dir = img_dir
        self.msk_dir = msk_dir

    def __call__(self, img_desc):
        img_dir, img_id, lbl_cat = img_desc
        img_unique_id = counter.increment()
        coco_ann = []
        # Load the annotation
        with Image.open(path.join(self.msk_base_dir, img_dir, img_id + "_" + lbl_cat + _INSTANCE_EXT)) as lbl_img:
            lbl = np.array(lbl_img)
            lbl_size = lbl_img.size

        ids = np.unique(lbl)

        # Compress the labels and compute cat
        lbl_out = np.ones(lbl.shape, np.uint8)*255
        cat = [255]
        iscrowd = [0]
        for city_id in ids:
            if city_id < 1000:
                # Stuff or group
                cls_i = city_id
                iscrowd_i = cs_labels[cls_i].hasInstances
            else:
                # Instance
                cls_i = city_id // 1000
                iscrowd_i = False

            # If it's a void class just skip it
            if cs_labels[cls_i].trainId == 255 or cs_labels[cls_i].trainId == -1:
                continue

            # Extract all necessary information
            iss_class_id = cs_labels[cls_i].trainId
            
            mask_i = lbl == city_id

           
            lbl_out[mask_i] = iss_class_id

            # Compute COCO detection format annotation
            if cs_labels[cls_i].hasInstances:
                category_info = {"id": iss_class_id, "is_crowd": iscrowd_i}
                coco_ann_i = pct.create_annotation_info(
                    counter.increment(), img_unique_id, category_info, mask_i, lbl_size, tolerance=2)
                if coco_ann_i is not None:
                    coco_ann.append(coco_ann_i)

        # COCO detection format image annotation
        coco_img = pct.create_image_info(img_unique_id, img_id + ".png", lbl_size)

        # Write output
        Image.fromarray(lbl_out).save(path.join(self.msk_dir, img_id + ".png"))
        shutil.copy(path.join(self.img_base_dir, img_dir, img_id + _IMAGE_EXT),
                    path.join(self.img_dir, img_id + ".png"))


        return coco_img, coco_ann


def _init_counter(c):
    global counter
    counter = c


class _Counter:
    def __init__(self, initval=0):
        self.val = Value('i', initval)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            val = self.val.value
            self.val.value += 1
        return val


if __name__ == "__main__":
    main(parser.parse_args())
