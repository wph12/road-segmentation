import os

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.DirBuilder import dir_builder

RAW_DIR = "/Users/mouse/Documents/GitHub/CS-433_ML_Projects/MLProject2/dataset/raw"
PROCESSED_DIR = "/Users/mouse/Documents/GitHub/CS-433_ML_Projects/MLProject2/dataset/processed"

# classes: {'background': 0, 'Unlabelled': 1, 'Bedroom': 2, 'Foyer': 3, 'DiningRoom': 4, 'Wall': 5,
# 'Corridor': 6, 'Operable Wall': 7, 'Sliding Door': 8, 'Toilet': 9, 'Glass': 10, 'Window': 11, 'LivingRoom': 12,
# 'Kitchen': 13, 'Garage': 14, 'Stairs': 15, 'Door': 16}

# classes: {'background': 0, 'Wall': 1, 'Operable Wall': 2, 'Sliding Door': 3, 'Glass': 4, 'one_room': 5, 'Window': 6, 'Door': 7}


class PreProcess:
    def __init__(self):
        self.classes = {'non_road': 0, 'road': 1}

    def _binarize_mask(self, mask):
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        (thresh, mask) = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255
        return mask

    def process_single_img(self, src_in_fpath, gt_in_fpath, src_out_fpath, gt_out_fpath):
        src = cv2.imread(src_in_fpath)
        cv2.imwrite(src_out_fpath, img=src)

        if "test" not in gt_in_fpath:
            gt = cv2.imread(gt_in_fpath)
            gt = self._binarize_mask(gt)
            np.save(gt_out_fpath, gt)

    def process_single_type(self, type_in_dir, type_out_dir):
        src_in_dir = f"{type_in_dir}/src"
        gt_in_dir = f"{type_in_dir}/gt"

        src_out_dir = f"{type_out_dir}/src"
        gt_out_dir = f"{type_out_dir}/gt"

        dir_builder(src_out_dir)
        dir_builder(gt_out_dir)

        print(f"Processing {type_in_dir.split('/')[-1]}")
        # print("type_in_dir:", type_in_dir)
        for img_name in tqdm([x for x in os.listdir(f"{type_in_dir}/src") if ".png" in x]):
            src_in_fpath = f"{src_in_dir}/{img_name}"
            gt_in_fpath = f"{gt_in_dir}/{img_name}"

            src_out_fpath = f"{src_out_dir}/{img_name}"
            gt_out_fpath = f"{gt_out_dir}/{img_name.split('.')[0]}.npy"

            self.process_single_img(src_in_fpath, gt_in_fpath, src_out_fpath, gt_out_fpath)

    def run(self, in_dir, out_dir):
        _types = ["train", "val", "test"]
        for t in _types:
            self.process_single_type(f"{in_dir}/{t}", f"{out_dir}/{t}")


if __name__ == "__main__":
    PreProcess().run(RAW_DIR, PROCESSED_DIR)