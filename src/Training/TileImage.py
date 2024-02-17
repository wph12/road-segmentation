import os

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.tiling_and_stitching import tile_img

SRC_DIR = r"/Users/mouse/Documents/GitHub/CS-433_ML_Projects/MLProject2/dataset/processed/val/src"
GT_DIR = r"/Users/mouse/Documents/GitHub/CS-433_ML_Projects/MLProject2/dataset/processed/val/gt"
SRC_OUTPUT_DIR = r"/Users/mouse/Documents/GitHub/CS-433_ML_Projects/MLProject2/tilings/val/src"
GT_OUTPUT_DIR = r"/Users/mouse/Documents/GitHub/CS-433_ML_Projects/MLProject2/tilings/val/gt"
NUM_CLASSES = 2
TILE_SIZE = 512


class Tile:
    def __init__(self, src_dir, gt_dir, src_output_dir, gt_output_dir, img_size, train=True):
        self.src_dir = src_dir
        self.gt_dir = gt_dir
        self.src_output_dir = src_output_dir
        print("gt_output_dir:", gt_output_dir)
        self.gt_output_dir = gt_output_dir
        self.img_size = img_size
        self.train = train

    def _switch_ext(self, fpath):
        if not os.path.isfile(fpath):
            if ".jpg" in fpath:
                return fpath.replace(".jpg", ".png")
            else:  # .png in fpath
                return fpath.replace(".png", ".jpg")
        else:
            return fpath

    def _tile_single(self, src_dir, gt_dir, fname, src_output_dir, gt_output_dir):
        src_fpath = f"{src_dir}/{fname}"
        # print(f"tiling img & gt: {fname}")
        # print("src_fpath:", src_fpath)
        src = cv2.imread(src_fpath)

        tiles_src = tile_img(src, (self.img_size, self.img_size, 3))

        if self.train:
            gt_fpath = f"{gt_dir}/{fname.split('.')[0] + '.npy'}"
            if not os.path.isfile(gt_fpath):
                gt_fpath = f"{gt_dir}/{fname.split('.')[0] + '.png'}"
                gt = cv2.imread(gt_fpath)
            else:
                gt = np.load(gt_fpath)
            # print("gt_fpath:", gt_fpath)

            tiles_mask = tile_img(gt, (self.img_size, self.img_size, 3))

        idx_to_remove = []

        for r_id, r in enumerate(tiles_src):
            for s_id, s in enumerate(r):
                if self.train and np.unique(s[0])[0] == 7:
                    idx_to_remove.append((r_id, s_id))

        if self.train:
            if not os.path.isdir(gt_output_dir):
                os.mkdir(gt_output_dir)

            for col_idx, col in enumerate(tiles_mask):
                for row_idx, gt_tple in enumerate(col):
                    gt, coords = gt_tple
                    mask_fname = fname.split(".")[0] + f"_gt_r{row_idx}_c{col_idx}"
                    if coords is not None:
                        y0, y1, x0, x1 = coords
                        mask_fname += f"_coords_x0{x0}_x1{x1}_y0{y0}_y1{y1}"

                    # print("fname:", fname)
                    mask_fname = os.path.join(gt_output_dir, mask_fname)
                    # print("gt unique:", np.unique(gt))
                    mask_fname += ".npy"
                    np.save(mask_fname, gt)
                    # print(f"saved mask {mask_fname}")
                    # self.total_num_tiles += 1

        if not os.path.isdir(src_output_dir):
            os.mkdir(src_output_dir)
        for col_idx, col in enumerate(tiles_src):
            for row_idx, src_tple in enumerate(col):
                src, coords = src_tple
                src_fname = src_output_dir + "/" + fname.split(".")[0] + f"_src_r{row_idx}_c{col_idx}"
                if coords is not None:
                    y0, y1, x0, x1 = coords
                    src_fname += f"_coords_x0{x0}_x1{x1}_y0{y0}_y1{y1}"

                src_fname += ".npy"
                np.save(src_fname, src)
                # print(f"saved src {src_fname}")

    def run(self):
        # fname = r"uplan-02.png"
        srcname_list = os.listdir(self.src_dir)
        # gtname_list = os.listdir(self.gt_dir)
        #print("tiling")

        for src_name in tqdm(srcname_list):
            if src_name != ".DS_Store":
                self._tile_single(self.src_dir, self.gt_dir, src_name, self.src_output_dir, self.gt_output_dir)


if __name__ == "__main__":
    Tile(SRC_DIR, GT_DIR, SRC_OUTPUT_DIR, GT_OUTPUT_DIR, TILE_SIZE).run()
