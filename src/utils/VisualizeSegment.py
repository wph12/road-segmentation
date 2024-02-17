from datetime import datetime

import numpy as np
from skimage.draw import polygon2mask


import os

import cv2
from tqdm import tqdm

from src.utils.colors import get_colors


SRC_DIR = r"/Users/mouse/Documents/GitHub/CS-433_ML_Projects/MLProject2/tilings/val/src"
MASK_DIR = r"/Users/mouse/Documents/GitHub/CS-433_ML_Projects/MLProject2/tilings/val/gt"
OUT = r"/Users/mouse/Documents/GitHub/CS-433_ML_Projects/MLProject2/visualizations/val"

NUM_CLASSES = 2


def visualize_mask_overlay(src, gt, out, num_cls):
    gt = gt.astype(np.int64)
    # print("gt shape:", gt.shape)
    binary = False
    if num_cls <=2:
        binary = True
    colors = np.array(get_colors(binary))
    # print("colors shape:", colors.shape)
    # print("gt unique:", np.unique(gt))
    if len(gt.shape) == 3:
        gt = gt[:, :, 0]
    # print("gt shape:", gt.shape)
    # print("colors.shape", colors.shape)
    colored_image = colors[gt]
    colored_image = np.where(colored_image == 0, src, colored_image)
    # print("colored_image.shape:", colored_image.shape)
    out = out.replace(".npy", ".png")
    # print("out:", out)
    cv2.imwrite(out, colored_image)


def convert_npy_to_png(img_dir, viz_dir, assign_color=False):
    colors = np.array(get_colors())
    for img_name in tqdm(os.listdir(img_dir)):
        if ".npy" in img_name:
            img = np.load(img_dir + "/" + img_name)
            # print(img.shape)
            # print(colors.shape)
            if assign_color:
                if len(img.shape) == 3:
                    img = img[:, :, 0]
                    img = img.astype(np.int64)
                    # print(img.shape)
                img = colors[img]
            cv2.imwrite(viz_dir + "/" + img_name.replace(".npy", ".png"), img)


def generate_layered_masks(coordinates_lst, src_image_shape, class_index, class_name):
    for coordinates in coordinates_lst:
        print(
            f"creating mask for class {class_name} | {datetime.now().strftime('%H:%M:%S')}")
        print("coordinates_lst pre:", coordinates_lst)
        print("len coordinates_lst pre:", len(coordinates_lst))
        coordinates = [[y, x] for [x, y] in coordinates]  # polygon2mask coordinates are in y,x order.
        polygon = np.array(coordinates)
        mask = polygon2mask(src_image_shape, polygon).astype(int)
        mask *= class_index

        return mask


def output_masked_segment(fname, class_len, img_dir, mask, output_np=False):
    mask = (mask / class_len * 255).astype("uint8")[:, :, 0]

    if not output_np:
        result_path = os.path.join(img_dir, fname + '.png')
        cv2.imwrite(result_path, mask)
        #plt.savefig(result_path)
        print(f"visualized {result_path}")
    else:
        result_path = os.path.join(img_dir, fname + '.npy')
        print("mask.shape:", mask.shape)
        np.save(result_path, mask)


if __name__ == "__main__":
    for fname in tqdm(os.listdir(SRC_DIR)):
        if ".jpg" in fname or ".png" in fname:
            src_path = SRC_DIR + "/" + fname
            # print("src_path:", src_path)
            if not os.path.isfile(src_path):
                if ".jpg" in src_path:
                    src_path = src_path.replace(".jpg", ".png")
                else:
                    src_path = src_path.replace(".png", ".jpg")
            src = cv2.imread(src_path)
            mask_path = MASK_DIR + "/" + fname.split(".")[0] + ".npy"
            # print("mask_pathP:", mask_path)
            if not os.path.isfile(mask_path):
                if ".jpg" in mask_path:
                    mask_path = mask_path.replace(".jpg", ".png")
                else:
                    mask_path = mask_path.replace(".png", ".jpg")
            mask = np.load(MASK_DIR + "/" + fname.split(".")[0] + ".npy")
            visualize_mask_overlay(src, mask, OUT + "/" + fname, NUM_CLASSES)
        elif ".npy" in fname:
            src_path = SRC_DIR + "/" + fname
            # print("src_path:", src_path)
            src = np.load(src_path)
            mask_path = MASK_DIR + "/" + fname.split(".")[0].replace("src", "gt") + ".npy"
            # print("mask_pathP:", mask_path)
            if not os.path.isfile(mask_path):
                if ".jpg" in mask_path:
                    mask_path = mask_path.replace(".jpg", ".png")
                else:
                    mask_path = mask_path.replace(".png", ".jpg")
            mask = np.load(mask_path)
            # print("mask shape", mask.shape)
            visualize_mask_overlay(src, mask, OUT + "/" + fname, NUM_CLASSES)
