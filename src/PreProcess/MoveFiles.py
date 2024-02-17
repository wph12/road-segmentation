import os
import shutil

from tqdm import tqdm

SRC_DIR = "/Users/mouse/Documents/GitHub/CS-433_ML_Projects/MLProject2/dataset/test_set_images"
src_imgs = [f"{SRC_DIR}/{x}/{x}.png" for x in os.listdir(SRC_DIR) if x != ".DS_STORE"]

DEST_DIR = "/Users/mouse/Documents/GitHub/CS-433_ML_Projects/MLProject2/dataset/test"
for src_img in tqdm(src_imgs):
    shutil.move(src_img, f"{DEST_DIR}/{src_img.split('/')[-1]}")


