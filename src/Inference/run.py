import os
import shutil

import numpy as np
import tensorflow as tf
from keras.utils import custom_object_scope
from tqdm import tqdm

from src.Training.Losses import dice_loss
from src.Training.Metrics import dice_coefficient, iou
from src.Training.TileImage import Tile
from src.utils.DirBuilder import dir_builder
from src.utils.post_process import post_process
from src.utils.tiling_and_stitching import stitch
from pathlib import Path
import matplotlib.image as mpimg
import re
base_dir = str(Path(os.getcwd()).parent.parent.absolute())

# base_dir = r"/Users/mouse/Documents/GitHub/CS-433_ML_Projects/ml-project-2-theasiandudes"
MODEL_NAME = r"/CbamUnet_last"

temp_dir = base_dir + r"/testing/temp"


TEST_TILED_IMG_DIR = temp_dir + "/tiled/src"
TEST_TILED_GT_DIR = temp_dir + "/tiled/gt"
VIZ_TILES_DIR = temp_dir + r"/viz"
RESULTS_DIR = base_dir + r"/results"
SUBS_DIR = base_dir + r"/submissions/"

data_dir = base_dir + "/dataset/dataset"
TEST_IMG_DIR = data_dir + r"/test/src"
TEST_GT_DIR = data_dir + r"/test/gt"

TILE_SIZE = 400
NUM_CLS = 2
MODEL_DIR = base_dir + r"/model"
MODEL_PATH = MODEL_DIR + MODEL_NAME

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename.split("/")[-1]).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))

def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


def run(test_tiled_img_dir, test_tiled_gt_dir, results_dir, test_img_dir, test_gt_dir, tile_size, model_path, viz_tiles_dir, num_classes):
    # Cleanup and Build Temp Directories
    temp_dirs = [test_tiled_img_dir, test_tiled_gt_dir, VIZ_TILES_DIR, results_dir]
    print("Cleaning Temp and Rebuilding Directories...")
    for d in tqdm(temp_dirs):
        if os.path.isdir(d):
            shutil.rmtree(d)
        dir_builder(d)

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    # Tile Test Dataset
    print("Tiling Test Dataset")
    Tile(test_img_dir, test_gt_dir, test_tiled_img_dir, test_tiled_gt_dir, tile_size, train=False).run()

    mirrored_strategy = tf.distribute.MirroredStrategy()

    dice = dice_coefficient

    with mirrored_strategy.scope():
        with custom_object_scope({"dice_coefficient": dice, "iou": iou, "dice_loss": dice_loss}):
            model = tf.keras.models.load_model(model_path)

            # Predict
            print("Predicting...")
            for sample_f in tqdm(os.listdir(test_tiled_img_dir)):
                # print("sample_f:", sample_f)
                sample = np.load(test_tiled_img_dir + "/" + sample_f)
                # sample_gt = np.load(test_tiled_gt_dir + "/" + sample_f.replace("src", "gt"))

                prediction = model.predict(np.array([sample / 255]))
                post_process(np.zeros(sample.shape), prediction, viz_tiles_dir, sample_f, num_classes)

            stitch(viz_tiles_dir, results_dir, tile_size, tile_size)

    submission_filename = SUBS_DIR + '/submission.csv'
    image_filenames = []
    for i in range(1, 51):
        image_filename = RESULTS_DIR + "/test" + str(i) + 's_merged.png'
        print(image_filename)
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)
    print("Submission file saved to", submission_filename)


if __name__ == "__main__":
    run(
        test_tiled_img_dir=TEST_TILED_IMG_DIR,
        test_tiled_gt_dir=TEST_TILED_GT_DIR,
        viz_tiles_dir=VIZ_TILES_DIR,
        results_dir=RESULTS_DIR,
        test_img_dir=TEST_IMG_DIR,
        test_gt_dir=TEST_GT_DIR,
        tile_size=TILE_SIZE,
        model_path=MODEL_PATH,
        num_classes=NUM_CLS
    )
