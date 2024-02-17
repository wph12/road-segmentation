# imports
import json
import math
import os
import pickle
import shutil

import numpy as np
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm

import src.Training.Losses
from IngestDataset import Dataset
from Losses import dice_loss
from src.Training.Metrics import dice_coefficient
from src.utils.DescribeClass import describe
from src.utils.DirBuilder import dir_builder
from src.utils.post_process import post_process
from src.utils.tiling_and_stitching import stitch
from TileImage import Tile
import tensorflow as tf
# from keras_unet_collection import models
from src.Architectures.Models.BaseModel import Unet
from src.Architectures.Models.CBAMUnet import CbamUnet
# from src.Architectures.Models.CBAMResUnet import CbamResUnet
# from src.Architectures.Models.CBAMResAttentionUnet import CbamResAttentionUnet
# from src.Architectures.Models.CBAMAttentionUnet import CbamAttentionUnet
# from src.Architectures.Models.SeiCBAMUnet import SeiCbamUnet
# from src.Architectures.Models.SeCbamUnet import SeCbamUnet
# from src.Architectures.Models.SeiCbamAttentionUnet import SeiCbamAttentionUnet

from pathlib import Path
base_dir = str(Path(os.getcwd()).parent.parent.absolute())
data_dir = base_dir + "/dataset"

TRAIN_IMG_DIR = data_dir + "/train/src"""
TRAIN_GT_DIR = data_dir + "/train/gt"
VAL_IMG_DIR = data_dir + "/val/src"
VAL_GT_DIR = data_dir + "/val/gt"
TEST_IMG_DIR = data_dir + "/test/src"
TEST_GT_DIR = base_dir + "/test/gt"

temp_dir = base_dir + "/temp"
tiles_dir = temp_dir + "/tiles"
TRAIN_TILED_IMG_DIR = tiles_dir + "/train/src"
TRAIN_TILED_GT_DIR = tiles_dir + "/train/gt"
VAL_TILED_IMG_DIR = tiles_dir + "/val/src"
VAL_TILED_GT_DIR = tiles_dir + "/val/gt"
TEST_TILED_IMG_DIR = tiles_dir + "/test/src"
TEST_TILED_GT_DIR = tiles_dir + "/test/gt"

VIZ_TILES_DIR = tiles_dir + "/viz"

MODEL_DIR = base_dir + "/models"
# MODEL_PATH = MODEL_DIR + "/Unet"

RESULTS_DIR = base_dir + "/results"

PATIENCE = 10
BATCH_SIZE = 8
NUM_EPOCHS = 400
NUM_CLASSES = 2
TILE_SIZE = 400
AUG = True


def train(train_img_dir, train_gt_dir,
          val_img_dir, val_gt_dir,
          test_img_dir, test_gt_dir,
          train_tiled_img_dir, train_tiled_gt_dir,
          val_tiled_img_dir, val_tiled_gt_dir,
          test_tiled_img_dir, test_tiled_gt_dir,
          viz_tiles_dir,
          num_classes, tile_size,
          batch_size, num_epochs,
          results_dir):
    # Cleanup and Build Temp Directories
    temp_dirs = [train_tiled_img_dir, train_tiled_gt_dir, val_tiled_img_dir, val_tiled_gt_dir,
                 test_tiled_img_dir, test_tiled_gt_dir, VIZ_TILES_DIR, results_dir]
    print("Cleaning Temp and Rebuilding Directories...")
    for d in tqdm(temp_dirs):
        if os.path.isdir(d):
            shutil.rmtree(d)
        dir_builder(d)

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    # Tile Train Dataset
    print("Tiling Train Dataset")
    Tile(train_img_dir, train_gt_dir, train_tiled_img_dir, train_tiled_gt_dir, tile_size).run()

    # Tile Val Dataset
    print("Tiling Val Dataset")
    Tile(val_img_dir, val_gt_dir, val_tiled_img_dir, val_tiled_gt_dir, tile_size).run()

    # Tile Test Dataset
    print("Tiling Test Dataset")
    Tile(test_img_dir, test_gt_dir, test_tiled_img_dir, test_tiled_gt_dir, tile_size, train=False).run()

    # Ingest Dataset
    train = Dataset(
        img_dir=train_tiled_img_dir,
        gt_dir=train_tiled_gt_dir,
        height=tile_size,
        width=tile_size,
        num_classes=num_classes,
        aug=AUG
    ).ingest(batch_size)
    val = Dataset(
        img_dir=val_tiled_img_dir,
        gt_dir=val_tiled_gt_dir,
        height=tile_size,
        width=tile_size,
        num_classes=num_classes,
        aug=False
    ).ingest(batch_size)

    print("batches train", len(train), "len:", len(train)*batch_size)
    print("batches val", len(val), "len:", len(val)*batch_size)

    cls_weights = describe(train_tiled_gt_dir)

    if type(cls_weights) == dict:
        cls_d = cls_weights
    else:
        cls_d = {}

        for idx, w in enumerate(cls_weights):
            cls_d[idx] = w

    weights_lst = []
    cls_lst = list(cls_d.keys())
    cls_lst.sort()
    for cls in cls_lst:
        weights_lst.append(int(math.ceil(cls_d[cls])))
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        dice = dice_coefficient
        if num_classes > 2:
            _dice_loss = dice_loss
            iou = tf.metrics.OneHotMeanIoU(num_classes=num_classes)
        else:
            _dice_loss = dice_loss
            iou = src.Training.Metrics.iou
        loss_fn = _dice_loss
        METRIC_FNS = [dice, iou]
        # Initialize Model
        if num_classes > 2:
            _num_cls = num_classes
        else:
            _num_cls = 1
        model = CbamUnet(starting_filter_size=64, num_classes=2, dropout_rate=0.3, tile_size=tile_size,
                         num_input_channels=3)()
        print(model.summary())

        # Compile Model
        model.compile(optimizer=tf.optimizers.Adam(0.001),
                      loss=loss_fn,
                      metrics=METRIC_FNS)
        # Model callbacks
        # es = EarlyStopping(monitor='val_loss', mode='min', patience=PATIENCE)
        model_fpath = MODEL_DIR + "/" + model.name
        print("model_fpath:", model_fpath)
        """mc = ModelCheckpoint(model_fpath + "_chkpt", monitor='val_loss', mode='min', save_best_only=True,
                             save_freq=10)"""
        """reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                      patience=10, verbose=0)"""

        cb = []
        print("training", model.name)

        # Train Model
        model_history = model.fit(train, validation_data=val, epochs=num_epochs, batch_size=batch_size,
                                  callbacks=cb, class_weight=cls_weights)

        # Save Model
        model.save(model_fpath + "_last")
        print("saved model to", model_fpath + "_last")

        # Get the dictionary containing each metric and the loss for each epoch
        history_dict = model_history.history
        history_fpath = model_fpath + "_history.pickle"
        print("saved history to", history_fpath)

        # Save it under the form of a pickle file
        with open(history_fpath, 'wb') as hist_file:
            pickle.dump(history_dict, hist_file)

        # Predict
        print("Predicting...")
        for sample_f in tqdm(os.listdir(test_tiled_img_dir)):
            # print("sample_f:", sample_f)
            sample = np.load(test_tiled_img_dir + "/" + sample_f)
            # sample_gt = np.load(test_tiled_gt_dir + "/" + sample_f.replace("src", "gt"))

            prediction = model.predict(np.array([sample / 255]))
            post_process(np.zeros(sample.shape), prediction, viz_tiles_dir, sample_f, num_classes)

        stitch(viz_tiles_dir, results_dir, tile_size, tile_size)


if __name__ == "__main__":
    train(TRAIN_IMG_DIR, TRAIN_GT_DIR,
          VAL_IMG_DIR, VAL_GT_DIR,
          TEST_IMG_DIR, TEST_GT_DIR,
          TRAIN_TILED_IMG_DIR, TRAIN_TILED_GT_DIR,
          VAL_TILED_IMG_DIR, VAL_TILED_GT_DIR,
          TEST_TILED_IMG_DIR, TEST_TILED_GT_DIR,
          VIZ_TILES_DIR,
          NUM_CLASSES, TILE_SIZE,
          BATCH_SIZE, NUM_EPOCHS,
          RESULTS_DIR)
