import os

import cv2
import keras.backend
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.utils.VisualizeSegment import visualize_mask_overlay

IMG_DIR = "/Users/mouse/Documents/GitHub/CS-433_ML_Projects/MLProject2/tilings/train/src"
GT_DIR = "/Users/mouse/Documents/GitHub/CS-433_ML_Projects/MLProject2/tilings/train/gt"
HEIGHT = 400
WIDTH = 400
NUM_CLASSES = 2
BATCH_SIZE = 16
AUG = True


class Augment(tf.keras.layers.Layer):
    def __init__(self, tile_size, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        random_gen = tf.random.Generator.from_seed(seed)
        factor = random_gen.uniform(shape=[1,], minval=0, maxval=359)[0]
        print("factor:", factor)
        # Rotate
        self.rotate_src = tf.keras.layers.RandomRotation(factor=factor, seed=seed)
        self.rotate_gt = tf.keras.layers.RandomRotation(factor=factor, seed=seed)

        # Brightness
        self.brightness_src = tf.keras.layers.RandomBrightness(factor=0.001, seed=seed)

        # Contrast
        self.contrast_src = tf.keras.layers.RandomContrast(factor=0.5, seed=seed)

    def call(self, src, gt):
        print("augmenting")
        # Rotate
        src = self.rotate_src(src)
        gt = self.rotate_gt(gt)

        # Brightness
        src = self.brightness_src(src)

        # Contrast
        src = self.contrast_src(src)

        return src, gt


class Dataset:
    def __init__(self, img_dir, gt_dir, height, width, num_classes, aug=False):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.aug = aug

    def load_data(self):
        img_paths = [x for x in os.listdir(self.img_dir) if ".npy" in x]
        if len(img_paths) == 0:
            raise Exception(f"Image Path List is Empty. Path: {self.img_dir}")
        gt_paths = [x for x in os.listdir(self.gt_dir) if ".npy" in x]
        if len(gt_paths) == 0:
            raise Exception(f"GT Path List is Empty. Path: {self.gt_dir}")

        img_paths.sort()
        gt_paths.sort()

        # ("img_paths[0]:", img_paths[0])
        # print("gt_paths[0]:", gt_paths[0])

        # print("len imgs:", len(img_paths))
        # print("len gts:", len(gt_paths))

        return img_paths

    def tf_parse(self, fname: str, _):
        def _parse(img_fname):
            # print("img_fname:", img_fname)
            img = np.load(self.img_dir + "/" + img_fname.decode())
            gt = np.load(self.gt_dir + "/" + img_fname.decode().replace("src", "gt"))

            img = img.astype(np.float64)
            gt = gt.astype(np.float64)
            if self.num_classes == 2:
                # print("squeezing dim3 to dim2")
                gt = np.expand_dims(gt[:, :, 0], axis=-1)

            img = img / 255

            return img, gt
        src, mask = tf.numpy_function(_parse, [fname], [tf.float64, tf.float64])
        src.set_shape([self.height, self.width, 3])
        mask.set_shape([self.height, self.width, 1])
        if self.num_classes > 2:
            mask = keras.backend.one_hot(mask[:, :, 0], self.num_classes)

        # print("src shape:", src.shape)
        # print("mask shape:", mask.shape)
        return src, mask

    def augment(self):
        data_augmentation = tf.keras.Sequential([
            keras.layers.RandomFlip("horizontal_and_vertical", seed=42),
            keras.layers.RandomRotation(0.2, seed=42),
        ])

    def ingest(self, batch_size):
        img_paths = self.load_data()
        # print("loaded image paths")
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, img_paths))
        # print("len dataset:", len(dataset))
        dataset = dataset.map(self.tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
        # print("mapped dataset")
        # print("len dataset:", len(dataset))
        if self.aug:
            print("augmented setting True")
            dataset = dataset.map(Augment(self.height), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000)
        print("shuffled dataset")
        # print("len dataset:", len(dataset))
        dataset = dataset.batch(batch_size)
        # print("batched dataset")
        # print("len dataset:", len(dataset))

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        # print("len dataset:", len(dataset))
        return dataset


if __name__ == "__main__":
    d = Dataset(img_dir=IMG_DIR, gt_dir=GT_DIR, num_classes=NUM_CLASSES, height=HEIGHT, width=WIDTH, aug=AUG).ingest(BATCH_SIZE)
    counter = 1
    for batched_images, batched_masks in tqdm(d.take(90)):
        print(batched_masks.shape)
        for idx, image in enumerate(batched_images):
            sample_image, sample_mask = image * 255, batched_masks[idx]
            sample_mask = sample_mask.numpy()[:, :, 0]
            # print("sample_mask.shape:", sample_mask.shape)
            visualize_mask_overlay(gt=sample_mask, src=sample_image.numpy(), num_cls=NUM_CLASSES,
                                   out=f"/Users/mouse/Documents/GitHub/CS-433_ML_Projects/MLProject2/visualizations/dataset_viz/train/{counter}.png")

            counter += 1
    print("done")

