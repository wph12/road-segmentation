import os

import numpy as np

from src.utils.VisualizeSegment import visualize_mask_overlay


def post_process(sample, prediction, viz_tiles_dir, fname, NUM_CLASSES):
    if len(prediction.shape) == 4:
        pred = prediction[0, :, :, :]
    else:
        pred = prediction

    if NUM_CLASSES > 2:
        pred = np.argmax(pred, axis=-1)
    else:
        pred = (pred >= 0.5).astype(np.uint8)
    # pred = pred[:, :, 0]
    # pred = np.dstack((pred, pred, pred))
    # print("sample_gt classes:", np.unique(sample_gt))
    # print("pred classes:", list(np.unique(pred)))
    # print("pred shape:", pred.shape)

    if not os.path.isdir(viz_tiles_dir):
        os.mkdir(viz_tiles_dir)

    out_fpath = viz_tiles_dir + "/" + fname.replace(".npy", ".png")
    # print('visualizing tile:', out_fpath)
    sample = np.zeros(sample.shape)
    visualize_mask_overlay(sample, pred, out_fpath, NUM_CLASSES)