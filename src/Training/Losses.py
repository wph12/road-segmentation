from src.Training.Metrics import dice_coefficient, iou_scaled_binary, iou, iou_roi
import tensorflow as tf


def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * (1 - dice_coefficient(y_true, y_pred))


def iou_loss(y_true, y_pred):
    return 1.0 - iou(y_true, y_pred)


def iou_roi_scaled_binary_loss(roi_weight):
    def loss_function(y_true, y_pred):
        return 1 - iou_scaled_binary(roi_weight)(y_true, y_pred)
    return loss_function


def iou_roi_loss(y_true, y_pred):
    return 1.0 - iou_roi(y_true, y_pred)


