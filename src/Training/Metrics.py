from keras import backend as K
import tensorflow as tf


def iou(y_true, y_pred):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + 1e-7) / (union + 1e-7)


def iou_roi(y_true, y_pred):
    y_true = tf.keras.backend.flatten(y_true[..., 1:])
    y_pred = tf.keras.backend.flatten(y_pred[..., 1:])
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + 1e-7) / (union + 1e-7)


def iou_bg(y_true, y_pred):
    y_true = tf.keras.backend.flatten(y_true[..., 0])
    y_pred = tf.keras.backend.flatten(y_pred[..., 0])
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + 1e-7) / (union + 1e-7)

def iou_scaled_binary(roi_weight=0.5):
    def metric_fn(y_true, y_pred):
        return (1 - roi_weight) * iou_bg(y_true, y_pred) + roi_weight * iou(y_true[..., 1:], y_pred[..., 1:])

    return metric_fn


def dice_coefficient(y_true, y_pred, smooth=0.00001):
    """Dice loss originates from Sørensen–Dice coefficient, which is a statistic developed in 1940s to gauge the similarity between two samples.
    Variant: Over all pixels
    Credits documentation: https://github.com/mlyg

    Parameters
    ----------
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))