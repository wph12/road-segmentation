import tensorflow as tf


class MLP(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio, num_input_channels):
        super(MLP, self).__init__()
        squeeze_val = num_input_channels // reduction_ratio
        self.relu = tf.keras.layers.Dense(squeeze_val, activation="relu", use_bias=False)
        self.fcn = tf.keras.layers.Dense(num_input_channels, use_bias=False)

    def __call__(self, inputs, *args, **kwargs):
        x = self.relu(inputs)
        x = self.fcn(x)
        return x


class AvgPool(tf.keras.layers.Layer):
    def __init__(self, axis=None):
        super(AvgPool, self).__init__()
        self.axis = axis
        if self.axis is None:
            self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()

    def __call__(self, inputs, *args, **kwargs):
        if self.axis is None:
            x = self.global_avg_pool(inputs)
        else:
            x = tf.reduce_mean(inputs, axis=self.axis)
            x = tf.expand_dims(x, axis=self.axis)
        return x


class MaxPool(tf.keras.layers.Layer):
    def __init__(self, axis=None):
        super(MaxPool, self).__init__()
        self.axis = axis
        if self.axis is None:
            self.global_max_pool = tf.keras.layers.GlobalMaxPooling2D()

    def __call__(self, inputs, *args, **kwargs):
        if self.axis is None:
            x = self.global_max_pool(inputs)
        else:
            x = tf.reduce_max(inputs, axis=self.axis)
            x = tf.expand_dims(x, axis=self.axis)
        return x


class ChannelAttentionModule(tf.keras.layers.Layer):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.avg_pool = AvgPool()
        self.mlp = MLP(self.reduction_ratio, num_channels)

        self.max_pool = MaxPool()

        self.add = tf.keras.layers.Add()
        self.activation = tf.keras.layers.Activation("sigmoid")
        self.multiply = tf.keras.layers.Multiply()

    def __call__(self, inputs, *args, **kwargs):

        # Avg pool
        x = self.avg_pool(inputs)
        avg_pooled = self.mlp(x)

        # Max Pool
        x = self.max_pool(inputs)
        max_pooled = self.mlp(x)

        # Sum mlp_avg and mlp_max and apply sigmoid
        add = self.add([avg_pooled, max_pooled])
        weights = self.activation(add)
        scaled_features = self.multiply([inputs, weights])
        return scaled_features


class SpatialAttentionModule(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        # Pooling along axis found to be effective in paper
        self.avg_pool = AvgPool(axis=-1)
        self.max_pool = MaxPool(axis=-1)

        self.concat = tf.keras.layers.Concatenate()

        self.conv = tf.keras.layers.Conv2D(1, kernel_size=7, padding="same",
                                           activation="sigmoid")  # kernel size of 7 is the best according to the paper
        self.multiply = tf.keras.layers.Multiply()

    def __call__(self, inputs, *args, **kwargs):
        avg_pooled = self.avg_pool(inputs)
        max_pooled = self.max_pool(inputs)

        concatenated = self.concat([avg_pooled, max_pooled])
        weights = self.conv(concatenated)
        scaled_features = self.multiply([weights, inputs])
        return scaled_features


class CBAM(tf.keras.layers.Layer):
    def __init__(self, num_channels, *args, **kwargs):
        super(CBAM, self).__init__(*args, **kwargs)
        self.channel_attn = ChannelAttentionModule(num_channels)
        self.spatial_attn = SpatialAttentionModule()

    def __call__(self, inputs, *args, **kwargs):
        # Sequential arrangement with channel attn first found to be more effective by paper
        x = self.channel_attn(inputs)
        x = self.spatial_attn(x)
        return x
