import tensorflow as tf


class DoubleConv(tf.keras.layers.Layer):
    def __init__(self, n_filters):
        super(DoubleConv, self).__init__()
        # Conv followed by ReLU
        self.conv1 = tf.keras.layers.Conv2D(n_filters, 3, padding="same", activation="relu",
                                            kernel_initializer="he_normal")
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(n_filters, 3, padding="same", activation="relu",
                                            kernel_initializer="he_normal")
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

    def __call__(self, inputs, *args, **kwargs):
        # Double Convolution
        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x


class DownSample(tf.keras.layers.Layer):
    def __init__(self, n_filters, dropout_rate):
        super(DownSample, self).__init__()
        self.conv_block = self.get_conv_block(n_filters)
        self.max_pool = tf.keras.layers.MaxPool2D(2)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.n_filters = n_filters

    def get_conv_block(self, n_filters):
        return DoubleConv(n_filters)

    def __call__(self, inputs, *args, **kwargs):
        skip = self.conv_block(inputs)
        x = self.max_pool(skip)
        x = self.dropout(x)
        return x, skip


class Encoder(tf.keras.layers.Layer):
    def __init__(self, starting_filter_size=64, dropout_rate=0.3):
        super(Encoder, self).__init__()
        self.filter_sizes = [(2 ** x) * starting_filter_size for x in range(4)]  # [64, 128, 256, 512]
        self.encoder_layers = [self.get_downsample_block(x, dropout_rate) for x in self.filter_sizes]

    def get_downsample_block(self, filter_size, dropout_rate):
        return DownSample(filter_size, dropout_rate)

    def __call__(self, inputs=None, *args, **kwargs):
        # print("self.encoder_layers[0]:", self.encoder_layers[0])
        if not isinstance(self.encoder_layers[0], tf.keras.layers.Layer) and tf.keras.backend.is_keras_tensor(self.encoder_layers[0]):
            x, skip_1 = self.encoder_layers[0], self.encoder_layers[0]
            x, skip_2 = self.encoder_layers[1], self.encoder_layers[1]
            x, skip_3 = self.encoder_layers[2], self.encoder_layers[2]
            x, skip_4 = self.encoder_layers[3], self.encoder_layers[3]
        else:
            x, skip_1 = self.encoder_layers[0](inputs)
            x, skip_2 = self.encoder_layers[1](x)
            x, skip_3 = self.encoder_layers[2](x)
            x, skip_4 = self.encoder_layers[3](x)
        # print("x.shape:", x.shape)
        # print("skip_4.shape:", skip_4.shape)
        return x, [skip_1, skip_2, skip_3, skip_4]

    def get_num_layers(self):
        return len(self.encoder_layers)


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_size):
        super(BottleNeck, self).__init__()
        self.conv_block = self.get_conv_block(filter_size)

    def get_conv_block(self, filter_size):
        return DoubleConv(filter_size)

    def __call__(self, inputs, *args, **kwargs):
        return self.conv_block(inputs)


class UpSample(tf.keras.layers.Layer):
    def __init__(self, filter_size, dropout_rate):
        super(UpSample, self).__init__()
        # Up-conv 2x2
        self.up_conv = tf.keras.layers.Conv2DTranspose(filter_size, 3, 2, padding="same")

        # Skip-connection
        self.concat = tf.keras.layers.Concatenate()

        # dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # conv 3x3, ReLU
        self.conv_block = self.get_conv_block(filter_size)

    def get_conv_block(self, filter_size):
        return DoubleConv(filter_size)


    def __call__(self, inputs, skip_conn, *args, **kwargs):
        x = self.up_conv(inputs)
        x = self.concat([x, skip_conn])
        x = self.dropout(x)
        x = self.conv_block(x)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, starting_filter_size, dropout_rate):
        super(Decoder, self).__init__()
        self.filter_sizes = [(2 ** x) * starting_filter_size for x in range(4)]
        self.filter_sizes.reverse()  # [512, 256, 128, 64]
        print("decoder filter_sizes:", self.filter_sizes)
        self.decoder_layers = [self.get_upsample_block(x, dropout_rate) for x in self.filter_sizes]

    def get_upsample_block(self, n_filters, dropout_rate):
        return UpSample(n_filters, dropout_rate=dropout_rate)

    def __call__(self, inputs, skip_connections, *args, **kwargs):
        decoder_1 = self.decoder_layers[0](inputs, skip_connections[3])
        decoder_2 = self.decoder_layers[1](decoder_1, skip_connections[2])
        decoder_3 = self.decoder_layers[2](decoder_2, skip_connections[1])
        decoder_4 = self.decoder_layers[3](decoder_3, skip_connections[0])
        return decoder_4
