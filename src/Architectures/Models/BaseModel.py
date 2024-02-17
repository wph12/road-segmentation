import tensorflow as tf
from src.Architectures.Components.HighLevel.unet import Encoder, Decoder, BottleNeck


class Unet:
    name = "Unet"

    def __init__(self, starting_filter_size, num_classes, dropout_rate, tile_size, num_input_channels):
        self._input_shape = (tile_size, tile_size, num_input_channels)
        self.inputs = tf.keras.Input(self._input_shape)
        self.encoder, num_encoder_layers = self.get_encoder(starting_filter_size, dropout_rate)
        self.bottleneck = self.get_bottleneck(starting_filter_size, num_encoder_layers)
        self.decoder = self.get_decoder(starting_filter_size, dropout_rate)
        if num_classes > 2:
            activation_fn = "softmax"
        else:
            activation_fn = "sigmoid"
            num_classes = 1
        self.outputs = tf.keras.layers.Conv2D(num_classes, 1, padding="same", activation=activation_fn)
        self.outputs = self.call()

    def get_encoder(self, starting_filter_size, dropout_rate):
        encoder = Encoder(starting_filter_size, dropout_rate)
        return encoder, encoder.get_num_layers()

    def get_bottleneck(self, starting_filter_size, num_encoder_layers):
        filter_size = (2 ** num_encoder_layers) * starting_filter_size
        # print("bottleneck filter size:", filter_size)
        return BottleNeck(filter_size)

    def get_decoder(self, starting_filter_size, dropout_rate):
        return Decoder(starting_filter_size, dropout_rate)

    def call(self):
        x, skip_conns = self.encoder(self.inputs)
        x = self.bottleneck(x)
        x = self.decoder(inputs=x, skip_connections=skip_conns)
        x = self.outputs(x)
        return x

    def __call__(self, *args, **kwargs):
        print(self.inputs)
        return tf.keras.Model(inputs=self.inputs, outputs=self.outputs, name=self.name)


if __name__ == "__main__":
    model = Unet(64, 2, 0.3, 512, 3)()
    print(model.summary())
