from src.Architectures.Components.HighLevel.unet import DownSample, Encoder, BottleNeck, UpSample, Decoder
from src.Architectures.Components.LowLevel.cbam import CBAM


class CBAMDownSample(DownSample):
    def __init__(self, n_filters, dropout_rate):
        super(CBAMDownSample, self).__init__(n_filters, dropout_rate)
        self.cbam = CBAM(n_filters)

    def __call__(self, inputs, *args, **kwargs):
        conv = self.conv_block(inputs)
        skip = self.cbam(conv)
        x = self.max_pool(skip)
        x = self.dropout(x)
        return x, skip


class CBAMEncoder(Encoder):
    def get_downsample_block(self, filter_size, dropout_rate):
        return CBAMDownSample(filter_size, dropout_rate=dropout_rate)


class CBAMBottleNeck(BottleNeck):
    def __init__(self, filter_size):
        super(CBAMBottleNeck, self).__init__(filter_size)
        self.cbam = CBAM(filter_size)

    def __call__(self, inputs, *args, **kwargs):
        x = self.conv_block(inputs)
        return self.cbam(x)


class CBAMUpSample(UpSample):
    def __init__(self, filter_size, dropout_rate):
        super(CBAMUpSample, self).__init__(filter_size, dropout_rate)
        self.cbam = CBAM(filter_size)

    def __call__(self, inputs, skip_conn, *args, **kwargs):
        x = self.up_conv(inputs)
        x = self.concat([x, skip_conn])
        x = self.dropout(x)
        x = self.conv_block(x)
        x = self.cbam(x)
        return x


class CBAMDecoder(Decoder):
    def __init__(self, starting_filter_size, dropout_rate):
        super().__init__(starting_filter_size, dropout_rate)

    def get_upsample_block(self, n_filters, dropout_rate):
        return CBAMUpSample(n_filters, dropout_rate=dropout_rate)



