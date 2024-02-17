from src.Architectures.Models.BaseModel import Unet
from src.Architectures.Components.HighLevel.cbam_unet import CBAMEncoder, CBAMBottleNeck, CBAMDecoder


class CbamUnet(Unet):
    name = "CbamUnet"

    def get_encoder(self, starting_filter_size, dropout_rate):
        encoder = CBAMEncoder(starting_filter_size, dropout_rate)
        return encoder, encoder.get_num_layers()

    def get_bottleneck(self, starting_filter_size, num_encoder_layers):
        filter_size = (2 ** num_encoder_layers) * starting_filter_size
        return CBAMBottleNeck(filter_size)

    def get_decoder(self, starting_filter_size, dropout_rate):
        return CBAMDecoder(starting_filter_size, dropout_rate)


if __name__ == "__main__":
    model = CbamUnet(64, 2, 0.3, 512, 3)()
    print(model.summary())