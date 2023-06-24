import torch
import torch.nn as nn

from torch_library.models.conv import Conv, ConvTranspose


class UnetGenerator(nn.Module):
    def __init__(self, input_n_ch: int = 3, output_n_ch: int = 3):
        super(UnetGenerator, self).__init__()

        # Initial convolution block
        self.model_down = nn.ModuleList()

        # Downsampling
        down_features = [(input_n_ch, 64), (64, 128), (128, 256), (256, 512), (512, 512), (512, 512), (512, 512)]
        for in_features, out_features in down_features:
            self.model_down.append(Conv(in_features, out_features))
        self.model_down.append(nn.Sequential(nn.Conv2d(512, 512, 4, stride=2, padding=1),
                                             nn.ReLU(inplace=True)))

        # Upsampling
        self.model_up = nn.ModuleList()
        up_features = [(512, 512), (1024, 512), (1024, 512), (1024, 512), (1024, 256), (512, 128), (256, 64)]
        for in_features, out_features in up_features:
            self.model_up.append(ConvTranspose(in_features, out_features))

        # Output layer
        self.last = (nn.Sequential(nn.ConvTranspose2d(128, output_n_ch, kernel_size=4, padding=1, stride=2),
                                   nn.Tanh()))

    def forward(self, x):
        skip_connections = []
        for idx, down in enumerate(self.model_down):
            x = down(x)
            skip_connections.append(x)
        # TODO tu skonczylem, dopasuj up do down, zeby wymiar sie zgadzal. Pewnie bedzie concat 512x256 i to trzeba na gorze napisac
        skips = reversed(skip_connections[:-1])
        for idx, (up, skip) in enumerate(zip(self.model_up, skips)):
            x = up(x)
            x = torch.concat([x, skip], axis=1)
        x = self.last(x)

        return x
