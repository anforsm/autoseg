import torch
import torch.nn as nn


def UNetResAdd(a, b):
    """Center crop a so that it can be added with b
    a: [b, c, z, y, x]
    b: [b, c, z, y, x]
    """
    a_shape = a.shape
    b_shape = b.shape

    # Calculate offsets for center cropping
    z_diff = a_shape[2] - b_shape[2]
    y_diff = a_shape[3] - b_shape[3]
    x_diff = a_shape[4] - b_shape[4]

    z_start = z_diff // 2
    y_start = y_diff // 2
    x_start = x_diff // 2

    # Perform center cropping
    a_cropped = a[
        :,
        :,
        z_start : z_start + b_shape[2],
        y_start : y_start + b_shape[3],
        x_start : x_start + b_shape[4],
    ]

    # Concatenate along the channel dimension
    return a_cropped + b


def UNetResConcat(a, b):
    """Center crop a so that it can be concatenated with b
    a: [b, c, z, y, x]
    b: [b, c, z, y, x]
    """
    a_shape = a.shape
    b_shape = b.shape

    # Calculate offsets for center cropping
    z_diff = a_shape[2] - b_shape[2]
    y_diff = a_shape[3] - b_shape[3]
    x_diff = a_shape[4] - b_shape[4]

    z_start = z_diff // 2
    y_start = y_diff // 2
    x_start = x_diff // 2

    # Perform center cropping
    a_cropped = a[
        :,
        :,
        z_start : z_start + b_shape[2],
        y_start : y_start + b_shape[3],
        x_start : x_start + b_shape[4],
    ]

    # Concatenate along the channel dimension
    return torch.cat([a_cropped, b], dim=1)


class ConvPassBase(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3)
        self.act1 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        return x


class DownsampleBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.MaxPool3d((1, 3, 3))

    def forward(self, x):
        return self.down(x)


class UpsampleBase(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 3, 3)
        )

    def forward(self, x):
        return self.up(x)


class UNeXt(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_fmaps=12,
        ConvPass=ConvPassBase,
        Downsample=DownsampleBase,
        Upsample=UpsampleBase,
        fmap_inc_factor=5,
        debug=False,
    ):
        super().__init__()
        self.debug = debug
        print(self.debug)

        self.enc1 = ConvPass(in_channels, num_fmaps)
        self.down1 = Downsample()

        self.enc2 = ConvPass(num_fmaps, num_fmaps * fmap_inc_factor**1)
        self.down2 = Downsample()

        self.enc3 = ConvPass(
            num_fmaps * fmap_inc_factor**1, num_fmaps * fmap_inc_factor**2
        )
        self.down3 = Downsample()

        self.enc4 = ConvPass(
            num_fmaps * fmap_inc_factor**2, num_fmaps * fmap_inc_factor**3
        )

        self.up1 = Upsample(
            num_fmaps * fmap_inc_factor**3, num_fmaps * fmap_inc_factor**2
        )
        self.uenc1 = ConvPass(
            2 * num_fmaps * fmap_inc_factor**2, num_fmaps * fmap_inc_factor**2
        )

        self.up2 = Upsample(
            num_fmaps * fmap_inc_factor**2, num_fmaps * fmap_inc_factor**1
        )
        self.uenc2 = ConvPass(
            2 * num_fmaps * fmap_inc_factor**1, num_fmaps * fmap_inc_factor**1
        )

        self.up3 = Upsample(num_fmaps * fmap_inc_factor**1, num_fmaps)
        self.uenc3 = ConvPass(2 * num_fmaps, num_fmaps)

    def forward(self, x):
        res1 = self.enc1(x)
        if self.debug:
            print("res1", res1.shape)
        x = self.down1(res1)
        if self.debug:
            print("x1", x.shape)

        res2 = self.enc2(x)
        if self.debug:
            print("res2", res2.shape)
        x = self.down2(res2)
        if self.debug:
            print("x2", x.shape)

        res3 = self.enc3(x)
        if self.debug:
            print("res3", res3.shape)
        x = self.down3(res3)
        if self.debug:
            print("x2", x.shape)

        x = self.enc4(x)
        if self.debug:
            print("x3", x.shape)

        x = self.up1(x)
        x = UNetResConcat(res3, x)
        x = self.uenc1(x)

        x = self.up2(x)
        x = UNetResConcat(res2, x)
        x = self.uenc2(x)

        x = self.up3(x)
        x = UNetResConcat(res1, x)
        x = self.uenc3(x)

        return x
