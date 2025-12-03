import torch
import torch.nn as nn


# ============================================================
# Discriminator
# ============================================================
class Discriminator(nn.Module):
    """
    Standard DCGAN discriminator.

    The network progressively downsamples a 64×64 RGB image into a single
    scalar probability indicating whether the input is real or generated.

    channels_img: number of channels in the input image (e.g. 3 for RGB)
    features_d:   base width multiplier for feature maps (e.g. 64)
    """

    def __init__(self, channels_img: int, features_d: int):
        super().__init__()

        # Input: (N, channels_img, 64, 64)
        self.disc = nn.Sequential(
            # Initial convolution – no BatchNorm per DCGAN paper
            nn.Conv2d(
                in_channels=channels_img,
                out_channels=features_d,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # Downsample: 32×32 → 16×16
            self._block(features_d, features_d * 2, 4, 2, 1),

            # 16×16 → 8×8
            self._block(features_d * 2, features_d * 4, 4, 2, 1),

            # 8×8 → 4×4
            self._block(features_d * 4, features_d * 8, 4, 2, 1),

            # Final conv: 4×4 → 1×1 (prediction)
            nn.Conv2d(
                in_channels=features_d * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0
            ),
            nn.Sigmoid()  # Output: (N, 1, 1, 1)
        )

    @staticmethod
    def _block(in_c, out_c, kernel_size, stride, padding):
        """
        A DCGAN convolutional downsampling block:
        Conv → BatchNorm → LeakyReLU
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.disc(x)


# ============================================================
# Generator
# ============================================================
class Generator(nn.Module):
    """
    Standard DCGAN generator.

    Starts from a latent vector z of shape (N, z_dim, 1, 1)
    and progressively upsamples it to a 64×64 image with pixel
    values in [-1, 1].

    z_dim:        dimension of latent noise vector (e.g. 100)
    channels_img: number of channels in output (e.g. 3 for RGB)
    features_g:   base width multiplier for feature maps
    """

    def __init__(self, z_dim: int, channels_img: int, features_g: int):
        super().__init__()

        # Input: (N, z_dim, 1, 1)
        self.gen = nn.Sequential(
            # 1×1 → 4×4
            self._block(z_dim, features_g * 16, kernel_size=4, stride=1, padding=0),

            # 4×4 → 8×8
            self._block(features_g * 16, features_g * 8, 4, 2, 1),

            # 8×8 → 16×16
            self._block(features_g * 8, features_g * 4, 4, 2, 1),

            # 16×16 → 32×32
            self._block(features_g * 4, features_g * 2, 4, 2, 1),

            # 32×32 → 64×64 (final RGB image)
            nn.ConvTranspose2d(
                in_channels=features_g * 2,
                out_channels=channels_img,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()  # Output: (N, channels_img, 64, 64)
        )

    @staticmethod
    def _block(in_c, out_c, kernel_size, stride, padding):
        """
        A DCGAN upsampling block:
        ConvTranspose → BatchNorm → ReLU
        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.gen(x)


# ============================================================
# Weight Initialization (DCGAN style)
# ============================================================
def initialize_weights(model: nn.Module):
    """
    Initialize model weights according to DCGAN recommendations:
    - Conv / ConvTranspose: normal mean=0, std=0.02
    - BatchNorm:           weight mean=1, std=0.02; bias=0
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)


# ============================================================
# Quick Architectural Shape Test
# ============================================================
def test():
    N, C, H, W = 8, 3, 64, 64
    z_dim = 100
    features_d = 64
    features_g = 64

    # ------------------ Discriminator ------------------
    x = torch.randn(N, C, H, W)
    disc = Discriminator(channels_img=C, features_d=features_d)
    initialize_weights(disc)

    out_d = disc(x)
    print("Discriminator output shape:", out_d.shape)
    assert out_d.shape == (N, 1, 1, 1)

    # ------------------ Generator ------------------
    z = torch.randn(N, z_dim, 1, 1)
    gen = Generator(z_dim=z_dim, channels_img=C, features_g=features_g)
    initialize_weights(gen)

    out_g = gen(z)
    print("Generator output shape:", out_g.shape)
    assert out_g.shape == (N, C, H, W)

    print("Architecture OK ✔")


if __name__ == "__main__":
    test()
