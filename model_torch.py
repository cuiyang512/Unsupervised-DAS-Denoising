import math
import torch
import torch.nn as nn


class ECABlock(nn.Module):
    """PyTorch version
    """

    def __init__(self, channels: int, b: int = 1, gama: int = 8):
        super().__init__()
        in_channel = 1
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(kernel_size, 1)

        self.conv1 = nn.Conv1d(channels, 1, kernel_size=kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(1, 1, kernel_size=kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L], and L=1 in the original notebook.
        pooled = x.mean(dim=2, keepdim=True)  # [B, C, 1]
        gate = self.sigmoid(self.conv2(self.relu(self.conv1(pooled))))  # [B,1,1]
        return x * gate


class ConvBranchBlock(nn.Module):
    """Two parallel Conv1D branches + channel mixing used repeatedly."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(out_channels),
        )

        k2 = kernel_size + 3  # Keras padding='same' also supports even kernels.
        # Explicit asymmetric padding reproduces TensorFlow SAME for stride=1.
        total_pad = k2 - 1
        left = total_pad // 2
        right = total_pad - left
        self.alt_pad = nn.ConstantPad1d((left, right), 0.0)
        self.alt_conv = nn.Conv1d(in_channels, out_channels, k2, padding=0)
        self.alt_prelu = nn.PReLU(out_channels)

        # Keras Dense(out_channels) acts on the final/channel dimension.
        # In channels-first PyTorch this is equivalent to a 1x1 Conv1D.
        self.mix = nn.Conv1d(2 * out_channels, out_channels, kernel_size=1)
        self.mix_prelu = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.main(x)
        b = self.alt_prelu(self.alt_conv(self.alt_pad(x)))
        x = torch.cat([b, a], dim=1)
        x = self.mix_prelu(self.mix(x))
        return x


class DIPAutoencoder1D(nn.Module):

    def __init__(self, input_size: int, d1: int = 128,
                 d2: int = 32, d3: int = 8, kernel_size: int = 3):
        super().__init__()

        self.enc1 = ConvBranchBlock(input_size, d1, kernel_size)
        self.att1 = ECABlock(d1)

        self.enc2 = ConvBranchBlock(d1, d2, kernel_size)
        self.att2 = ECABlock(d2)

        self.enc3 = ConvBranchBlock(d2, d3, kernel_size)
        self.att3 = ECABlock(d3)

        self.dec3 = ConvBranchBlock(d3, d3, kernel_size)
        self.dec2 = ConvBranchBlock(2 * d3, d2, kernel_size)
        self.dec1 = ConvBranchBlock(2 * d2, d1, kernel_size)

        self.out1 = nn.Conv1d(2 * d1, input_size // 3, kernel_size=1)
        self.out2 = nn.Conv1d(input_size // 3, input_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected [batch, input_size], got {tuple(x.shape)}")

        # Keras: input_img[:, tf.newaxis] => [B, 1, input_size].
        # PyTorch Conv1d uses [B, C, L], hence [B, input_size, 1].
        x = x.unsqueeze(-1)

        e1 = self.enc1(x)
        s1 = self.att1(e1)

        e2 = self.enc2(e1)
        s2 = self.att2(e2)

        e3 = self.enc3(e2)
        s3 = self.att3(e3)

        d3 = self.dec3(e3)
        d3 = torch.cat([d3, s3], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, s2], dim=1)

        d1 = self.dec1(d2)
        d1 = torch.cat([d1, s1], dim=1)

        y = self.out2(self.out1(d1))
        return y.flatten(start_dim=1)



class InceptionRes(nn.Module):
    def __init__(self, channels: int = 1):
        super().__init__()
        self.e1 = nn.Sequential(
            nn.Conv1d(channels, 1, 1),
            nn.Conv1d(1, 1, 3, padding=1), nn.Tanh(),
            nn.Conv1d(1, 1, 3, padding=1), nn.Tanh(),
            nn.BatchNorm1d(1),
        )
        self.e2 = nn.Sequential(
            nn.Conv1d(channels, 1, 1),
            nn.Conv1d(1, 1, 3, padding=1), nn.Tanh(),
            nn.BatchNorm1d(1),
        )
        self.e3 = nn.Sequential(nn.Conv1d(channels, 1, 1), nn.BatchNorm1d(1))
        self.merge = nn.Conv1d(3, 1, 1)

    def forward(self, x):
        return self.merge(torch.cat([self.e1(x), self.e2(x), self.e3(x)], dim=1)) + x


class MFFBlock(nn.Module):
    def __init__(self, in_channels: int, filters: int, kernel_size: int, stride: int):
        super().__init__()
        self.e1 = nn.Sequential(
            nn.Conv1d(in_channels, filters, kernel_size, stride=stride,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(filters), nn.PReLU(filters),
        )
        k2 = kernel_size + 3
        self.e2 = nn.Sequential(
            nn.Conv1d(in_channels, filters, k2, stride=stride + 2,
                      padding=k2 // 2),
            nn.BatchNorm1d(filters), nn.PReLU(filters),
        )
        self.out = nn.Sequential(nn.Conv1d(2 * filters, 1, 1), nn.Tanh())

    def forward(self, x):
        e1, e2 = self.e1(x), self.e2(x)
        if e1.shape[-1] != e2.shape[-1]:
            raise ValueError("mff_block branches have different sequence lengths for these strides")
        y = self.out(torch.cat([e2, e1], dim=1))
        if y.shape != x.shape:
            raise ValueError("mff_block residual add requires output and input shapes to match")
        return y + x
