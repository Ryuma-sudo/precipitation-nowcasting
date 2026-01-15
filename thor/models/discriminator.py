import torch.nn as nn

from thor.models.conv_lstm import ConvLSTM
from thor.utils import get_norm_layer


# -----------------------------
# Discriminator
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = get_norm_layer()

        kw = 4
        padw = 1

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=False,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=False,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
            nn.AdaptiveAvgPool2d((1, 1)),
            # Removed Sigmoid here
        ]

        self.model = nn.Sequential(*sequence)

        # ConvLSTM for temporal modeling
        self.conv_lstm = ConvLSTM(
            input_dim=input_nc, hidden_dim=input_nc, kernel_size=(3, 3), padding=1
        )

    def forward(self, input):
        """
        input: [B, C, H, W] or [B, T, C, H, W]
        output: [B, 1, 1, 1] probability for each batch element (before sigmoid)
        """
        if input.dim() == 5:  # [B, T, C, H, W]
            B, T, C, H, W = input.shape
            h, c = self.conv_lstm.init_hidden(B, (H, W))
            for t in range(T):
                h, (h, c) = self.conv_lstm(input[:, t], (h, c))
            x = h  # last timestep
        else:  # [B, C, H, W]
            x = input

        return self.model(x)
