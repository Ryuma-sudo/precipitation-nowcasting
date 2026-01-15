# =========================================================
# Motion Field Estimator - Standard Neural Network
# =========================================================
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# Motion Dynamics Block (Replaces ODE)
# =========================================================
class MotionDynamicsBlock(nn.Module):
    """Iterative refinement block for motion features"""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.GroupNorm(8, channels // 2),
            nn.SiLU(),
            nn.Conv2d(channels // 2, channels, 3, padding=1),
        )

    def forward(self, z):
        # Residual connection for stable refinement
        return z + self.net(z)


# =========================================================
# Motion Field Estimator (Standard NN Architecture)
# =========================================================
class MotionFieldNN(nn.Module):
    def __init__(self, in_channels=6, latent_channels=32, num_refinement_steps=4):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, stride=2),  # 128x128
            nn.ReLU(inplace=True),
            nn.Conv2d(32, latent_channels, 3, padding=1, stride=2),  # 64x64
            nn.ReLU(inplace=True),
        )

        # Refinement blocks (replaces ODE integration)
        self.refinement_blocks = nn.ModuleList(
            [MotionDynamicsBlock(latent_channels) for _ in range(num_refinement_steps)]
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 32, 4, stride=2, padding=1),  # 128x128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 2, 4, stride=2, padding=1),  # 256x256
        )

    def forward(self, x):
        # Encode to latent space
        z = self.encoder(x)  # [B, latent, 64, 64]

        # Iterative refinement (replaces ODE solving)
        for block in self.refinement_blocks:
            z = block(z)

        # Decode to motion field
        out = self.decoder(z)
        return out
