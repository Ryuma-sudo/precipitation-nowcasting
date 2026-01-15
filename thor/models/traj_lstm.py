import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Safe GroupNorm
# ---------------------------
def safe_gn(num_channels, max_groups=8):
    for g in reversed(range(1, max_groups + 1)):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


# ---------------------------
# Memory-efficient warp
# ---------------------------
def warp(feat, flow):
    B, C, H, W = feat.shape
    device = feat.device
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    flow_x = flow[:, 0] / ((W - 1.0) / 2.0)
    flow_y = flow[:, 1] / ((H - 1.0) / 2.0)
    flow_norm = torch.stack((flow_x, flow_y), dim=-1)

    sampling_grid = (base_grid + flow_norm).clamp(-1.0, 1.0)
    return F.grid_sample(
        feat, sampling_grid, mode="bilinear", padding_mode="border", align_corners=True
    )


# ---------------------------
# Memory-efficient TrajLSTM
# ---------------------------
class TrajLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, H, W, n_scales=1, refiner=False, use_separable=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.H, self.W = H, W
        self.n_scales = n_scales
        self.refiner = refiner
        self.use_separable = use_separable

        mid = max(16, hidden_dim // 4)
        self.feat_x = nn.Sequential(
            nn.Conv2d(input_dim, mid, 3, padding=1, bias=False),
            safe_gn(mid),
            nn.ReLU(inplace=True),
        )
        self.feat_h = nn.Sequential(
            nn.Conv2d(hidden_dim, mid, 3, padding=1, bias=False),
            safe_gn(mid),
            nn.ReLU(inplace=True),
        )

        if use_separable:
            self.flow_head = nn.Sequential(
                nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=True),
                nn.Conv2d(mid, 2, 1, bias=True),
                nn.ReLU(inplace=True),
            )
        else:
            self.flow_head = nn.Conv2d(mid, 2, 3, padding=1)

        if refiner:
            self.refine_net = nn.Sequential(
                nn.Conv2d(mid + 2, mid, 3, padding=1, bias=False),
                safe_gn(mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, 2, 3, padding=1),
            )

        fusion_in_ch = hidden_dim
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_ch, hidden_dim, 1, bias=False),
            safe_gn(hidden_dim),
            nn.ReLU(inplace=True),
        )

        gates_ch = hidden_dim * 4
        if use_separable:
            self.conv_x = nn.Sequential(
                nn.Conv2d(input_dim, gates_ch, 3, padding=1, groups=input_dim, bias=False),
                safe_gn(gates_ch),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(hidden_dim, gates_ch, 3, padding=1, groups=hidden_dim, bias=False),
                safe_gn(gates_ch),
            )
        else:
            self.conv_x = nn.Conv2d(input_dim, gates_ch, 3, padding=1, bias=False)
            self.conv_h = nn.Conv2d(hidden_dim, gates_ch, 3, padding=1, bias=False)

    def _downsample_flow_and_feature(self, x, scale):
        if scale == 0:
            return x
        factor = 2**scale
        return F.adaptive_avg_pool2d(x, (max(1, self.H // factor), max(1, self.W // factor)))

    def forward(self, x_t, h_prev, c_prev):
        fx = self.feat_x(x_t)
        fh = self.feat_h(h_prev)
        shared = fx + fh

        d_init = self.flow_head(shared)
        if self.refiner:
            refine_in = torch.cat([d_init, shared], dim=1)
            d_full = d_init + self.refine_net(refine_in)
            del refine_in
        else:
            d_full = d_init
        del fx, fh, shared, d_init

        weighted_h = torch.zeros_like(h_prev)
        for s in range(self.n_scales):
            h_s = self._downsample_flow_and_feature(h_prev, s) if s else h_prev
            flow_s = self._downsample_flow_and_feature(d_full, s)
            warped_h_s = warp(h_s, flow_s)
            if s != 0:
                warped_h_s = F.interpolate(
                    warped_h_s,
                    size=(self.H, self.W),
                    mode="bilinear",
                    align_corners=True,
                )
            weighted_h += warped_h_s / self.n_scales
            del h_s, flow_s, warped_h_s

        fused_h = self.fusion(weighted_h)

        gx = self.conv_x(x_t)
        gh = self.conv_h(fused_h)
        gates = gx + gh
        i, g, f, o = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)

        return h_t, c_t, d_full
