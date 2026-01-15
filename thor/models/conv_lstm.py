import torch
import torch.nn as nn


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=(3, 3), padding=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, hidden):
        # x: [B, C, H, W]
        # hidden: (h, c), both [B, hidden_dim, H, W]
        h, c = hidden
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)
        (cc_i, cc_f, cc_o, cc_g) = torch.split(conv_out, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)

    def init_hidden(self, batch_size, spatial_size):
        H, W = spatial_size
        device = self.conv.weight.device
        h = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        return h, c
