import torch
import torch.nn as nn


class ConvLSTM2D(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=(3, 3), padding=1, bias=True):
        super(ConvLSTM2D, self).__init__()
        self.hidden_dim = hidden_dim
        self.padding = padding

        # reduce the number of filters in conv layer
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(self, input_tensor, current_state):
        h_current, c_current = current_state
        combined = torch.cat((input_tensor, h_current), dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_current + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, (h_next, c_next)

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        device = self.conv.weight.device
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
        )
