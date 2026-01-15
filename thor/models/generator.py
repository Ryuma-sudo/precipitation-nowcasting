import torch
import torch.nn as nn
from axial_attention import AxialAttention, AxialPositionalEmbedding

from thor.models.conv_lstm_2d import ConvLSTM2D
from thor.models.flow_estimator import FastPrecipitationFlowEstimator
# from thor.models.flow_estimator import PrecipitationFlowEstimator

from thor.models.motion import MotionFieldNN
from thor.models.traj_lstm import TrajLSTM
from thor.velocity import simulate_velocity


# ===============================
# Main Generator (ODE-Free)
# ===============================
class Generator(nn.Module):
    def __init__(self, T_in=6, T_out=12, H=128, W=128):
        super().__init__()
        self.T_in = T_in
        self.T_out = T_out
        self.H = H
        self.W = W

        self.module1 = MotionFieldNN(in_channels=T_in)  # standard NN version
        self.motion_guider = FastPrecipitationFlowEstimator()  # PrecipitationFlowEstimator()

        self.conv3d = nn.Sequential(
            nn.Conv3d(6, 6, (3, 3, 3), padding=1),
            nn.BatchNorm3d(6),
            nn.ReLU(inplace=True),
        )

        self.convlstm = ConvLSTM2D(input_dim=6, hidden_dim=12, kernel_size=(3, 3), padding=1)

        # sequential LSTM for temporal evolution (replaces ODE)
        # TrajLSTM expects: forward(x_t, h_prev, c_prev) -> (h_t, c_t, d_full)
        self.traj_lstm = TrajLSTM(input_dim=12, hidden_dim=12, H=H, W=W)

        # Additional refinement ConvLSTM
        self.convlstm2 = ConvLSTM2D(input_dim=12, hidden_dim=1, kernel_size=(3, 3), padding=1)

        self.conv3d_reduce = nn.Conv3d(in_channels=12, out_channels=1, kernel_size=1)

        # Simplified attention
        self.axial_pos_emb = AxialPositionalEmbedding(dim=T_in + T_out, shape=(H, W))
        self.axial_attn = AxialAttention(dim=T_in + T_out, heads=T_in + T_out, dim_index=1)
        self.layer_norm = nn.LayerNorm([T_in + T_out, H, W])

        self.out = nn.Conv2d(
            in_channels=T_out, out_channels=T_out, kernel_size=3, stride=1, padding=1
        )
        self.relu = nn.ReLU()

    def forward(self, input_tensor):
        B = input_tensor.size(0)

        # motion estimation
        V_t = self.module1(input_tensor)
        V_seq = simulate_velocity(V_t)
        V_gt = simulate_velocity(self.motion_guider(input_tensor))

        # Conv3D encoding
        temp = input_tensor.unsqueeze(2)
        conv3d_out = self.conv3d(temp).squeeze(2)

        # ConvLSTM temporal encoding
        h, c = self.convlstm.init_hidden(B, (self.H, self.W))
        lstm_out, (h, c) = self.convlstm(conv3d_out, (h, c))

        # sequential trajectory evolution (replaces ODE integration)
        h_trajectory = []
        h_t, c_t = lstm_out, c

        for t in range(self.T_out):
            # TrajLSTM.forward expects: (x_t, h_prev, c_prev)
            # We use h_t as the input (feeding hidden state back as input)
            h_t, c_t, flow = self.traj_lstm(h_t, h_t, c_t)
            h_trajectory.append(h_t)

        # stack trajectory: [B, C, T, H, W]
        h_3d = torch.stack(h_trajectory, dim=2)

        # reduce channels
        h_reduced = self.conv3d_reduce(h_3d)
        ode_out = h_reduced.permute(0, 2, 1, 3, 4).squeeze(2)

        # axial attention refinement
        x_concat = torch.cat([input_tensor, ode_out], dim=1)
        x = self.axial_pos_emb(x_concat)
        x = self.axial_attn(x)
        x = self.layer_norm(x)
        out = x[:, self.T_in :, :, :]
        x = self.out(out)
        x = self.relu(x)

        return x, V_seq, V_gt
