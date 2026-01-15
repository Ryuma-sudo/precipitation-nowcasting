import torch
import torch.nn as nn
import torch.nn.functional as F


class HornSchunckRegularization(nn.Module):
    def __init__(self):
        super().__init__()

    def spatial_gradient(self, tensor):
        # tensor: (B, 1, H, W) or (B, H, W)
        # compute forward differences and replicate-pad to original size
        if tensor.dim() == 4 and tensor.shape[1] == 1:
            x = tensor[:, 0]
        else:
            x = tensor  # (B,H,W)
        # grad along width (x)
        grad_x = x[..., :, 1:] - x[..., :, :-1]  # (B,H,W-1)
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode="replicate")  # back to (B,H,W)
        # grad along height (y)
        grad_y = x[..., 1:, :] - x[..., :-1, :]  # (B,H-1,W)
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode="replicate")
        # return with shape (B,1,H,W) for consistency
        return grad_x.unsqueeze(1), grad_y.unsqueeze(1)

    def forward(self, V: torch.Tensor, R: torch.Tensor):
        """
        V: (T, B, 2, H, W) as in your code (we keep semantics)
        R: (B, T, H, W)
        """
        device = V.device
        # take first timestep velocity (mimics your original)
        V_t = V[0]  # (B, 2, H, W)
        u_t = V_t[:, 0:1, :, :]  # (B,1,H,W)
        v_t = V_t[:, 1:2, :, :]  # (B,1,H,W)

        # compute R_current like original: use R[:, 1:, :, :]
        R_current = R[:, 1:, :, :]  # (B, T-1, H, W)
        # For weight w(R_t) = min(24, R_t + 1)
        # expand to match spatial grads: use last dimension collapse across time by mean (approx) to produce shape (B,1,H,W)
        # but to keep compatibility we take average across time dimension to create spatial weight
        w_R = torch.minimum(R_current + 1.0, torch.tensor(24.0, device=device))
        w_R = torch.clamp(w_R, min=0.0)
        # average over time axis -> (B,H,W)
        w_mean = w_R.mean(dim=1)  # (B, H, W)
        sqrt_w = torch.sqrt(w_mean + 1e-6).unsqueeze(1)  # (B,1,H,W)

        # spatial grads
        grad_u_x, grad_u_y = self.spatial_gradient(u_t)
        grad_v_x, grad_v_y = self.spatial_gradient(v_t)

        loss_u = (grad_u_x * sqrt_w).pow(2).mean()
        loss_v = (grad_v_x * sqrt_w).pow(2).mean()
        loss_u += (grad_u_y * sqrt_w).pow(2).mean()
        loss_v += (grad_v_y * sqrt_w).pow(2).mean()

        loss = loss_u + loss_v
        return torch.clamp(loss, min=1e-6, max=100.0 - 1e-6)
