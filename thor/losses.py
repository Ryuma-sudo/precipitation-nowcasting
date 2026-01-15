import torch
import torch.nn as nn
import torch.nn.functional as F

from thor.models.horn_schunk_regularization import HornSchunckRegularization
from thor.utils import enumerate_tensor


class AdvectionDiffusionLoss(nn.Module):
    """
    Physics-informed loss enforcing the advection–diffusion PDE constraint:
        ∂R/∂t = -u∂R/∂x - v∂R/∂y + ν(∂²R/∂x² + ∂²R/∂y²)

    Args:
        dt (float): time step between frames.
        nu (float): diffusion coefficient.
        eps (float): small value to prevent division by zero.
    """

    def __init__(self, dt=1.0, nu=0.01, eps=1e-6):
        super().__init__()
        self.dt = float(dt)
        self.nu = float(nu)
        self.eps = float(eps)

        # Finite-difference kernels registered as buffers
        self.register_buffer(
            "dx_kernel", torch.tensor([[[[-0.5, 0.0, 0.5]]]], dtype=torch.float32)
        )  # (1,1,1,3)
        self.register_buffer(
            "dy_kernel", torch.tensor([[[[-0.5], [0.0], [0.5]]]], dtype=torch.float32)
        )  # (1,1,3,1)
        self.register_buffer(
            "dxx_kernel", torch.tensor([[[[1.0, -2.0, 1.0]]]], dtype=torch.float32)
        )
        self.register_buffer(
            "dyy_kernel", torch.tensor([[[[1.0], [-2.0], [1.0]]]], dtype=torch.float32)
        )

    def forward(self, R: torch.Tensor, V: torch.Tensor):
        """
        Args:
            R: (B, T, H, W) - intensity/rainfall field sequence
            V: (T, B, 2, H, W) - velocity field (u,v)
        Returns:
            scalar physics-informed loss
        """
        device = R.device
        dtype = R.dtype
        B, T, H, W = R.shape

        # Handle degenerate sequence
        if T < 2:
            return torch.zeros([], device=device, dtype=dtype)

        # Compute temporal derivative
        R_prev = R[:, :-1]  # (B, T-1, H, W)
        R_curr = R[:, 1:]  # (B, T-1, H, W)
        dR_dt = (R_curr - R_prev) / self.dt

        # Flatten for vectorized spatial derivatives
        BT = B * (T - 1)
        R_curr_flat = R_curr.reshape(BT, 1, H, W)

        # Ensure kernels match precision & device (critical for AMP)
        dx_kernel = self.dx_kernel.to(device=device, dtype=R_curr_flat.dtype)
        dy_kernel = self.dy_kernel.to(device=device, dtype=R_curr_flat.dtype)
        dxx_kernel = self.dxx_kernel.to(device=device, dtype=R_curr_flat.dtype)
        dyy_kernel = self.dyy_kernel.to(device=device, dtype=R_curr_flat.dtype)

        # Compute spatial derivatives
        dR_dx_flat = F.conv2d(R_curr_flat, dx_kernel, padding=(0, 1))
        dR_dy_flat = F.conv2d(R_curr_flat, dy_kernel, padding=(1, 0))
        d2R_dx2_flat = F.conv2d(R_curr_flat, dxx_kernel, padding=(0, 1))
        d2R_dy2_flat = F.conv2d(R_curr_flat, dyy_kernel, padding=(1, 0))

        # Reshape back
        dR_dx = dR_dx_flat.reshape(B, T - 1, H, W)
        dR_dy = dR_dy_flat.reshape(B, T - 1, H, W)
        d2R_dx2 = d2R_dx2_flat.reshape(B, T - 1, H, W)
        d2R_dy2 = d2R_dy2_flat.reshape(B, T - 1, H, W)

        # Velocity components
        V_bt = V[: T - 1].permute(1, 0, 2, 3, 4)  # (B, T-1, 2, H, W)
        u, v = V_bt[..., 0, :, :], V_bt[..., 1, :, :]

        # PDE right-hand side
        advection = -u * dR_dx - v * dR_dy
        diffusion = self.nu * (d2R_dx2 + d2R_dy2)
        RHS = advection + diffusion

        # Residual and mean reduction
        res = (dR_dt - RHS).pow(2)
        res_mean_spatial = res.flatten(2).mean(dim=2).clamp_min(self.eps)

        # Physics-informed importance weighting
        J_t = res_mean_spatial[:, 0].clamp_min(self.eps).clamp_max(1.0)
        p = torch.exp(J_t)
        physics_loss_per_sample = res_mean_spatial.mean(dim=1)

        total_loss = (physics_loss_per_sample / p).mean()
        return total_loss


class WeightedMAE(nn.Module):
    def __init__(self, min_weight=1.0, max_weight=10.0, eps=1e-6):
        super().__init__()
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # Expect pred & target on same device
        # compute per-sample normalization to avoid huge global scaling domination
        # compute max per-sample (B) to keep weights numerically stable
        B = pred.shape[0]
        # reduce over spatial dims but keep batch dim
        target_abs = target.abs()
        target_max = (
            target_abs.reshape(B, -1)
            .max(dim=1)
            .values.clamp_min(self.eps)
            .view(B, *([1] * (target_abs.dim() - 1)))
        )
        normalized_target = target_abs / target_max

        weights = self.min_weight + (self.max_weight - self.min_weight) * normalized_target
        loss = (pred - target).abs() * weights
        # sum over non-batch dims and average by batch
        return loss.reshape(B, -1).sum(dim=1).mean() / (weights.numel() / B + self.eps)


class WeightedMSE(nn.Module):
    def __init__(self, min_weight=1.0, max_weight=10.0, eps=1e-6):
        super().__init__()
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        B = pred.shape[0]
        target_abs = target.abs()
        target_max = (
            target_abs.reshape(B, -1)
            .max(dim=1)
            .values.clamp_min(self.eps)
            .view(B, *([1] * (target_abs.dim() - 1)))
        )
        normalized_target = target_abs / target_max
        weights = self.min_weight + (self.max_weight - self.min_weight) * normalized_target
        loss = (pred - target).pow(2) * weights
        return loss.reshape(B, -1).sum(dim=1).mean() / (weights.numel() / B + self.eps)


adversarial_loss = nn.BCEWithLogitsLoss()
icl_loss = nn.CrossEntropyLoss()
huber_loss = nn.HuberLoss(delta=1.0)
l1_loss = WeightedMAE(min_weight=1, max_weight=10.0)
l2_loss = WeightedMSE(min_weight=1, max_weight=5.0)

advection_diffusion_loss = AdvectionDiffusionLoss(dt=1.0, nu=0.01).to(
    next(iter(l1_loss.parameters()), torch.device("cpu"))
    if isinstance(l1_loss, nn.Module)
    else torch.device("cpu")
)


def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss


def generator_loss(
    R_log1p,
    generated_log1p,
    target_log1p,
    G_logits,
    real_target_label,
    velocity,
    velo_gt,
    lambda_nn=10.0,
):
    """
    Inputs:
      - R_log1p: R in log1p domain, shape (B, T, H, W)  (as in your code)
      - generated_log1p, target_log1p: model outputs and GT in log1p domain (B, 1, H, W) or (B, H, W)
      - G_logits: discriminator logits for fake inputs (what you previously passed as G)
      - real_target_label: label tensor (same shape as G_logits), typically ones
      - velocity: V in shape (T, B, 2, H, W)
      - velo_gt: ground-truth velocity (B, T?, 2, H, W) or shape expected by l1_loss
    Returns:
      final_loss, NN_loss, V_loss, phy_loss
    """
    device = generated_log1p.device

    # --- Adversarial term (discriminator logits -> BCEWithLogitsLoss)
    # G_logits is expected to be discriminator output for generated images
    gen_loss_adv = adversarial_loss(G_logits, real_target_label)

    # --- NN image losses: l1, l2, huber
    # convert to same dtype & device (assume both are already on same device)
    generated = generated_log1p
    target = target_log1p

    # weighted L1/L2 using classes defined earlier
    l1_l = l1_loss(generated, target)
    l2_l = l2_loss(generated, target)
    huber = huber_loss(generated, target) * 8.0

    # --- classification loss (per-pixel) (optimized)
    # enumerate both target and predicted; predict logits from predicted intensity distance to centers
    icl_l = icl_loss(enumerate_tensor(generated), enumerate_tensor(target))
    NN_loss = 0.4 * l1_l + 0.05 * l2_l + 0.15 * icl_l + 0.4 * huber

    # --- velocity regularization & matching
    # horn schunck reg expects V in your same layout
    # ensure velo_gt and velocity are on same device
    velo_gt = velo_gt.to(device)
    velocity = velocity.to(device)
    horn_schunck_regularization = HornSchunckRegularization()

    V_reg = horn_schunck_regularization(velocity, torch.expm1(R_log1p))
    # match velo_gt and velocity with weighted L1 (reuse l1_loss but it expects image-shaped)
    # If velo_gt shape matches velocity shape for l1_loss call; otherwise sum over time dimension
    try:
        l1_on_velocity = l1_loss(velo_gt, velocity)
    except Exception:
        # fallback: compute simple L1 between flattened
        l1_on_velocity = F.l1_loss(velo_gt, velocity)

    V_loss = 0.25 * V_reg + l1_on_velocity

    # --- motion penalty (encourage magnitude)
    # velocity expected shape (T,B,2,H,W) or (B, T, 2, H, W)
    # normalize to (B, T, 2, H, W)
    if velocity.dim() == 5 and velocity.shape[0] != generated.shape[0]:
        # velocity layout (T,B,2,H,W) -> convert to (B,T,2,H,W)
        vel_bt = velocity.permute(1, 0, 2, 3, 4)
        mag = torch.sqrt(vel_bt[..., 0, :, :].pow(2) + vel_bt[..., 1, :, :].pow(2) + 1e-6)
        magnitude = mag.mean(dim=1)  # average over time -> (B, H, W)
        motion_penalty = -magnitude.mean()
    else:
        # assume velocity is (B, T, 2, H, W) or (B, 2, H, W)
        if velocity.dim() == 5:
            mag = torch.sqrt(velocity[..., 0, :, :].pow(2) + velocity[..., 1, :, :].pow(2) + 1e-6)
            magnitude = mag.mean(dim=1)
            motion_penalty = -magnitude.mean()
        elif velocity.dim() == 4:
            mag = torch.sqrt(velocity[:, 0, :, :].pow(2) + velocity[:, 1, :, :].pow(2) + 1e-6)
            motion_penalty = -mag.mean()
        else:
            motion_penalty = torch.tensor(0.0, device=device)

    # --- physics advection-diffusion loss
    phy_loss = advection_diffusion_loss(torch.expm1(R_log1p), velocity)

    final_loss = (V_loss + phy_loss - 0.1 * motion_penalty) + 10.0 * NN_loss + gen_loss_adv

    return final_loss, NN_loss, V_loss, phy_loss
