import torch
import torch.nn.functional as F

# ================================
#  Stable Velocity Simulation (Functional)
# ================================


def spatial_gradient(tensor, boundary="replicate"):
    """Compute spatial gradients along x and y."""
    # Assuming tensor shape is [B, C, H, W]
    grad_x = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
    grad_y = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]

    # Pad to original size
    grad_x = F.pad(grad_x, (0, 1, 0, 0), mode=boundary)
    grad_y = F.pad(grad_y, (0, 0, 0, 1), mode=boundary)
    return grad_x, grad_y


def laplacian(tensor, boundary="replicate"):
    """Compute Laplacian using convolution kernel."""
    kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=tensor.dtype, device=tensor.device
    )
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape [1,1,3,3]
    B, C, H, W = tensor.shape
    # Ensure padding is correct for 3x3 kernel
    lap = F.conv2d(tensor.view(B * C, 1, H, W), kernel, padding=1)
    return lap.view(B, C, H, W)


def advect(V, boundary="replicate"):
    """Advection term: (V · ∇)V"""
    # V shape is [B, 2, H, W]
    u, v = V[:, 0:1], V[:, 1:2]
    grad_u_x, grad_u_y = spatial_gradient(u, boundary)
    grad_v_x, grad_v_y = spatial_gradient(v, boundary)
    adv_u = u * grad_u_x + v * grad_u_y
    adv_v = u * grad_v_x + v * grad_v_y
    return torch.cat([adv_u, adv_v], dim=1)


def update_velocity(V_t, dt=0.1, mu=0.01, boundary="replicate"):
    """One step of advection-diffusion velocity update."""
    # V_t shape is [B, 2, H, W]
    adv = advect(V_t, boundary)
    diff = laplacian(V_t, boundary)
    return V_t + dt * (-adv + mu * diff)


def simulate_velocity(V_t, dt=0.1, mu=0.01, steps=12, boundary="replicate"):
    """
    Simulate velocity field over multiple timesteps.
    Args:
        V_t: tensor [B, 2, H, W]
        dt: time step
        mu: diffusion coefficient
        steps: number of iterations
        boundary: 'replicate' or 'circular'
    """
    hard_clamp = torch.nn.Hardtanh(min_val=-1.0, max_val=1.0)
    velocities = [V_t]
    V = V_t.clone()

    for _ in range(steps):
        V = update_velocity(V, dt, mu, boundary)
        V = hard_clamp(V)
        velocities.append(V)

    return torch.stack(velocities, dim=0)
