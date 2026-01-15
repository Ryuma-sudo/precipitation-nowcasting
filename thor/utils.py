import functools

import torch
import torch.nn as nn


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(model, "weight"):
            torch.nn.init.normal_(model.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(model.weight, 1.0, 0.02)
        torch.nn.init.zeros_(model.bias)


def get_norm_layer(norm_type="batch"):
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    else:
        raise NotImplementedError(f"Normalization {norm_type} not implemented")


def enumerate_tensor(img_1):
    # Reverse the log1p transformation
    img_1 = torch.expm1(img_1)
    # Apply classification rules
    enumerate_matrix = torch.where(
        img_1 == 0,
        torch.tensor(0, dtype=torch.float32),
        torch.where(
            (img_1 > 0) & (img_1 <= 1),
            torch.tensor(1, dtype=torch.float32),
            torch.where(
                (img_1 > 1) & (img_1 <= 4),
                torch.tensor(2, dtype=torch.float32),
                torch.where(
                    (img_1 > 4) & (img_1 <= 8),
                    torch.tensor(3, dtype=torch.float32),
                    torch.tensor(4, dtype=torch.float32),  # for img_1 > 8
                ),
            ),
        ),
    )
    return enumerate_matrix
