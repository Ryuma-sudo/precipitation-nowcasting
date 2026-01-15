import numpy as np
import torch
from torch.utils.data import Dataset


class PrecipitationNowcastingDataset(Dataset):
    def __init__(self, images, T_in, T_out):
        self.T_in = T_in
        self.T_out = T_out

        # check if images is already an array or convert it to one
        if isinstance(images, np.ndarray):
            self.frames = images
        elif isinstance(images, torch.Tensor):
            self.frames = images.numpy()  # Convert torch tensor to numpy
        else:
            self.frames = np.array(images)

        # verify frames is an array, not a scalar
        if self.frames.ndim == 0:  # It's a scalar
            raise ValueError("Input 'images' must be a sequence of frames, not a scalar value")

        # clip values to the range [0, 76]
        self.frames = np.clip(self.frames, 0, 76)
        self.frames = np.log1p(self.frames)

    def __len__(self):
        return max(0, len(self.frames) - (self.T_in + self.T_out) + 1)

    def __getitem__(self, idx):
        input_frames = torch.tensor(self.frames[idx : idx + self.T_in], dtype=torch.float32)
        target_frames = torch.tensor(
            self.frames[idx + self.T_in : idx + self.T_in + self.T_out],
            dtype=torch.float32,
        )
        return input_frames, target_frames
