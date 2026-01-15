from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TrainConfig:
    T_in: int = 6
    T_out: int = 12
    height: int = 256
    width: int = 256
    batch_size: int = 4
    split_ratio: float = 0.8
    num_epochs: int = 30
    learning_rate: float = 0.0110207
    max_grad_norm = 1.0
    data_path: Path = Path("data/preprocessed-data/cmax_256x256.npy")
    best_model_path: Path = Path("saved-models/best_gen_model.pth")
    final_model_path: Path = Path("saved-models/final_gen_model.pth")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42


cfg = TrainConfig()
