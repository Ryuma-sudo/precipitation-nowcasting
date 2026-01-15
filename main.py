import dataclasses
import os

import numpy as np
import torch
import torch.nn.functional as F
import gc
from torch.utils.data import DataLoader, Subset

import wandb
from config import cfg
from thor.data import PrecipitationNowcastingDataset
from thor.models.discriminator import Discriminator
from thor.models.generator import Generator
from thor.trainer import Trainer
from thor.utils import weights_init
from dotenv import load_dotenv
load_dotenv()

torch.backends.cudnn.benchmark = True


def init_wandb() -> None:
    wandb.login(key="ae8ab4414cf9c9052ba0b950a538c17ff254d492")
    wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project=os.getenv("WANDB_PROJECT"),
        name=os.getenv("WANDB_RUN_NAME"),
        config=dataclasses.asdict(cfg),
    )


def main():
    init_wandb()

    data_numpy = np.load(cfg.data_path)
    data = torch.from_numpy(data_numpy).float()
    if data.ndim == 3:
        data = data.unsqueeze(1)
    data = F.interpolate(data, size=(256, 256), mode='bilinear', align_corners=False)
    precipitation_data = data.squeeze(1)

    del data
    gc.collect()

    dataset = PrecipitationNowcastingDataset(precipitation_data, cfg.T_in, cfg.T_out)
    idxs = np.arange(len(dataset))

    np.random.seed(cfg.seed)  # for reproducibility
    np.random.shuffle(idxs)

    train_idxs = idxs[: int(cfg.split_ratio * len(idxs))]
    test_idxs = idxs[int(cfg.split_ratio * len(idxs)) :]

    train_dataset = Subset(dataset, train_idxs)
    test_dataset = Subset(dataset, test_idxs)

    train_loader = DataLoader(
        train_dataset,
        cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,  # keeps workers alive between epochs
        prefetch_factor=4,  # how many batches each worker prefetches
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset, cfg.batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    generator = Generator(cfg.T_in, cfg.T_out, cfg.height, cfg.width).to(cfg.device)
    discriminator = (
        Discriminator(input_nc=2 + cfg.T_out, ndf=64, n_layers=3).apply(weights_init).to(cfg.device)
    )

    trainer = Trainer(generator, discriminator, cfg)
    trainer.train(train_loader, test_loader)


if __name__ == "__main__":
    main()
