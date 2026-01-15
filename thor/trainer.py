import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.regression import CriticalSuccessIndex
from tqdm import tqdm

import wandb
from thor.losses import discriminator_loss, generator_loss, huber_loss


class Trainer:
    def __init__(self, generator, discriminator, cfg):
        # parameters
        self.device = cfg.device
        self.learning_rate = cfg.learning_rate
        self.batch_size = cfg.batch_size
        self.T_in = cfg.T_in
        self.T_out = cfg.T_out
        self.max_grad_norm = cfg.max_grad_norm
        self.num_epochs = cfg.num_epochs
        self.best_model_path = cfg.best_model_path
        self.final_model_path = cfg.final_model_path

        self.generator = generator
        self.discriminator = discriminator

        self.G_optimizer = optim.AdamW(
            generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999)
        )
        self.D_optimizer = optim.AdamW(
            discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999)
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.G_optimizer, mode="min", patience=7
        )

        # metrics
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)
        self.csi1 = CriticalSuccessIndex(1).to(self.device)
        self.csi8 = CriticalSuccessIndex(8).to(self.device)
        self.mse_loss = nn.MSELoss().to(self.device)

        # scaler
        self.scaler = GradScaler()

    def train_one_epoch(self, train_loader, epoch):
        self.generator.train()
        self.discriminator.train()

        self.ssim.reset()
        self.csi1.reset()
        self.csi8.reset()

        D_losses, G_losses, MSE_losses, V_losses, physics_losses = (
            [],
            [],
            [],
            [],
            [],
        )

        real_target = torch.full((self.batch_size, 1, 1, 1), 0.99, device=self.device)
        fake_target = torch.full((self.batch_size, 1, 1, 1), 0.01, device=self.device)

        for batch_idx, (input_img, target_img) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
        ):
            input_img = input_img.to(self.device, non_blocking=True)
            target_img = target_img.to(self.device, non_blocking=True)

            with autocast():
                generated_image, velocity, velo_gt = self.generator(input_img)

            # D must be trainable here
            for p in self.discriminator.parameters():
                p.requires_grad_(True)

            # train discriminator
            self.D_optimizer.zero_grad()

            disc_inp_fake = torch.cat(
                (input_img[:, self.T_in - 2 :, :, :], generated_image.detach()), 1
            )
            disc_inp_real = torch.cat((input_img[:, self.T_in - 2 :, :, :], target_img), 1)

            D_fake = self.discriminator(disc_inp_fake)
            D_real = self.discriminator(disc_inp_real)

            D_total_loss = discriminator_loss(D_real, real_target) + discriminator_loss(
                D_fake, fake_target
            )

            if not torch.isnan(D_total_loss) and not torch.isinf(D_total_loss):
                D_losses.append(D_total_loss.item())

            D_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.D_optimizer.step()

            # freeze D so we don't compute grads for it during G step
            for p in self.discriminator.parameters():
                p.requires_grad_(False)

            # train generator
            self.G_optimizer.zero_grad()

            with autocast():
                fake_gen = torch.cat((input_img[:, self.T_in - 2 :, :, :], generated_image), 1)
                G_output = self.discriminator(fake_gen)

                G_loss, g_nn_loss, g_v_loss, g_phy_loss = generator_loss(
                    fake_gen,
                    generated_image,
                    target_img,
                    G_output,
                    real_target,
                    velocity,
                    velo_gt,
                )

            self.scaler.scale(G_loss).backward()
            self.scaler.unscale_(self.G_optimizer)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm)
            self.scaler.step(self.G_optimizer)
            self.scaler.update()

            # compute metrics
            with torch.inference_mode():
                mse = huber_loss(generated_image, target_img)
                MSE_losses.append(mse.item())

                # update metrics in float
                self.ssim.update(fake_gen.detach().float(), disc_inp_real.detach().float())
                self.csi1.update(
                    torch.expm1(fake_gen).detach(), torch.expm1(disc_inp_real).detach()
                )
                self.csi8.update(
                    torch.expm1(fake_gen).detach(), torch.expm1(disc_inp_real).detach()
                )

            G_losses.append(G_loss.item())
            V_losses.append(g_v_loss.item())
            physics_losses.append(g_phy_loss.item())

            wandb.log(
                {
                    "D_loss": D_total_loss.item(),
                    "G_loss": G_loss.item(),
                    "MSE_loss": mse.item(),
                    "Velocity_loss": g_v_loss.item(),
                    "physics_loss": g_phy_loss.item(),
                }
            )

        # return averaged metrics
        return {
            "D_loss": sum(D_losses) / len(D_losses),
            "G_loss": sum(G_losses) / len(G_losses),
            "MSE_loss": sum(MSE_losses) / len(MSE_losses),
            "V_loss": sum(V_losses) / len(V_losses),
            "physics_loss": sum(physics_losses) / len(physics_losses),
        }

    def validate(self, test_loader):
        self.generator.eval()
        self.discriminator.eval()

        self.ssim.reset()
        self.csi1.reset()
        self.csi8.reset()

        val_G_losses, val_MSE_losses = [], []

        with torch.inference_mode():
            for input_img, target_img in test_loader:
                input_img, target_img = input_img.to(self.device), target_img.to(self.device)

                generated_image, velocity, velo_gt = self.generator(input_img)
                fake_gen = torch.cat((input_img[:, self.T_in - 2 :, :, :], generated_image), 1)
                G_output = self.discriminator(fake_gen)

                real_target = torch.ones(input_img.size(0), 1, 1, 1).to(self.device)
                G_loss, _, _, _ = generator_loss(
                    fake_gen, generated_image, target_img, G_output, real_target, velocity, velo_gt
                )
                mse = huber_loss(generated_image, target_img)

                val_G_losses.append(G_loss.item())
                val_MSE_losses.append(mse.item())

                # Update metrics
                disc_inp_real = torch.cat((input_img[:, self.T_in - 2 :, :, :], target_img), 1)
                self.ssim.update(fake_gen.float(), disc_inp_real.float())
                self.csi1.update(torch.expm1(fake_gen), torch.expm1(disc_inp_real))
                self.csi8.update(torch.expm1(fake_gen), torch.expm1(disc_inp_real))

        return {
            "val_G_loss": sum(val_G_losses) / len(val_G_losses),
            "val_MSE_loss": sum(val_MSE_losses) / len(val_MSE_losses),
            "SSIM": self.ssim.compute().item(),
            "CSI1": self.csi1.compute().item(),
            "CSI8": self.csi8.compute().item(),
        }

    def train(
        self,
        train_loader,
        test_loader,
    ):
        best_validation_loss = float("inf")
        for epoch in range(self.num_epochs):
            start_time = time.time()
            train_metrics = self.train_one_epoch(train_loader, epoch)
            validation_metrics = self.validate(test_loader)

            # scheduler step
            self.scheduler.step(validation_metrics["val_G_loss"])

            # save best model
            if validation_metrics["val_G_loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["val_G_loss"]
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "generator_state_dict": self.generator.state_dict(),
                        "discriminator_state_dict": self.discriminator.state_dict(),
                        "gen_optimizer_state_dict": self.G_optimizer.state_dict(),
                        "disc_optimizer_state_dict": self.D_optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "best_validation_loss": best_validation_loss,
                    },
                    self.best_model_path,
                )

                wandb.log_artifact(
                    self.best_model_path.as_posix(), name="best-gen-model", type="model"
                )

                print(
                    f"Best model saved at epoch {epoch + 1} with val loss: {best_validation_loss:.4f}"
                )

            # epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch [{epoch + 1}/{self.num_epochs}] - Time: {epoch_time:.2f}s")
            print(
                f"Train G_loss: {train_metrics['G_loss']:.4f}, D_loss: {train_metrics['D_loss']:.4f}, MSE: {train_metrics['MSE_loss']:.4f}"
            )
            print(
                f"Val G_loss: {validation_metrics['val_G_loss']:.4f}, MSE: {validation_metrics['val_MSE_loss']:.4f}, SSIM: {validation_metrics['SSIM']:.4f}, CSI1: {validation_metrics['CSI1']:.4f}, CSI8: {validation_metrics['CSI8']:.4f}"
            )

            wandb.log(
                {
                    **train_metrics,
                    **validation_metrics,
                    "Epoch": epoch + 1,
                    "Epoch_time_sec": epoch_time,
                    "Learning_rate": self.G_optimizer.param_groups[0]["lr"],
                }
            )

        # save final model
        torch.save(
            {
                "epoch": self.num_epochs,
                "generator_state_dict": self.generator.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "gen_optimizer_state_dict": self.G_optimizer.state_dict(),
                "disc_optimizer_state_dict": self.D_optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            self.final_model_path,
        )

        wandb.log_artifact(self.final_model_path.as_posix(), name="final-gen-model", type="model")

        print(f"Final model saved as: {self.final_model_path}")
        wandb.finish()
