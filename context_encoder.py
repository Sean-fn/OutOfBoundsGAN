import os
import threading

import torch
from torch.amp import GradScaler, autocast
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
# from carbs import ObservationInParam

from datasets import get_dataloader
from models import Generator, Discriminator
from utils import Logger

class GANTrainer:
    def __init__(self, config, writer):
        self.config = config
        self.early_stopping = False
        self.generator = Generator(channels=config.opt.channels)
        self.discriminator = Discriminator(channels=config.opt.channels)
        self.adversarial_loss = nn.MSELoss()
        self.pixelwise_loss = nn.L1Loss()
        
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), 
            lr=config.opt.lr, 
            betas=(config.opt.b1, config.opt.b2)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=config.opt.lr, 
            betas=(config.opt.b1, config.opt.b2)
        )

        self.scheduler_G = CyclicLR(
            self.optimizer_G, 
            base_lr=config.opt.lr_min, 
            max_lr=config.opt.lr_max,
            step_size_up=config.opt.step_size_up,
            mode='exp_range',
            gamma=config.opt.lr_gamma
        )
        self.scheduler_D = CyclicLR(
            self.optimizer_D, 
            base_lr=config.opt.lr_min, 
            max_lr=config.opt.lr_max,
            step_size_up=config.opt.step_size_up,
            mode='exp_range',
            gamma=config.opt.lr_gamma
        )

        self.scheduler_G_plateau = ReduceLROnPlateau(
            self.optimizer_G, 
            mode='min', 
            factor=config.opt.plateau_factor, 
            patience=config.opt.plateau_patience, 
            verbose=True
        )
        self.scheduler_D_plateau = ReduceLROnPlateau(
            self.optimizer_D, 
            mode='min', 
            factor=config.opt.plateau_factor, 
            patience=config.opt.plateau_patience, 
            verbose=True
        )

        if config.cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()
            self.pixelwise_loss.cuda()
        
        self.dataloader = get_dataloader(config)
        self.test_dataloader = get_dataloader(config, mode="val")

        self.writer = writer
        self.logger = Logger(
            self.writer,
            self.generator,
            self.discriminator,
            self.optimizer_G,
            self.optimizer_D,
            self.test_dataloader,
            self.config
        )

    def process_batch(self, imgs, masked_imgs, masked_parts):
        imgs = imgs.type(self.config.Tensor)
        masked_imgs = masked_imgs.type(self.config.Tensor)
        masked_parts = masked_parts.type(self.config.Tensor)

        batch_size = imgs.shape[0]
        label_size = int(self.config.opt.mask_size / 2 ** 3)
        valid = torch.ones(batch_size, 1, label_size, label_size).type(self.config.Tensor)
        fake = torch.zeros(batch_size, 1, label_size, label_size).type(self.config.Tensor)


        with autocast(device_type="cuda" if self.config.cuda else "cpu"):
            gen_parts = self.generator(masked_imgs)
            g_adv = self.adversarial_loss(self.discriminator(gen_parts), valid)
            g_pixel = self.pixelwise_loss(gen_parts, masked_parts)
            g_loss = 0.001 * g_adv + 0.999 * g_pixel

            real_loss = self.adversarial_loss(self.discriminator(masked_parts), valid)
            fake_loss = self.adversarial_loss(self.discriminator(gen_parts.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

        return gen_parts, g_adv, g_pixel, g_loss, d_loss

    def train(self, epoch):
        # Update optimizer with new learning rate
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = self.config.opt.lr
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = self.config.opt.lr
    
        print(f"Epoch {epoch}/{self.config.opt.n_epochs} - 調整後的超參數:")
        print(f"  學習率 (lr): {self.config.opt.lr}")
        print(f"  最小學習率 (lr_min): {self.config.opt.lr_min}")
        print(f"  最大學習率 (lr_max): {self.config.opt.lr_max}")
        print(f"  步進大小上升 (step_size_up): {self.config.opt.step_size_up}")
        print(f"  降低因子 (plateau_factor): {self.config.opt.plateau_factor}")
        print(f"  降低耐心 (plateau_patience): {self.config.opt.plateau_patience}")
        
        self.generator.train()
        self.discriminator.train()

        prev_g_pixel = float('inf')
        
        scaler = GradScaler()
        # BUG: Dosen't resume training on the specific batch
        for i, (imgs, masked_imgs, masked_parts) in enumerate(self.dataloader, start=int(self.config.opt.resume_start_num)+1):
            gen_parts, g_adv, g_pixel, g_loss, d_loss = self.process_batch(imgs, masked_imgs, masked_parts)

            self.optimizer_G.zero_grad()
            scaler.scale(g_loss).backward()
            scaler.step(self.optimizer_G)
            scaler.update()

            self.optimizer_D.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.step(self.optimizer_D)
            scaler.update()

            print(
                f"[Epoch {epoch}/{self.config.opt.n_epochs}] [Batch {i}/{len(self.dataloader)}] "
                f"[D loss: {d_loss.item():.6f}] [G adv: {g_adv.item():.3f}, pixel: {g_pixel.item():.3f}]"
            )

            if g_pixel < prev_g_pixel:
                early_stopping = 0
            else:
                early_stopping += 1
                prev_g_pixel = g_pixel

            if early_stopping >= self.config.opt.early_stopping:
                self.early_stopping = True
                print(f"Early stopping at batch {i}")
                return

            global_step = epoch * len(self.dataloader) + i
            self.logger.log_training_progress(d_loss, g_adv, g_pixel, g_loss, global_step)
            self.scheduler_G.step()
            self.scheduler_D.step()

        # Validate after each epoch
        val_loss = self.validate(global_step)

        self.scheduler_G_plateau.step(val_loss)
        self.scheduler_D_plateau.step(val_loss)

    def validate(self,global_step):
        self.generator.eval()
        self.discriminator.eval()
        total_loss = 0.0
        with torch.no_grad():
            for imgs, masked_imgs, masked_parts in self.test_dataloader:
                gen_parts, g_adv, g_pixel, g_loss, _ = self.process_batch(imgs, masked_imgs, masked_parts)
                total_loss += g_loss.item()

        avg_loss = total_loss / len(self.test_dataloader)
        self.generator.train()
        self.discriminator.train()
        print(f"Validation loss: {avg_loss:.4f}")
        self.logger.log_validation_loss(avg_loss, global_step)
        return avg_loss
