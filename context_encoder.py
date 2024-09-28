import datetime
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
from carbs import ObservationInParam

from datasets import ImageDataset
from models import Generator, Discriminator

class GANTrainer:
    def __init__(self, config, writer):
        self.config = config
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
            mode='triangular'
        )
        self.scheduler_D = CyclicLR(
            self.optimizer_D, 
            base_lr=config.opt.lr_min, 
            max_lr=config.opt.lr_max,
            step_size_up=config.opt.step_size_up,
            mode='triangular'
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
        
        self.dataloader = self.get_dataloader()
        self.test_dataloader = self.get_dataloader(mode="val")

        self.epochs_list = []
        self.d_losses = []
        self.g_adv_losses = []
        self.g_pixel_losses = []

        self.writer = writer
        
    def get_dataloader(self, mode="train"):
        transforms_ = [
            transforms.Resize((self.config.opt.img_size, self.config.opt.img_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        dataset = ImageDataset(f"data/{self.config.opt.dataset_name}", transforms_=transforms_, mode=mode)
        dataloader = DataLoader(
            dataset,
            self.config.opt.batch_size if mode == "train" else 12,
            shuffle=True,
            num_workers=self.config.opt.n_cpu if mode == "train" else 1,
        )
        return dataloader
    
    def save_sample(self, batches_done):
        save_dir = 'images'
        os.makedirs(save_dir, exist_ok=True)

        samples, masked_samples, i = next(iter(self.test_dataloader))
        samples = samples.type(self.config.Tensor)
        masked_samples = masked_samples.type(self.config.Tensor)
        i = 32      # TODO: make i a variable passed to the model
        center_mask = self.config.opt.mask_size // 2 
        gen_fram = self.generator(masked_samples)
        center_part = masked_samples.clone()
        gen_fram[:, :, i:i+center_mask, i:i+center_mask] = center_part[:, :, i:i+center_mask, i:i+center_mask]
        sample = torch.cat((masked_samples, gen_fram, samples), -2)
        save_path = os.path.join(save_dir, f"{batches_done}.png")
        save_image(sample, save_path, nrow=6, normalize=True)
    #     data = {
    #         'epochs': np.array(self.epochs_list),
    #         'd_losses': np.array(self.d_losses),
    #         'g_adv_losses': np.array(self.g_adv_losses),
    #         'g_pixel_losses': np.array(self.g_pixel_losses)
    #     }
    #     os.makedirs('./config_data', exist_ok=True)
    #     np.save('./config_data/training_data.npy', data)

    def log_training_progress(self, epoch, batch_idx, imgs, masked_imgs, masked_parts, 
                              d_loss, g_adv, g_pixel, g_loss, global_step):
        self.writer.add_scalar('Loss/Discriminator', d_loss.item(), global_step)
        self.writer.add_scalar('Loss/Generator/Adversarial', g_adv.item(), global_step)
        self.writer.add_scalar('Loss/Generator/Pixel', g_pixel.item(), global_step)
        self.writer.add_scalar('Loss/Generator/Total', g_loss.item(), global_step)

        if global_step % self.config.opt.sample_interval == 0:
            self.log_images(imgs, masked_imgs, masked_parts, global_step)

        # lr
        self.writer.add_scalar('Learning Rate/Generator', self.optimizer_G.param_groups[0]['lr'], global_step)
        self.writer.add_scalar('Learning Rate/Discriminator', self.optimizer_D.param_groups[0]['lr'], global_step)

        self.log_gradients(global_step)

    def log_images(self, imgs, masked_imgs, masked_parts, global_step):
        with torch.no_grad():
            fake_imgs = self.generator(masked_imgs)
            img_grid = self.save_image_grid(fake_imgs, f"images/{global_step}.png")
            self.writer.add_image('Generated Images', img_grid, global_step)

            real_grid = self.save_image_grid(imgs, f"images/real_{global_step}.png")
            masked_grid = self.save_image_grid(masked_imgs, f"images/masked_{global_step}.png")
            self.writer.add_image('Real Images', real_grid, global_step)
            self.writer.add_image('Masked Images', masked_grid, global_step)

    def log_gradients(self, global_step):
        for name, param in self.generator.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'Generator Gradients/{name}', param.grad, global_step)
        
        for name, param in self.discriminator.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'Discriminator Gradients/{name}', param.grad, global_step)

    def log_epoch_end(self, epoch):
        self.writer.add_histogram('Generator Weights', self.generator.state_dict()['weight'], epoch)
        self.writer.add_histogram('Discriminator Weights', self.discriminator.state_dict()['weight'], epoch)

    @staticmethod
    def save_image_grid(img_tensor, filename, nrow=8):
        img_grid = torchvision.utils.make_grid(img_tensor, nrow=nrow, normalize=True)
        torchvision.utils.save_image(img_grid, filename)
        return img_grid


    def save_weights(self, epoch):
        weights_dir = 'weights'
        os.makedirs(weights_dir, exist_ok=True)

        if os.path.exists(os.path.join(weights_dir, 'generator_latest.pth')):
            os.rename(os.path.join(weights_dir, 'generator_latest.pth'), os.path.join(weights_dir, f'generator_epoch_{epoch - self.config.opt.sample_interval}.pth'))
            os.rename(os.path.join(weights_dir, 'discriminator_latest.pth'), os.path.join(weights_dir, f'discriminator_epoch_{epoch - self.config.opt.sample_interval}.pth'))
        torch.save(self.generator.state_dict(), os.path.join(weights_dir, 'generator_latest.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(weights_dir, 'discriminator_latest.pth'))

        print(f"Saved model weights for epoch {epoch}")

    def process_batch(self, imgs, masked_imgs, masked_parts):
        imgs = imgs.type(self.config.Tensor)
        masked_imgs = masked_imgs.type(self.config.Tensor)
        masked_parts = masked_parts.type(self.config.Tensor)

        batch_size = imgs.shape[0]
        label_size = int(self.config.opt.mask_size / 2 ** 3)
        valid = torch.ones(batch_size, 1, label_size, label_size).type(self.config.Tensor)
        fake = torch.zeros(batch_size, 1, label_size, label_size).type(self.config.Tensor)

        # Forward pass
        gen_parts = self.generator(masked_imgs)

        # Compute loss
        g_adv = self.adversarial_loss(self.discriminator(gen_parts), valid)
        g_pixel = self.pixelwise_loss(gen_parts, masked_parts)
        g_loss = 0.001 * g_adv + 0.999 * g_pixel

        # Discriminator loss
        real_loss = self.adversarial_loss(self.discriminator(masked_parts), valid)
        fake_loss = self.adversarial_loss(self.discriminator(gen_parts.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        return gen_parts, g_adv, g_pixel, g_loss, d_loss

    def train(self, epoch, carbs):
        # Suggest new learning rate from CARBS
        suggestion = carbs.suggest().suggestion
        self.config.opt.lr = suggestion.get('lr', self.config.opt.lr)
        self.config.opt.lr_min = suggestion.get('lr_min', self.config.opt.lr_min)
        self.config.opt.lr_max = suggestion.get('lr_max', self.config.opt.lr_max)
        self.config.opt.step_size_up = suggestion.get('step_size_up', self.config.opt.step_size_up)
        self.config.opt.plateau_factor = suggestion.get('plateau_factor', self.config.opt.plateau_factor)
        self.config.opt.plateau_patience = suggestion.get('plateau_patience', self.config.opt.plateau_patience)

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
        
        # BUG: Dosen't resume training on the specific batch
        for i, (imgs, masked_imgs, masked_parts) in enumerate(self.dataloader, start=int(self.config.opt.resume_start_num)+1):
            gen_parts, g_adv, g_pixel, g_loss, d_loss = self.process_batch(imgs, masked_imgs, masked_parts)

            self.optimizer_G.zero_grad()
            g_loss.backward()
            self.optimizer_G.step()

            self.optimizer_D.zero_grad()
            d_loss.backward()
            self.optimizer_D.step()

            print(
                f"[Epoch {epoch}/{self.config.opt.n_epochs}] [Batch {i}/{len(self.dataloader)}] "
                f"[D loss: {d_loss.item():.6f}] [G adv: {g_adv.item():.3f}, pixel: {g_pixel.item():.3f}]"
            )

            # Save loss data into arrays
            # self.epochs_list.append(epoch + i / len(self.dataloader))
            # self.d_losses.append(d_loss.item())
            # self.g_adv_losses.append(g_adv.item())
            # self.g_pixel_losses.append(g_pixel.item())

            # global_step = epoch * len(self.dataloader) + i
            # self.log_training_progress(epoch, i, imgs, masked_imgs, masked_parts, 
            #                             d_loss, g_adv, g_pixel, g_loss, global_step)

            self.scheduler_G.step()
            self.scheduler_D.step()

            batches_done = epoch * len(self.dataloader) + i
            if batches_done % self.config.opt.sample_interval == 0:
                self.save_sample(batches_done)
                self.save_weights(batches_done)
                # self.save_training_data()

        # Validate after each epoch
        val_loss = self.validate()

        self.scheduler_G_plateau.step(val_loss)
        self.scheduler_D_plateau.step(val_loss)

        # self.log_epoch_end(epoch)
        observed_value = g_loss.item()
        carbs.observe(ObservationInParam(
            input=suggestion,
            output=observed_value,
            cost=epoch
        ))

    def validate(self):
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
        return avg_loss