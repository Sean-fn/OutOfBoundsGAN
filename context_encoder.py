import datetime
import os
import threading

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from datasets import ImageDataset
from models import Generator, Discriminator

class GANTrainer:
    def __init__(self, config):
        self.config = config
        self.generator = Generator(channels=config.opt.channels)
        self.discriminator = Discriminator(channels=config.opt.channels)
        self.adversarial_loss = nn.MSELoss()
        self.pixelwise_loss = nn.L1Loss()
        
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=config.opt.lr, betas=(config.opt.b1, config.opt.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=config.opt.lr, betas=(config.opt.b1, config.opt.b2))

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

        self.writer = SummaryWriter(log_dir=f'logs/{config.opt.run_name}/CNN_BatchSize=4_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
        
    def get_dataloader(self, mode="train"):
        transforms_ = [
            transforms.Resize((self.config.opt.img_size, self.config.opt.img_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        dataset = ImageDataset(f"data/{self.config.opt.dataset_name}", transforms_=transforms_, mode=mode)
        dataloader = DataLoader(
            dataset,
            self.config.opt.batch_size if mode == "train" else 4,
            shuffle=True,
            num_workers=self.config.opt.n_cpu if mode == "train" else 1,
        )
        return dataloader
    
    def log_to_tensorboard(self, func, *args):
        thread = threading.Thread(target=func, args=args)
        thread.start()

    def log_training_progress(self, d_loss, g_adv, g_pixel, g_loss, global_step):
        self.log_to_tensorboard(self.writer.add_scalar, 'Loss/Discriminator', d_loss.item(), global_step)
        self.log_to_tensorboard(self.writer.add_scalar, 'Loss/Generator/Adversarial', g_adv.item(), global_step)
        self.log_to_tensorboard(self.writer.add_scalar, 'Loss/Generator/Pixel', g_pixel.item(), global_step)
        self.log_to_tensorboard(self.writer.add_scalar, 'Loss/Generator/Total', g_loss.item(), global_step)
        self.log_to_tensorboard(self.writer.add_scalar, 'Learning Rate/Generator', self.optimizer_G.param_groups[0]['lr'], global_step)
        self.log_to_tensorboard(self.writer.add_scalar, 'Learning Rate/Discriminator', self.optimizer_D.param_groups[0]['lr'], global_step)

        if global_step % self.config.opt.sample_interval == 0:
            self.log_images(global_step)
            self.log_gradients(global_step)
            self.save_weights(global_step)


    def log_gradients(self, global_step):
        for name, param in self.generator.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'Generator Gradients/{name}', param.grad, global_step)
        
        for name, param in self.discriminator.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'Discriminator Gradients/{name}', param.grad, global_step)

    def log_images(self, global_step):
        with torch.no_grad():
            samples, masked_samples, masked_parts = next(iter(self.test_dataloader))

            samples = samples.type(self.config.Tensor)
            masked_samples = masked_samples.type(self.config.Tensor)
            i = 32  # TODO: make i a variable, passed to the model
            center_mask = self.config.opt.mask_size // 2
            gen_parts = self.generator(masked_samples)
            center_part = masked_samples.clone()
            gen_parts[:, :, i:i+center_mask, i:i+center_mask] = center_part[:, :, i:i+center_mask, i:i+center_mask]
            sample_imgs = torch.cat((masked_samples, gen_parts, samples), -2)

            img_grid = torchvision.utils.make_grid(sample_imgs, nrow=4, normalize=True)
            torchvision.utils.save_image(img_grid, f"images/{global_step}.png")
            self.writer.add_image('Generated Images', img_grid, global_step)


    def save_weights(self, epoch):
        weights_dir = 'weights'
        os.makedirs(weights_dir, exist_ok=True)

        # if os.path.exists(os.path.join(weights_dir, 'generator_latest.pth')):
        #     os.rename(os.path.join(weights_dir, 'generator_latest.pth'), os.path.join(weights_dir, f'generator_epoch_{epoch - self.config.opt.sample_interval}.pth'))
        #     os.rename(os.path.join(weights_dir, 'discriminator_latest.pth'), os.path.join(weights_dir, f'discriminator_epoch_{epoch - self.config.opt.sample_interval}.pth'))
        torch.save(self.generator.state_dict(), os.path.join(weights_dir, 'generator_latest.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(weights_dir, 'discriminator_latest.pth'))

        print(f"Saved model weights for epoch {epoch}")

    def train(self, epoch):
        for i, (imgs, masked_imgs, masked_parts) in enumerate(self.dataloader, start=int(self.config.opt.resume_start_num)+1):
            valid = torch.ones(imgs.shape[0], 1, int(self.config.opt.mask_size / 2 ** 3), int(self.config.opt.mask_size / 2 ** 3)).type(self.config.Tensor)
            fake = torch.zeros(imgs.shape[0], 1, int(self.config.opt.mask_size / 2 ** 3), int(self.config.opt.mask_size / 2 ** 3)).type(self.config.Tensor)

            imgs = imgs.type(self.config.Tensor)
            masked_imgs = masked_imgs.type(self.config.Tensor)
            masked_parts = masked_parts.type(self.config.Tensor)

            # Train Generator
            self.optimizer_G.zero_grad()
            gen_parts = self.generator(masked_imgs)
            g_adv = self.adversarial_loss(self.discriminator(gen_parts), valid)
            g_pixel = self.pixelwise_loss(gen_parts, masked_parts)
            g_loss = 0.001 * g_adv + 0.999 * g_pixel
            g_loss.backward()
            self.optimizer_G.step()

            # Train Discriminator
            self.optimizer_D.zero_grad()
            real_loss = self.adversarial_loss(self.discriminator(masked_parts), valid)
            fake_loss = self.adversarial_loss(self.discriminator(gen_parts.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            self.optimizer_D.step()

            print(
                f"[Epoch {epoch}/{self.config.opt.n_epochs}] [Batch {i}/{len(self.dataloader)}] "
                f"[D loss: {d_loss.item()}] [G adv: {g_adv.item()}, pixel: {g_pixel.item()}]"
            )

            global_step = epoch * len(self.dataloader) + i
            self.log_training_progress(d_loss, g_adv, g_pixel, g_loss, global_step)

        observed_value = g_loss.item()
        # carbs.observe(ObservationInParam(
        #     input=suggestion,
        #     output=observed_value,
        #     cost=epoch
        # ))
        self.writer.close()
