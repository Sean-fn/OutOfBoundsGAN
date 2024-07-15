import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from datasets import ImageDataset
from models import Generator, Discriminator

class GANTrainer:
    def __init__(self, config):
        self.config = config
        self.generator = Generator(channels=config.opt.channels)
        self.discriminator = Discriminator(channels=config.opt.channels)
        self.adversarial_loss = nn.MSELoss()
        self.pixelwise_loss = nn.L1Loss()
        
        if config.cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()
            self.pixelwise_loss.cuda()
        
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=config.opt.lr, betas=(config.opt.b1, config.opt.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=config.opt.lr, betas=(config.opt.b1, config.opt.b2))
        
        self.dataloader = self.get_dataloader()
        self.test_dataloader = self.get_dataloader(mode="val")
        
    def get_dataloader(self, mode="train"):
        transforms_ = [
            transforms.Resize((self.config.opt.img_size, self.config.opt.img_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        dataset = ImageDataset(f"data/{self.config.opt.dataset_name}", transforms_=transforms_, mode=mode)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.opt.batch_size if mode == "train" else 12,
            shuffle=True,
            num_workers=self.config.opt.n_cpu if mode == "train" else 1,
        )
        return dataloader
    
    def save_sample(self, batches_done):
        samples, masked_samples, i = next(iter(self.test_dataloader))
        samples = samples.type(self.config.Tensor)
        masked_samples = masked_samples.type(self.config.Tensor)
        i = 32
        center_mask = self.config.opt.mask_size // 2 
        gen_mask = self.generator(masked_samples)
        filled_samples = masked_samples.clone()
        gen_mask[:, :, i:i+center_mask, i:i+center_mask] = filled_samples[:, :, i:i+center_mask, i:i+center_mask]
        sample = torch.cat((masked_samples, gen_mask, samples), -2)
        save_image(sample, f"images/{batches_done}.png", nrow=6, normalize=True)
    
    def train(self):
        for epoch in range(self.config.opt.n_epochs):
            for i, (imgs, masked_imgs, masked_parts) in enumerate(self.dataloader):
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

                batches_done = epoch * len(self.dataloader) + i
                if batches_done % self.config.opt.sample_interval == 0:
                    self.save_sample(batches_done)