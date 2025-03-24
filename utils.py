import os
import datetime
import torch
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
import torchvision
import threading
from torch.utils.tensorboard import SummaryWriter

def load_weights(config, trainer):
    """
    Load pre-trained weights if resume_num is not None
    """
    resume_num = config.opt.resume_num
    if resume_num is not None:
        generator_path = os.path.join('weights', f'generator_{resume_num}.pth')
        discriminator_path = os.path.join('weights', f'discriminator_{resume_num}.pth')

        if os.path.exists(discriminator_path):
            trainer.discriminator.load_state_dict(torch.load(discriminator_path))
            trainer.generator.load_state_dict(torch.load(generator_path))
            print("**Loaded pre-trained weights**")
        else:
            resume_num = 0
            print("Fetching the pre-trained weights, but nothing found. Starting from scratch.")

def create_optim(model_G, model_D, config):
    """Create optimizers and schedulers for both generator and discriminator"""
    optimizer_G = torch.optim.Adam(
        model_G.parameters(),
        lr=config.opt.lr,
        betas=(config.opt.b1, config.opt.b2)
    )
    optimizer_D = torch.optim.Adam(
        model_D.parameters(),
        lr=config.opt.lr,
        betas=(config.opt.b1, config.opt.b2)
    )

    # scheduler_G = CyclicLR(
    #     optimizer_G,
    #     base_lr=config.opt.lr_min,
    #     max_lr=config.opt.lr_max,
    #     step_size_up=config.opt.step_size_up,
    #     mode='exp_range',
    #     gamma=config.opt.lr_gamma
    # )
    # scheduler_D = CyclicLR(
    #     optimizer_D,
    #     base_lr=config.opt.lr_min,
    #     max_lr=config.opt.lr_max,
    #     step_size_up=config.opt.step_size_up,
    #     mode='exp_range',
    #     gamma=config.opt.lr_gamma
    # )

    # scheduler_G_plateau = ReduceLROnPlateau(
    #     optimizer_G,
    #     mode='min',
    #     factor=config.opt.plateau_factor,
    #     patience=config.opt.plateau_patience,
    #     verbose=True
    # )
    # scheduler_D_plateau = ReduceLROnPlateau(
    #     optimizer_D,
    #     mode='min',
    #     factor=config.opt.plateau_factor,
    #     patience=config.opt.plateau_patience,
    #     verbose=True
    # )

    return (optimizer_G, optimizer_D, )
            # scheduler_G, scheduler_D,
            # scheduler_G_plateau, scheduler_D_plateau)

class Logger:
    def __init__(self, generator, discriminator, optimizer_G, optimizer_D, test_dataloader, config):
        self.writer = SummaryWriter(log_dir=f'logs/{config.opt.run_name}_BatchSize={config.opt.batch_size}_{datetime.datetime.now().strftime("%m%d-%H%M")}')
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.test_dataloader = test_dataloader
        self.config = config
        self.device = config.device  # Use the device from config instead of creating new one

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

    def log_validation_loss(self, avg_loss, global_step):
        self.writer.add_scalar('Loss/Validation', avg_loss, global_step)

    def log_images(self, global_step):
        with torch.no_grad():
            samples, masked_samples, _ = next(iter(self.test_dataloader))
            samples = samples.to(self.device)
            masked_samples = masked_samples.to(self.device)
            
            i = 32  # TODO: make i a variable
            center_mask = self.config.opt.mask_size // 2
            gen_parts = self.generator(masked_samples)
            center_part = masked_samples.clone()
            gen_parts[:, :, i:i+center_mask, i:i+center_mask] = center_part[:, :, i:i+center_mask, i:i+center_mask]
            sample_imgs = torch.cat((masked_samples.cpu(), gen_parts.cpu(), samples.cpu()), -2)

            img_grid = torchvision.utils.make_grid(sample_imgs, nrow=4, normalize=True)
            os.makedirs('images', exist_ok=True)
            torchvision.utils.save_image(img_grid, f"images/{global_step}.png")
            self.writer.add_image('Generated Images', img_grid, global_step)

    def save_weights(self, epoch):
        weights_dir = f'weights/{self.config.opt.run_name}'
        os.makedirs(weights_dir, exist_ok=True)
        torch.save(self.generator.state_dict(), os.path.join(weights_dir, 'generator_latest.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(weights_dir, 'discriminator_latest.pth'))
        print(f"Saved model weights for epoch {epoch}")

    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()
