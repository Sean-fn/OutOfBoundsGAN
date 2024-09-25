import argparse
import torch

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        # TODO: early stopping
        parser.add_argument("--n_epochs", type=int, default=200)
        parser.add_argument("--batch_size", type=int, default=310)
        # parser.add_argument("--dataset_name", type=str, default="img_align_celeba")
        parser.add_argument("--dataset_name", type=str, default="street")
        parser.add_argument("--lr", type=float, default=0.00009)
        parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--n_cpu", type=int, default=24, help="number of cpu threads to use during batch generation")
        parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
        parser.add_argument("--img_size", type=int, default=128)
        parser.add_argument("--mask_size", type=int, default=128)
        parser.add_argument("--channels", type=int, default=3)
        parser.add_argument("--sample_interval", type=int, default=100)
        parser.add_argument("--run_name", type=str, default="GAN_experiment")
        parser.add_argument("--resume_num", type=str, default="10")
        self.opt = parser.parse_args()
        
        self.cuda = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
