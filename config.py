import argparse

import torch
import os

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        # TODO: early stopping
        parser.add_argument("--n_epochs", type=int, default=3)
        parser.add_argument("--batch_size", type=int, default=10)
        # parser.add_argument("--dataset_name", type=str, default="img_align_celeba")
        parser.add_argument("--dataset_name", type=str, default="street")
        parser.add_argument("--lr", type=float, default=9e-4)
        parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--n_cpu", type=int, default=24, help="number of cpu threads to use during batch generation")
        parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
        parser.add_argument("--img_size", type=int, default=128)
        parser.add_argument("--mask_size", type=int, default=128)
        parser.add_argument("--channels", type=int, default=3)
        parser.add_argument("--sample_interval", type=int, default=1000)
        parser.add_argument("--run_name", type=str, default="ViT")
        parser.add_argument("--last_log", type=str, default=max((os.path.join("logs", f) for f in os.listdir("logs")) if os.path.exists("logs") and os.listdir("logs") else ["logs/default"], key=os.path.getmtime))
        # TODO: resume training with one argument
        parser.add_argument("--resume_num", type=str, default="latest")
        parser.add_argument("--resume_start_num", type=int, default=0, help="Set to 0 if not resuming")
        parser.add_argument("--early_stopping", type=int, default=100)

        parser.add_argument("--lr_min", type=float, default=9e-5)
        parser.add_argument("--lr_max", type=float, default=9e-4)
        parser.add_argument("--lr_gamma", type=float, default=0.99999)
        parser.add_argument("--step_size_up", type=int, default=150)
        parser.add_argument("--plateau_factor", type=float, default=0.5)
        parser.add_argument("--plateau_patience", type=int, default=5)

        self.opt = parser.parse_args()
        
        self.cuda = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda" if self.cuda else "cpu")
