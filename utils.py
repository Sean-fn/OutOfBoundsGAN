import os
import torch

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
            print("Loaded pre-trained weights")
        else:
            resume_num = 0
            print("Fetching the pre-trained weights, but nothing found. Starting from scratch.")
