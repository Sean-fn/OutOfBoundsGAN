
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from models import Generator
from datasets import ImageDataset
from config import Config

def main():
    config = Config()
    device = config.device

    generator = Generator(channels=3).to(device)
    generator.eval()

    generator.load_state_dict(torch.load('weights/generator_latest.pth', map_location=device))

    transforms_ = [
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    dataset = ImageDataset(config.opt.dataset_name, transforms_=transforms_, mode="val")
    dataloader = DataLoader(dataset, batch_size=12, shuffle=False)

    imgs, masked_imgs, _ = next(iter(dataloader))
    imgs = imgs.to(device)
    masked_imgs = masked_imgs.to(device)

    # Inference
    with torch.no_grad():
        gen_parts = generator(masked_imgs)

        center_part = masked_imgs.clone()
        i = 32
        center_mask = config.opt.mask_size // 2
        gen_parts[:, :, i:i+center_mask, i:i+center_mask] = center_part[:, :, i:i+center_mask, i:i+center_mask]

    sample_imgs = torch.cat((masked_imgs.cpu(), center_part.cpu(), imgs.cpu()), -2)
    img_grid = make_grid(sample_imgs, nrow=4, normalize=True)

    plt.figure(figsize=(12, 8))
    plt.imshow(img_grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

    save_option = input("Do you want to save the generated images? (y/n): ")
    if save_option.lower() == 'y':
        os.makedirs('inference_results', exist_ok=True)
        save_image(img_grid, 'inference_results/generated.png')
        print("Images saved to 'inference_results/generated.png'.")

if __name__ == "__main__":
    main()