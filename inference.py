
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

    try:
        generator = Generator().to(device)
        generator.eval()

        state_dict = torch.load('./weights/ViT/generator_latest.pth', map_location=device, weights_only=True)
        state_dict = torch.load('./weights/CNN_DynamicLR/generator_latest.pth', map_location=device, weights_only=True)
        state_dict = torch.load('./weights/CNN/generator_latest.pth', map_location=device, weights_only=True)
        generator.load_state_dict(state_dict)
        print("Model loaded successfully!")

    except Exception as e:
        print(f"Error loading model: {e}")

    transforms_ = [
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    # TODO: To better gives the path to the dataset
    dataset = ImageDataset("data/street", transforms_=transforms_, mode="val")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    imgs, masked_imgs, _ = next(iter(dataloader))
    imgs = imgs.to(device)
    masked_imgs = masked_imgs.to(device)

    # Inference
    print('Generating images...')
    with torch.no_grad():
        gen_parts = generator(masked_imgs)

        center_part = masked_imgs.clone()
        i = 32
        center_mask = 128 // 2
        gen_parts[:, :, i:i+center_mask, i:i+center_mask] = center_part[:, :, i:i+center_mask, i:i+center_mask]

    print('Image generated successfully!')
    sample_imgs = torch.cat((masked_imgs.cpu(), gen_parts.cpu(), imgs.cpu()), -2)
    img_grid = make_grid(sample_imgs, nrow=4, normalize=True)

    os.makedirs('inference_results', exist_ok=True)
    save_image(img_grid, 'inference_results/generated.png')
    print("Images saved to 'inference_results/generated.png'.")

if __name__ == "__main__":
    main()