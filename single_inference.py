import os
import torch
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from models import Generator
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

    transforms_ = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load single image
    image_path = "inference_results/org.jpeg"  # Replace with your image path
    img = Image.open(image_path).convert('RGB')
    img_tensor = transforms_(img).unsqueeze(0).to(device)
    
    # Create masked version
    masked_img = img_tensor.clone()
    i = 128 // 2
    center_mask = 128
    masked_img[:, :, i:i+center_mask, i:i+center_mask] = 1  # White mask in center

    # Inference
    print('Generating images...')
    with torch.no_grad():
        gen_parts = generator(masked_img)

        center_part = masked_img.clone()
        gen_parts[:, :, i:i+center_mask, i:i+center_mask] = center_part[:, :, i:i+center_mask, i:i+center_mask]

    print('Image generated successfully!')
    sample_imgs = torch.cat((masked_img.cpu(), gen_parts.cpu(), img_tensor.cpu()), -2)
    img_grid = make_grid(sample_imgs, nrow=3, normalize=True)

    os.makedirs('inference_results', exist_ok=True)
    save_image(img_grid, 'inference_results/generated.png')
    print("Images saved to 'inference_results/generated.png'.")

if __name__ == "__main__":
    main()