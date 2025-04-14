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

        # state_dict = torch.load('./weights/CNN_DynamicLR/generator_latest.pth', map_location=device, weights_only=True)
        # state_dict = torch.load('./weights/CNN/generator_latest.pth', map_location=device, weights_only=True)
        state_dict = torch.load('./weights/ViT/generator_latest.pth', map_location=device, weights_only=True)
        generator.load_state_dict(state_dict)
        print("Model loaded successfully!")

    except Exception as e:
        print(f"Error loading model: {e}")

    transforms_ = transforms.Compose([
        transforms.Resize((128, 128), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load single image
    image_path = "inference_results/org2.jpeg"  # Replace with your image path
    img = Image.open(image_path).convert('RGB')
    img_tensor = transforms_(img).unsqueeze(0).to(device)
    
    # Create masked version
    mask = torch.ones_like(img_tensor)
    # masked_img = img_tensor.clone()
    i = 64 // 2
    center_mask = 64
    mask[:, :, i:i+center_mask, i:i+center_mask] = 0  # make center 0
    
    masked_img = img_tensor * (1 - mask) + mask

    # Inference
    print('Generating images...')
    with torch.no_grad():
        gen_parts = generator(masked_img)

        center_part = img_tensor.clone()
        gen_parts[:, :, i:i+center_mask, i:i+center_mask] = center_part[:, :, i:i+center_mask, i:i+center_mask]

    print('Image generated successfully!')
    sample_imgs = torch.cat((masked_img.cpu(), gen_parts.cpu(), img_tensor.cpu()), -2)
    sample_imgs = (sample_imgs + 1) / 2 # Normalize to [0, 1]

    os.makedirs('inference_results', exist_ok=True)
    save_image(sample_imgs, 'inference_results/generated.png')
    print("Images saved to 'inference_results/generated.png'.")

if __name__ == "__main__":
    main()