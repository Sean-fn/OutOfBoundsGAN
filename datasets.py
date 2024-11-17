import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=128, mask_size=64, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        # self.files = sorted(glob.glob("%s/*.jpg" % root))
        self.files = sorted(glob.glob("%s/*.jpeg" % root))
        self.files = self.files[:-4000] if mode == "train" else self.files[-4000:]

    def apply_random_mask(self, img):
        """Randomly masks image"""
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part

    def apply_center_mask(self, img):
        """Mask center part of image(for face generation)"""
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1

        return masked_img, i

    def apply_external_random_mask(self, img):
        """Randomly masks the external part of the image"""
        mask = torch.ones_like(img)
        
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        
        mask[:, y1:y2, x1:x2] = 0
        
        masked_img = img * (1 - mask) + mask
        
        masked_part = img * mask
        return masked_img, masked_part
    
    def apply_external_center_mask(self, img):
        """Masks the external part of the image, keeping the center visible"""
        mask = torch.ones_like(img)
        
        i = (self.img_size - self.mask_size) // 2
        
        mask[:, i:i+self.mask_size, i:i+self.mask_size] = 0
        
        masked_img = img * (1 - mask) + mask
        masked_part = img * mask
        
        return masked_img, masked_part
    
    def apply_frame_mask(self, img):
        """Mask edge part of image(for frame generation)"""
        mask = torch.ones_like(img)
        i = (self.img_size - self.mask_size) // 2
        mask[:, i : i + self.mask_size, i : i + self.mask_size] = 0
        
        masked_img = img * (1 - mask) + mask
        
        return masked_img, i

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        if self.mode == "train":
            masked_img, aux = self.apply_external_random_mask(img)
        else:
            masked_img, aux = self.apply_external_center_mask(img)

        return img, masked_img, aux

    def __len__(self):
        return len(self.files)

#     def visualize_samples(self, num_samples=5, save_dir='./'):
#         os.makedirs(save_dir, exist_ok=True)
#         for i in range(num_samples):
#             img, masked_img, frame = self[i]
            
#             def tensor_to_pil(t):
#                 return transforms.ToPILImage()(t.clamp(0, 1))
            
#             original_img = tensor_to_pil(img)
#             masked_img = tensor_to_pil(masked_img)
#             # frame = tensor_to_pil(frame)
            
#             combined_img = Image.new('RGB', (self.img_size * 3, self.img_size))
#             combined_img.paste(original_img, (0, 0))
#             combined_img.paste(masked_img, (self.img_size, 0))
#             # combined_img.paste(frame, (self.img_size * 2, 0))
            
#             combined_img.save(os.path.join(save_dir, f'sample_{i}.png'))
        
#         print(f"Save to: {save_dir}")

# if __name__ == "__main__":
#     transforms_ = [
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#     ]

#     dataset = ImageDataset("data/img_align_celeba", transforms_=transforms_)
    
#     dataset.visualize_samples(save_dir='./samples')

#     dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

#     for i, (imgs, masked_imgs, aux) in enumerate(dataloader):
#         print("Images shape:", imgs.shape)
#         print("Masked images shape:", masked_imgs.shape)
#         print("Aux shape:", aux.shape)
#         break
