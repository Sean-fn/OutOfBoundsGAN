import torch.nn as nn
import torch.nn.functional as F
from torch import is_grad_enabled
from einops import rearrange
from torch.nn import Transformer
from torch.utils.checkpoint import checkpoint

class VisionTransformerUpsample(nn.Module):
    def __init__(self, in_feat, out_feat, upsample_scale=2, nhead=4, num_layers=2, dim_feedforward=1024):
        super(VisionTransformerUpsample, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_feat, out_feat,
            kernel_size=upsample_scale * 2,
            stride=upsample_scale,
            padding=upsample_scale // 2
        )
        self.transformer = Transformer(
            d_model=out_feat,
            nhead=nhead,
            num_encoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)  # 上採樣
        b, c, h, w = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer(x, x)  # 自注意力機制
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.relu(x)
        return x

class Generator(nn.Module):
    def __init__(self, channels=3):
        super(Generator, self).__init__()

        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.Sequential(
            *downsample(channels, 64, normalize=False),
            *downsample(64, 64),
            *downsample(64, 128),
            *downsample(128, 256),
            *downsample(256, 512),
            nn.Conv2d(512, 4000, 1),
        )
        
        self.upsample_blocks = nn.ModuleList([
            VisionTransformerUpsample(4000, 256),
            VisionTransformerUpsample(256, 128, upsample_scale=4),
            VisionTransformerUpsample(128, 64, upsample_scale=4),
        ])
        
        self.final_layers = nn.Sequential(
            nn.Conv2d(64, channels, 3, 1, 1),
            nn.Tanh()
        )
# ViT use this forward
    def forward(self, x):
        x = self.model(x)
        for block in self.upsample_blocks:
            if is_grad_enabled():  # Only use checkpointing during training
                x = checkpoint(block, x, use_reentrant=True)
            else:
                x = block(x)  # Directly run without checkpointing in inference  #without checkpoint, it will cause OOM
        return self.final_layers(x)


        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers

        self.model = nn.Sequential(
            *downsample(channels, 64, normalize=False),
            *downsample(64, 64),
            *downsample(64, 128),
            *downsample(128, 256),
            *downsample(256, 512),
            nn.Conv2d(512, 4000, 1),
            *upsample(4000, 512),
            *upsample(512, 256),
            *upsample(256, 128),
            *upsample(128, 128),    # For face common this
            *upsample(128, 64),
            nn.Conv2d(64, channels, 3, 1, 1),
            nn.Tanh()
        )

# CNN use this forward
    # def forward(self, x):
    #     return self.model(x)



class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)