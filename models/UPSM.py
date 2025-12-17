import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import cv2
import numpy as np


class GaussianBlur2d(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__()  
        self.sigma = sigma
        self.kernel_size = kernel_size  
        self.kernel = self._create_gaussian_kernel(kernel_size, sigma)

    def _create_gaussian_kernel(self, kernel_size, sigma):
        x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        gauss /= gauss.sum()
        return gauss.view(1, 1, 1, -1), gauss.view(1, 1, -1, 1)

    def forward(self, x):
        weight_h, weight_w = self.kernel
        # Repeat the kernel for each channel
        weight_h = weight_h.repeat(x.size(1), 1, 1, 1)
        weight_w = weight_w.repeat(x.size(1), 1, 1, 1)

        # Apply horizontal convolution
        x_blur_h = F.conv2d(x, weight_h.to(x.device), padding=(0, self.kernel_size // 2), groups=x.shape[1])
        # Apply vertical convolution
        x_blur = F.conv2d(x_blur_h, weight_w.to(x.device), padding=(self.kernel_size // 2, 0), groups=x.shape[1])

        return x_blur


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_features=32):  
        super(UNet, self).__init__()
        
        self.encoder1 = self._block(in_channels, num_features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = self._block(num_features, num_features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = self._block(num_features * 2, num_features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = self._block(num_features * 4, num_features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(num_features * 8, num_features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((num_features * 4) * 2, num_features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(num_features * 4, num_features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((num_features * 2) * 2, num_features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(num_features * 2, num_features, kernel_size=2, stride=2)
        self.decoder1 = self._block(num_features * 2, num_features)
        
        self.conv = nn.Conv2d(num_features, out_channels, kernel_size=1)
        
    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        output = self.conv(dec1)
        return output


class PatchSmoothingModule(nn.Module):
    def __init__(self, embed_dim=768, out_chans=3, input_size=128, output_size=192, patch_size=16, overlap_size=8,
                 bias=True):
        super().__init__()
        self.use_bias = bias
        self.patch_size = patch_size
        self.input_size = input_size
        self.output_size = output_size

        self.embed_dim = embed_dim
        kernel_size = patch_size + overlap_size * 2
        padding_size = overlap_size
        self.proj = nn.ConvTranspose2d(embed_dim, out_chans, bias=False, kernel_size=kernel_size, stride=patch_size,
                                       padding=padding_size)

        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(1, out_chans, kernel_size, kernel_size),
                                           requires_grad=True)
            nn.init.constant_(self.bias, 0)

        self.unet = UNet(out_chans, out_chans)

        self.mask = torch.ones(1, 1, output_size // patch_size, output_size // patch_size)
        p = ((output_size - input_size) // 2) // patch_size
        self.mask[:, :, p:-p, p:-p] = 0

        self.mask_weight = F.conv_transpose2d(self.mask.detach(), torch.ones([1, out_chans, kernel_size, kernel_size]),
                                              bias=None, stride=patch_size, padding=padding_size)
        self.mask_weight[self.mask_weight != 0] = 1 / self.mask_weight[self.mask_weight != 0]

        self.padding_size = padding_size

    def forward(self, x, gt_inner):
        assert x.size(1) == (self.output_size // self.patch_size) ** 2
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.output_size // self.patch_size)
        x = self.proj(x)
        
        # Apply UNet for feature extraction and fusion
        x = self.unet(x)
        
        if self.use_bias:
            bias = F.conv_transpose2d(self.mask.detach().to(x.device), self.bias, bias=None, stride=self.patch_size,
                                    padding=self.padding_size)
            x = x + bias
        
        x_original = x.clone()
        
        x_weighted = x * self.mask_weight.to(x.device)
        
        p = (self.output_size - self.input_size) // 2
        outer_mask = torch.ones_like(x)
        outer_mask[:, :, p:-p, p:-p] = 0
        x = x_weighted * (1 - outer_mask) + x_original * outer_mask
        
        x[:, :, p:-p, p:-p] = gt_inner[:, :, p:-p, p:-p]
        
        
        return x


# Test the module
if __name__ == "__main__":
    x1 = torch.randn([1, 144, 768])
    x2 = torch.randn([1, 3, 192, 192])

    model = PatchSmoothingModule(embed_dim=768, out_chans=3, input_size=128, output_size=192, patch_size=16,
                                 overlap_size=8, bias=True)
    output = model(x1, x2)
    print(f"Final Output Shape: {output.shape}")



