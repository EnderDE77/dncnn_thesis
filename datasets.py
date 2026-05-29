import torch
import numpy as np
import cv2
import os
from torch.utils.data import Dataset
from config import SEED

class ImageDomainDataset(Dataset):
    def __init__(self, image_dir, patch_size=64, sigma=25, augment=True):
        self.patches = []
        self.sigma = sigma
        for fname in sorted(os.listdir(image_dir)):
            if not fname.lower().endswith(('.png', '.bmp', '.jpg')):
                continue
            img = cv2.imread(os.path.join(image_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = img.astype(np.float32) / 255.0
            h, w = img.shape
            for i in range(0, h - patch_size, patch_size // 2):
                for j in range(0, w - patch_size, patch_size // 2):
                    patch = img[i:i+patch_size, j:j+patch_size]
                    self.patches.append(patch)
                    if augment:
                        self.patches.append(np.fliplr(patch).copy())
                        self.patches.append(np.flipud(patch).copy())

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        clean = torch.tensor(self.patches[idx]).unsqueeze(0)
        noise = torch.randn_like(clean) * (self.sigma / 255.0)
        return clean + noise, clean


class ComplexFourierDataset(Dataset):
    def __init__(self, image_dir, patch_size=64, sigma=25, augment=True):
        self.patches = []
        self.sigma = sigma
        for fname in sorted(os.listdir(image_dir)):
            if not fname.lower().endswith(('.png', '.bmp', '.jpg')):
                continue
            img = cv2.imread(os.path.join(image_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = img.astype(np.float32) / 255.0

            # Full complex FFT — keep real and imaginary
            fft = np.fft.fftshift(np.fft.fft2(img))
            real = np.real(fft).astype(np.float32)
            imag = np.imag(fft).astype(np.float32)

            # Normalize each channel independently
            real = (real - real.min()) / (real.max() - real.min() + 1e-8)
            imag = (imag - imag.min()) / (imag.max() - imag.min() + 1e-8)

            two_channel = np.stack([real, imag], axis=0)  # (2, H, W)

            h, w = real.shape
            for i in range(0, h - patch_size, patch_size // 2):
                for j in range(0, w - patch_size, patch_size // 2):
                    patch = two_channel[:, i:i+patch_size, j:j+patch_size]
                    self.patches.append(patch)
                    if augment:
                        self.patches.append(np.flip(patch, axis=2).copy())
                        self.patches.append(np.flip(patch, axis=1).copy())

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        clean = torch.tensor(self.patches[idx])  # (2, 64, 64)
        # Add independent noise to real and imaginary channels
        noise = torch.randn_like(clean) * (self.sigma / 255.0)
        return clean + noise, clean


def get_dataset(domain, image_dir, patch_size, sigma, augment=True):
    if domain == 'image':
        return ImageDomainDataset(image_dir, patch_size, sigma, augment)
    elif domain == 'fourier_complex':
        return ComplexFourierDataset(image_dir, patch_size, sigma, augment)
    else:
        raise ValueError(f"Unknown domain: {domain}")