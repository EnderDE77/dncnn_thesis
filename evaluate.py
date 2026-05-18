import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class DnCNN(nn.Module):
    def __init__(self, depth=17, channels=1):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(channels, 64, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, channels, kernel_size=3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)

def evaluate():
    device = torch.device('cuda')
    sigma = 25

    model = DnCNN(depth=17, channels=1).to(device)
    model.load_state_dict(torch.load('models/dncnn_best.pth'))
    model.eval()

    test_dir = 'testsets/Set12'
    os.makedirs('results', exist_ok=True)

    psnr_vals, ssim_vals = [], []

    with torch.no_grad():
        for fname in sorted(os.listdir(test_dir)):
            if not fname.lower().endswith(('.png', '.bmp')):
                continue
            img = cv2.imread(os.path.join(test_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            clean = img.astype(np.float32) / 255.0
            np.random.seed(42)
            noisy = clean + np.random.randn(*clean.shape) * (sigma / 255.0)
            noisy_clipped = np.clip(noisy, 0, 1)

            noisy_t = torch.tensor(noisy).unsqueeze(0).unsqueeze(0).float().to(device)
            out = model(noisy_t).squeeze().cpu().numpy()
            out = np.clip(out, 0, 1)

            p = psnr(clean, out, data_range=1.0)
            s = ssim(clean, out, data_range=1.0)
            psnr_vals.append(p)
            ssim_vals.append(s)
            print(f"{fname}: PSNR={p:.2f} dB | SSIM={s:.4f}")

            # Save visual comparison
            comparison = np.hstack([
                (noisy_clipped * 255).astype(np.uint8),
                (out * 255).astype(np.uint8),
                img
            ])
            cv2.imwrite(f'results/{fname}_comparison.png', comparison)

    print(f"\nSet12 Average PSNR: {np.mean(psnr_vals):.2f} dB")
    print(f"Set12 Average SSIM: {np.mean(ssim_vals):.4f}")

if __name__ == '__main__':
    evaluate()