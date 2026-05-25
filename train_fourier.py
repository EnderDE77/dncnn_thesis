import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import json
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

def to_fourier(img):
    # Convert image to log magnitude of shifted FFT
    # as specified by professor: log(fftshift(fft2(img)))
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    # Normalize to [0, 1]
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    return magnitude.astype(np.float32)

def from_fourier_eval(clean_img, denoised_fourier, noisy_fourier):
    # For evaluation we compare in Fourier domain directly
    # PSNR/SSIM computed on Fourier magnitude maps
    return denoised_fourier

class FourierDenoisingDataset(Dataset):
    def __init__(self, image_dir, patch_size=64, sigma=25, augment=True):
        self.patches = []
        self.sigma = sigma
        for fname in os.listdir(image_dir):
            if fname.lower().endswith(('.png', '.bmp', '.jpg')):
                img = cv2.imread(os.path.join(image_dir, fname), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = img.astype(np.float32) / 255.0
                # Convert full image to Fourier domain
                fourier = to_fourier(img)
                h, w = fourier.shape
                for i in range(0, h - patch_size, patch_size // 2):
                    for j in range(0, w - patch_size, patch_size // 2):
                        patch = fourier[i:i+patch_size, j:j+patch_size]
                        self.patches.append(patch)
                        if augment:
                            self.patches.append(np.fliplr(patch).copy())
                            self.patches.append(np.flipud(patch).copy())

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        clean = torch.tensor(self.patches[idx]).unsqueeze(0)
        noise = torch.randn_like(clean) * (self.sigma / 255.0)
        noisy = clean + noise
        return noisy, clean

def evaluate_fourier(model, test_dir, sigma, device):
    model.eval()
    psnr_vals, ssim_vals = [], []
    with torch.no_grad():
        for fname in os.listdir(test_dir):
            if not fname.lower().endswith(('.png', '.bmp')):
                continue
            img = cv2.imread(os.path.join(test_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_f = img.astype(np.float32) / 255.0
            clean_fourier = to_fourier(img_f)
            np.random.seed(42)
            noise = np.random.randn(*clean_fourier.shape) * (sigma / 255.0)
            noisy_fourier = np.clip(clean_fourier + noise, 0, 1)
            noisy_t = torch.tensor(noisy_fourier).unsqueeze(0).unsqueeze(0).float().to(device)
            out = model(noisy_t).squeeze().cpu().numpy()
            out = np.clip(out, 0, 1)
            psnr_vals.append(psnr(clean_fourier, out, data_range=1.0))
            ssim_vals.append(ssim(clean_fourier, out, data_range=1.0))
    return np.mean(psnr_vals), np.mean(ssim_vals)

def train_fourier(sigma=25):
    device = torch.device('cuda')
    epochs = 30
    batch_size = 64
    lr = 1e-3

    print(f"\n{'='*50}")
    print(f"Training Fourier-domain DnCNN for sigma={sigma}")
    print(f"{'='*50}")

    dataset = FourierDenoisingDataset('testsets/BSD68', sigma=sigma)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Dataset size: {len(dataset)} patches")

    model = DnCNN(depth=17, channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()

    log = {'sigma': sigma, 'domain': 'fourier', 'epochs': [], 'train_loss': [],
           'bsd68_psnr': [], 'bsd68_ssim': [], 'set12_psnr': [], 'set12_ssim': []}
    best_psnr = 0

    for epoch in range(epochs):
        model.train()
        losses = []
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            loss = criterion(output, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        avg_loss = np.mean(losses)
        bsd68_psnr, bsd68_ssim = evaluate_fourier(model, 'testsets/BSD68', sigma, device)
        set12_psnr, set12_ssim = evaluate_fourier(model, 'testsets/Set12', sigma, device)

        log['epochs'].append(epoch + 1)
        log['train_loss'].append(float(avg_loss))
        log['bsd68_psnr'].append(float(bsd68_psnr))
        log['bsd68_ssim'].append(float(bsd68_ssim))
        log['set12_psnr'].append(float(set12_psnr))
        log['set12_ssim'].append(float(set12_ssim))

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | BSD68 PSNR: {bsd68_psnr:.2f} | Set12 PSNR: {set12_psnr:.2f} | SSIM: {set12_ssim:.4f}")

        if bsd68_psnr > best_psnr:
            best_psnr = bsd68_psnr
            torch.save(model.state_dict(), f'models/dncnn_fourier_sigma{sigma}_best.pth')
            print(f"  Saved best model")

    with open(f'results/training_log_fourier_sigma{sigma}.json', 'w') as f:
        json.dump(log, f)

    print(f"\nFourier sigma={sigma} complete. Best BSD68 PSNR: {best_psnr:.2f} dB")
    return log

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    all_logs = {}
    for sigma in [15, 25, 50]:
        all_logs[sigma] = train_fourier(sigma)
    with open('results/all_training_logs_fourier.json', 'w') as f:
        json.dump(all_logs, f)
    print("\nAll Fourier training complete.")