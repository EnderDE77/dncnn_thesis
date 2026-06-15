import torch
import numpy as np
import cv2
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def evaluate_image_domain(model, test_dir, sigma, device):
    model.eval()
    psnr_vals, ssim_vals, mse_vals, mae_vals = [], [], [], []
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
            noisy_t = torch.tensor(noisy).unsqueeze(0).unsqueeze(0).float().to(device)
            out = model(noisy_t).squeeze().cpu().numpy()
            out = np.clip(out, 0, 1)
            psnr_vals.append(psnr(clean, out, data_range=1.0))
            ssim_vals.append(ssim(clean, out, data_range=1.0))
            mse_vals.append(float(np.mean((clean - out) ** 2)))
            mae_vals.append(float(np.mean(np.abs(clean - out))))
    return {
        'psnr': float(np.mean(psnr_vals)),
        'ssim': float(np.mean(ssim_vals)),
        'mse': float(np.mean(mse_vals)),
        'rmse': float(np.sqrt(np.mean(mse_vals))),
        'mae': float(np.mean(mae_vals))
    }

def evaluate_fourier_complex(model, test_dir, sigma, device):
    model.eval()
    psnr_vals, ssim_vals, mse_vals, mae_vals = [], [], [], []
    with torch.no_grad():
        for fname in sorted(os.listdir(test_dir)):
            if not fname.lower().endswith(('.png', '.bmp')):
                continue
            img = cv2.imread(os.path.join(test_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_f = img.astype(np.float32) / 255.0
            fft = np.fft.fftshift(np.fft.fft2(img_f))
            real = np.real(fft).astype(np.float32)
            imag = np.imag(fft).astype(np.float32)
            real_n = (real - real.min()) / (real.max() - real.min() + 1e-8)
            imag_n = (imag - imag.min()) / (imag.max() - imag.min() + 1e-8)
            clean_2ch = np.stack([real_n, imag_n], axis=0)
            np.random.seed(42)
            noise = np.random.randn(*clean_2ch.shape) * (sigma / 255.0)
            noisy_2ch = clean_2ch + noise
            noisy_t = torch.tensor(noisy_2ch).unsqueeze(0).float().to(device)
            out = model(noisy_t).squeeze().cpu().numpy()
            out = np.clip(out, 0, 1)
            psnr_vals.append(psnr(clean_2ch, out, data_range=1.0))
            ssim_vals.append(ssim(clean_2ch[0], out[0], data_range=1.0))
            mse_vals.append(float(np.mean((clean_2ch - out) ** 2)))
            mae_vals.append(float(np.mean(np.abs(clean_2ch - out))))
    return {
        'psnr': float(np.mean(psnr_vals)),
        'ssim': float(np.mean(ssim_vals)),
        'mse': float(np.mean(mse_vals)),
        'rmse': float(np.sqrt(np.mean(mse_vals))),
        'mae': float(np.mean(mae_vals))
    }

def evaluate_model(model, test_dir, sigma, device, domain):
    if domain == 'image':
        return evaluate_image_domain(model, test_dir, sigma, device)
    elif domain == 'fourier_complex':
        return evaluate_fourier_complex(model, test_dir, sigma, device)
    else:
        raise ValueError(f"Unknown domain: {domain}")