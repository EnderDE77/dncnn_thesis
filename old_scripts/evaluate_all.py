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

def to_fourier(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    return magnitude.astype(np.float32)

def evaluate_model(model, test_dir, sigma, device, fourier=False):
    model.eval()
    psnr_vals, ssim_vals, mse_vals, mae_vals, rmse_vals = [], [], [], [], []
    
    with torch.no_grad():
        for fname in sorted(os.listdir(test_dir)):
            if not fname.lower().endswith(('.png', '.bmp')):
                continue
            img = cv2.imread(os.path.join(test_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            img_f = img.astype(np.float32) / 255.0
            
            if fourier:
                clean = to_fourier(img_f)
            else:
                clean = img_f
                
            np.random.seed(42)
            noisy = clean + np.random.randn(*clean.shape) * (sigma / 255.0)
            noisy = np.clip(noisy, 0, 1)
            
            noisy_t = torch.tensor(noisy).unsqueeze(0).unsqueeze(0).float().to(device)
            out = model(noisy_t).squeeze().cpu().numpy()
            out = np.clip(out, 0, 1)
            
            mse = np.mean((clean - out) ** 2)
            mae = np.mean(np.abs(clean - out))
            rmse = np.sqrt(mse)
            
            psnr_vals.append(psnr(clean, out, data_range=1.0))
            ssim_vals.append(ssim(clean, out, data_range=1.0))
            mse_vals.append(mse)
            mae_vals.append(mae)
            rmse_vals.append(rmse)
    
    return {
        'psnr': float(np.mean(psnr_vals)),
        'ssim': float(np.mean(ssim_vals)),
        'mse': float(np.mean(mse_vals)),
        'rmse': float(np.mean(rmse_vals)),
        'mae': float(np.mean(mae_vals))
    }

def save_visual(model, test_dir, sigma, device, domain, fourier=False):
    out_dir = f'results/visuals_{domain}_sigma{sigma}'
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    count = 0
    with torch.no_grad():
        for fname in sorted(os.listdir(test_dir)):
            if not fname.lower().endswith(('.png', '.bmp')):
                continue
            if count >= 3:  # save 3 sample images per model
                break
            img = cv2.imread(os.path.join(test_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_f = img.astype(np.float32) / 255.0
            if fourier:
                clean = to_fourier(img_f)
            else:
                clean = img_f
            np.random.seed(42)
            noisy = clean + np.random.randn(*clean.shape) * (sigma / 255.0)
            noisy = np.clip(noisy, 0, 1)
            noisy_t = torch.tensor(noisy).unsqueeze(0).unsqueeze(0).float().to(device)
            out = model(noisy_t).squeeze().cpu().numpy()
            out = np.clip(out, 0, 1)
            comparison = np.hstack([
                (noisy * 255).astype(np.uint8),
                (out * 255).astype(np.uint8),
                (clean * 255).astype(np.uint8)
            ])
            cv2.imwrite(f'{out_dir}/{fname}_comparison.png', comparison)
            count += 1

def main():
    device = torch.device('cuda')
    os.makedirs('results', exist_ok=True)

    print("="*80)
    print("FULL EVALUATION — IMAGE DOMAIN vs FOURIER DOMAIN")
    print("="*80)
    print(f"\n{'Domain':<12} {'Sigma':<8} {'Dataset':<10} {'PSNR':<10} {'SSIM':<10} {'RMSE':<10} {'MAE':<10}")
    print("-"*80)

    results = {}

    for sigma in [15, 25, 50]:
        for domain, fourier, model_path in [
            ('Image', False, f'models/dncnn_sigma{sigma}_best.pth'),
            ('Fourier', True, f'models/dncnn_fourier_sigma{sigma}_best.pth')
        ]:
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                continue

            model = DnCNN(depth=17, channels=1).to(device)
            model.load_state_dict(torch.load(model_path))

            for test_name, test_dir in [('BSD68', 'testsets/BSD68'), ('Set12', 'testsets/Set12')]:
                metrics = evaluate_model(model, test_dir, sigma, device, fourier=fourier)
                key = f"{domain}_sigma{sigma}_{test_name}"
                results[key] = metrics
                print(f"{domain:<12} {sigma:<8} {test_name:<10} {metrics['psnr']:<10.2f} {metrics['ssim']:<10.4f} {metrics['rmse']:<10.5f} {metrics['mae']:<10.5f}")

            # Save visual comparisons from Set12
            save_visual(model, 'testsets/Set12', sigma, device, f'{domain.lower()}', fourier=fourier)

    print("="*80)
    print("\nVisual comparisons saved to results/visuals_* folders")
    
    # Save results to JSON
    import json
    with open('results/final_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Full results saved to results/final_evaluation.json")

if __name__ == '__main__':
    main()