import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import time
from datasets import get_dataset
from evaluate import evaluate_model
from config import get_all_configs, EPOCHS, PATCH_SIZE, LEARNING_RATE
from config import TRAIN_DIR, TEST_DIR_VAL, TEST_DIR_FINAL
import urllib.request

NTFY_TOPIC = "erlis_dncnn_thesis_2026"  # change to your topic

def notify(title, message):
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode('utf-8'),
            headers={'Title': title, 'Priority': 'default'},
            method='POST'
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass  # silent fail — don't interrupt training

class DnCNN(nn.Module):
    def __init__(self, depth=17, channels=1, kernel_size=3, activation='relu'):
        super().__init__()
        padding = kernel_size // 2

        def get_activation():
            if activation == 'relu':
                return nn.ReLU(inplace=True)
            elif activation == 'leakyrelu':
                return nn.LeakyReLU(negative_slope=0.1, inplace=True)
            elif activation == 'prelu':
                return nn.PReLU()
            elif activation == 'elu':
                return nn.ELU(inplace=True)
            else:
                raise ValueError(f"Unknown activation: {activation}")

        layers = []
        layers.append(nn.Conv2d(channels, 64, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(get_activation())
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(get_activation())
        layers.append(nn.Conv2d(64, channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)
    

def train_single(config):
    device = torch.device('cuda')
    name = config['name']
    sigma = config['sigma']
    kernel_size = config['kernel_size']
    activation = config['activation']
    batch_size = config['batch_size']
    domain = config['domain']
    channels = 2 if domain == 'fourier_complex' else 1

    print(f"\n{'='*70}")
    print(f"Training: {name}")
    print(f"Sigma={sigma} | Kernel={kernel_size}x{kernel_size} | "
          f"Act={activation} | Batch={batch_size} | Domain={domain}")
    print(f"{'='*70}")

    # Check if already trained
    model_path = f"models/{name}_best.pth"
    log_path = f"results/logs/log_{name}.json"
    if os.path.exists(log_path):
        print(f"Already trained — skipping. Delete {log_path} to retrain.")
        with open(log_path) as f:
            return json.load(f)

    dataset = get_dataset(domain, TRAIN_DIR, PATCH_SIZE, sigma)
    loader = DataLoader(dataset, batch_size=batch_size,
                       shuffle=True, num_workers=0, pin_memory=True)
    print(f"Dataset: {len(dataset)} patches | Batch size: {batch_size} | "
          f"Steps per epoch: {len(loader)}")

    model = DnCNN(depth=17, channels=channels,
                  kernel_size=kernel_size, activation=activation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()

    log = {
        'name': name,
        'sigma': sigma,
        'kernel_size': kernel_size,
        'activation': activation,
        'batch_size': batch_size,
        'domain': domain,
        'channels': channels,
        'epochs': [],
        'train_loss': [],
        'val_psnr': [],
        'val_ssim': [],
        'val_rmse': [],
        'val_mae': [],
        'set12_psnr': [],
        'set12_ssim': [],
        'set12_rmse': [],
        'set12_mae': [],
        'epoch_times': [],
    }

    best_psnr = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
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
        avg_loss = float(np.mean(losses))

        val = evaluate_model(model, TEST_DIR_VAL, sigma, device, domain)
        s12 = evaluate_model(model, TEST_DIR_FINAL, sigma, device, domain)
        epoch_time = time.time() - epoch_start

        log['epochs'].append(epoch + 1)
        log['train_loss'].append(avg_loss)
        log['val_psnr'].append(val['psnr'])
        log['val_ssim'].append(val['ssim'])
        log['val_rmse'].append(val['rmse'])
        log['val_mae'].append(val['mae'])
        log['set12_psnr'].append(s12['psnr'])
        log['set12_ssim'].append(s12['ssim'])
        log['set12_rmse'].append(s12['rmse'])
        log['set12_mae'].append(s12['mae'])
        log['epoch_times'].append(round(epoch_time, 2))

        print(f"Ep {epoch+1:02d}/{EPOCHS} | "
              f"Loss: {avg_loss:.6f} | "
              f"Val PSNR: {val['psnr']:.2f} | "
              f"Set12 PSNR: {s12['psnr']:.2f} | "
              f"SSIM: {s12['ssim']:.4f} | "
              f"Time: {epoch_time:.1f}s")

        if val['psnr'] > best_psnr:
            best_psnr = val['psnr']
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved best model (PSNR: {best_psnr:.2f})")

    total_time = time.time() - start_time
    log['total_runtime_seconds'] = round(total_time, 2)
    log['total_runtime_minutes'] = round(total_time / 60, 2)
    log['best_val_psnr'] = round(best_psnr, 4)

    # Final evaluation with best model
    model.load_state_dict(torch.load(model_path,
                          weights_only=True))
    log['final_val'] = evaluate_model(model, TEST_DIR_VAL,
                                      sigma, device, domain)
    log['final_set12'] = evaluate_model(model, TEST_DIR_FINAL,
                                        sigma, device, domain)

    os.makedirs('results/logs', exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\n Done: {name}")
    print(f"  Runtime: {total_time/60:.1f} min | "
          f"Best Val PSNR: {best_psnr:.2f} dB")
    return log

if __name__ == '__main__':
    notify(
    "Pipeline Started",
    f"48 experiments starting\n"
    f"Est. total time: ~32 hours\n"
    f"Started: {time.strftime('%H:%M:%S')}"
    )
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)

    configs = get_all_configs()
    total = len(configs)
    print(f"\nTotal experiments: {total}")
    print(f"Estimated time: {total * 40 / 60:.1f} hours\n")

    all_results = {}
    pipeline_start = time.time()

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{total}] Starting: {config['name']}")

        notify(
        f"Starting [{i+1}/{total}]",
        f"{config['name']}\n"
        f"σ={config['sigma']} | "
        f"k={config['kernel_size']} | "
        f"{config['activation']} | "
        f"bs={config['batch_size']} | "
        f"{config['domain']}"
        )
        
        log = train_single(config)
        all_results[config['name']] = log

        fs = log.get('final_set12', {})
        notify(
        f"Done [{i+1}/{total}] — "
        f"{log.get('total_runtime_minutes', 0):.1f}min",
        f"{config['name']}\n"
        f"Set12 PSNR: {fs.get('psnr', 0):.2f} dB\n"
        f"SSIM: {fs.get('ssim', 0):.4f}\n"
        f"Runtime: {log.get('total_runtime_minutes', 0):.1f} min\n"
        f"Est. remaining: "
        f"{(total - i - 1) * 40 / 60:.1f} hours"
    )

        # Save progress after each experiment
        with open('results/all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)

        elapsed = (time.time() - pipeline_start) / 3600
        remaining = (total - i - 1) * 40 / 60
        print(f"Progress: {i+1}/{total} | "
              f"Elapsed: {elapsed:.1f}h | "
              f"Est. remaining: {remaining:.1f}h")

    pipeline_time = time.time() - pipeline_start
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Total pipeline time: {pipeline_time/3600:.2f} hours")
    print(f"Results saved to results/all_results.json")
    print(f"{'='*70}")
    notify(
    "Pipeline Complete!",
    f"All 48 experiments done\n"
    f"Total time: {elapsed:.2f} hours\n"
    f"Check results/all_results.json"
    )