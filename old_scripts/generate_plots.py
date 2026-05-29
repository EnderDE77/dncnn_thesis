import json
import numpy as np
import matplotlib.pyplot as plt
import os

def load_log(path):
    with open(path) as f:
        return json.load(f)

def plot_metric(logs_image, logs_fourier, metric_key, ylabel, title, filename):
    plt.figure(figsize=(10, 5))
    sigmas = [15, 25, 50]
    colors = ['blue', 'green', 'red']
    for sigma, color in zip(sigmas, colors):
        key = str(sigma)
        if key in logs_image and metric_key in logs_image[key]:
            plt.plot(logs_image[key]['epochs'], logs_image[key][metric_key],
                    color=color, linestyle='-', label=f'Image domain σ={sigma}')
        if key in logs_fourier and metric_key in logs_fourier[key]:
            plt.plot(logs_fourier[key]['epochs'], logs_fourier[key][metric_key],
                    color=color, linestyle='--', label=f'Fourier domain σ={sigma}')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/{filename}.png', dpi=150)
    plt.close()
    print(f"Saved results/{filename}.png")

def plot_image_only(logs_image, metric_key, ylabel, title, filename):
    plt.figure(figsize=(10, 5))
    sigmas = [15, 25, 50]
    colors = ['blue', 'green', 'red']
    for sigma, color in zip(sigmas, colors):
        key = str(sigma)
        if key in logs_image and metric_key in logs_image[key]:
            plt.plot(logs_image[key]['epochs'], logs_image[key][metric_key],
                    color=color, linestyle='-', label=f'σ={sigma}')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/{filename}.png', dpi=150)
    plt.close()
    print(f"Saved results/{filename}.png")

def print_summary_table(logs_image, logs_fourier):
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"{'Domain':<15} {'Sigma':<8} {'BSD68 PSNR':<14} {'Set12 PSNR':<14} {'Set12 SSIM':<12}")
    print("-"*70)
    for sigma in [15, 25, 50]:
        key = str(sigma)
        if key in logs_image:
            log = logs_image[key]
            best_idx = np.argmax(log['bsd68_psnr'])
            print(f"{'Image':<15} {sigma:<8} {log['bsd68_psnr'][best_idx]:<14.2f} {log['set12_psnr'][best_idx]:<14.2f} {log['set12_ssim'][best_idx]:<12.4f}")
    print("-"*70)
    for sigma in [15, 25, 50]:
        key = str(sigma)
        if key in logs_fourier:
            log = logs_fourier[key]
            best_idx = np.argmax(log['bsd68_psnr'])
            print(f"{'Fourier':<15} {sigma:<8} {log['bsd68_psnr'][best_idx]:<14.2f} {log['set12_psnr'][best_idx]:<14.2f} {log['set12_ssim'][best_idx]:<12.4f}")
    print("="*70)

def main():
    os.makedirs('results', exist_ok=True)

    # Load image domain logs
    img_log_path = 'results/all_training_logs.json'
    fourier_log_path = 'results/all_training_logs_fourier.json'

    logs_image = {}
    logs_fourier = {}

    if os.path.exists(img_log_path):
        with open(img_log_path) as f:
            raw = json.load(f)
        # normalize keys to strings
        logs_image = {str(k): v for k, v in raw.items()}

    if os.path.exists(fourier_log_path):
        with open(fourier_log_path) as f:
            raw = json.load(f)
        logs_fourier = {str(k): v for k, v in raw.items()}

    if not logs_image and not logs_fourier:
        print("No training logs found. Run training first.")
        return

    # Plot training loss
    if logs_image:
        plot_image_only(logs_image, 'train_loss', 'MSE Loss', 
                       'Training Loss per Epoch (Image Domain)', 'plot_train_loss')

    # Plot BSD68 PSNR
    if logs_image or logs_fourier:
        plot_metric(logs_image, logs_fourier, 'bsd68_psnr', 'PSNR (dB)',
                   'BSD68 PSNR per Epoch — Image vs Fourier Domain', 'plot_bsd68_psnr')

    # Plot Set12 PSNR
    if logs_image or logs_fourier:
        plot_metric(logs_image, logs_fourier, 'set12_psnr', 'PSNR (dB)',
                   'Set12 PSNR per Epoch — Image vs Fourier Domain', 'plot_set12_psnr')

    # Plot Set12 SSIM
    if logs_image or logs_fourier:
        plot_metric(logs_image, logs_fourier, 'set12_ssim', 'SSIM',
                   'Set12 SSIM per Epoch — Image vs Fourier Domain', 'plot_set12_ssim')

    # Print summary table
    print_summary_table(logs_image, logs_fourier)

if __name__ == '__main__':
    main()