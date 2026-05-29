# Master experiment configuration
# All combinations are generated from these settings

SIGMAS = [15, 25, 50]
KERNEL_SIZES = [3, 5]
ACTIVATIONS = ['relu', 'leakyrelu']
BATCH_SIZES = [8, 16, 32, 64]
DOMAINS = ['image', 'fourier_complex']
EPOCHS = 30
PATCH_SIZE = 64
LEARNING_RATE = 1e-3
TRAIN_DIR = 'testsets/BSD68'
TEST_DIR_VAL = 'testsets/BSD68'
TEST_DIR_FINAL = 'testsets/Set12'
SEED = 42

# Generate all combinations
def get_all_configs():
    configs = []
    for sigma in SIGMAS:
        for kernel in KERNEL_SIZES:
            for activation in ACTIVATIONS:
                for batch_size in BATCH_SIZES:
                    for domain in DOMAINS:
                        name = f"sigma{sigma}_k{kernel}_act{activation}_bs{batch_size}_{domain}"
                        configs.append({
                            'name': name,
                            'sigma': sigma,
                            'kernel_size': kernel,
                            'activation': activation,
                            'batch_size': batch_size,
                            'domain': domain,
                        })
    return configs

# Published benchmarks for comparison
PUBLISHED_BENCHMARKS = {
    'BM3D': {'sigma15': 31.73, 'sigma25': 28.61, 'sigma50': 25.62},
    'DnCNN': {'sigma15': 31.73, 'sigma25': 29.23, 'sigma50': 26.23},
    'FFDNet': {'sigma15': 31.63, 'sigma25': 29.19, 'sigma50': 26.05},
}