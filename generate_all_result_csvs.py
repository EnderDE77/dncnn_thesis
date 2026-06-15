import json
import csv

data = {}

with open('results/all_results.json', 'r') as f:
  data = json.load(f)

data_image = {k:data[k] for k in data if 'image' in k}
data_fourier = {k:data[k] for k in data if 'fourier' in k}

print(data[list(data.keys())[0]].keys())

#get metadata from each data, using data
metadata = []
metadata.append([
  'Name',
  'Sigma',
  'Kernel Size',
  'Activator',
  'Batch Size',
  'Domain',
  'Channels',
  'Total runtime (seconds)',
  'Total runtime (minutes)',
  'Best Validation PSNR',
  'Final Validation PSNR',
  'Final Validation SSIM',
  'Final Validation RMSE',
  'Final Validation MAE',
  'Final Validation MSE',
  'Final Set12 PSNR',
  'Final Set12 SSIM',
  'Final Set12 RMSE',
  'Final Set12 MAE',
  'Final Set12 MSE'
])

for model in data.values():
  model_row = [
  model['name'],
  model['sigma'],
  model['kernel_size'],
  model['activation'],
  model['batch_size'],
  model['domain'],
  model['channels'],
  model['total_runtime_seconds'],
  model['total_runtime_minutes'],
  model['best_val_psnr'],
  model['final_val']['psnr'],
  model['final_val']['ssim'],
  model['final_val']['rmse'],
  model['final_val']['mae'],
  model['final_val']['mse'],
  model['final_set12']['psnr'],
  model['final_set12']['ssim'],
  model['final_set12']['rmse'],
  model['final_set12']['mae'],
  model['final_set12']['mse']
 ]
  metadata.append(model_row)

all_model_results = {'metadata' : metadata }

def get_results_hyp(param, domain):

  group = []
  dt = {}
  if domain == 'image':
    dt = data_image
  elif domain == 'fourier':
    dt = data_fourier
  else:
    raise ValueError("Domain must be either 'image' or 'fourier'")

  group.append(['Epoch'])
  group[0].extend(list(dt.keys()))

  for i in range(30):
    row = []
    row.append(i + 1)
    for model in dt.values():
      row.append(model[param][i])
    group.append(row)
  return group

metrics = ['train_loss', 'val_psnr', 'val_ssim', 'val_rmse', 'val_mae', 'set12_psnr', 'set12_ssim', 'set12_rmse', 'set12_mae', 'epoch_times']
domains = ['image', 'fourier']

combos = [(metric, domain) for metric in metrics for domain in domains]

for metric, domain in combos:
  results = get_results_hyp(metric, domain)
  all_model_results[f'{metric}_{domain}'] = results

for key, value in all_model_results.items():
  with open(f'results/csvs/{key}.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(value)

