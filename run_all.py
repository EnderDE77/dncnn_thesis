import subprocess
import time
import os
import json
import urllib.request

NTFY_TOPIC = "erlis_dncnn_thesis_2026"

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
        pass

def run_script(script_name):
    print(f"\n{'='*70}")
    print(f"STARTING: {script_name}")
    print(f"Time: {time.strftime('%H:%M:%S')}")
    print(f"{'='*70}")

    start = time.time()
    result = subprocess.run(
        ['python', script_name],
        capture_output=False,
        text=True
    )
    elapsed = time.time() - start

    status = 'SUCCESS' if result.returncode == 0 else 'FAILED'
    print(f"\n{status}: {script_name}")
    print(f"Runtime: {elapsed/60:.1f} minutes")
    notify(
      f"{status}: {script_name}",
      f"Runtime: {elapsed/60:.1f} min"
  )
    return {
        'script': script_name,
        'status': status,
        'runtime_minutes': round(elapsed / 60, 2),
        'returncode': result.returncode
    }

if __name__ == '__main__':
    print("\n" + "="*70)
    print("DnCNN FULL EXPERIMENT PIPELINE")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    pipeline_start = time.time()
    run_log = []

    # Step 1 — Install dependencies
    print("\nInstalling dependencies...")
    subprocess.run(['pip', 'install', 'python-docx', '-q'])

    # # Step 2 — Run all training experiments
    # result = run_script('train_all.py')
    # run_log.append(result)

    # Step 3 — Generate all plots and tables
    result = run_script('generate_all_plots.py')
    run_log.append(result)

    # Step 4 — Git commit
    print("\nCommitting to GitHub...")
    os.system('git add .')
    os.system('git commit -m "Full experiment pipeline complete"')
    os.system('git push')

    # Step 5 — Save pipeline log
    total_time = time.time() - pipeline_start
    pipeline_summary = {
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_runtime_hours': round(total_time / 3600, 2),
        'scripts': run_log
    }

    with open('results/pipeline_log.json', 'w') as f:
        json.dump(pipeline_summary, f, indent=2)

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"Total runtime: {total_time/3600:.2f} hours")
    print(f"Results: results/all_results.json")
    print(f"Plots: results/plots/")
    print(f"Tables: results/tables/")
    print(f"Word tables: results/tables/all_experiment_tables.docx")
    print(f"{'='*70}")