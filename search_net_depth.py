"""
Launch 16 fine-tuning tasks in parallel across 8 GPUs.
Each GPU runs 2 concurrent tasks (tox21 + toxcast).
Each task logs to: logs/{logdir}.log

Requirements:
- 8 GPUs (IDs 0–7)
- Conda env 'atomprop' activated
- Sufficient GPU memory for 2 concurrent jobs per GPU

Usage:
    conda activate atomprop
    python launch_16_parallel.py
"""

import os
import sys
import multiprocessing as mp
from itertools import product

# Parameter mapping: config index -> (geat_num_layers, aggr_num_layers)
PARAM_DICT = {
    0: (3, 1),
    1: (4, 1),
    2: (4, 2),
    3: (5, 2),
    4: (5, 3),
    5: (6, 1),
    6: (6, 2),
    7: (6, 3)
}

DATASETS = ["tox21", "toxcast"]
LOG_DIR = "logs"

def run_task_on_gpu(args):
    """
    Run a single fine-tuning task on a specified GPU and log output to a file.
    
    Args:
        args (tuple): (gpu_id, geat_l, aggr_l, dataset)
    """
    gpu_id, geat_l, aggr_l, dataset = args

    # Set visible GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Construct logdir and logfile name
    logdir_name = f"finetune_geat_{dataset}_{geat_l}{aggr_l}"
    log_file_path = os.path.join(LOG_DIR, f"{logdir_name}.log")

    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # Redirect stdout and stderr to log file for this process
    with open(log_file_path, "w") as log_file:
        # Save original stdout/stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Redirect
        sys.stdout = log_file
        sys.stderr = log_file

        try:
            print(f"[GPU {gpu_id}] Starting task: GeAT={geat_l}, Aggr={aggr_l}, Dataset={dataset}")
            print(f"Log file: {log_file_path}")
            print("-" * 60)

            # Import inside subprocess to avoid cross-process contamination
            import configs.config_finetune as cfg
            from finetune_geat import main

            # Override config
            cfg.geat_num_layers = geat_l
            cfg.aggr_num_layers = aggr_l
            cfg.device_str = "cuda:0"  # Because CUDA_VISIBLE_DEVICES maps to logical cuda:0
            cfg.logdir = logdir_name
            cfg.no_pretrain = True  # Adjust if needed

            # Run training
            main(ft_dataset=dataset)

            print(f"[GPU {gpu_id}] Task completed successfully: {logdir_name}")
        except Exception as e:
            print(f"[GPU {gpu_id}] ERROR in task ({geat_l},{aggr_l},{dataset}): {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Restore original stdout/stderr (though process will exit anyway)
            sys.stdout = original_stdout
            sys.stderr = original_stderr

def main():
    # Build 16 tasks: 8 configs × 2 datasets
    tasks = []
    for idx in range(8):
        geat_l, aggr_l = PARAM_DICT[idx]
        gpu_id = idx
        for dataset in DATASETS:
            tasks.append((gpu_id, geat_l, aggr_l, dataset))

    assert len(tasks) == 16, "Expected exactly 16 tasks"

    print(f"Launching {len(tasks)} tasks. Logs will be saved to '{LOG_DIR}/'")
    print("WARNING: Two processes will share each GPU. Monitor memory usage!")

    # Use 'spawn' for CUDA compatibility
    mp.set_start_method("spawn", force=True)

    # Launch all 16 processes in parallel
    with mp.Pool(processes=16) as pool:
        pool.map(run_task_on_gpu, tasks)

    print("All tasks finished. Check logs in './logs/'")

if __name__ == "__main__":
    main()