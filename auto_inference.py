#!/usr/bin/env python3
import argparse
import os
import subprocess
import time
import glob
import traceback

from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

from inference import inference_checkpoint_all_datasets

# Configurable parameters
GPU_UTIL_THRESHOLD = 10  # If GPU utilization is above this (in %), we wait.
CHECK_INTERVAL = 60  # Seconds to wait between GPU checks

# For the experiment directory, you may compute this based on your structure.
EXPERIMENT_RUN_DIR = '/inference/2025-02/23/00.04.32/'


def is_gpu_idle():
    """
    Check GPU utilization using nvidia-smi.
    Returns True if all GPUs are below the utilization threshold.
    """
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        # If multiple GPUs, we check each one.
        utilizations = [int(x.strip()) for x in result.strip().split('\n') if x.strip().isdigit()]
        for util in utilizations:
            if util > GPU_UTIL_THRESHOLD:
                print(f"GPU utilization is {util}%, which is above threshold.")
                return False
        return True
    except Exception as e:
        print("Error checking GPU utilization:", e)
        return False


def run_inference(checkpoint_path, experiment_name):
    """
    Wait until GPU is idle and run inference on the given checkpoint.
    Then, create a marker file to indicate that this checkpoint has been processed.
    """
    # Wait for GPU to be idle
    while not is_gpu_idle():
        print("Waiting for GPU to be idle...")
        time.sleep(CHECK_INTERVAL)

    # Run inference takes directory, not file
    if checkpoint_path.endswith('model.safetensors'):
        checkpoint_path = os.path.dirname(checkpoint_path)

    print(f"Running inference on checkpoint: {checkpoint_path}")

    results_dir = inference_checkpoint_all_datasets(checkpoint_path, experiment_name)
    try:
        # After successful inference, mark the checkpoint as processed
        marker_file = create_marker_file_str(checkpoint_path)
        with open(marker_file, 'w') as f:
            f.write(results_dir)
        print(f"Inference completed for '{checkpoint_path}' and marker file '{marker_file}' created.\n\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running inference on {checkpoint_path}: {e}")
    except Exception as e:
        traceback.print_exc()
        print(f"Unexpected error running inference on {checkpoint_path}: {e}")


def create_marker_file_str(checkpoint_dir):
    return os.path.join(checkpoint_dir, 'inference_run_dir.txt')


def has_marker_file(checkpoint):
    cp_dir = os.path.dirname(checkpoint)
    marker_file = create_marker_file_str(cp_dir)
    return os.path.exists(marker_file)


def find_unprocessed_checkpoints(root_dir, ignore_markfiles):
    """
    Recursively search for checkpoint files (model.safetensors) in a folder structure like:
    <root_dir>/.../checkpoints/<something>/model.safetensors
    and return those that have not been processed (i.e. lack the marker file).
    """
    pattern = os.path.join(root_dir, '**', 'checkpoints', '*', 'model.safetensors')
    checkpoint_files = glob.glob(pattern, recursive=True)
    unprocessed = []
    for cp in checkpoint_files:
        cp_dirname = os.path.dirname(cp)

        if cp_dirname.split("/")[-1] == "colbert":
            print(f"Skipping {cp} directory because batch step count is not there.")
            continue

        if ignore_markfiles or not has_marker_file(cp):
            unprocessed.append(cp_dirname)
    return sorted(unprocessed)  # Sort for consistent processing


class CheckpointEventHandler(FileSystemEventHandler):
    """
    Watchdog event handler that reacts to new checkpoint files.
    """

    def __init__(self, root_dir, experiment_name):
        super().__init__()
        self.root_dir = root_dir
        self.experiment_name = experiment_name

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('model.safetensors'):
            print(f"New checkpoint detected: {event.src_path}")
            run_inference(os.path.dirname(event.src_path), self.experiment_name)
        else:
            print(f"Ignoring event: {event}")


def main():
    parser = argparse.ArgumentParser(description='Automated Inference Runner')
    parser.add_argument('--experiment_name', type=str,
                        help='Name of current experiment to look for data in experiments/')
    parser.add_argument('--ignore-markfiles', action='store_true',)
    args = parser.parse_args()

    root_dir = os.path.join('experiments', args.experiment_name, 'train')

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)  # Directory for watching might not exist yet

    # Process any existing unprocessed checkpoints
    unprocessed = find_unprocessed_checkpoints(root_dir, args.ignore_markfiles)
    print(f"Found {len(unprocessed)} unprocessed checkpoint(s):")
    print('\n\t'.join(unprocessed))
    for cp_path in unprocessed:
        run_inference(cp_path, args.experiment_name)

    # Set up watchdog observer to monitor for new checkpoint files
    event_handler = CheckpointEventHandler(root_dir, args.experiment_name)
    observer = PollingObserver()
    observer.schedule(event_handler, path=root_dir, recursive=True)
    observer.start()
    print(f"Watching directory {root_dir} for new checkpoints...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == '__main__':
    main()
