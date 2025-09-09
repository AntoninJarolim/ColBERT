#!/usr/bin/env python3
import argparse
import os
import subprocess
import time
import glob
import traceback
from collections import defaultdict
from datetime import timedelta

from tqdm import tqdm
from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

from evaluate.extractions import downsample_full_fidelity
from inference import inference_checkpoint_all_datasets, re_evaluate_all

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


def run_inference(checkpoint_path, run_eval, ignore_mark_files, extractions_only_datasets):
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

    datasets = None
    results_dir = inference_checkpoint_all_datasets(datasets, checkpoint_path,
                                                    overwrite_inference=ignore_mark_files,
                                                    run_eval=run_eval,
                                                    extractions_only_datasets=extractions_only_datasets)
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


def find_unprocessed_checkpoints(root_dir, ignore_markfiles, max_per_experiment=5):
    """
    Recursively search for checkpoint files (model.safetensors) in a folder structure like:
    <root_dir>/.../checkpoints/<something>/model.safetensors
    and return those that have not been processed (i.e. lack the marker file).
    """
    pattern = os.path.join(root_dir, '**', 'checkpoints', '*', 'model.safetensors')
    checkpoint_files = glob.glob(pattern, recursive=True)

    # Skip checkpoints with experiments that have way too many checkpoints
    paths_by_experiment = defaultdict(list)
    for cp in checkpoint_files:
        experiment_name = cp.split("/")[3]
        steps = int(cp.split("/")[-2].split('-')[1])
        paths_by_experiment[experiment_name].append((cp, steps))

    # sort by steps
    paths_by_experiment = {k: sorted(v, key=lambda x: x[1]) for k, v in paths_by_experiment.items()}

    # Subsample if too many checkpoints
    for experiment_name, values in paths_by_experiment.items():
        paths = [v[0] for v in values]
        if len(values) > 0 > max_per_experiment:  # -1 means no limit
            paths = downsample_full_fidelity(paths, max_per_experiment)
        paths_by_experiment[experiment_name] = paths

    # Flatten the dictionary back to a list of paths
    checkpoint_files = [item for sublist in paths_by_experiment.values() for item in sublist]

    # Remove already processed checkpoints
    unprocessed = []
    for cp in checkpoint_files:
        cp_dirname_path = os.path.dirname(cp)

        dir_cp = cp_dirname_path.split("/")[-1]
        if dir_cp == "colbert":
            print(f"Skipping {cp} directory because batch step count is not there.")
            continue

        if ignore_markfiles or not has_marker_file(cp):
            unprocessed.append(cp_dirname_path)

    unprocessed = sorted(unprocessed)  # Sort for consistent processing
    return unprocessed


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
            run_inference(os.path.dirname(event.src_path), True, False, False)
        else:
            print(f"Ignoring event: {event}")


def arg_parser():

    parser = argparse.ArgumentParser(description='Automated Inference Runner')
    parser.add_argument('--experiment_name', type=str,
                        help='Name of current experiment to look for data in experiments/')
    parser.add_argument('--ignore-markfiles', action='store_true',)
    parser.add_argument('--extractions-only-datasets', action='store_true',)
    parser.add_argument('--max-per-experiment', type=int, default=5,
                        help='Maximum number of checkpoints to process per experiment (default: 5)'
                        ' -1 means no limit.')
    parser.add_argument('--re-eval-all', action='store_true',
                        help='Re-evaluate all results after processing unprocessed checkpoints.')
    args = parser.parse_args()
    return args

def main():
    args = arg_parser()
    if args.experiment_name == "all":
        root_dir = 'experiments/'
    else:
        root_dir = os.path.join('experiments', args.experiment_name, 'train')

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)  # Directory for watching might not exist yet

    # Configuration from arguments
    ignore_markfiles = args.ignore_markfiles
    extractions_only_datasets = args.extractions_only_datasets
    experiment_name = args.experiment_name
    max_per_experiment = args.max_per_experiment
    re_eval_all = args.re_eval_all

    # Process any existing unprocessed checkpoints
    run_auto_inference(extractions_only_datasets, ignore_markfiles, root_dir, max_per_experiment)
    print("All checkpoints have been processed.")

    # Optionally re-evaluate all results, use when evaluation metrics are updated,
    # and you do not mind waiting for it to finish before watching for new checkpoints
    if re_eval_all:
        re_evaluate_all()

    # Remain active to watch for new checkpoints
    watch_new_checkpoints(experiment_name, root_dir)
    print("Stopped watching for new checkpoints.")


def watch_new_checkpoints(experiment_name, root_dir):
    # Set up watchdog observer to monitor for new checkpoint files
    event_handler = CheckpointEventHandler(root_dir, experiment_name)
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


def run_auto_inference(extractions_only_datasets, ignore_markfiles, root_dir, max_per_experiment):
    while unprocessed := find_unprocessed_checkpoints(root_dir, ignore_markfiles, max_per_experiment):
        print(f"Found {len(unprocessed)} unprocessed checkpoint(s):", '\n\t'.join(unprocessed))

        unprocessed = unprocessed[186:]
        for cp_path in tqdm(unprocessed, desc="Running inference", unit="checkpoint"):
            tqdm.write(f"Running inference on {cp_path}")

            time_before = time.time()
            run_eval = False
            run_inference(cp_path, run_eval, ignore_markfiles, extractions_only_datasets)
            time_after = time.time()
            duration = timedelta(seconds=(time_after - time_before))
            print(f"\n Inference completed in {duration} (HH:MM:SS)\n")


if __name__ == '__main__':
    main()
