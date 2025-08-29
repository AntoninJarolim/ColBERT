import json
import os
import re
from collections import defaultdict
import torch

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from torch import nn

import wandb
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import precision_recall_curve

from colbert.data.extractions import ExtractionResults
from colbert.training.training import extraction_stats
from evaluate.wandb_logging import wandb_log_pr_curve, wandb_log_extraction_accuracy

sns.set_theme()


def downsample_full_fidelity(data, total_points=1000):
    if len(data) < total_points:
        return data

    # Number of bins and samples per bin
    num_bins = 100 if len(data) > 100 else 1
    samples_per_bin = total_points // num_bins

    # Split data into bins
    bins = np.array_split(data, num_bins)

    # Sample points from each bin
    sampled_data = []
    for bin in bins:
        rnd_indexes = np.random.choice(np.array(list(range(1, len(bin) - 1))),
                                       size=min(samples_per_bin, len(bin)) - 2,  # for first and last
                                       replace=False)
        sampled_data.extend(bin[rnd_indexes])

        first = bin[0]
        sampled_data.append(first)

        last = bin[-1]
        sampled_data.append(last)
    return sampled_data


def max_f1_score(precision, recall):
    # Compute F1 scores for all threshold values
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # Avoid division by zero

    # Return the maximum F1 score
    arg_max_f1 = np.argmax(f1_scores)
    return f1_scores[arg_max_f1], arg_max_f1


def _get_pr_data(all_extractions, all_max_scores):
    all_max_scores = torch.nn.Sigmoid()(torch.tensor(all_max_scores)).detach().numpy()
    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(
        np.array(all_extractions), all_max_scores, drop_intermediate=True
    )

    # Prepare data for wandb Table
    data = list(zip(recall, precision, thresholds))
    data = downsample_full_fidelity(data)

    best_f1, arg_max_f1 = max_f1_score(precision, recall)

    best_f1_recall = recall[arg_max_f1]
    best_f1_threshold = thresholds[arg_max_f1]
    return data, best_f1, best_f1_recall, best_f1_threshold


def _log_extr_accuracy_wandb(target_extractions, all_max_scores, checkpoint_steps, collection_name):
    data = []
    counter = 0
    for ext, max_scr in zip(target_extractions, all_max_scores):
        ext = torch.tensor(ext).float()
        max_scr = torch.tensor(max_scr).float()

        if sum(ext) == 0:
            continue  # Skip if no positive extractions

        assert len(ext) == len(max_scr)
        ex_loss = nn.BCEWithLogitsLoss()(max_scr, ext)
        d = extraction_stats(ex_loss, max_scr, ext)
        data.append(d)

        counter += 1

    # Count nr of non-nan values np.count_nonzero(~np.isnan(data))
    non_nans = {k: np.count_nonzero([~np.isnan(d[k]) for d in data]) for k in data[0].keys()}
    assert [x > (len(target_extractions) * 0.99) for x in non_nans.values()]

    # Log the data to wandb
    wandb_log_extraction_accuracy(data, checkpoint_steps, collection_name)


def _evaluate_extractions(all_extractions, all_max_scores, checkpoint_steps, is_first_call):
    # Compute precision-recall curve for real max scores (requires flat data)
    all_extractions_flat = np.array([x for sublist in all_extractions for x in sublist])
    all_max_scores_flat = np.array([x for sublist in all_max_scores for x in sublist])

    data_max, best_f1, best_f1_recall, best_f1_threshold = _get_pr_data(all_extractions_flat, all_max_scores_flat)

    data_rnd = []
    if is_first_call:
        # Compute precision-recall curve for baselines (random and select all)
        random_scores = np.random.random_sample(len(all_max_scores_flat))
        data_rnd, _, _, _ = _get_pr_data(all_extractions_flat, random_scores)

    # Iterate over data types and store recall/precision values
    df_data = []
    data_type_pairs = [
        (data_rnd, "Random"),
        (data_max, f"Max-Values Checkpoint {checkpoint_steps}"),
    ]

    for data, type_name in data_type_pairs:
        for recall, precision, threshold in data:
            df_data.append((recall, precision, threshold, type_name))

    return df_data, best_f1, best_f1_recall, best_f1_threshold


def _cp_palette(df_data):
    df = pd.DataFrame(df_data, columns=["Recall", "Precision", "Threshold", "Type"])
    df_cp = df.copy()[df['Type'].str.contains('Max-Values')]
    df_cp["Step"] = df_cp["Type"].str.extract(r"(\d+)").astype(int)
    norm = mcolors.Normalize(vmin=df_cp["Step"].min(), vmax=df_cp["Step"].max())
    cmap = plt.cm.copper
    color_mapping = {t: cmap(norm(s)) for t, s in zip(df_cp["Type"].unique(), df_cp["Step"].unique())}
    color_mapping['Random'] = 'red'
    return color_mapping, norm, cmap


def update_extractions_figures(evaluation_path, run_name):
    pattern = re.compile(r'col_name=(.+)\.nbits=\d+\.steps=(\d+)\.extraction_scores\.jsonl$')

    files_by_coll = defaultdict(list)
    for file in os.listdir(evaluation_path):
        match = pattern.search(file)
        if match:
            collection_name = match.group(1)
            files_by_coll[collection_name].append((file, int(match.group(2))))

    f1_scores = defaultdict(dict)
    all_pr_data = defaultdict(list)
    for collection_name, files_to_eval in files_by_coll.items():
        df_data = []

        # Sort files by steps
        files_to_eval = sorted(files_to_eval, key=lambda x: int(x[1]))
        for i, (file, steps) in enumerate(files_to_eval):
            # Evaluation
            max_ranking_path = os.path.join(evaluation_path, file)
            max_ranking_results = ExtractionResults.cast(max_ranking_path)

            all_extractions = [[score for score in line['extraction_binary']] for line in max_ranking_results]
            all_max_scores = [[score for score in line['max_scores']] for line in max_ranking_results]

            pr_data, best_f1, recall_best_f1, best_f1_threshold = _evaluate_extractions(
                all_extractions, all_max_scores, steps, i == 0
            )
            df_data.extend(pr_data)

            # Compute accuracy etc. for each datapoint
            _log_extr_accuracy_wandb(all_extractions, all_max_scores, steps, collection_name)

            f1_scores[collection_name][steps] = best_f1

            all_pr_data[collection_name].append(
                {
                    'best_f1': best_f1,
                    'best_f1_threshold': best_f1_threshold,
                    'pr_data': pr_data,
                    'checkpoint_steps': steps,
                    'run_name': run_name,
                    'recall_best_f1': recall_best_f1,
                }
            )

        # Log the PR curve to wandb
        color_mapping, norm, cmap = _cp_palette(df_data)
        wandb_log_pr_curve(collection_name, df_data, color_mapping=color_mapping, norm=norm, cmap=cmap)

    # Log the F1 scores to wandb
    _log_f1_scores_wandb(f1_scores)

    # get best results for each collection by f1 value
    all_pr_data = add_dev_thresholded_f1s(all_pr_data)

    # Save all PR data to a file
    pr_data_path = os.path.join(evaluation_path, 'aggregated_pr_data.json')
    json.dump(all_pr_data, open(pr_data_path, 'w'))

    return all_pr_data


def _log_f1_scores_wandb(f1_scores):
    # Fill missing steps with 0.0
    all_steps = sorted(list(set([s for v in f1_scores.values() for s in v.keys()])))
    for key in f1_scores.keys():
        for step in all_steps:
            if step not in f1_scores[key]:
                f1_scores[key][step] = 0.0

    # collect column names and f1 scores
    collection_keys = [f"F1 scores {k}" for k in f1_scores.keys()]
    f1_scores_values = [[f1_scores[k][step] for step in all_steps] for k in f1_scores.keys()]

    assert len(collection_keys) == len(f1_scores_values)

    for coll_name, f1 in zip(collection_keys, f1_scores_values):
        data = [all_steps, f1]
        data = list(zip(*data))  # Transpose
        table = wandb.Table(columns=["Steps", coll_name],
                            data=data)
        wandb.log({f"f1_scores {coll_name}": table})


def add_dev_thresholded_f1s(all_pr_data):
    def find_best_dev_threshold(all_pr_data):
        best_f1s = get_best_pr_data_by_f1(all_pr_data)
        dev_f1 = [best_f1 for best_f1 in best_f1s if best_f1['collection_name'] == 'official_dev_small'][0]
        dev_f1_threshold = dev_f1['best_f1_threshold']
        return dev_f1_threshold

    def find_f1_by_threshold(pr_data, threshold):
        ds = []

        for p, r, t, n in pr_data:
            if n == 'Random':
                continue

            ds.append((p, r, t))

        ds = sorted(ds, key=lambda x: x[2])
        ts = np.array([x[2] for x in ds]) > threshold
        first = np.argmax(ts)
        recall_first = ds[first][1]
        precision_first = ds[first][0]

        # compute F1 score
        return 2 * (precision_first * recall_first) / (precision_first + recall_first + 1e-10)

    try:
        dev_f1_threshold = find_best_dev_threshold(all_pr_data)
    except IndexError:
        print("Warning: No development data found. Skipping thresholding.")
        dev_f1_threshold = None

    new_data = defaultdict(list)
    for dataset_name, all_checkpoints_prs in all_pr_data.items():
        if dataset_name == 'official_dev_small':
            continue

        # For all checkpoint pr data
        for cp_prs in all_checkpoints_prs:

            if dev_f1_threshold is None:
                f1 = None
            else:
                f1 = find_f1_by_threshold(cp_prs['pr_data'], dev_f1_threshold)

            f1_updated_sample = {
                'f1_dev_thresholded': f1,
                **cp_prs
            }
            new_data[dataset_name].append(f1_updated_sample)

    return new_data


def get_best_pr_data_by_f1(all_pr_data, f1_key='best_f1'):

    # Remove f1_key that equals None (so sorting works)
    all_pr_data = {k: [v for v in v if v[f1_key] is not None] for k, v in all_pr_data.items()}

    # Sort by f1_key and get the best one
    best_pr_curves = {k: sorted(v, key=lambda x: x[f1_key], reverse=True)[0] for k, v in all_pr_data.items()}

    # reshape 'collection: [{data}, {data}]' to 'collection: {data}', as we only have one best
    return [{'collection_name': k, **v} for k, v in best_pr_curves.items()]


