import json
import os
import re
from collections import defaultdict
import torch

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from torch import nn
from numba import njit

import wandb
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from tqdm import tqdm

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


def _extraction_baseline_random(targets):
    # Flatten targets if necessary
    if not isinstance(targets, np.ndarray):
        targets = np.array([x for sublist in targets for x in sublist])

    # Compute precision-recall curve for baselines (random and select all)
    random_scores = np.random.random_sample(len(targets))
    data_rnd, _, _, _ = _get_pr_data(targets, random_scores)

    df_random_data = []
    for recall, precision, thr in data_rnd:
        df_random_data.append((recall, precision, thr, "Random"))

    return df_random_data


def _micro_f1(all_extractions, all_max_scores, checkpoint_steps):
    # Compute precision-recall curve for real max scores (requires flat data)
    all_extractions_flat = np.array([x for sublist in all_extractions for x in sublist])
    all_max_scores_flat = np.array([x for sublist in all_max_scores for x in sublist])

    data_max, best_f1, best_f1_recall, best_f1_threshold = _get_pr_data(all_extractions_flat, all_max_scores_flat)

    type_name = f"Max-Values Checkpoint {checkpoint_steps}"

    df_data = [
        (recall, precision, thr, type_name)
        for recall, precision, thr in data_max
    ]

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


def _mid_thresholds(values, return_last=False):
    """
    Given a list/array of numeric values, return thresholds
    that lie between each pair of adjacent sorted unique values.
    """

    # Flatten the list of lists if necessary
    if not isinstance(values, np.ndarray):
        values = (score for sublist in values for score in sublist)
        values = np.fromiter(values, dtype=float)

    values = np.unique(values)  # sorted + deduplicated
    thresholds = (values[:-1] + values[1:]) / 2
    thresholds = thresholds if return_last else thresholds[:-1]
    return thresholds


def pad_and_stack(sequences, pad_value=0):
    """
    Pad sequences to the same length and stack them into a 2D array.
    """
    max_len = max(len(seq) for seq in sequences)
    return np.vstack([
        np.pad(np.asarray(seq), (0, max_len - len(seq)), constant_values=pad_value)
        for seq in sequences
    ])


def _f1(threshold: float, t, predictions: np.array):
    """
    Compute precision, recall, F1 for a given threshold.
    """
    # Convert scores to binary predictions using threshold
    p = (predictions >= threshold)

    TP = (t & p).sum()
    FP = (~t & p).sum()
    FN = (t & ~p).sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return precision, recall, f1


@njit
def f1_batched_numba(threshold, t, predictions):
    B, L = t.shape
    f1_sum = 0.0
    prec_sum = 0.0
    rec_sum = 0.0

    for i in range(B):
        TP = 0
        FP = 0
        FN = 0
        for j in range(L):
            pred = predictions[i, j] >= threshold
            true = t[i, j]
            if pred and true:
                TP += 1
            elif pred and not true:
                FP += 1
            elif not pred and true:
                FN += 1

        precision = TP / (TP + FP) if TP + FP > 0 else 0.0
        recall = TP / (TP + FN) if TP + FN > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        f1_sum += f1
        prec_sum += precision
        rec_sum += recall

    macro_precision = prec_sum / B
    macro_recall = rec_sum / B
    macro_f1 = f1_sum / B

    return macro_precision, macro_recall, macro_f1


def _f1_batched(threshold, t, predictions: np.array):
    """
    Compute precision, recall, F1 for a given threshold.
    t and predictions are 2D arrays (batch_size, seq_len).
    """
    # Convert scores to binary predictions using threshold
    p = (predictions >= threshold)

    # Compute TP, FP, FN per example (sum over last axis)
    TP = (t & p).sum(-1)
    FP = (~t & p).sum(-1)
    FN = (t & ~p).sum(-1)

    # Initialize precision and recall arrays with zeros
    precision = np.zeros_like(TP, dtype=float)
    recall = np.zeros_like(TP, dtype=float)

    # Mask for division without warning if denominator is 0
    # zeroes_like -> default values are 0
    mask_prec = TP + FP > 0
    mask_rec = TP + FN > 0

    # Compute only where denominator > 0
    precision[mask_prec] = TP[mask_prec] / (TP[mask_prec] + FP[mask_prec])
    recall[mask_rec] = TP[mask_rec] / (TP[mask_rec] + FN[mask_rec])

    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return precision, recall, f1


def _macro_f1(t, targets, predictions):
    assert targets.shape == predictions.shape, \
        f"Targets shape {targets.shape} and predictions shape {predictions.shape} must be the same."

    # precisions_b, recalls_b, f1s_b = _f1_batched(t, targets, predictions)
    macro_precision_nu, macro_recall_nu, macro_f1_nu = f1_batched_numba(t, targets, predictions)

    # macro_precision = np.mean(precisions_b)
    # macro_recall = np.mean(recalls_b)
    # macro_f1 = np.mean(f1s_b)

    # assert np.isclose(macro_precision, macro_precision_nu), f"{macro_precision} vs {macro_precision_nu}"
    # assert np.isclose(macro_recall, macro_recall_nu), f"{macro_recall} vs {macro_recall_nu}"
    # assert np.isclose(macro_f1, macro_f1_nu), f"{macro_f1} vs {macro_f1_nu}"

    return macro_recall_nu, macro_precision_nu, macro_f1_nu


def update_extractions_figures(evaluation_dir, run_name):
    pattern = re.compile(r'col_name=(.+)\.nbits=\d+\.steps=(\d+)\.extraction_scores\.jsonl$')

    files_by_coll = defaultdict(list)
    for file in os.listdir(evaluation_dir):
        match = pattern.search(file)
        if match:
            max_ranking_path = os.path.join(evaluation_dir, file)
            extraction_results = ExtractionResults.cast(max_ranking_path)

            collection_name = match.group(1)
            files_by_coll[collection_name].append(
                {
                    'extraction_results': extraction_results,
                    'steps': int(match.group(2)),
                }
            )

    all_pr_data = defaultdict(list)
    for collection_name, files_to_eval in files_by_coll.items():

        # Add random baseline - first get targets - it is the same for all checkpoints, so just use the first
        targets = [
            np.array(line['extraction_binary'], dtype=bool)
            for line in files_to_eval[0]['extraction_results']
        ]

        # Find targets with no positive labels
        no_positive_labels = find_no_positive_labels(collection_name, targets)

        # Remove targets with no positive labels
        targets = [t for i, t in enumerate(targets) if i not in no_positive_labels]
        targets_b = pad_and_stack(targets, pad_value=0)

        pr_curves_by_cp = []
        # Sort files by steps
        files_to_eval = sorted(files_to_eval, key=lambda x: x['steps'])
        for i, file_data in enumerate(files_to_eval):
            extraction_results = file_data['extraction_results']
            steps = file_data['steps']

            # Get predictions and remove those with no positive labels
            predictions = [np.array(line['max_scores'])
                           for i, line in enumerate(extraction_results)
                           if i not in no_positive_labels]
            predictions_b = pad_and_stack(predictions, pad_value=0)
            thrs = _mid_thresholds(predictions)

            data = []
            for t in tqdm(thrs, desc=f"Computing macro F1 for collection-{collection_name}", total=len(thrs)):
                macro_recall, macro_precision, macro_f1 = _macro_f1(t, targets_b, predictions_b)
                data.append({
                    "Threshold": t,
                    "macro-Precision": macro_precision,
                    "macro-Recall": macro_recall,
                    "macro-F1": macro_f1,
                })

            max_by_f1 = sorted(data, key=lambda x: x['macro-F1'], reverse=True)[0]

            pr_data, best_f1, recall_best_f1, best_f1_threshold = _micro_f1(
                targets, predictions, steps
            )
            pr_curves_by_cp.extend(pr_data)

            # Compute accuracy etc. for each datapoint
            _log_extr_accuracy_wandb(targets, predictions, steps, collection_name)

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

        # add random scores baseline
        pr_curves_by_cp.extend(_extraction_baseline_random(targets))

        # Log the PR curve to wandb
        color_mapping, norm, cmap = _cp_palette(pr_curves_by_cp)
        wandb_log_pr_curve(collection_name, pr_curves_by_cp, color_mapping=color_mapping, norm=norm, cmap=cmap)

    # Log the F1 scores to wandb
    f1_scores = _agg_f1_by_collection(all_pr_data)
    _log_f1_scores_wandb(f1_scores)

    # get best results for each collection by f1 value
    all_pr_data = add_dev_thresholded_f1s(all_pr_data)

    # Save all PR data to a file
    pr_data_path = os.path.join(evaluation_dir, 'aggregated_pr_data.json')
    json.dump(all_pr_data, open(pr_data_path, 'w'))

    return all_pr_data


def _agg_f1_by_collection(all_pr_data):
    return {
        collection_name: {
            entry['checkpoint_steps']: entry['best_f1']
            for entry in entries
        }
        for collection_name, entries in all_pr_data.items()
    }


def find_no_positive_labels(collection_name, targets):
    no_positive_labels = [i for i, t in enumerate(targets) if np.sum(t.astype(int)) == 0]
    if no_positive_labels:
        print(
            f"Warning: {len(no_positive_labels)} samples in collection '{collection_name}' have no positive labels. "
            f"They will be ignored in macro F1 computation.")

    return no_positive_labels


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
