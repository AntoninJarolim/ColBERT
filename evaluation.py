import json
import os
import re
from collections import defaultdict

import jsonlines
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
from utility.evaluate.msmarco_passages import evaluate_ms_marco_ranking


def _downsample_full_fidelity(data, total_points=1000):
    # Number of bins and samples per bin
    num_bins = 100
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
    return np.max(f1_scores)


def _get_pr_data(all_extractions, all_max_scores):
    all_max_scores = torch.nn.Sigmoid()(torch.tensor(all_max_scores)).detach().numpy()
    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(
        np.array(all_extractions), all_max_scores, drop_intermediate=True
    )
    # Prepare data for wandb Table
    data = list(zip(recall, precision))
    data = _downsample_full_fidelity(data)

    best_f1 = max_f1_score(precision, recall)
    return data, best_f1


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

    # Get mean of all data
    data_agg = {f"{k}-{collection_name}": np.nanmean([d[k] for d in data]) for k in data[0].keys()}

    # Log the data to wandb
    collection_cp_steps = f"checkpoint_steps-{collection_name}"
    wandb.define_metric(collection_cp_steps)
    for k in data_agg.keys():
        wandb.define_metric(k, step_metric=collection_cp_steps)

    data_agg[collection_cp_steps] = checkpoint_steps
    wandb.log(data_agg)


def _evaluate_extractions(all_extractions, all_max_scores, checkpoint_steps, is_first_call):
    # Compute precision-recall curve for real max scores (requires flat data)
    all_extractions_flat = np.array([x for sublist in all_extractions for x in sublist])
    all_max_scores_flat = np.array([x for sublist in all_max_scores for x in sublist])

    data_max, best_f1 = _get_pr_data(all_extractions_flat, all_max_scores_flat)

    data_rnd = []
    if is_first_call:
        # Compute precision-recall curve for baselines (random and select all)
        random_scores = np.random.random_sample(len(all_max_scores_flat))
        data_rnd, _ = _get_pr_data(all_extractions_flat, random_scores)

    # Iterate over data types and store recall/precision values
    df_data = []
    data_type_pairs = [
        (data_rnd, "Random"),
        (data_max, f"Max-Values Checkpoint {checkpoint_steps}"),
    ]

    for data, type_name in data_type_pairs:
        for recall, precision in data:
            df_data.append((recall, precision, type_name))

    return df_data, best_f1


def evaluate_retrieval(ranking_path, qrels_path, collection_path, index_name):
    # Make evaluation of current run
    out_json = evaluate_ms_marco_ranking(collection_path, qrels_path, ranking_path)
    ranking_path_dir = os.path.dirname(ranking_path)
    out_json_path = os.path.join(ranking_path_dir, f"{index_name}.retrieval_evaluation.json")

    # Save the evaluation to json
    with open(out_json_path, "w") as f:
        json.dump(out_json, f, indent=4)


def _cp_palette(df_data):
    # Convert to DataFrame
    df = pd.DataFrame(df_data, columns=["Recall", "Precision", "Type"])
    df_cp = df.copy()[df['Type'].str.contains('Max-Values')]
    # Create custom colours
    df_cp["Step"] = df_cp["Type"].str.extract(r"(\d+)").astype(int)
    # Normalize steps for colormap mapping
    norm = mcolors.Normalize(vmin=df_cp["Step"].min(), vmax=df_cp["Step"].max())
    # Create a gradient colormap (e.g., "viridis" or "coolwarm")
    cmap = plt.cm.copper
    color_mapping = {t: cmap(norm(s)) for t, s in zip(df_cp["Type"].unique(), df_cp["Step"].unique())}
    color_mapping.update(
        {
            'Random': 'red'
        }
    )
    return color_mapping


def update_extractions_figures(evaluation_path, run_name):
    pattern = re.compile(r'col_name=(.+)\.nbits=\d+\.steps=(\d+)\.extraction_scores\.jsonl$')

    files_by_coll = defaultdict(list)
    for file in os.listdir(evaluation_path):
        match = pattern.search(file)
        if match:
            collection_name = match.group(1)
            files_by_coll[collection_name].append((file, int(match.group(2))))

    f1_scores = defaultdict(dict)
    best_pr_curves = defaultdict(list)
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

            pr_data, best_f1 = _evaluate_extractions(all_extractions, all_max_scores,  steps, i == 0)
            df_data.extend(pr_data)

            # Compute accuracy etc. for each datapoint
            _log_extr_accuracy_wandb(all_extractions, all_max_scores, steps, collection_name)

            f1_scores[collection_name][steps] = best_f1

            best_pr_curves[collection_name].append(
                {'best_f1': best_f1, 'pr_data': pr_data, 'checkpoint_steps': steps, 'run_name': run_name}
            )

        # Log the PR curve to wandb

        _log_pr_curve_wandb(collection_name, df_data, _cp_palette(df_data))

    # Log the F1 scores to wandb
    _log_f1_scores_wandb(f1_scores)

    # get best results for each collection by f1 value
    best_pr_curves = {k: sorted(v, key=lambda x: x['best_f1'], reverse=True)[0] for k, v in best_pr_curves.items()}

    return [{'collection_name': k, **v} for k, v in best_pr_curves.items()]


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


def _log_pr_curve_wandb(collection_name, data, color_mapping=None):
    df = pd.DataFrame(data, columns=["Recall", "Precision", "Type"])
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    sns.lineplot(data=df, x="Recall", y="Precision", hue="Type", ax=ax, palette=color_mapping)
    ax.set_title(f"PR Curve {collection_name}")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1)  # Force y-axis from 0 to 1
    plt.tight_layout()
    wandb.log({
        f"pr_curve_all_{collection_name}": wandb.Image(fig)
    })
    plt.close(fig)


def update_retrieval_figures(evaluation_path, qrels_path, collection_path):
    # Generate evaluation jsons from ranking.tsv
    pattern = re.compile(r'col_name=(.+)\.nbits=\d+\.steps=(\d+)\.ranking.tsv')

    for file in os.listdir(evaluation_path):
        match = pattern.search(file)
        if match:
            col_name = match.group(1)
            if col_name == '35_samples': # Skip 35 samples dataset - not for retrieval
                continue
            full_evaluation_path = os.path.join(evaluation_path, file)
            assert file.endswith('.ranking.tsv')
            index_name = file[:-len('.ranking.tsv')]
            evaluate_retrieval(full_evaluation_path, qrels_path, collection_path, index_name)

    # Use all jsons in the directory to generate figures/tables
    pattern = re.compile(r'col_name=(.+)\.nbits=\d+\.steps=(\d+)\.retrieval_evaluation\.json')

    file_datas = defaultdict(list)
    for file in os.listdir(evaluation_path):
        match = pattern.search(file)
        if match:
            steps = int(match.group(2))
            collection_name = match.group(1)
            with open(os.path.join(evaluation_path, file), "r") as matched_file:
                data = json.load(matched_file)
            data['batch_steps'] = steps
            file_datas[collection_name].append(data)

    for collection_name, data in file_datas.items():
        # Log the evaluation to wandb
        data = sorted(data, key=lambda x: x['batch_steps'])
        values = [file_data.values() for file_data in data]
        tab = wandb.Table(columns=list(data[0].keys()), data=values)
        wandb.log({f"retrieval_evaluation_{collection_name}": tab})

        # Craete mrr@10 line plot and upload it to wandb
        wandb.log({f"mrr@10_steps_{collection_name}": wandb.plot.line(tab, "batch_steps", "mrr@10", title="MRR@10 over steps")})
        wandb.log({f"recall@50_steps_{collection_name}": wandb.plot.line(tab, "batch_steps", "recall@50", title="Recall@10 over steps")})


def log_best_pr_curve_wandb(best_pr_curves):
    # Save the best PR curves
    print("\n\n Extraction aggregated f1 results:")
    all_pr_data = defaultdict(list)
    all_f1_data = defaultdict(list)
    for best_pr in best_pr_curves:
        name = f"{best_pr['run_name']} {best_pr['checkpoint_steps']}"
        all_f1_data[best_pr['collection_name']].append((best_pr['best_f1'], name))

        for p, r, _ in best_pr['pr_data']:
            all_pr_data[best_pr['collection_name']].append((p, r, name))

    # Print the best F1 scores
    for collection, data in all_f1_data.items():
        print(f"Best F1 score for {collection}:")
        for f1, name in data:
            print(f"\t{name}: {f1:.4f}")
        print()

    # Log the best PR curves
    for collection, data in all_pr_data.items():
        fig_name = f'PR curve {collection}'
        _log_pr_curve_wandb(fig_name, data)
