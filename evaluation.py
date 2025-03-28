import json
import os
import re
from collections import defaultdict

import jsonlines
import torch

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

import wandb
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import precision_recall_curve

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


def _get_pr_data(all_extractions, all_max_scores):
    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(
        np.array(all_extractions), all_max_scores, drop_intermediate=True
    )
    # Prepare data for wandb Table
    data = list(zip(recall, precision))
    data = _downsample_full_fidelity(data)
    return data


def _evaluate_extractions(max_ranking_path, checkpoint_steps, is_first_call):
    # Evaluation
    with jsonlines.open(max_ranking_path) as reader:
        max_ranking_results = list(reader)

    all_extractions = [score for line in max_ranking_results for score in line['extraction_binary']]

    all_max_scores = [score for line in max_ranking_results for score in line['max_scores']]
    all_max_scores = torch.nn.Sigmoid()(torch.tensor(all_max_scores)).detach().numpy()

    # Compute precision-recall curve for real max scores
    data_max = _get_pr_data(all_extractions, all_max_scores)

    data_rnd = data_ones = []
    if is_first_call:
        # Compute precision-recall curve for baselines (random and select all)
        random_scores = np.random.random_sample(len(all_max_scores))
        data_rnd = _get_pr_data(all_extractions, random_scores)

    # Iterate over data types and store recall/precision values
    df_data = []
    data_type_pairs = [
        (data_rnd, "Random"),
        (data_max, f"Max-Values Checkpoint {checkpoint_steps}"),
    ]

    for data, type_name in data_type_pairs:
        for recall, precision in data:
            df_data.append((recall, precision, type_name))

    return df_data


def evaluate_retrieval(ranking_path, qrels_path, collection_path, index_name):
    # Make evaluation of current run
    out_json = evaluate_ms_marco_ranking(collection_path, qrels_path, ranking_path)
    ranking_path_dir = os.path.dirname(ranking_path)
    out_json_path = os.path.join(ranking_path_dir, f"{index_name}.retrieval_evaluation.json")

    # Save the evaluation to json
    with open(out_json_path, "w") as f:
        json.dump(out_json, f, indent=4)


def update_extractions_figures(evaluation_path):
    pattern = re.compile(r'nbits=\d+\.steps=(\d+)\.col_name=(.+)\.extraction_scores\.jsonl$')

    files_by_coll = defaultdict(list)
    for file in os.listdir(evaluation_path):
        match = pattern.search(file)
        if match:
            collection_name = match.group(2)
            files_by_coll[collection_name].append((file, int(match.group(1))))

    for collection_name, files_to_eval in files_by_coll.items():
        df_data = []
        files_to_eval = sorted(files_to_eval, key=lambda x: int(x[1]))
        for i, (file, steps) in enumerate(files_to_eval):
            data = _evaluate_extractions(os.path.join(evaluation_path, file), steps, i == 0)
            df_data.extend(data)

        _log_pr_curve_wandb(collection_name, df_data)


def _log_pr_curve_wandb(collection_name, df_data):
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
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    sns.lineplot(data=df, x="Recall", y="Precision", hue="Type", ax=ax, palette=color_mapping)
    ax.set_title(f"PR Curve {collection_name}")
    wandb.log({
        f"pr_curve_all_{collection_name}": wandb.Image(fig)
    })


def update_retrieval_figures(evaluation_path, qrels_path, collection_path):
    # Generate evaluation jsons from ranking.tsv
    pattern = re.compile(r'nbits=\d+\.steps=(\d+)\.col_name=(.+)\.ranking.tsv')

    for file in os.listdir(evaluation_path):
        match = pattern.search(file)
        if match:
            full_evaluation_path = os.path.join(evaluation_path, file)
            assert file.endswith('.ranking.tsv')
            index_name = file[:-len('.ranking.tsv')]
            evaluate_retrieval(full_evaluation_path, qrels_path, collection_path, index_name)

    # Use all jsons in the directory to generate figures/tables
    pattern = re.compile(r'nbits=\d+\.steps=(\d+)\.col_name=(.+).retrieval_evaluation\.json')

    file_datas = defaultdict(list)
    for file in os.listdir(evaluation_path):
        match = pattern.search(file)
        if match:
            steps = int(match.group(1))
            collection_name = int(match.group(2))
            with open(os.path.join(evaluation_path, file), "r") as matched_file:
                data = json.load(matched_file)
            data['batch_steps'] = steps
            file_datas['collection_name'].append(data)

    for collection_name, data in file_datas.items():
        # Log the evaluation to wandb
        data = sorted(data, key=lambda x: x['batch_steps'])
        values = [file_data.values() for file_data in data]
        tab = wandb.Table(columns=list(data[0].keys()), data=values)
        wandb.log({f"retrieval_evaluation_{collection_name}": tab})

        # Craete mrr@10 line plot and upload it to wandb
        wandb.log({f"mrr@10_steps_{collection_name}": wandb.plot.line(tab, "batch_steps", "mrr@10", title="MRR@10 over steps")})
        wandb.log({f"recall@50_steps_{collection_name}": wandb.plot.line(tab, "batch_steps", "recall@50", title="Recall@10 over steps")})
