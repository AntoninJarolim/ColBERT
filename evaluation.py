import json
import os
import re

import jsonlines
import torch

import wandb
import numpy as np
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
        rnd_indexes = np.random.choice(len(bin), size=min(samples_per_bin, len(bin)), replace=False)
        sampled_data.extend(bin[rnd_indexes])
    return sampled_data


def _evaluate_extractions(max_ranking_path, figure_id):
    """
    Use checkpoint steps as figure_id
    """

    if type(figure_id) is int:
        figure_id = str(figure_id)

    assert type(figure_id) is str and figure_id.isalnum(), f"'{figure_id}' is not a valid figure_id"

    # Evaluation
    with jsonlines.open(max_ranking_path) as reader:
        max_ranking_results = list(reader)

    all_extractions = [score for line in max_ranking_results for score in line['extraction_binary']]

    all_max_scores = [score for line in max_ranking_results for score in line['max_scores']]
    all_max_scores = torch.nn.Sigmoid()(torch.tensor(all_max_scores)).detach().numpy()

    # Compute precision-recall curve for real max scores
    data_max = _get_pr_data(all_extractions, all_max_scores)

    # Compute precision-recall curve for baselines (random and select all)
    random_scores = np.random.random_sample(len(all_max_scores))
    data_rnd = _get_pr_data(all_extractions, random_scores)

    all_scores = np.ones(len(all_max_scores))
    data_ones = _get_pr_data(all_extractions, all_scores)

    table = wandb.Table(columns=["Recall", "Precision", "Type"])

    for data, type_name in [(data_max, 0), (data_rnd, 1), (data_ones, 2)]:
    # for data, type_name in [(data_max, "Max-Values"), (data_rnd, "Random"), (data_ones, "Select-All")]:
        for recall, precision in data:
            table.add_data(recall, precision, type_name)

    # Log the PR curve
    figure_name = f"pr_curve_{figure_id}"
    wandb.log({
        figure_name: wandb.plot_table(
            vega_spec_name="wandb/area-under-curve/v0",
            data_table=table,
            fields={"x": "Recall", "y": "Precision", "class": "Type"},
            string_fields={
                "title": f"Precision-Recall Curve Checkpoint {figure_id}",
                "x-axis-title": "Recall",
                "y-axis-title": "Precision",
            },
        )
    })


def _get_pr_data(all_extractions, all_max_scores):
    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(
        np.array(all_extractions), all_max_scores, drop_intermediate=True
    )
    # Prepare data for wandb Table
    data = list(zip(recall, precision))
    data = _downsample_full_fidelity(data)
    return data


def evaluate_retrieval(ranking_path, qrels_path, collection_path, index_name):
    # Make evaluation of current run
    out_json = evaluate_ms_marco_ranking(collection_path, qrels_path, ranking_path)
    ranking_path_dir = os.path.dirname(ranking_path)
    out_json_path = os.path.join(ranking_path_dir, f"{index_name}.retrieval_evaluation.json")

    # Save the evaluation to json
    with open(out_json_path, "w") as f:
        json.dump(out_json, f, indent=4)


def update_extractions_figures(evaluation_path):
    pattern = re.compile(r'nbits=\d+\.steps=(\d+)\.extraction_scores\.jsonl$')

    files_to_eval = []
    for file in os.listdir(evaluation_path):
        match = pattern.search(file)
        if match:
            files_to_eval.append((file, int(match.group(1))))

    files_to_eval = sorted(files_to_eval, key=lambda x: int(x[1]))
    for file, steps in files_to_eval:
        _evaluate_extractions(os.path.join(evaluation_path, file), steps)


def update_retrieval_figures(evaluation_path, qrels_path, collection_path):
    # Generate evaluation jsons from ranking.tsv
    pattern = re.compile(r'nbits=\d+\.steps=(\d+)\.ranking.tsv')

    for file in os.listdir(evaluation_path):
        match = pattern.search(file)
        if match:
            full_evaluation_path = os.path.join(evaluation_path, file)
            assert file.endswith('.ranking.tsv')
            index_name = file[:-len('.ranking.tsv')]
            evaluate_retrieval(full_evaluation_path, qrels_path, collection_path, index_name)

    # Use all jsons in the directory to generate figures/tables
    pattern = re.compile(r'nbits=\d+\.steps=(\d+).retrieval_evaluation\.json')

    file_datas = []
    for file in os.listdir(evaluation_path):
        match = pattern.search(file)
        if match:
            steps = int(match.group(1))
            with open(os.path.join(evaluation_path, file), "r") as matched_file:
                data = json.load(matched_file)
            data['batch_steps'] = steps
            file_datas.append(data)

    # Log the evaluation to wandb
    file_datas = sorted(file_datas, key=lambda x: x['batch_steps'])
    values = [file_data.values() for file_data in file_datas]
    tab = wandb.Table(columns=list(file_datas[0].keys()), data=values)
    wandb.log({f"retrieval_evaluation": tab})

    # Craete mrr@10 line plot and upload it to wandb
    wandb.log({f"mrr@10_steps": wandb.plot.line(tab, "batch_steps", "mrr@10", title="MRR@10 over steps")})
    wandb.log({f"recall@50_steps": wandb.plot.line(tab, "batch_steps", "recall@50", title="Recall@10 over steps")})

