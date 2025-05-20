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
from utility.evaluate.msmarco_passages import evaluate_ms_marco_ranking

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


def evaluate_retrieval(ranking_path, qrels_path, collection_path, index_name):
    # Make evaluation of current run
    out_json = evaluate_ms_marco_ranking(collection_path, qrels_path, ranking_path, silent=True)
    ranking_path_dir = os.path.dirname(ranking_path)
    out_json_path = os.path.join(ranking_path_dir, f"{index_name}.retrieval_evaluation.json")

    # Save the evaluation to json
    with open(out_json_path, "w") as f:
        json.dump(out_json, f, indent=4)


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

            pr_data, best_f1, recall_best_f1, best_f1_threshold = _evaluate_extractions(
                all_extractions, all_max_scores, steps, i == 0
            )
            df_data.extend(pr_data)

            # Compute accuracy etc. for each datapoint
            _log_extr_accuracy_wandb(all_extractions, all_max_scores, steps, collection_name)

            f1_scores[collection_name][steps] = best_f1

            best_pr_curves[collection_name].append(
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
        _log_pr_curve_wandb(collection_name, df_data, color_mapping=color_mapping, norm=norm, cmap=cmap)

    # Log the F1 scores to wandb
    _log_f1_scores_wandb(f1_scores)

    # get best results for each collection by f1 value
    return best_pr_curves


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


def _log_pr_curve_wandb(collection_name, data, color_mapping=None, norm=None, cmap=None):
    run_name = wandb.run.name
    title = f"PR Curve {collection_name} {run_name}"

    if len(data[0]) == 3:
        columns = ["Recall", "Precision", "Type"]
    elif len(data[0]) == 4:
        len(data[0]) == 3
        columns = ["Recall", "Precision", "Threshold", "Type"]
    else:
        raise ValueError("Data must have 3 or 4 columns")

    df = pd.DataFrame(data, columns=columns)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    if color_mapping:
        # Plot each line individually to allow custom colors (and not to label checkpoint lines)
        for t in df["Type"].unique():
            df_sub = df[df["Type"] == t]
            color = color_mapping[t] if color_mapping and t in color_mapping else None
            label = None if "Max-Values" in t else t  # Only label 'Random'
            sns.lineplot(data=df_sub, x="Recall", y="Precision", ax=ax, color=color, label=label)
    else:
        # Plot all lines with default colors (hue will do the trick)
        sns.lineplot(data=df, x="Recall", y="Precision", hue="Type", ax=ax)

    ax.set_title(title)
    ax.set_ylim(0, 1)

    # Create colorbar as a gradient legend for Max-Values
    if norm and cmap:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Only needed for older matplotlib versions
        cbar = fig.colorbar(sm, ax=ax)

        # Force showing min and max step values
        min_step = int(norm.vmin)
        max_step = int(norm.vmax)
        cbar.set_ticks(np.linspace(min_step, max_step, num=5, dtype=int))
        cbar.set_ticklabels([str(int(x)) for x in np.linspace(min_step, max_step, num=5)])

        cbar.set_label("Checkpoint Step")

    # Add manual red marker for Random
    # ax.plot([], [], color='red', label='Random Baseline')

    # if cmap is not None, we have a colorbar and only one entry in a legend (random baseline)
    if len(df["Type"].unique()) < 6 or cmap is not None:
        legend_loc = "lower right"
        bbox = (1, 0)
    else:
        legend_loc = "upper left"
        bbox = (1, 1)

    ax.legend(loc=legend_loc, bbox_to_anchor=bbox)

    plt.tight_layout()
    plt.savefig(f"experiments/eval/fig/{title}.pdf", dpi=900)
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
            # Skip 35 samples dataset - not for retrieval
            if col_name == '35_samples' or col_name == 'human_explained':
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
        wandb.log({f"mrr@10_steps_{collection_name}": wandb.plot.line(tab, "batch_steps", "mrr@10",
                                                                      title="MRR@10 over steps")})
        wandb.log({f"recall@50_steps_{collection_name}": wandb.plot.line(tab, "batch_steps", "recall@50",
                                                                         title="Recall@50 over steps")})

    return file_datas


def log_best_pr_curve_wandb(best_pr_curves_data, group_name):
    # Save the best PR curves
    print("\n\n Extraction aggregated f1 results:")
    all_pr_data = defaultdict(list)
    all_f1_data = defaultdict(list)
    all_f1_data_dev_thr = defaultdict(list)

    # Prepare data for logging
    for best_pr in best_pr_curves_data:
        name = f"{best_pr['run_name']} {best_pr['checkpoint_steps']}"

        all_f1_data[best_pr['collection_name']].append((best_pr['best_f1'], best_pr['recall_best_f1'], name))

        if best_pr['f1_dev_thresholded'] is not None:
            all_f1_data_dev_thr[best_pr['collection_name']].append((best_pr['f1_dev_thresholded'], None, name))

        for p, r, _, _ in best_pr['pr_data']:
            all_pr_data[best_pr['collection_name']].append((p, r, name))

    # Print the best F1 scores
    for collection, f1_data in all_f1_data.items():
        print(f"Best F1 score for {collection} thresholded by best:")
        f1_data = sorted(f1_data, key=lambda x: x[0], reverse=True)
        for f1, recall_best_f1, name in f1_data:
            print(f"\t{name}: {f1:.4f} - at recall {recall_best_f1:.4f}")
        print()

    # Print the best F1 scores dev thresholded
    for collection, f1_data in all_f1_data_dev_thr.items():
        print(f"Best F1 score for {collection} thresholded by dev:")
        f1_data = sorted(f1_data, key=lambda x: x[0], reverse=True)
        for f1, _, name in f1_data:
            print(f"\t{name}: {f1:.4f}")
        print()

    # Log the best PR curves
    for collection, data in all_pr_data.items():
        fig_name = f'PR curve {collection} {group_name}'
        _log_pr_curve_wandb(fig_name, data)


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


def mark_pareto_front(data, x_key='best_f1', y_key='recall@10'):
    """
    Marks Pareto optimal points in the data.
    Maximizes x_key and y_key (e.g., best_f1 and recall@10).
    """
    pareto = []
    for i, d in enumerate(data):
        is_dominated = False
        for j, other in enumerate(data):
            if i != j:
                if other[x_key] >= d[x_key] and other[y_key] >= d[y_key]:
                    if other[x_key] > d[x_key] or other[y_key] > d[y_key]:
                        is_dominated = True
                        break
        d['is_pareto_optimal'] = not is_dominated
        pareto.append(d)
    return pareto


def log_pareto_optimal_solutions(combined_stats_all, group_name):
    combined_stats_all = mark_pareto_front(combined_stats_all)
    df = pd.DataFrame(combined_stats_all)

    # Normalize steps for opacity (alpha)
    df['Training progress'] = df.groupby('run_name')['steps'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    )

    # Marker: dot for non-optimal, X for optimal
    # df['marker'] = df['is_pareto_optimal'].map({True: 'x', False: 'o'})

    # Plot
    rename_map = {'best_f1': 'best extraction f1', 'is_pareto_optimal': 'Is pareto optimal'}
    df.rename(columns=rename_map, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    sns.scatterplot(
        data=df,
        x='best extraction f1',
        y='recall@10',
        hue='run_name',
        size='Training progress',
        style='Is pareto optimal',
        # markers={True: 'x', False: 'o'},
        s=100,  # Size of the markers
        ax=ax
    )

    title = f"Pareto Optimal Solutions {group_name}"
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"test {group_name}.png")

    wandb.log({f'pareto_optimal_solutions_{group_name}': wandb.Image(fig)})

    plt.savefig(f"experiments/eval/fig/{title}.pdf", dpi=900)
    plt.close(fig)
