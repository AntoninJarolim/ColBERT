from jsonlines import jsonlines

from evaluate.wandb_logging import (
    wandb_log_pr_curve,
    wandb_connect_running,
    wandb_log_figure
)

from collections import defaultdict

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns



def get_dev_thresholded_filename(save_all_best_pr_curves):
    return save_all_best_pr_curves.replace('.jsonl', '_dev_thresholded.jsonl')

def load_all_stats(save_all_experiments_stats, save_all_best_pr_curves):
    with jsonlines.open(save_all_experiments_stats, 'r') as reader:
        combined_stats_all = list(reader)

    with jsonlines.open(save_all_best_pr_curves, 'r') as reader:
        best_pr_curves = list(reader)

    with jsonlines.open(get_dev_thresholded_filename(save_all_best_pr_curves), 'r') as reader:
        best_pr_curves_dev_thresholded = list(reader)

    return best_pr_curves, combined_stats_all, best_pr_curves_dev_thresholded

def run_aggregated_eval(save_all_experiments_stats, save_all_best_pr_curves):
    """
    Creates aggregated figures for all runs
    """
    best_pr_curves, combined_stats_all, best_pr_curves_dev_thr = load_all_stats(
        save_all_experiments_stats,
        save_all_best_pr_curves
    )

    # Connect to WandB run that's logging all runs
    agg_wandb_name = 'eval-all'
    wandb_connect_running(agg_wandb_name)

    # Define groups to create figures with subsets of runs
    always_exclude = ['denormalized-e80-r20', 'add_extraction_ffn', 'add_extraction_ffn-bert']
    name_filter_groups = {
        'from bert': lambda name: 'bert' in name,
        'from pretrained': lambda name: 'bert' not in name,
        'all': lambda name: True,
    }

    def filter_group(data, filter_function):
        data = [s for s in data if filter_function(s['run_name'])]
        data = [s for s in data if all(excl != s['run_name'] for excl in always_exclude)]
        return data

    for group_name, group_filter in name_filter_groups.items():
        print(f"Evaluating group: {group_name}")
        filtered_stats = filter_group(combined_stats_all, group_filter)
        filtered_pr_curves = filter_group(best_pr_curves, group_filter)
        filtered_pr_curves_dev_thr = filter_group(best_pr_curves_dev_thr, group_filter)

        log_pareto_optimal_solutions(filtered_stats, group_name)
        aggregated_extraction_eval(filtered_pr_curves, group_name)
        # log_best_pr_curve_wandb(filtered_pr_curves_dev_thr, group_name, 'dev threshold')


def aggregated_extraction_eval(best_pr_curves_data, extraction_id):
    """
    Prints F1 scores and creates PR curves
    parameters:
     - extraction_id:
    """
    aggregated_f1_printing(best_pr_curves_data)
    log_aggregated_pr_data(best_pr_curves_data, extraction_id)


def log_aggregated_pr_data(best_pr_curves_data, extraction_id):
    collected_pr_data = defaultdict(list)
    for best_pr in best_pr_curves_data:
        name = f"{best_pr['run_name']} {best_pr['checkpoint_steps']}"

        for p, r, _, _ in best_pr['pr_data']:
            collected_pr_data[best_pr['collection_name']].append((p, r, name))
    # Log the best PR curves
    for collection, data in collected_pr_data.items():
        fig_name = f'PR curve {collection} {extraction_id}'
        wandb_log_pr_curve(fig_name, data)


def aggregated_f1_printing(best_pr_curves_data):
    # Save the best PR curves
    print("\n\n Extraction aggregated f1 results:")
    all_f1_data = defaultdict(list)
    all_f1_data_dev_thr = defaultdict(list)
    # Prepare data for logging
    for best_pr in best_pr_curves_data:
        name = f"{best_pr['run_name']} {best_pr['checkpoint_steps']}"

        all_f1_data[best_pr['collection_name']].append((best_pr['best_f1'], best_pr['recall_best_f1'], name))

        if best_pr['f1_dev_thresholded'] is not None:
            all_f1_data_dev_thr[best_pr['collection_name']].append((best_pr['f1_dev_thresholded'], None, name))
    # Print the best F1 scores
    print_best_f1_scores(all_f1_data, mode="best")
    # Print the best F1 scores dev thresholded
    print_best_f1_scores(all_f1_data_dev_thr, mode="dev")


def print_best_f1_scores(all_f1_data, mode="best"):
    """
    Print best F1 scores for collections.
    """
    assert mode in ["best", "dev"], "mode must be either 'best' or 'dev'"

    for collection, f1_data in all_f1_data.items():
        print(f"Best F1 score for {collection} thresholded by {mode}:")
        f1_data = sorted(f1_data, key=lambda x: x[0], reverse=True)

        if mode == "best":
            for f1, recall_best_f1, name in f1_data:
                print(f"\t{name}: {f1:.4f} - at recall {recall_best_f1:.4f}")

        else:
            for f1, _, name in f1_data:
                print(f"\t{name}: {f1:.4f}")


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

    wandb_log_figure(f'pareto_optimal_solutions_{group_name}', fig)

    plt.savefig(f"experiments/eval/fig/{title}.pdf", dpi=900)
    plt.close(fig)
