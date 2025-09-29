from jsonlines import jsonlines

from evaluate.wandb_logging import (
    wandb_log_pr_curve,
    wandb_connect_running,
    wandb_log_figure, log_wandb_table
)

from collections import defaultdict
from jinja2 import Template
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

def aggregate_eval(save_all_experiments_stats, save_all_best_pr_curves):
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

        log_pareto_optimal_solutions(filtered_stats, group_name, 'best_f1_macro')
        log_pareto_optimal_solutions(filtered_stats, group_name, 'best_f1_micro')
        aggregated_extraction_eval(filtered_pr_curves, group_name)
        # log_best_pr_curve_wandb(filtered_pr_curves_dev_thr, group_name, 'dev threshold')


def aggregated_extraction_eval(best_pr_curves_data, group_name):
    aggregated_f1_printing(best_pr_curves_data, group_name)
    log_aggregated_pr_data(best_pr_curves_data, group_name)


def log_aggregated_pr_data(best_pr_curves_data, extraction_id):
    collected_pr_data = defaultdict(list)
    for best_pr in best_pr_curves_data:
        name = f"{best_pr['run_name']} {best_pr['checkpoint_steps']}"

        for p, r, _ in best_pr['macro_results']['pr_data']:
            collected_pr_data[best_pr['collection_name']].append((p, r, name))
    # Log the best PR curves
    for collection, data in collected_pr_data.items():
        fig_name = f'PR curve {collection} {extraction_id}'
        wandb_log_pr_curve(fig_name, data)


def aggregated_f1_printing(best_pr_curves_data, group_name):
    # Prepare data for logging
    data = defaultdict(list)
    for best_pr in best_pr_curves_data:

        if best_pr['collection_name'] in [
            '35_samples', 'human_explained_0', 'human_explained_1', 'human_explained_2',
            'accuracy_dataset_pos', 'accuracy_dataset_neg', 'md2d-sample'
        ]:
            continue

        collection = best_pr['collection_name']

        def safe_get(d, key):
            return d[key] if d[key] else None

        data[collection].append(
            {
                'steps': best_pr['checkpoint_steps'],
                'run_name': best_pr['run_name'],
                'micro_f1': safe_get(best_pr['micro_results'], 'best_f1'),
                'micro_f1_dev': safe_get(best_pr['micro_results'], 'f1_dev_thresholded'),
                'macro_f1': safe_get(best_pr['macro_results'], 'best_f1'),
                'macro_f1_dev': safe_get(best_pr['macro_results'], 'f1_dev_thresholded'),
            }
        )

    tables = []
    for collection_name, records in data.items():
        df = pd.DataFrame(records)
        table_html = df.to_html(index=False, border=0)
        tables.append((collection_name, table_html))

    create_html_table_for_f1_scores(tables, group_name)



def create_html_table_for_f1_scores(tables, group_name):
    template_filename = "visualizations/f1_table.html"
    with open(template_filename) as f:
        template = Template(f.read())

    title = f"Aggregated F1 Scores for group {group_name}"
    html = template.render(tables=tables, title=title)

    with open("experiments/eval/f1-agg-table.html", "w") as f:
        f.write(html)

    log_wandb_table(html, title)


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


def log_pareto_optimal_solutions(combined_stats_all, group_name, x_key='best_f1_micro'):
    combined_stats_all = mark_pareto_front(combined_stats_all, x_key=x_key)
    df = pd.DataFrame(combined_stats_all)

    # Normalize steps for opacity (alpha)
    df['Training progress'] = df.groupby('run_name')['steps'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    )

    # Marker: dot for non-optimal, X for optimal
    # df['marker'] = df['is_pareto_optimal'].map({True: 'x', False: 'o'})

    # Plot
    rename_map = {
        'best_f1_micro': 'best extraction f1 micro',
        'best_f1_macro': 'best extraction f1 macro',
        'is_pareto_optimal': 'Is pareto optimal'
    }
    df.rename(columns=rename_map, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    sns.scatterplot(
        data=df,
        x=rename_map.get(x_key, x_key),
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
