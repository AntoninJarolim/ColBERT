import argparse
import glob
import json
import os.path
import sys
import time
from datetime import timedelta

from datasets import tqdm

import wandb
from jsonlines import jsonlines

from colbert import Indexer
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

from evaluate.aggregated_runs import aggregate_eval, get_dev_thresholded_filename
from evaluate.extractions import update_extractions_figures, add_dev_thresholded_f1s, get_best_pr_data_by_f1, \
    downsample_full_fidelity
from evaluate.retrieval import update_retrieval_figures, get_coll_agg_retrieval_data
from evaluate.wandb_logging import wandb_connect_running


def load_datasets(json_path: str):
    """Load datasets from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        datasets = json.load(f)
    return datasets


def get_datasets(json_path: str, dataset_names):
    """Return datasets specified in dataset_names."""
    datasets = load_datasets(json_path)

    # If no dataset names are provided, return all datasets
    if not dataset_names:
        return datasets

    # Filter datasets based on provided names
    filtered = {}
    for name in dataset_names:
        if name not in datasets:
            raise ValueError(f"Dataset '{name}' not found. Available: {list(datasets.keys())}")
        filtered[name] = datasets[name]

    return filtered


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def index_dataset(checkpoint, collection_path, root_folder, nbits, index_name, overwrite=False):
    checkpoint = os.path.realpath(checkpoint)
    config = ColBERTConfig(
        nbits=nbits,
        root=root_folder
    )
    indexer = Indexer(
        checkpoint=checkpoint,
        config=config)
    indexer.index(name=index_name,
                  overwrite='reuse' if not overwrite else True,
                  collection=collection_path)


def get_max_ranking_name(index_name):
    return f"{index_name}.extraction_scores.jsonl"


def get_ranking_name(index_name):
    return f"{index_name}.ranking.tsv"


def search_dataset(root_folder, queries_path, extraction_path, index_name, overwrite=False):
    max_ranking_name = get_max_ranking_name(index_name)
    ranking_name = get_ranking_name(index_name)

    max_ranking_path = Run().get_results_path(max_ranking_name)
    ranking_path = Run().get_results_path(ranking_name)

    run_inference = not os.path.exists(max_ranking_path) or not os.path.exists(ranking_path)

    if not run_inference and not overwrite:
        print(
            f"Both max ranking and ranking already exist at {max_ranking_path} and {ranking_path}, skipping search step.")
        return ranking_path, max_ranking_path

    config = ColBERTConfig(root=root_folder)
    searcher = Searcher(index=index_name, config=config)
    queries = Queries(queries_path)

    max_ranking = searcher.search_extractions(queries, extraction_path)
    max_ranking_path = max_ranking.save(max_ranking_name)

    ranking = searcher.search_all(queries, k=3)
    ranking_path = ranking.save(ranking_name)

    return ranking_path, max_ranking_path


def get_run_name(checkpoint):
    assert checkpoint.startswith("experiments/")
    experiment_name, dir_type, run_name, *_ = remove_prefix(checkpoint, "experiments/").split('/')
    assert dir_type in ['train', 'inference', 'index'], dir_type
    return experiment_name, run_name


def get_checkpoint_steps(checkpoint):
    # remove last slash for consistency
    if checkpoint[-1] == '/':
        checkpoint = checkpoint[:-1]

    # get the last part of the path
    # example: 'experiments/exp_name/train/restful-deluge-83/checkpoints/colbert-200'
    checkpoint_steps = checkpoint.split('/')[-1]
    if checkpoint_steps == 'colbert':
        raise AssertionError(f"Checkpoint steps not found in '{checkpoint}'")
    try:
        checkpoint_steps = int(checkpoint_steps.split('-')[-1])
    except ValueError:
        raise AssertionError(f"Checkpoint steps not found in '{checkpoint}'")
    return checkpoint_steps


def inference_checkpoint_all_datasets(
        datasets, checkpoint,
        overwrite_inference=False,
        run_eval=True,
        extractions_only_datasets=False
):
    if datasets is None:
        datasets = load_datasets("datasets.json")

    if extractions_only_datasets:
        # qrels_path is None -> dataset does not have retrieval annotations
        datasets = {k: v for k, v in datasets.items() if v['qrels_path'] is None}

    eval_dir = None
    for idx_dataset, (collection_name, data) in enumerate(datasets.items()):
        new_eval_dir = inference_checkpoint_one_dataset(
            checkpoint,
            collection_name,
            data['collection_path'],
            data['queries_path'],
            data['extraction_path'],
            data['qrels_path'],
            overwrite_inference,
            run_eval=run_eval and idx_dataset == len(datasets) - 1
        )

        if eval_dir is None:
            eval_dir = new_eval_dir

        assert eval_dir == new_eval_dir, "All eval datasets must be in the same directory"

    return eval_dir


def inference_checkpoint_one_dataset(
        checkpoint,
        collection_name,
        collection_path,
        queries_path,
        extraction_path,
        qrels_path,
        overwrite_inference,
        run_eval=True
):
    # inference
    root_folder = 'experiments'
    experiment, run_name = get_run_name(checkpoint)
    print(f"Current run name: {run_name}")

    nbits = 2
    checkpoint_steps = get_checkpoint_steps(checkpoint)
    config_search = {
        'checkpoint': checkpoint,
        'collection_path': collection_path,
        'queries_path': queries_path,
        'extraction_path': extraction_path,
        'root_folder': root_folder,
        'nbits': nbits,
        'index_name': f"col_name={collection_name}.nbits={nbits}.steps={checkpoint_steps}",
        'checkpoint_steps': checkpoint_steps
    }

    config_run = {
        'nranks': 1,
        'experiment': experiment,
        'overwrite': True,
    }

    run_config = RunConfig(
        nranks=config_run['nranks'],
        experiment=config_run['experiment'],
        overwrite=config_run['overwrite'],
        name=run_name
    )

    # Allow run only evaluation
    with Run().context(run_config):
        # Run indexing and retrieval
        index_dataset(
            config_search['checkpoint'],
            config_search['collection_path'],
            config_search['root_folder'],
            config_search['nbits'],
            config_search['index_name'],
            overwrite=overwrite_inference
        )
        ranking_path, max_ranking_path = search_dataset(
            config_search['root_folder'],
            config_search['queries_path'],
            config_search['extraction_path'],
            config_search['index_name'],
            overwrite=overwrite_inference
        )

    assert ranking_path is not None, "Inference failed"

    log_config = {
        **config_search,
        **config_run,
        'ranking_path': ranking_path,
        'max_ranking_path': max_ranking_path
    }
    wandb_connect_running(run_name)
    wandb.config.update(log_config, allow_val_change=True)

    assert os.path.dirname(max_ranking_path) == os.path.dirname(ranking_path)
    eval_dir = os.path.dirname(max_ranking_path)

    if qrels_path is not None:
        update_retrieval_figures(eval_dir, qrels_path, collection_path)

    if run_eval:
        update_extractions_figures(eval_dir, run_name)

    wandb.finish()
    return eval_dir


def find_all_results_dirs():
    base_dir = os.getcwd()
    pattern = os.path.join(base_dir, 'experiments', '*', 'results', '*')
    matching_paths = [path for path in glob.glob(pattern, recursive=True) if os.path.isdir(path)]
    return matching_paths


def parse_args():
    parser = argparse.ArgumentParser(description='Inference runner')

    # Run inference on concrete checkpoint
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint', default=None)
    parser.add_argument(
        '--overwrite_inference',
        action='store_true',
        default=False,
        help='Whether to overwrite existing inference results.'
    )

    parser.add_argument(
        '--save_all_experiments_stats',
        type=str,
        default='experiments/eval/all_experiments_stats.jsonl',
        help='Path to save all experiments statistics.'
    )

    parser.add_argument(
        '--save_all_best_pr_curves',
        type=str,
        default='experiments/eval/all_best_pr_curves.jsonl',
        help='Path to save all best precision-recall curves.'
    )

    parser.add_argument("--datasets_json", default="datasets.json", help="Path to datasets JSON file")
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        help="Dataset name(s) to filter. Filtering will not be applied if not provided."
    )

    parser.add_argument(
        '--aggregate_results',
        action='store_true',
        default=True,
        help='Whether to aggregate results from all found evaluation directories.'
    )

    parser.add_argument(
        '--re_evaluate_all',
        action='store_true',
        default=False,
        help='Whether to re-evaluate all found evaluation directories.'
    )

    args = parser.parse_args()

    if args.re_evaluate_all and args.checkpoint is not None:
        raise ValueError("Cannot use --checkpoint with --re_evaluate_all. Please choose one.")

    args.datasets = get_datasets(args.datasets_json, args.eval_datasets)
    return args


def run_name_from_path(evaluation_path):
    return evaluation_path.strip('/').split('/')[-1]


def load_run_extraction_results(evaluation_path):
    pr_data_path = os.path.join(evaluation_path, 'aggregated_pr_data.json')
    try:
        return json.load(open(pr_data_path, 'r'))
    except FileNotFoundError:
        return None


def aggregate_results(eval_dirs, save_all_experiments_stats, save_all_best_pr_curves):
    # Find the best checkpoint for each run
    # and save data needed to plot precision-recall curves
    best_pr_curves = []

    # For each run, for each checkpoint, for each dataset
    # save recall - F1 combinations
    combined_stats_all = []

    # Find the best F1 value based on dev-set threshold
    best_pr_curves_dev_thresholded = []

    for eval_dir in tqdm(eval_dirs, desc="Aggregated evaluation", unit="dir", total=len(eval_dirs)):
        tqdm.write(f"Current dir: {eval_dir}")

        run_name = run_name_from_path(eval_dir)

        time_before = time.time()

        wandb_connect_running(run_name)
        all_pr_data = load_run_extraction_results(eval_dir)
        if all_pr_data is None:
            print(f"Warning: No extraction results found in {eval_dir}, skipping")
            continue

        if 'official_dev_small' not in all_pr_data:
            print(f"Warning: No 'official_dev_small' data found in {eval_dir}, skipping")
            continue

        try:
            best_pr_data_dev_f1 = get_best_pr_data_by_f1(all_pr_data, f1_key='f1_dev_thresholded')
            best_pr_curves_dev_thresholded.extend(best_pr_data_dev_f1)
        except IndexError:
            print("Warning: No dev thresholded F1 available, skipping")

        # Just to log the best checkpoint CURVE for each run
        best_pr_data = get_best_pr_data_by_f1(all_pr_data)
        best_pr_curves.extend(best_pr_data)

        all_retrieval_data = get_coll_agg_retrieval_data(eval_dir)

        # Get (recall - F1) pairs for each checkpoint, so it can be plotted in the same figure
        combined_stats = combine_retrieval_and_extraction_stats(all_pr_data, all_retrieval_data, run_name)
        combined_stats_all.extend(combined_stats)

        wandb.finish()
        print(f"\n Inference completed in {timedelta(seconds=(time.time() - time_before))} (HH:MM:SS)\n")

    save_all_stats(
        save_all_experiments_stats,
        save_all_best_pr_curves,
        combined_stats_all,
        best_pr_curves,
        best_pr_curves_dev_thresholded
    )
    return best_pr_curves, combined_stats_all, best_pr_curves_dev_thresholded


def combine_retrieval_and_extraction_stats(all_pr_data, all_retrieval_data, run_name):
    combined_stats = []
    for data_cp in all_pr_data['official_dev_small']:
        steps = data_cp['checkpoint_steps']
        assert run_name == data_cp['run_name']

        # Get recall with corresponding batch steps from all_retrieval_data
        recall = [
            d['recall@50']
            for d in all_retrieval_data['official_dev_small']
            if d['batch_steps'] == steps
        ][0]

        combined_stats.append(
            {
                'best_f1_micro': data_cp['micro_results']['best_f1'],
                'best_f1_macro': data_cp['macro_results']['best_f1'],
                'recall@10': recall,
                'steps': steps,
                'run_name': run_name
            }
        )
    # Get only 10 checkpoints evaluation for a figure not to be that crowded
    combined_stats = downsample_full_fidelity(combined_stats, total_points=10)
    return combined_stats


def save_all_stats(
        save_all_experiments_stats,
        save_all_best_pr_curves,
        combined_stats_all,
        best_pr_curves,
        best_pr_curves_dev_thresholded
):
    with jsonlines.open(save_all_experiments_stats, 'w') as writer:
        writer.write_all(combined_stats_all)

    with jsonlines.open(save_all_best_pr_curves, 'w') as writer:
        writer.write_all(best_pr_curves)

    with jsonlines.open(get_dev_thresholded_filename(save_all_best_pr_curves), 'a') as writer:
        writer.write_all(best_pr_curves_dev_thresholded)


def safe_eval_all(eval_dir, retrieval_datasets, run_name):
    try:
        for collection_name, collection in retrieval_datasets.items():
            update_retrieval_figures(eval_dir, collection['qrels_path'], collection['collection_path'])

        update_extractions_figures(eval_dir, run_name)

    except Exception as e:
        print(f"Error during re-evaluation of {eval_dir}: {e}", file=sys.stderr)

def re_evaluate_all():
    eval_dirs = find_all_results_dirs()
    datasets = load_datasets("datasets.json")

    retrieval_datasets = {k: v for k, v in datasets.items() if v['qrels_path'] is not None}

    for eval_dir in tqdm(eval_dirs,
                         desc="Re-evaluating all directories",
                         total=len(eval_dirs)):
        tqdm.write(f"Current dir: {eval_dir}")

        run_name = run_name_from_path(eval_dir)
        wandb_connect_running(run_name)
        tqdm.write(f"\nCurrent run name: {run_name}")

        # Run evaluation for all datasets with retrieval annotations
        safe_eval_all(eval_dir, retrieval_datasets, run_name)

        wandb.finish()


def run_aggregate_results(save_all_experiments_stats, save_all_best_pr_curves):
    eval_dirs = find_all_results_dirs()
    print("Evaluating found directories:\n" + '\n'.join([f'\t{e_dir}' for e_dir in eval_dirs]))
    aggregate_results(eval_dirs, save_all_experiments_stats, save_all_best_pr_curves)
    aggregate_eval(save_all_experiments_stats, save_all_best_pr_curves)


def main():
    args = parse_args()

    # Evaluate specific checkpoint
    if args.checkpoint is not None:
        datasets = load_datasets(args.datasets_json)
        inference_checkpoint_all_datasets(
            datasets,
            args.checkpoint,
            overwrite_inference=args.overwrite_inference,
        )

    if args.re_evaluate_all:
        re_evaluate_all()

    if args.aggregate_results:
        run_aggregate_results(
            args.save_all_experiments_stats,
            args.save_all_best_pr_curves,
        )


if __name__ == '__main__':
    main()
