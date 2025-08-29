import argparse
import glob
import json
import os.path
import time
from datetime import timedelta

import wandb
from jsonlines import jsonlines

from colbert import Indexer
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

from evaluate.aggregated_runs import run_aggregated_eval, get_dev_thresholded_filename
from evaluate.extractions import update_extractions_figures, add_dev_thresholded_f1s, get_best_pr_data_by_f1, \
    downsample_full_fidelity
from evaluate.retrieval import update_retrieval_figures, get_coll_agg_retrieval_data
from evaluate.wandb_logging import wandb_connect_running


def load_datasets(json_path: str):
    """Load datasets from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        datasets = json.load(f)
    return datasets


def get_datasets(json_path: str, dataset_names: list[str]):
    """Return datasets specified in dataset_names."""
    datasets = load_datasets(json_path)
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


def index_dataset(checkpoint, collection_path, root_folder, nbits, index_name):
    checkpoint = os.path.realpath(checkpoint)
    config = ColBERTConfig(
        nbits=nbits,
        root=root_folder
    )
    indexer = Indexer(
        checkpoint=checkpoint,
        config=config)
    indexer.index(name=index_name,
                  overwrite=True,  # 'reuse'
                  collection=collection_path)


def search_dataset(root_folder, queries_path, extraction_path, index_name):
    config = ColBERTConfig(
        root=root_folder
    )
    searcher = Searcher(index=index_name,
                        config=config)
    queries = Queries(queries_path)

    max_ranking = searcher.search_extractions(queries, extraction_path)

    ranking = searcher.search_all(queries, k=3)
    max_ranking_path = max_ranking.save(f"{index_name}.extraction_scores.jsonl")
    ranking_path = ranking.save(f"{index_name}.ranking.tsv")

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


def inference_checkpoint(
        datasets, checkpoint, run_eval=True, extractions_only_datasets=False):
    eval_datasets = []

    if extractions_only_datasets:
        # qrels_path is None -> dataset does not have retrieval annotations
        eval_sets = {k: v for k, v in datasets.items() if v['qrels_path'] is None}
    else:
        eval_sets = datasets

    for idx_dataset, (collection_name, data) in enumerate(eval_sets.items()):
        eval_dataset = inference_checkpoint_one_dataset(
            checkpoint,
            collection_name,
            data['collection_path'],
            data['queries_path'],
            data['extraction_path'],
            data['qrels_path'],
            run_eval=run_eval and idx_dataset == len(datasets) - 1
        )
        eval_datasets.append(eval_dataset)

    if len(eval_datasets) > 1:
        assert eval_datasets[0] == eval_datasets[1]

    return eval_datasets[0]


def inference_checkpoint_one_dataset(
        checkpoint,
        collection_name,
        collection_path,
        queries_path,
        extraction_path,
        qrels_path,
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
            config_search['index_name']
        )
        ranking_path, max_ranking_path = search_dataset(
            config_search['root_folder'],
            config_search['queries_path'],
            config_search['extraction_path'],
            config_search['index_name']
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

    if run_eval:
        if qrels_path is not None:
            update_retrieval_figures(eval_dir, qrels_path, collection_path)
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
    parser.add_argument('--load_inference_dir', type=str, default=None,
                        help='Path to dir with ranking and evaluation. Inference wont be run if provided')

    parser.add_argument('--load_eval_stats', action='store_true', default=False)

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
        required=True,
        nargs="+",
        help="Dataset name(s) to filter"
    )

    parsed_args = parser.parse_args()
    parsed_args.datasets = get_datasets(parsed_args.datasets_json, parsed_args.eval_datasets)
    return parsed_args


def load_run_extraction_results(evaluation_path):
    pr_data_path = os.path.join(evaluation_path, 'aggregated_pr_data.json')
    all_pr_data = json.load(open(pr_data_path, 'r'))
    return all_pr_data



def evaluate_all_dirs(eval_dirs, save_all_experiments_stats, save_all_best_pr_curves):
    # Find the best checkpoint for each run
    # and save data needed to plot precision-recall curves
    best_pr_curves = []

    # For each run, for each checkpoint, for each dataset
    # save recall - F1 combinations
    combined_stats_all = []

    # Find the best F1 value based on dev-set threshold
    best_pr_curves_dev_thresholded = []

    for i, eval_dir in enumerate(eval_dirs):
        print(f"\n\n\n\n ({i + 1} / {len(eval_dirs)}) Evaluation of dir: {eval_dir}")
        run_name = eval_dir.strip('/').split('/')[-1]

        time_before = time.time()

        wandb_connect_running(run_name)
        all_pr_data = load_run_extraction_results(eval_dir)

        # Thresholding with dev-set best F1 threshold
        # must be done after dev-set F1 is calculated todo: consider moving this to update_extractions_figures
        # and ensure that dev-set F1 is calculated
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
                'best_f1': data_cp['best_f1'],
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


def inference_checkpoint_all_datasets(
        checkpoint_path,
        run_eval,
        extractions_only_datasets):

    datasets = load_datasets('datasets.json')
    inference_checkpoint(
        datasets,
        checkpoint_path,
        run_eval=run_eval,
        extractions_only_datasets=extractions_only_datasets)

def main():
    args = parse_args()

    # Evaluate specific checkpoint
    if args.checkpoint is not None:
        inference_checkpoint(args.datasets, args.checkpoint)

    if args.evaluate_all_dirs:
        eval_dirs = find_all_results_dirs()
        print("Evaluating found directories:\n" + '\n'.join([f'\t{e_dir}' for e_dir in eval_dirs]))
        evaluate_all_dirs(eval_dirs, args.save_all_experiments_stats, args.save_all_best_pr_curves)

    run_aggregated_eval(args.datasets, args.save_all_experiments_stats)


if __name__ == '__main__':
    main()
