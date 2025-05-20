import argparse
import glob
import os.path
import time
from datetime import timedelta

import wandb
from jsonlines import jsonlines

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
from evaluation import (
    update_retrieval_figures,
    update_extractions_figures,
    log_best_pr_curve_wandb,
    get_best_pr_data_by_f1,
    downsample_full_fidelity, log_pareto_optimal_solutions, add_dev_thresholded_f1s
)

DATASETS = {
    'md2d-sample': {
        'collection_path': 'data/evaluation/collection.md2d_dataset.tsv',
        'queries_path': 'data/evaluation/queries.eval.accuracy_dataset.tsv',
        'extraction_path': 'data/evaluation/extracted_relevancy_md2d.tsv',
        'qrels_path': None,
    },
    # 'accuracy_dataset_pos': {
    #     'collection_path': 'data/evaluation/collection.accuracy_dataset.tsv',
    #     'queries_path': 'data/evaluation/queries.eval.accuracy_dataset.tsv',
    #     'extraction_path': 'data/evaluation/extracted_relevancy_accuracy_dataset_first_30_lines.tsv',
    #     'qrels_path': None,
    # },
    # 'accuracy_dataset_neg': {
    #     'collection_path': 'data/evaluation/collection.accuracy_dataset.tsv',
    #     'queries_path': 'data/evaluation/queries.eval.accuracy_dataset.tsv',
    #     'extraction_path': 'data/evaluation/extracted_relevancy_accuracy_dataset_last_30_lines.tsv',
    #     'qrels_path': None,
    # },
    # 'official_dev_small': {
    #     'collection_path': 'data/evaluation/collection.dev.small_50-25-25.tsv',
    #     'queries_path': 'data/evaluation/queries.dev.small_ex_only.tsv',
    #     'extraction_path': 'data/evaluation/extracted_relevancy_qrels.dev.small.tsv',
    #     'qrels_path': 'data/evaluation/qrels.dev.small_ex_only.tsv',
    # },
    # '35_samples': {
    #     'collection_path': 'data/evaluation/collection.35_sample_dataset.tsv',
    #     'queries_path': 'data/evaluation/queries.eval.35_sample_dataset.tsv',
    #     'extraction_path': 'data/evaluation/extracted_relevancy_35_sample_dataset.tsv',
    #     'qrels_path': None
    # },
    # 'human_explained': {
    #     'collection_path': 'data/evaluation/collection.ms-marco-human-explained.tsv',
    #     'queries_path': 'data/evaluation/queries.eval.ms-marco-human-explained.tsv',
    #     'extraction_path': 'data/evaluation/extracted_relevancy_ms-marco-human-explained.tsv',
    #     'qrels_path': None
    # }
}


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


def connect_running_wandb(run_name):
    api = wandb.Api()

    # Replace with your project name
    project_name = 'eval-' + Run().config.project_name
    entity = Run().config.wandb_entity
    run_name = 'eval_' + run_name

    # Fetch all runs in the project
    runs = api.runs(f"{entity}/{project_name}")

    # Find the run with the specified name
    for run in runs:
        if run.name == run_name:
            print(f"Resuming found run ID: {run.id}")
            wandb.init(project=project_name, id=run.id, resume="must")

    print("\n\nStarting new eval run with name:", run_name)
    wandb.init(project=project_name, name=run_name)


def inference_checkpoint_all_datasets(checkpoint, run_eval=True, extractions_only_datasets=False):
    eval_datasets = []

    if extractions_only_datasets:
        # qrels_path is None -> dataset does not have retrieval annotations
        eval_sets = {k: v for k, v in DATASETS.items() if v['qrels_path'] is None}
    else:
        eval_sets = DATASETS

    for idx_dataset, (collection_name, data) in enumerate(eval_sets.items()):
        eval_dataset = inference_checkpoint_one_dataset(
            checkpoint,
            collection_name,
            data['collection_path'],
            data['queries_path'],
            data['extraction_path'],
            data['qrels_path'],
            run_eval=run_eval and idx_dataset == len(DATASETS) - 1
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
    connect_running_wandb(run_name)
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
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Path to dir with ranking and evaluation. Inference wont be run if provided')
    parser.add_argument('--evaluate_all', action='store_true', default=False)
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

    return parser.parse_args()


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

        connect_running_wandb(run_name)
        all_pr_data = update_extractions_figures(eval_dir, run_name)
        all_pr_data = add_dev_thresholded_f1s(all_pr_data)

        # Thresholding with dev-set best F1 threshold
        try:
            best_pr_data_dev_f1 = get_best_pr_data_by_f1(all_pr_data, f1_key='f1_dev_thresholded')
            best_pr_curves_dev_thresholded.extend(best_pr_data_dev_f1)
        except IndexError:
            print("Warning: No dev thresholded F1 available, skipping")

        # Just to log the best checkpoint CURVE for each run
        best_pr_data = get_best_pr_data_by_f1(all_pr_data)
        best_pr_curves.extend(best_pr_data)

        all_retrieval_data = update_retrieval_figures(
            eval_dir,
            DATASETS['official_dev_small']['qrels_path'],
            DATASETS['official_dev_small']['collection_path']
        )

        # Get recall - F1 combinations for each checkpoint, so it can be plotted in the same figure
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


def get_dev_thresholded_filename(save_all_best_pr_curves):
    return save_all_best_pr_curves.replace('.jsonl', '_dev_thresholded.jsonl')


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


def load_all_stats(save_all_experiments_stats, save_all_best_pr_curves):
    with jsonlines.open(save_all_experiments_stats, 'r') as reader:
        combined_stats_all = list(reader)

    with jsonlines.open(save_all_best_pr_curves, 'r') as reader:
        best_pr_curves = list(reader)

    with jsonlines.open(get_dev_thresholded_filename(save_all_best_pr_curves), 'r') as reader:
        best_pr_curves_dev_thresholded = list(reader)

    return best_pr_curves, combined_stats_all, best_pr_curves_dev_thresholded


def main():
    args = parse_args()

    eval_dir = None
    if args.checkpoint is not None:
        eval_dir = inference_checkpoint_all_datasets(args.checkpoint)
    elif args.results_dir is not None:
        eval_dir = args.results_dir

    if eval_dir is None:
        assert args.evaluate_all
        eval_dirs = find_all_results_dirs()
    else:
        eval_dirs = [eval_dir]

    print("Evaluating found directories:\n" + '\n'.join([f'\t{e_dir}' for e_dir in eval_dirs]))
    if args.load_eval_stats:
        print("Evaluation of each directory was skipped. Loading all stats from files instead.")
        best_pr_curves, combined_stats_all, best_pr_curves_dev_thr = load_all_stats(
            args.save_all_experiments_stats, args.save_all_best_pr_curves
        )
    else:
        best_pr_curves, combined_stats_all, best_pr_curves_dev_thr = evaluate_all_dirs(
            eval_dirs, args.save_all_experiments_stats, args.save_all_best_pr_curves
        )

    connect_running_wandb('eval-all')
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

        # log_pareto_optimal_solutions(filtered_stats, group_name)
        log_best_pr_curve_wandb(filtered_pr_curves, group_name)
        # log_best_pr_curve_wandb(filtered_pr_curves_dev_thr, group_name, 'dev threshold')

    wandb.finish()


if __name__ == '__main__':
    main()
