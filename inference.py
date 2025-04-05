import argparse
import glob
import os.path

import wandb

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
from evaluation import update_retrieval_figures, update_extractions_figures


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


def inference_checkpoint_all_datasets(checkpoint):
    datasets = {
        'official_dev_small': {
            'collection_path': 'data/evaluation/collection.dev.small_50-25-25.tsv',
            'queries_path': 'data/evaluation/queries.dev.small_ex_only.tsv',
            'extraction_path': 'data/evaluation/extracted_relevancy_qrels.dev.small.tsv',
            'qrels_path': 'data/evaluation/qrels.dev.small_ex_only.tsv',
        },
        '35_samples': {
            'collection_path': 'data/evaluation/collection.35_sample_dataset.tsv',
            'queries_path': 'data/evaluation/queries.eval.35_sample_dataset.tsv',
            'extraction_path': 'data/evaluation/extracted_relevancy_35_sample_dataset.tsv',
            'qrels_path': None
        }
    }

    eval_datasets = []
    for collection_name, data in datasets.items():
        eval_dataset = inference_checkpoint_one_dataset(
            checkpoint,
            collection_name,
            data['collection_path'],
            data['queries_path'],
            data['extraction_path'],
            data['qrels_path']
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

    eval_dir = os.path.dirname(max_ranking_path)
    assert eval_dir == os.path.dirname(ranking_path)

    if qrels_path is not None:
        update_retrieval_figures(eval_dir, qrels_path, collection_path)
    update_extractions_figures(eval_dir)
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
    return parser.parse_args()


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

    for eval_dir in eval_dirs:
        run_name = eval_dir.strip('/').split('/')[-1]
        connect_running_wandb(run_name)
        update_extractions_figures(eval_dir)
        # todo: check why it is not possible to also evaluate retrieval again
        wandb.finish()


if __name__ == '__main__':
    main()
