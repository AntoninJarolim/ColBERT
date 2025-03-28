import argparse
import os.path

from pandas.core.algorithms import rank
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
    project_name = "eval-llm2colbert-BCE"
    entity = "jarolim-antonin-brno-university-of-technology"  # todo: put me to env/config
    run_name = 'eval_' + run_name

    # Fetch all runs in the project
    runs = api.runs(f"{entity}/{project_name}")

    # Find the run with the specified name
    for run in runs:
        if run.name == run_name:
            print(f"Resuming found run ID: {run.id}")
            wandb.init(project=project_name, id=run.id, resume="must")

    print("Starting a new eval run")
    wandb.init(project=project_name, name=run_name)


def inference_checkpoint_all_datasets(checkpoint, experiment):
    datasets = {
        'official_dev_small': {
            'collection_path': 'data/evaluation/collection.dev.small_50-25-25.tsv',
            'queries_path': 'data/evaluation/queries.dev.small_ex_only.tsv',
            'extraction_path': 'data/evaluation/extracted_relevancy_qrels.dev.small.tsv',
            'qrels_path': 'data/evaluation/qrels.dev.small_ex_only.tsv',
        },
        '35_samples': {
            'collection_path': 'data/evaluation/collection.35_sample_dataset.tsv',
            'queries_path': 'daqueries.eval.35_sample_dataset.tsv',
            'extraction_path': 'data/evaluation/extracted_relevancy_35_sample_dataset.tsv',
            'qrels_path': None
        }
    }

    eval_datasets = []
    for collection_name, data in datasets.items():
        eval_dataset = inference_checkpoint_one_dataset(
            checkpoint,
            experiment,
            collection_name,
            data['collection_path'],
            data['queries_path'],
            data['extraction_path'],
            data['qrels_path']
        )
        eval_datasets.append(eval_dataset)

    assert eval_datasets[0] == eval_datasets[1]

    return eval_datasets[0]


def inference_checkpoint_one_dataset(
        checkpoint,
        experiment,
        collection_name,
        collection_path,
        queries_path,
        extraction_path,
        qrels_path,
):
    # inference
    root_folder = 'experiments'
    ex_name, run_name = get_run_name(checkpoint)
    assert ex_name == experiment, (ex_name, run_name)
    print(f"Current run name: {Run().config.name}")

    nbits = 2
    checkpoint_steps = get_checkpoint_steps(checkpoint)
    config_search = {
        'checkpoint': checkpoint,
        'collection_path': collection_path,
        'queries_path': queries_path,
        'extraction_path': extraction_path,
        'root_folder': root_folder,
        'nbits': nbits,
        'index_name': f"nbits={nbits}.steps={checkpoint_steps}.col_name={collection_name}",
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

    return eval_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Inference runner')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--experiment', type=str, help='Path to the current experiment')
    parser.add_argument('--results_dir', type=str,
                        help='Path to dir with ranking and evaluation. Inference wont be run if provided',
                        default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    eval_dir = args.results_dir
    if eval_dir is None:
        eval_dir = inference_checkpoint_all_datasets(
            args.checkpoint, args.experiment
        )

    if wandb.run is None:
        run_name = eval_dir.strip('/').split('/')[-1]
        connect_running_wandb(run_name)

    # update_retrieval_figures(eval_dir, qrels_path, collection_path)
    # not possible to do now, because we don't have qrels_path and collection_path
    # matters only if you would like to compute more retrieval metrics

    update_extractions_figures(eval_dir)
    wandb.run.finish()


if __name__ == '__main__':
    main()
