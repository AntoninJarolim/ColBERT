import argparse
import os.path

from pandas.core.algorithms import rank
import wandb

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
from evaluation import evaluate_extractions


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


def search_dataset(root_folder, queries_path, index_name):
    config = ColBERTConfig(
        root=root_folder
    )
    searcher = Searcher(index=index_name,
                        config=config)
    queries = Queries(queries_path)

    ranking = searcher.search_all(queries, k=3)
    max_ranking = searcher.search_extractions(queries,
                                              'data/evaluation/extracted_relevancy_qrels.dev.small.tsv',
                                              "data/evaluation/collection.dev.small_50-25-25.translate_dict.json")

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


def connect_running_wandb(run_name, config):
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
    wandb.init(project=project_name, name=run_name, config=config)


def inference_qrels_small_dataset(checkpoint, experiment, ranking_path=None):
    # Data
    collection_path = 'data/evaluation/collection.dev.small_50-25-25.tsv'
    queries_path = 'data/evaluation/queries.dev.small_ex_only.tsv'

    # inference
    root_folder = 'experiments'
    ex_name, run_name = get_run_name(checkpoint)
    assert ex_name == experiment, (ex_name, run_name)
    print(f"Current run name: {Run().config.name}")

    nbits = 2
    config_search = {
        'checkpoint': checkpoint,
        'collection_path': collection_path,
        'queries_path': queries_path,
        'root_folder': root_folder,
        'nbits': nbits,
        'index_name': f"nbits={nbits}",
        'checkpoint_steps': get_checkpoint_steps(checkpoint)
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
    if ranking_path is None:
        with Run().context(run_config):
            # Run indexing and retrieval
            # todo: dont run indexing twice
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
                config_search['index_name']
            )

    assert ranking_path is not None, "Inference failed"

    log_config = {
        **config_search,
        **config_run,
        'ranking_path': ranking_path,
        'max_ranking_path': max_ranking_path
    }
    connect_running_wandb(run_name, log_config)

    # Evaluate extracted relevancy
    evaluate_extractions(max_ranking_path, config_search['checkpoint_steps'])

    # Evaluate retrieval
    wandb.run.finish()
    return os.path.dirname(ranking_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Inference runner')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--experiment', type=str, help='Path to the current experiment')
    parser.add_argument('--ranking_path', type=str, help='Path to the ranking file')
    return parser.parse_args()


def main():
    # 35 sample dataset
    # queries_path = 'data/35_sample_dataset/35/queries.tsv'
    # collection_path = 'colbert_data/35_sample_dataset/35/collection.tsv'

    args = parse_args()  # > Encoding
    # checkpoint = 'colbert-ir/colbertv1.9'

    inference_qrels_small_dataset(args.checkpoint, args.experiment, args.ranking_path)


if __name__ == '__main__':
    main()
