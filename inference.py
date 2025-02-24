import argparse
import os.path

from pandas.core.algorithms import rank

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def index_dataset(checkpoint, collection_path, root_folder):
    config = ColBERTConfig(
        nbits=2,
        root=root_folder
    )
    indexer = Indexer(
        checkpoint=checkpoint,
        config=config)
    indexer.index(name="msmarco.nbits=2", overwrite=True, collection=collection_path)


def search_dataset(root_folder, queries_path):
    config = ColBERTConfig(
        root=root_folder
    )
    searcher = Searcher(index="msmarco.nbits=2", config=config)
    queries = Queries(queries_path)
    ranking = searcher.search_all(queries, k=100)
    return ranking.save("msmarco.nbits=2.ranking.tsv")


def get_run_name(checkpoint):
    run_name = remove_prefix(checkpoint, "experiments/").split('/')[0]
    assert  run_name not in ['experiments', 'train', 'inference', 'index'], run_name
    return run_name


def inference_qrels_small_dataset(checkpoint, experiment):
    # Data
    collection_path = 'data/evaluation/collection.dev.small_50-25-25.tsv'
    queries_path = 'data/evaluation/queries.dev.small_ex_only.tsv'

    # inference
    root_folder = 'experiments'

    # Running
    with Run().context(RunConfig(nranks=1, experiment=experiment)):
        Run().config.name = get_run_name(checkpoint)
        print(f"Current run name: {Run().config.name}")
        checkpoint = os.path.realpath(checkpoint)
        index_dataset(checkpoint, collection_path, root_folder)
        search_results_path = search_dataset(root_folder, queries_path)
        results_path = os.path.dirname(search_results_path)

        # todo: python -m utility.evaluate.msmarco_passages --ranking "/path/to/msmarco.nbits=2.ranking.tsv" --qrels "/path/to/MSMARCO/qrels.dev.small.tsv"
        # todo: run my own evaluation script
        # todo: use Run().config.name to save the results in the right folder (take a look h and to wandb
    return results_path


def parse_args():
    parser = argparse.ArgumentParser(description='Inference runner')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--experiment', type=str, help='Path to the current experiment')
    return parser.parse_args()


def main():
    # 35 sample dataset
    # queries_path = 'data/35_sample_dataset/35/queries.tsv'
    # collection_path = 'colbert_data/35_sample_dataset/35/collection.tsv'

    args = parse_args()#> Encoding
    # checkpoint = 'colbert-ir/colbertv1.9'

    inference_qrels_small_dataset(args.checkpoint, args.experiment)


if __name__ == '__main__':
    main()
