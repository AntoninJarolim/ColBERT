from os import environ

import wandb

from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer


def train(experiment, ngpus, ex_lambda):
    with Run().context(RunConfig(nranks=ngpus, experiment=experiment)):
        triples_path = 'data/training/examples_with_relevancy.jsonl'
        queries_path = 'data/training/queries.train.tsv'
        collection_path = 'data/training/collection.tsv'
        extractions_path = 'data/training/extracted_relevancy_800k_unique.tsv'

        one_bsize = 128
        config = ColBERTConfig(bsize=one_bsize * ngpus, lr=1e-05, warmup=20_000,
                               doc_maxlen=180, dim=128, attend_to_mask_tokens=False,
                               return_max_scores=True, extractions_lambda=ex_lambda,
                               nway=64, accumsteps=2, similarity='cosine', use_ib_negatives=True)
        trainer = Trainer(triples=triples_path, queries=queries_path, collection=collection_path,
                          extractions=extractions_path, config=config)

        trainer.train(checkpoint='colbert-ir/colbertv1.9')  # or start from scratch, like `bert-base-uncased`


def arg_parse():
    import argparse
    parser = argparse.ArgumentParser(description='Train ColBERT with extractions')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--ngpus', type=int, default=2, help='Number of GPUs')
    parser.add_argument('--ex_lambda', type=float, default=0.5, help='Extractions lambda')
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    train(args.experiment, args.ngpus, args.ex_lambda)

