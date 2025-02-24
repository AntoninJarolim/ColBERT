from os import environ

import wandb

from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer


def train(experiment):
    with Run().context(RunConfig(nranks=2, experiment=experiment)):
        triples = 'data/examples_with_relevancy.jsonl'
        queries = 'data/queries.train.tsv'
        collection = 'data/collection.tsv'
        extractions = 'data/extracted_relevancy.tsv'

        config = ColBERTConfig(bsize=128, lr=1e-05, warmup=20_000, doc_maxlen=180, dim=128, attend_to_mask_tokens=False,
                               return_max_scores=True,
                               nway=64, accumsteps=32, similarity='cosine', use_ib_negatives=True)
        trainer = Trainer(triples=triples, queries=queries, collection=collection,
                          extractions=extractions, config=config)

        trainer.train(checkpoint='colbert-ir/colbertv1.9')  # or start from scratch, like `bert-base-uncased`


def arg_parse():
    import argparse
    parser = argparse.ArgumentParser(description='Train ColBERT with extractions')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    train(args.experiment)
