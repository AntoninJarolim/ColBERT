from os import environ

import wandb

from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer


def train(experiment, ngpus, ex_lambda, accumsteps, checkpoint, add_max_linear, epochs):
    with Run().context(RunConfig(nranks=ngpus, experiment=experiment)):
        triples_path = 'data/training/examples_with_relevancy.jsonl'
        queries_path = 'data/training/queries.train.tsv'
        collection_path = 'data/training/collection.tsv'
        extractions_path = 'data/training/extracted_relevancy_800k_unique.tsv'

        one_bsize = 64
        config = ColBERTConfig(bsize=one_bsize * ngpus, lr=1e-05, warmup=20_000,
                               doc_maxlen=180, dim=128, attend_to_mask_tokens=False, epochs=epochs,
                               return_max_scores=True, extractions_lambda=ex_lambda, add_max_linear=add_max_linear,
                               nway=64, accumsteps=accumsteps, similarity='cosine', use_ib_negatives=True)
        trainer = Trainer(triples=triples_path, queries=queries_path, collection=collection_path,
                          extractions=extractions_path, config=config)

        trainer.train(checkpoint=checkpoint)  # or start from scratch, like `bert-base-uncased`


def arg_parse():
    import argparse
    parser = argparse.ArgumentParser(description='Train ColBERT with extractions')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name')
    parser.add_argument('--ngpus', type=int, default=2, help='Number of GPUs')
    parser.add_argument('--epochs', type=int, default=1, help='Number epochs to run training on')
    parser.add_argument('--ex_lambda', type=float, default=0.5, help='Extractions lambda')
    parser.add_argument('--accumsteps', type=int, default=2, help='Number of gradient accumulation steps')
    parser.add_argument('--checkpoint', type=str,
                        default='colbert-ir/colbertv1.9', help='Checkpoint to start training from')
    parser.add_argument('--add_max_linear', action='store_true', help='Add max linear layer', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    train(
        args.experiment,
        args.ngpus,
        args.ex_lambda,
        args.accumsteps,
        args.checkpoint,
        args.add_max_linear,
        args.epochs
    )

