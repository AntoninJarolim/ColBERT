from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer


def train():
    with Run().context(RunConfig(nranks=2)):
        triples = 'data/examples_with_relevancy.jsonl'
        queries = 'data/queries.train.tsv'
        collection = 'data/collection.tsv'
        extractions = 'data/extracted_relevancy.tsv'

        config = ColBERTConfig(bsize=64, lr=1e-05, warmup=20_000, doc_maxlen=180, dim=128, attend_to_mask_tokens=False,
                               return_max_scores=True,
                               nway=64, accumsteps=16, similarity='cosine', use_ib_negatives=True)
        trainer = Trainer(triples=triples, queries=queries, collection=collection,
                          extractions=extractions, config=config)

        trainer.train(checkpoint='colbert-ir/colbertv1.9')  # or start from scratch, like `bert-base-uncased`


if __name__ == '__main__':
    train()
