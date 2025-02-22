from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher


if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="msmarco")):

        config = ColBERTConfig(
            nbits=2,
            root="data/experiments",

        )
        indexer = Indexer(checkpoint='colbert-ir/colbertv1.9',  config=config)
        # collection_path = 'data/msmarco-2m-triplets/10000/collection.tsv'
        collection_path = 'data/35_sample_dataset/35/collection.tsv'
        indexer.index(name="msmarco.nbits=2", overwrite=True, collection=collection_path)

    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config = ColBERTConfig(
            root="data/experiments",
        )
        searcher = Searcher(index="msmarco.nbits=2", config=config)
        # queries_path = 'data/msmarco-2m-triplets/10000/queries.tsv'
        queries_path = 'data/35_sample_dataset/35/queries.tsv'
        queries = Queries(queries_path)
        ranking = searcher.search_all(queries, k=10)
        # ranking.save("msmarco.nbits=2.ranking.tsv")

