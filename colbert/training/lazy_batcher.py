import os
import ujson

from functools import partial

from colbert.data.extractions import Extractions
from colbert.infra.config.config import ColBERTConfig
from colbert.utils.utils import print_message, zipstar
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from colbert.evaluation.loaders import load_collection

from colbert.data.collection import Collection
from colbert.data.queries import Queries
from colbert.data.examples import Examples

# from colbert.utils.runs import Run


class LazyBatcher():
    def __init__(self, config: ColBERTConfig, triples, queries, collection, extracted_spans, rank=0, nranks=1):
        self.bsize, self.accumsteps = config.bsize, config.accumsteps
        self.nway = config.nway

        self.query_tokenizer = QueryTokenizer(config)
        self.doc_tokenizer = DocTokenizer(config)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.position = 0

        self.has_extractions = True if extracted_spans else False

        self.triples = Examples.cast(triples, nway=self.nway, has_extractions=self.has_extractions).tolist(rank, nranks)
        self.extracted_spans = Extractions.cast(extracted_spans) if extracted_spans else None
        self.queries = Queries.cast(queries)
        self.collection = Collection.cast(collection)
        assert len(self.triples) > 0, "Received no triples on which to train."
        assert len(self.queries) > 0, "Received no queries on which to train."
        assert len(self.collection) > 0, "Received no collection on which to train."

    def __iter__(self):
        self.position = 0
        return self

    def __len__(self):
        return len(self.triples)

    def __next__(self):
        offset, endpos = self.position, min(self.position + self.bsize, len(self.triples))
        self.position = endpos

        if offset + self.bsize > self.__len__():
            raise StopIteration

        all_queries, all_passages, all_scores, all_extractions = [], [], [], []

        for position in range(offset, endpos):
            if self.has_extractions:
                query_id, extractions_pids, *pids = self.triples[position]
            else:
                query_id, *pids = self.triples[position]
                extractions_pids = []

            pids = pids[:self.nway]

            query = self.queries[query_id]

            try:
                pids, scores = zipstar(pids)
            except:
                scores = []

            passages = [self.collection[pid] for pid in pids]
            extractions = [self.extracted_spans[(query_id, pid)] for pid in extractions_pids]

            all_queries.append(query)
            all_passages.extend(passages)
            all_scores.extend(scores)
            all_extractions.extend(extractions)
        
        assert len(all_scores) in [0, len(all_passages)], len(all_scores)

        return self.collate(all_queries, all_passages, all_scores, all_extractions)

    def collate(self, queries, passages, scores, all_extractions):
        assert len(queries) == self.bsize
        assert len(passages) == self.nway * self.bsize, (len(passages), self.nway, self.bsize)

        return self.tensorize_triples(queries, passages, scores, all_extractions,
                                      self.bsize // self.accumsteps, self.nway)

    # def skip_to_batch(self, batch_idx, intended_batch_size):
    #     Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')
    #     self.position = intended_batch_size * batch_idx
