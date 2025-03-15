import os
import torch
import math

from tqdm import tqdm
from typing import Union

from colbert.data import Collection, Queries, Ranking, TranslateAbleCollection
from colbert.data.extractions import ExtractionResults, Extractions

from colbert.modeling.checkpoint import Checkpoint
from colbert.modeling.colbert import colbert_score
from colbert.modeling.tokenization import tensorize_triples
from colbert.search.index_storage import IndexScorer

from colbert.infra.provenance import Provenance
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert.infra.launcher import print_memory_stats

TextQueries = Union[str, 'list[str]', 'dict[int, str]', Queries]


class Searcher:
    def __init__(self, index, checkpoint=None, collection=None, config=None, index_root=None, extractions=None,
                 verbose: int = 3):
        self.verbose = verbose
        if self.verbose > 1:
            print_memory_stats()

        initial_config = ColBERTConfig.from_existing(config, Run().config)

        default_index_root = initial_config.index_root_
        index_root = index_root if index_root else default_index_root
        self.index = os.path.join(index_root, index)
        self.index_config = ColBERTConfig.load_from_index(self.index)

        self.checkpoint = checkpoint or self.index_config.checkpoint
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(self.checkpoint)
        self.config = ColBERTConfig.from_existing(self.checkpoint_config, self.index_config, initial_config)

        self.collection = Collection.cast(collection or self.config.collection)
        self.configure(checkpoint=self.checkpoint, collection=self.collection)

        self.checkpoint = Checkpoint(self.checkpoint, colbert_config=self.config, verbose=self.verbose)
        use_gpu = self.config.total_visible_gpus > 0
        if use_gpu:
            self.checkpoint = self.checkpoint.cuda()
        load_index_with_mmap = self.config.load_index_with_mmap
        if load_index_with_mmap and use_gpu:
            raise ValueError(f"Memory-mapped index can only be used with CPU!")
        self.ranker = IndexScorer(self.index, use_gpu, load_index_with_mmap)

        print_memory_stats()

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def encode(self, text: TextQueries, full_length_search=False):
        queries = text if type(text) is list else [text]
        bsize = 128 if len(queries) > 128 else None

        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        Q = self.checkpoint.queryFromText(queries, bsize=bsize, to_cpu=True, full_length_search=full_length_search)

        return Q

    def encode_queries(self, queries, full_length_search=False):
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        return self.encode(queries_, full_length_search=full_length_search)

    def search(self, text: str, k=10, filter_fn=None, full_length_search=False, pids=None):
        Q = self.encode(text, full_length_search=full_length_search)
        return self.dense_search(Q, k, filter_fn=filter_fn, pids=pids)

    def search_all(self, queries: TextQueries, k=10, filter_fn=None, full_length_search=False, qid_to_pids=None):
        Q = self.encode_queries(queries, full_length_search=full_length_search)
        return self._search_all_Q(queries, Q, k, filter_fn=filter_fn, qid_to_pids=qid_to_pids)

    def _search_all_Q(self, queries, Q, k, filter_fn=None, qid_to_pids=None):
        qids = list(queries.keys())

        if qid_to_pids is None:
            qid_to_pids = {qid: None for qid in qids}

        all_scored_pids = {
            qid: self.dense_search(
                Q[query_idx:query_idx + 1],
                k, filter_fn=filter_fn,
                pids=qid_to_pids[qid]
            )
            for query_idx, qid in tqdm(enumerate(qids),
                                       desc="searcing top-k docs for queries",
                                       total=len(qids))
        }

        # note: all_scored_pids may also contain max_scores
        ranking_data = {qid: (val['pids'], val['ranking'], val['scores']) for qid, val in all_scored_pids.items()}
        provenance = Provenance()
        provenance.source = 'Searcher::search_all'
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k

        ranking = Ranking(data=ranking_data, provenance=provenance)
        return ranking

    def dense_search(self, Q: torch.Tensor, k=10, filter_fn=None, pids=None):
        if k <= 10:
            if self.config.ncells is None:
                self.configure(ncells=1)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.5)
            if self.config.ndocs is None:
                self.configure(ndocs=256)
        elif k <= 100:
            if self.config.ncells is None:
                self.configure(ncells=2)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.45)
            if self.config.ndocs is None:
                self.configure(ndocs=1024)
        else:
            if self.config.ncells is None:
                self.configure(ncells=4)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.4)
            if self.config.ndocs is None:
                self.configure(ndocs=max(k * 4, 4096))

        pids, scores = self.ranker.rank(self.config, Q, filter_fn=filter_fn, pids=pids)

        if self.config.return_max_scores:
            scores, max_scores = scores

        return {
            'pids': pids[:k],
            'ranking': list(range(1, k + 1)),
            'scores': scores[:k],
            'max_scores': max_scores[:k] if self.config.return_max_scores else None
        }

    def search_extractions(self, queries, extractions, translate_dict_path):
        assert self.config.return_max_scores, "return_max_scores must be set to True for search_extractions."

        extractions = Extractions.cast(extractions)
        self.collection = TranslateAbleCollection.cast(self.collection)

        data = []
        for index, q_id in enumerate(queries.keys()):
            relevant_d_pids = extractions.get_psg_by_qid(q_id)
            relevant_pid = self.collection.translate_rev(relevant_d_pids)
            data.append(
                {
                    'q_id': q_id,
                    'psg_id': relevant_pid,
                    'query': queries[q_id],
                    'passage': self.collection[relevant_pid],
                    'extraction_spans': extractions[(q_id, relevant_d_pids)],
                }
            )

        # Collect data for the extraction search
        extraction_spans_all = [d['extraction_spans'] for d in data]
        passages = [d['passage'] for d in data]
        queries_text = [d['query'] for d in data]
        extractions_data = self.get_all_extractions(extraction_spans_all,
                                                    passages,
                                                    queries_text)

        # Each doc is different length, therefore we cannot do D @ Q, but need to compute scores one by one
        for i in tqdm(range(len(extractions_data['q_ids'])), desc="Finding max values for all queries"):

            # Get all data for this sample
            q_id = extractions_data['q_ids'][i]
            doc_id = extractions_data['doc_ids_masked'][i]
            doc_ids_colbert_mask = extractions_data['doc_ids_colbert_mask'][i]
            extractions_masked = extractions_data['binary_extractions_masked'][i]
            extractions = extractions_data['binary_extractions'][i]

            max_scores = self.extract_max_scores(doc_id, q_id)

            assert len(max_scores) == len(extractions_masked)

            # Create full length max_scores tensor
            max_scores_viz = torch.full_like(doc_ids_colbert_mask, torch.nan, dtype=max_scores.dtype)
            max_scores_viz[doc_ids_colbert_mask] = max_scores
            max_scores_viz = [(None if math.isnan(x) else x) for x in max_scores_viz.tolist()]  # replace nan -> None

            assert len(extractions) == len(max_scores_viz)

            data[i]['max_scores'] = max_scores
            data[i]['extraction_binary'] = extractions_masked

            data[i]['max_scores_full'] = max_scores_viz
            data[i]['extraction_full'] = extractions

        max_scoring = ExtractionResults.cast(data)
        return max_scoring

    def extract_max_scores(self, doc_id, q_id):
        q_id = q_id.unsqueeze(0)
        doc_id = doc_id.unsqueeze(0)

        q_mask = torch.ones_like(q_id).bool()
        Q = self.checkpoint.query(q_id, q_mask, normalize=False)

        d_mask = torch.ones_like(doc_id).bool()
        D = self.checkpoint.doc(doc_id, d_mask, normalize=False)

        _, max_scores = colbert_score(Q, D, d_mask, self.config)
        max_scores = max_scores.cpu()
        return max_scores.squeeze()

    def get_all_extractions(self, extraction_spans_all, passages, queries_text):
        nway = 1  # Because there is only one passage per query
        batch_size = len(passages)  # to get only one batch
        distill_scores = [None] * len(passages)  # something empty - will not be used anyway
        triplets = tensorize_triples(
            self.checkpoint.query_tokenizer,
            self.checkpoint.doc_tokenizer,
            queries_text,
            passages,
            distill_scores,
            extraction_spans_all,
            batch_size,
            nway)
        (q_ids, _), (doc_ids, doc_ids_pad_masks, _), _, binary_extractions = triplets[0]  # only one batch

        # Create colbert mask by masking doc_ids by skiplist and padding
        doc_ids_colbert_mask = self.checkpoint.mask(doc_ids, skiplist=self.checkpoint.skiplist)
        doc_ids_pad_masks = doc_ids_pad_masks.bool().cpu()
        doc_ids_colbert_mask = [torch.tensor(mask)[pad_mask]
                                for mask, pad_mask
                                in zip(doc_ids_colbert_mask, doc_ids_pad_masks)]

        def apply_masks(tensor, masks):
            out = []
            for i, t in enumerate(tensor):
                for mask in masks:
                    t = t[mask[i]]
                out.append(t)
            return out

        all_masks = [doc_ids_pad_masks, doc_ids_colbert_mask]
        out = dict(
            doc_ids_colbert_mask=doc_ids_colbert_mask,
            binary_extractions_masked=apply_masks(binary_extractions, all_masks),
            binary_extractions=apply_masks(binary_extractions, [doc_ids_pad_masks]),
            doc_ids_masked=apply_masks(doc_ids, all_masks),
            q_ids=q_ids,
        )

        return out
