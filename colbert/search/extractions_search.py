import torch
import ujson
from jsonlines import jsonlines

from colbert.modeling.tokenization.utils import extract_positions, get_span_masks


class ExtractionSearcher:
    def __init__(self, extraction_list):
        self.extraction_list = extraction_list

    def extract(self, search_term):
        return [extraction for extraction in self.extraction_list if search_term in extraction]

    def _load_extractions_result(self, path):
        data = {}
        with jsonlines.open(path) as reader:
            for line in reader:
                q_id = line['q_id']
                psg_id = line['psg_id']
                data[(q_id, psg_id)] = line['extractions']
        return data


    def get_collection(self):
        collection = {}
        with open("data/evaluation/collection.dev.small_50-25-25.tsv") as reader:
            for line in reader:
                psg_id, psg = line.strip().split('\t')
                collection[int(psg_id)] = psg
        return collection

    def _from_ranking(self, results, all_extractions, checkpoint):
        keep_list = []
        translation_dict = self.get_translation_dict()
        collection = self.get_collection()

        text_batches, sorted_length = checkpoint.doc_tokenizer.tensorize(
            list(collection.values()), bsize=64
        )
        collections_tokenized = torch.cat([val[0] for val in text_batches])[sorted_length]
        # collections_masks = torch.cat([val[1] for val in text_batches])[sorted_length]

        colbert_skip_list_mask = checkpoint.mask(collections_tokenized, skiplist=checkpoint.skiplist)

        scores_from = 2  # skip scores for [CLS] and [D]
        scores_to = -1  # skip scores for [SEP]
        collections_tokenized_list = [tokenized_text[mask][scores_from:scores_to]
                                      for mask, tokenized_text
                                      in zip(colbert_skip_list_mask, collections_tokenized)]

        # collection_lengths = [len(val) for val in collections_tokenized_list]

        for qid, val in results.items():
            for psg_id, rank, max_scores in zip(val['pids'], val['ranking'], val['max_scores']):
                try:
                    extractions = all_extractions[(qid, translation_dict[str(psg_id)])]
                except KeyError:
                    continue

                psg_text_toks = collections_tokenized_list[psg_id]
                psg_text = collection[psg_id]

                scores = [score for score in max_scores if -1000 < score < 1000][scores_from:scores_to]

                ids, mask, offsets, mask_special = checkpoint.doc_tokenizer.tensorize([psg_text])
                extractions_positions = extract_positions([psg_text], [extractions])
                extraction_binary = get_span_masks(offsets, extractions_positions).squeeze(0)[scores_from:scores_to]

                current_mask = colbert_skip_list_mask[psg_id][2:]
                # z extraction_binary potrebujes vymaskovat skiplist tokeny
                # skiplist toky mas v colbert_skip_list_mask[psg_id]
                # problem je ze colbert_skip_list_mask[psg_id] je fest velke - obsahuje i padding tokeny
                # padding tokeky by v colbert_skip_list_mask[psg_id] takze na toto muzes napsat assert
                # a maskovat prvni pulkou listu
                # extraction_masks je o jedno vetsi nez ma byt, cim by to mohlo byt, doslova
                # vybiras jen ty neskipnute
                before, after = current_mask[:len(extraction_binary)], current_mask[len(extraction_binary):]
                assert sum(after[1:]) == 0 and after[0], (
                    current_mask[len(extraction_binary):], " contains non zero values")
                extraction_masked = extraction_binary[before].tolist()

                assert len(psg_text_toks) == len(scores) == len(extraction_masked)

                keep_list.append({
                    'q_id': qid,
                    'psg_id': psg_id,
                    'rank': rank,
                    'max_score': scores,
                    'extraction_masked': extraction_masked
                })

        print(f"#> For {len(keep_list)}/{len(all_extractions)} were relevant extraction in top-k.")

        return keep_list
