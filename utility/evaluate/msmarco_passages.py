"""
    Evaluate MS MARCO Passages ranking.
"""

import os
import math
import tqdm
import ujson
import random

from argparse import ArgumentParser
from collections import defaultdict

from colbert.data import TranslateAbleCollection
from colbert.utils.utils import print_message, file_tqdm


def evaluate_ms_marco_ranking(collection_path, qrels_path, ranking_path,
                              output_annotations=None, annotate=None, silent=False):
    qid2positives = defaultdict(list)
    qid2ranking = defaultdict(list)
    qid2mrr = {}
    qid2recall = {depth: {} for depth in [50, 200, 1000, 5000, 10000]}

    collection = None
    if collection_path:
        collection = TranslateAbleCollection(path=collection_path)

    with open(qrels_path) as f:
        if not silent:
            print_message(f"#> Loading QRELs from {qrels_path} ..")
        for line in file_tqdm(f):
            qid, _, pid, label = map(int, line.strip().split())
            assert label == 1

            qid2positives[qid].append(pid)

    with open(ranking_path) as f:
        if not silent:
            print_message(f"#> Loading ranked lists from {ranking_path} ..")

        results = defaultdict(list)
        for line in file_tqdm(f):
            qid, *rest = line.strip().split('\t')
            results[int(qid)].append(rest)

        for qid, result_data in results.items():
            assert len(result_data) == 3
            pids = result_data[0]
            ranks = result_data[1]
            scores = result_data[2]

            for pid, rank, score in zip(pids, ranks, scores):
                if collection:
                    pid = collection.translate(int(pid))
                qid2ranking[qid].append((int(rank), int(pid), float(score)))

    assert set.issubset(set(qid2ranking.keys()), set(qid2positives.keys()))

    num_judged_queries = len(qid2positives)
    num_ranked_queries = len(qid2ranking)

    if num_judged_queries != num_ranked_queries:
        if not silent:
            print()
            print_message("#> [WARNING] num_judged_queries != num_ranked_queries")
            print_message(f"#> {num_judged_queries} != {num_ranked_queries}")
            print()

    if not silent:
        print_message(f"#> Computing MRR@10 for {num_judged_queries} queries.")

    for qid in tqdm.tqdm(qid2positives):
        ranking = qid2ranking[qid]
        positives = qid2positives[qid]

        for rank, (_, pid, _) in enumerate(ranking):
            rank = rank + 1  # 1-indexed

            if pid in positives:
                if rank <= 10:
                    qid2mrr[qid] = 1.0 / rank
                break

        for rank, (_, pid, _) in enumerate(ranking):
            rank = rank + 1  # 1-indexed

            if pid in positives:
                for depth in qid2recall:
                    if rank <= depth:
                        qid2recall[depth][qid] = qid2recall[depth].get(qid, 0) + 1.0 / len(positives)

    assert len(qid2mrr) <= num_ranked_queries, (len(qid2mrr), num_ranked_queries)

    mrr_10 = sum(qid2mrr.values()) / num_judged_queries
    mrr_10_ranked = sum(qid2mrr.values()) / num_ranked_queries
    if not silent:
        print()
        print_message(f"#> MRR@10 = {mrr_10}")
        print_message(f"#> MRR@10 (only for ranked queries) = {mrr_10_ranked}")
        print()

    dict_out = {"mrr@10": mrr_10}
    for depth in qid2recall:
        assert len(qid2recall[depth]) <= num_ranked_queries, (len(qid2recall[depth]), num_ranked_queries)

        metric_sum = sum(qid2recall[depth].values())
        if not silent:
            print()
            print_message(f"#> Recall@{depth} = {metric_sum / num_judged_queries}")
            print_message(f"#> Recall@{depth} (only for ranked queries) = {metric_sum / num_ranked_queries}")
            print()

        dict_out[f"recall@{depth}"] = metric_sum / num_judged_queries

    if annotate:
        if not silent:
            print_message(f"#> Writing annotations to {output_annotations} ..")

        with open(output_annotations, 'w') as f:
            for qid in tqdm.tqdm(qid2positives):
                ranking = qid2ranking[qid]
                positives = qid2positives[qid]

                for rank, (_, pid, score) in enumerate(ranking):
                    rank = rank + 1  # 1-indexed
                    label = int(pid in positives)

                    line = [qid, pid, rank, score, label]
                    line = [x for x in line if x is not None]
                    line = '\t'.join(map(str, line)) + '\n'
                    f.write(line)

    return dict_out


if __name__ == "__main__":
    parser = ArgumentParser(description="msmarco_passages.")

    # Input Arguments.
    parser.add_argument('--qrels', dest='qrels', required=True, type=str)
    parser.add_argument('--ranking', dest='ranking', required=True, type=str)
    parser.add_argument('--annotate', dest='annotate', default=False, action='store_true')
    parser.add_argument('--collection', dest='collection', type=str, default=None)

    args = parser.parse_args()

    args.output = None
    if args.annotate:
        args.output = f'{args.ranking}.annotated'
        assert not os.path.exists(args.output), args.output

    evaluate_ms_marco_ranking(args.collection,
                              args.qrels,
                              args.ranking,
                              annotate=args.annotate,
                              output_annotations=args.output)
