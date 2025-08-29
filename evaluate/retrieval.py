import json
import os
import re
from collections import defaultdict

import wandb

from evaluate.wandb_logging import wandb_log_retrieval_figs
from utility.evaluate.msmarco_passages import evaluate_ms_marco_ranking


def evaluate_retrieval_checkpoint(ranking_path, qrels_path, collection_path, index_name):
    # Make evaluation of current run
    out_json = evaluate_ms_marco_ranking(collection_path, qrels_path, ranking_path, silent=True)
    ranking_path_dir = os.path.dirname(ranking_path)
    out_json_path = os.path.join(ranking_path_dir, f"{index_name}.retrieval_evaluation.json")
    return out_json_path

    # Save the evaluation to json
    with open(out_json_path, "w") as f:
        json.dump(out_json, f, indent=4)


def update_retrieval_figures(evaluation_dir, qrels_path, collection_path):
    # Generate evaluation jsons from ranking.tsv
    pattern = re.compile(r'col_name=(.+)\.nbits=\d+\.steps=(\d+)\.ranking.tsv')

    for file in os.listdir(evaluation_dir):
        match = pattern.search(file)
        if match:

            col_name = match.group(1)
            steps = int(match.group(2))

            # Skip 35 samples dataset - not for retrieval
            if col_name == '35_samples' or col_name == 'human_explained':
                continue

            # Evaluate ranking
            ranking_path = os.path.join(evaluation_dir, file)
            out_json = evaluate_ms_marco_ranking(collection_path, qrels_path, ranking_path, silent=True)
            out_json["batch_steps"] = steps

            # Save the evaluation to json
            index_name = file[:-len('.ranking.tsv')]
            out_json_path = os.path.join(evaluation_dir, f"{index_name}.retrieval_evaluation.json")
            with open(out_json_path, "w") as f:
                json.dump(out_json, f, indent=4)

    agg_retrieval_data = get_coll_agg_retrieval_data(evaluation_dir)
    wandb_log_retrieval_figs(agg_retrieval_data)

    return agg_retrieval_data


def get_coll_agg_retrieval_data(evaluation_dir):
    # Use all evaluation jsons in the directory to generate figures/tables
    pattern = re.compile(r'col_name=(.+)\.nbits=\d+\.steps=(\d+)\.retrieval_evaluation\.json')
    file_datas = defaultdict(list)
    for file in os.listdir(evaluation_dir):
        match = pattern.search(file)
        if match:
            collection_name = match.group(1)
            with open(os.path.join(evaluation_dir, file), "r") as matched_file:
                data = json.load(matched_file)
            file_datas[collection_name].append(data)
    return file_datas



