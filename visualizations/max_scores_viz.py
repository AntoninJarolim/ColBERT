import torch
import streamlit as st
import os
import glob

# Allows having script in a folder
import sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from colbert.data import Collection, Queries
from colbert.data.extractions import ExtractionResults
from colbert.infra import ColBERTConfig
from colbert.modeling.tokenization import DocTokenizer

from visualizations.viz_utils import create_highlighted_passage

st.set_page_config(layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Layout config
_, data_column, _ = st.columns([2, 6, 2])

# DATA
collection_path = "data/evaluation/collection.dev.small_50-25-25.tsv"
queries_path = "data/evaluation/queries.dev.small_ex_only.tsv"


@st.cache_resource
def cache_collection(collection_path, queries_path):
    collection = Collection.cast(collection_path)
    queries = Queries.cast(queries_path)
    return collection, queries


def find_all_checkpoints(root_dir):
    pattern = os.path.join(root_dir, '**', 'train', '**', 'inference_run_dir.txt')
    checkpoint_files = glob.glob(pattern, recursive=True)
    checkpoints = dict()
    for file in checkpoint_files:
        # Create key
        file_dir = os.path.dirname(file)
        splitted = file_dir.split('/')
        exp = splitted[1]
        assert splitted[2] == 'train'
        run_name = splitted[-3]
        steps = int(splitted[-1].split('-')[1])
        cp_key = f"{exp}/{run_name}/{steps}"

        # get inference run dir from file
        with open(file, 'r') as f:
            inference_path = f.read().strip()

        # get time added for sorting
        time_added = os.path.getmtime(file)

        # Save to dict
        checkpoints[cp_key] = {
            'inference_path': inference_path,
            'checkpoint_path': file_dir,
            'steps': steps,
            'time_added': time_added
        }

    # Sort by time added
    return dict(sorted(checkpoints.items(), key=lambda item: item[1]['time_added'], reverse=True))


collection, queries = cache_collection(collection_path, queries_path)
checkpoints = find_all_checkpoints('experiments/')

# Default data configuration
set_data = {
    "start_data_id": 0,
    "nr_show_data": 10,
    "gt_label_colour": "#2222DD",
    "checkpoint": None
}

# CONFIGURATION SIDEBAR
with st.sidebar:
    "## Configuration"
    "### Data loading"
    set_data["start_data_id"] = st.selectbox('First id to show:',
                                             range(len(collection)), key="start_data_id")

    set_data["checkpoint"] = st.selectbox('Model checkpoint:',
                                          list(checkpoints.keys()), key="checkpoint")

    "### Visualization settings"
    set_data["nr_show_data"] = st.selectbox('Number of passages to show:',
                                            [10, 20, 50, 100], key="dialogue_index")


@st.cache_resource
def cache_init_tokenizer(checkpoint):
    config = ColBERTConfig(
        checkpoint=checkpoint,
    )
    return DocTokenizer(config)


def get_viz_data(inference_dir, steps):
    scores_file = None
    for file in os.listdir(inference_dir):
        if file.endswith('extraction_scores.jsonl') and f"steps={steps}" in file:
            scores_file = os.path.join(inference_dir, file)
            break

    assert scores_file is not None, f"Could not find scores file in {inference_dir} with steps={steps}"

    return ExtractionResults.cast(scores_file)


def show_one_example(idx, passage_tokens, gt_label_list, annotation_scores,
                     additional_text="", ln_colours=False, ):
    """
    :param gt_label_list: list of True/False, True - GT word, False - no GT
    :param passage_tokens: list of strings of same length as annotation_scores
    :param annotation_scores: list of numbers used to colour the passage_text
    :param idx: ID of the passage to show
    :param ln_colours: if True, then annotation_scores are scaled by log fn
    """
    assert not isinstance(passage_tokens, str)

    with st.container(border=True):
        # Print psg index number in colour if grounding
        f'#### :blue[{idx} - {additional_text}]'

        highlighted_passage = create_highlighted_passage(passage_tokens, gt_label_list, annotation_scores,
                                                         ln_colours)
        st.html("\n".join(highlighted_passage))


def show_examples(starting_index, nr_show_data, viz_data: ExtractionResults, collection, queries, doc_tokenizer):
    for i in range(starting_index, starting_index + nr_show_data):
        example = viz_data[i]

        query = queries[example['q_id']]
        passage = collection[example['psg_id']]
        extraction_full = example['extraction_full']
        max_scores_full = example['max_scores_full']

        passage_tokens = doc_tokenizer.tensorize([passage])[0][0]  # First batch, ids only
        passage_tokens = doc_tokenizer.tok.batch_decode(passage_tokens)

        show_one_example(i, passage_tokens, extraction_full, max_scores_full, additional_text=query)


def gradient_test():
    tok_len = 255
    tokens = list(range(-255, tok_len, 2))
    gt = [0] * tok_len
    gt_1 = [1] * tok_len
    max_scores_full = [(t_score - 50) / tok_len for t_score in tokens]
    tokens = [str(t) for t in tokens]

    show_one_example(-1, tokens, gt, max_scores_full, additional_text="Test gradient")
    show_one_example(-1, tokens, gt_1, max_scores_full, additional_text="Test gradient gt 1")


with data_column:
    if set_data["checkpoint"] is not None:
        current_cp = checkpoints[set_data["checkpoint"]]

        viz_data = get_viz_data(current_cp['inference_path'], current_cp['steps'])
        doc_tokenizer = cache_init_tokenizer(current_cp['checkpoint_path'])
        show_examples(set_data["start_data_id"], set_data["nr_show_data"], viz_data, collection, queries, doc_tokenizer)
    else:
        st.write("No checkpoint selected")
