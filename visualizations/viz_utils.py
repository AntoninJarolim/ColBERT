import re

import matplotlib
import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve


def create_highlighted_passage(passage_tokens, gt_label_list, annotation_scores,
                               ln_colours=False):

    highlighted_passage = []

    # Create default list of colours for each token
    colours_annotation_list = ["#00000000"] * len(passage_tokens)
    if annotation_scores is not None:
        colours_annotation_list = _annt_list_2_colours(annotation_scores, ln_colours)

    if gt_label_list is None:
        gt_label_list = [False] * len(passage_tokens)

    for bg_colour, token, gt_label in zip(colours_annotation_list, passage_tokens, gt_label_list):
        token = token.replace('$', '\$')
        span_text = _create_highlighted_word(token, bg_colour, gt_label)
        highlighted_passage.append(span_text)

    return highlighted_passage


def _create_highlighted_word(word, bg_color, highlight):
    # Create the HTML string with inline CSS for styling
    highlight_colour = "#060061"
    fg_colour = highlight_colour if highlight else "#000000"

    is_continuation = False
    if word.startswith('##'):
        is_continuation = True
        word = word[2:]

    left_padding = -0.25 if is_continuation else 0
    return f"""
        <span style="display: 
        inline-flex; 
        flex-direction: row; 
        align-items: center; 
        background: {bg_color}; 
        padding: 0.07rem 0.15rem;
        margin: 0.0rem 0.07rem 0.0rem {left_padding}rem;
        overflow: hidden; 
        line-height: 1; 
        color: {fg_colour};
        font-weight: {'bold' if highlight else 'normal'};
        ">                
        {word}
        </span>
    """


def num_to_colour(x):
    if torch.isnan(x):
        return f"#00000000"
    cmap = plt.cm.Wistia
    return str(matplotlib.colors.rgb2hex(cmap(x)))



def _annt_list_2_colours(annotation_list, ln_colours):
    if annotation_list is None:
        return None

    if not isinstance(annotation_list, torch.Tensor):
        annotation_list = [x if x is not None else torch.nan for x in annotation_list]
        annotation_list = torch.Tensor(annotation_list)

    min_val = np.nanmin(annotation_list.numpy())
    annotation_list = annotation_list - min_val

    max_val = np.nanmax(annotation_list.numpy())
    normalized_tensor_list = annotation_list / max_val
    if ln_colours:
        normalized_tensor_list = convert_scale_ln(normalized_tensor_list)

    # colour_range = torch.where(
    #     torch.isnan(normalized_tensor_list),  # Condition: Check for NaN
    #     torch.tensor(float('nan')),  # Keep NaN values as NaN
    #     (normalized_tensor_list * 255 + 0.5).to(torch.int64)  # Convert valid values to int64
    # )
    # print(colour_range)

    non_nan = normalized_tensor_list[~torch.isnan(normalized_tensor_list)]
    assert torch.all(non_nan >= 0) and torch.all(non_nan <= 1), f"Values out of range: {non_nan}"

    coloured_list = [num_to_colour(x) for x in normalized_tensor_list]
    return coloured_list


def convert_scale_ln(normalized_tensor_list):
    negative_index = torch.where(normalized_tensor_list < 0)
    normalized_tensor_list = torch.abs(normalized_tensor_list)
    transf_list = -torch.log(normalized_tensor_list)
    normalized_tensor_list = 1 - (transf_list / torch.max(transf_list))
    normalized_tensor_list[negative_index] = -normalized_tensor_list[negative_index]
    return normalized_tensor_list


def compute_all_recall_thresholds(viz_data):

    all_ex = []
    all_scores = []
    for d in viz_data:
        extractions = d['extraction_binary']
        scores = d['max_scores']
        assert len(extractions) == len(scores)

        all_ex.append(extractions)
        all_scores.append(scores)

    all_ex_flat = [x for xs in all_ex for x in xs]
    all_scores_flat = [x for xs in all_scores for x in xs]

    precision, recall, thresholds = precision_recall_curve(
        np.array(all_ex_flat), np.array(all_scores_flat), drop_intermediate=True
    )

    recall = recall[1:]  # Remove hard-code added '1' score

    recall, thresholds = recall.tolist(), thresholds.tolist()

    recall_th_dict = {r: t for r, t in zip(recall, thresholds)}

    return recall_th_dict


def hard_threshold(viz_data, threshold_value):
    for d in viz_data:
        d['max_scores_full'] = [
            None if x is None else float(x > threshold_value)
            for x
            in d['max_scores_full']
        ]

    return viz_data

