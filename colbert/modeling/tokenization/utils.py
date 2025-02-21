import torch
import string
from itertools import zip_longest


def tensorize_triples(query_tokenizer, doc_tokenizer, queries, passages, scores, extractions, bsize, nway):
    # assert len(passages) == len(scores) == bsize * nway
    # assert bsize is None or len(queries) % bsize == 0

    # N = len(queries)
    Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    D_ids, D_mask, offset_mapping, D_mask_special = doc_tokenizer.tensorize(passages)
    # D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)

    extraction_labels = mask_from_offsets(bsize, extractions, nway, offset_mapping, passages)
    assert extraction_labels.shape == (bsize, D_ids.size(1))


    # # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    # maxlens = D_mask.sum(-1).max(0).values

    # # Sort by maxlens
    # indices = maxlens.sort().indices
    # Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
    # D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]

    # (positive_ids, negative_ids), (positive_mask, negative_mask) = D_ids, D_mask

    query_batches = _split_into_batches(Q_ids, Q_mask, bsize)
    doc_batches = _split_into_batches3(D_ids, D_mask, D_mask_special, bsize * nway)
    extractions_batches = _split_into_batches2(extraction_labels, bsize)
    # positive_batches = _split_into_batches(positive_ids, positive_mask, bsize)
    # negative_batches = _split_into_batches(negative_ids, negative_mask, bsize)

    if len(scores):
        score_batches = _split_into_batches2(scores, bsize * nway)
    else:
        score_batches = [[] for _ in doc_batches]

    batches = []
    for Q, D, S, E in zip(query_batches, doc_batches, score_batches, extractions_batches):
        batches.append((Q, D, S, E))

    return batches


def mask_from_offsets(bsize, extractions, nway, offset_mapping, passages):
    # Get only first passage of each batch
    offset_mapping = torch.cat([offset_mapping[i * nway][None] for i in range(bsize)], dim=0)
    passages_first_pos = [passages[i * nway] for i in range(bsize)]

    # find starts and ends of spans
    extractions = extract_positions(passages_first_pos, extractions)
    # get masks by comparing with offset mapping
    extraction_masks = get_span_masks(offset_mapping, extractions)
    return extraction_masks


def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def _split_into_batches(ids, mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset + bsize], mask[offset:offset + bsize]))

    return batches



def _split_into_batches2(scores, bsize):
    batches = []
    for offset in range(0, len(scores), bsize):
        batches.append(scores[offset:offset + bsize])

    return batches

def _split_into_batches3(ids, mask, mask_special, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append(
            (ids[offset:offset + bsize],
             mask[offset:offset + bsize],
             mask_special[offset:offset + bsize]
             )
        )

    return batches

def _push_at_first_position(tensor: torch.Tensor, inserted: torch.Tensor):
    return torch.cat([tensor[:, :1], inserted, tensor[:, 1:]], dim=1)


def _insert_prefix_token(tensor: torch.Tensor, prefix_id: int):
    prefix_tensor = torch.full(
        (tensor.size(0), 1),
        prefix_id,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return _push_at_first_position(tensor, prefix_tensor)


def _insert_prefix_list(tensor: torch.Tensor, inserted: list):
    prefix_tensor = torch.tensor(
        inserted,
        device=tensor.device,
    ).repeat(tensor.size(0), 1, 1)
    return _push_at_first_position(tensor, prefix_tensor)


def extract_positions(passages, extractions):
    """
    Ensures that each extraction is a substring of the corresponding passage and has correct start and end positions
    """
    extractions_positions = []
    for psg, extraction in zip(passages, extractions):
        if extraction is None:
            continue

        extractions = find_spans(psg, extraction)
        extractions_positions.append(extractions)

    return extractions_positions


def get_span_masks(offset_mapping, extractions):
    device = offset_mapping.device
    longest_text_len = offset_mapping.max()
    span_starts = pad_longest([[e['start'] for e in e_list] for e_list in extractions], longest_text_len, device=device)
    span_ends = pad_longest([[e['end'] for e in e_list] for e_list in extractions], 0, device=device)

    # Encode and split offsets to starts and ends
    offset_mapping_starts = offset_mapping[:, :, 0]  # 0th dim are starts
    offset_mapping_ends = offset_mapping[:, :, 1]  # 1st dim are ends

    # Reshape for broadcasting each-to-each
    start_overlap = offset_mapping_starts[:, None] < span_ends[:, :, None]
    end_overlap = offset_mapping_ends[:, None] > span_starts[:, :, None]
    label_tensors = torch.any(start_overlap & end_overlap, dim=1).to(dtype=torch.float32)

    # Binary labels must match encoded text length in number of tokens
    return label_tensors


def pad_longest(list_data, pad_value, device=None):
    # Pad to the max row length using zip_longest
    padded_data = list(zip_longest(*list_data, fillvalue=pad_value))
    padded_data = list(map(list, zip(*padded_data)))   # Transpose back
    return torch.tensor(padded_data, device=device)


def find_span_start_index(text, span: str):
    """
    Find span sub-sequence in text and return text-wise indexes
    """
    if not type(span) is str:
        raise AssertionError("Invalid generated data structure.")

    text_len = len(text)
    span_len = len(span)

    # Loop through possible start indices in `text`
    for i in range(text_len - span_len + 1):
        # Check if the sub-sequence from `text` matches `span`
        if text[i:i + span_len] == span:
            return i  # Return the start index if a match is found

    return -1  # Return -1 if the span is not found in text


def find_spans(text, selected_spans):
    """
    Find spans in the text and return start index, length and end index
    :param text:
    :param selected_spans:
    """
    rationales = []

    if selected_spans is None or len(selected_spans) == 0:
        return rationales

    if not isinstance(selected_spans, list):
        print(selected_spans)
        raise AssertionError("Selected spans must be list!")

    if isinstance(selected_spans[0], dict):
        selected_spans = [span['text'] for span in selected_spans]
    for span in selected_spans:
        span_start = find_span_start_index(text, span)
        if span_start == -1:
            raise AssertionError(f"Selected span '{span}' was not found in the text: {text}")
        span_length = len(span)
        rationales.append(
            {
                'start': span_start,
                'length': span_length,
                'end': span_start + span_length,
            }
        )
    return rationales

def get_skiplist(tokenizer):
    skiplist = {w: True
                     for symbol in string.punctuation
                     for w in [symbol, tokenizer.encode(symbol, add_special_tokens=False)[0]]}
    return skiplist

def get_skiplist_ids(skiplist):
    return [t_id for t_id in skiplist.keys() if type(t_id) is int]