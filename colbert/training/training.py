import time
from os import environ

import torch
import random
import torch.nn as nn
import numpy as np
from jsonlines import jsonlines

from transformers import AdamW, get_linear_schedule_with_warmup
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher

from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker

from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints
from colbert.infra.run import Run

import wandb


def wandb_find_next_name(initial_run_name):
    api = wandb.Api()
    project_name = Run().config.project_name
    entity = Run().confing.wandb_entity

    # Fetch all runs in the project
    runs = api.runs(f"{entity}/{project_name}")

    # Find the run with the specified name
    new_run_name = initial_run_name
    max_runs = 100
    for run_index in range(max_runs):
        if all(run.name != new_run_name for run in runs):
            return new_run_name
        else:
            new_run_name = f"{initial_run_name}_{run_index}"

    raise ValueError(f"Experiment '{initial_run_name}' was ran more then {max_runs} times.")


def init_wandb(config):
    name = config.name if config.resume else wandb_find_next_name(config.experiment)

    wandb.init(
        project=Run().config.project_name,
        config=config.__dict__,
        resume="must" if config.resume else "allow",
        name=name,
    )
    Run().config.name = name  # Used as path to save checkpoints


def train(config: ColBERTConfig, triples, queries=None, collection=None, extracted_spans=None):
    config.checkpoint = config.checkpoint or 'bert-base-uncased'

    if config.rank < 1:
        config.help()
        init_wandb(config)

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks

    print("Using config.bsize =", config.bsize, "(per process) and config.accumsteps =", config.accumsteps)

    if collection is not None:
        if config.reranker:
            reader = RerankBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank),
                                   config.nranks)
        else:
            reader = LazyBatcher(config, triples, queries, collection, extracted_spans,
                                 (0 if config.rank == -1 else config.rank), config.nranks)
    else:
        raise NotImplementedError()

    if not config.reranker:
        colbert = ColBERT(name=config.checkpoint, colbert_config=config)
    else:
        colbert = ElectraReranker.from_pretrained(config.checkpoint)

    colbert = colbert.to(DEVICE)
    if not Run().config.is_debugging:
        colbert = torch.compile(colbert)
    colbert.train()

    colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[config.rank],
                                                        output_device=config.rank,
                                                        find_unused_parameters=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8)
    optimizer.zero_grad()

    scheduler = None
    if config.warmup is not None:
        print(f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps.")
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup,
                                                    num_training_steps=config.maxsteps)

    warmup_bert = config.warmup_bert
    if warmup_bert is not None:
        set_bert_grad(colbert, False)

    amp = MixedPrecisionManager(config.amp)
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = None
    train_loss_mu = 0.999

    start_batch_idx = 0

    # if config.resume:
    #     assert config.checkpoint is not None
    #     start_batch_idx = checkpoint['batch']

    #     reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

    st = StatsTracker()
    batch_idx = 0
    for e in range(config.epochs):
        for BatchSteps in reader:
            if config.rank < 1:
                print_message(batch_idx, train_loss)
                manage_checkpoints(config, colbert, optimizer, batch_idx, savepath=None)

            if (warmup_bert is not None) and warmup_bert <= batch_idx:
                set_bert_grad(colbert, True)
                warmup_bert = None

            this_batch_loss = 0.0

            for acc_step, batch in enumerate(BatchSteps):
                batch_logs = {}
                with amp.context():
                    try:
                        queries, passages, target_scores, target_extractions = batch
                        encoding = [queries, (passages[0], passages[1])]
                    except:
                        encoding, target_scores = batch
                        encoding = [encoding.to(DEVICE)]

                    outs = colbert(*encoding)
                    scores = outs['scores']
                    scores = scores.view(-1, config.nway)

                    if len(target_scores) and not config.ignore_scores:
                        target_scores = torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                        target_scores = target_scores * config.distillation_alpha
                        target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)

                        log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                        loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(log_scores, target_scores)
                        batch_logs['distillation_loss'] = loss.item()
                    else:
                        loss = nn.CrossEntropyLoss()(scores, labels[:scores.size(0)])

                    if config.use_ib_negatives:
                        ib_loss = outs['ib_loss']
                        if config.rank < 1:
                            print('\t\t\t\t', loss.item(), ib_loss.item())

                        batch_logs['ib_loss'] = ib_loss.item()
                        loss += ib_loss

                    if config.return_max_scores:
                        max_scores, max_scores_i = outs['max_scores']

                        # Extract scores for first (ie relevant) documents
                        first_doc_ids = [b * config.nway for b in range(int(passages[0].size(0) / config.nway))]
                        max_scores_first, max_scores_i_f = max_scores[first_doc_ids], max_scores_i[first_doc_ids]
                        doc_mask = ~passages[2][first_doc_ids]

                        ex_loss_unreduced = nn.BCEWithLogitsLoss(reduction='none')(max_scores_first, target_extractions)
                        ex_loss_unreduced = doc_mask * ex_loss_unreduced  # Set masked tokens scores to 0
                        ex_loss = torch.mean(ex_loss_unreduced.sum(dim=-1) / doc_mask.sum(dim=-1))

                        # mask documents all specials and skip tokens
                        masked_scores, targets_masked = max_scores_first[doc_mask], target_extractions[doc_mask]
                        batch_logs.update(extraction_stats(ex_loss, masked_scores, targets_masked))
                        st.save_idx(max_scores_i_f, doc_mask, queries[1])

                        loss = (1 - config.extractions_lambda) * loss + config.extractions_lambda * ex_loss

                    if config.rank < 1:
                        consumed = (batch_idx * config.bsize + acc_step * (config.bsize / config.accumsteps)) / len(reader)
                        wandb.log(dict({"total_loss": loss}, **batch_logs), step=consumed)
                    loss = loss / config.accumsteps

                if config.rank < 1:
                    print_progress(scores)

                amp.backward(loss)

                this_batch_loss += loss.item()

            train_loss = this_batch_loss if train_loss is None else train_loss
            train_loss = train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss

            amp.step(colbert, optimizer, scheduler)

            batch_idx += 1

    if config.rank < 1:
        print_message("#> Done with all triples!")
        ckpt_path = manage_checkpoints(config, colbert, optimizer, batch_idx, savepath=None,
                                       consumed_all_triples=True)

        return ckpt_path  # TODO: This should validate and return the best checkpoint, not just the last one.


class StatsTracker:
    def __init__(self):
        self.save_jsonl_path = 'max_scores_indices.jsonl'
        self.q_toks_f = []

    def save_idx(self, max_scores_indices, mask, q_mask_tokens):
        assert max_scores_indices.dim() == 2

        nr_tokens_q = q_mask_tokens.size(1)

        x = [torch.bincount(x[m], minlength=nr_tokens_q).cpu().tolist() for x, m in zip(max_scores_indices, mask)]
        x = [{'max_score_index_bin_counts': x, 'is_q_mask': q_mask.cpu().tolist()}
             for x, q_mask in zip(x, q_mask_tokens)]

        for y in x:
            len_max = len(y['max_score_index_bin_counts'])
            len_q = len(y['is_q_mask'])
            assert len_max == len_q, (len_max, len_q)

        self.q_toks_f.extend(x)

        if len(self.q_toks_f) % 100 == 0:
            with jsonlines.open(self.save_jsonl_path, 'a') as f:
                f.write_all(self.q_toks_f)
            self.q_toks_f = []


def extraction_stats(ex_loss, masked_scores, targets_masked):
    # Compute extractions stats
    probs = nn.Sigmoid()(masked_scores)
    thresholded = (probs > 0.5).to(dtype=torch.float32)
    acc = (thresholded == targets_masked).float().mean()
    recall = thresholded[targets_masked.bool()].float().sum() / targets_masked.float().sum()
    precision = thresholded[targets_masked.bool()].float().sum() / thresholded.sum()

    var, mean = torch.var_mean(probs[targets_masked.bool()])
    var0, mean0 = torch.var_mean(probs[~targets_masked.bool()])
    return {
        "extractions_loss": ex_loss,
        "extraction_accuracy": acc,
        "recall": recall,
        "precision": precision,

        "mean_extraction_prob": mean,
        "mean_extraction_prob_var": var,

        "mean_extraction_prob_0": mean0,
        "mean_extraction_prob_var0": var0,
    }


def set_bert_grad(colbert, value):
    try:
        for p in colbert.bert.parameters():
            assert p.requires_grad is (not value)
            p.requires_grad = value
    except AttributeError:
        set_bert_grad(colbert.module, value)
