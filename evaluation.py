import jsonlines
import wandb
import numpy as np
from sklearn.metrics import precision_recall_curve


def evaluate_extractions(max_ranking_path, figure_id):
    """
    Use checkpoint steps as figure_id
    """

    if type(figure_id) is int:
        figure_id = str(figure_id)

    assert type(figure_id) is str and figure_id.isalnum(), f"'{figure_id}' is not a valid figure_id"

    # todo: add select all and random sampling baselines
    figure_name = f"pr_curve_{figure_id}"
    # Evaluation
    with jsonlines.open(max_ranking_path) as reader:
        max_ranking_results = list(reader)

        all_max_scores = [score for line in max_ranking_results for score in line['max_score']]
        all_extractions = [score for line in max_ranking_results for score in line['extraction_masked']]

        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(
            np.array(all_extractions), np.array(all_max_scores)
        )

        # Prepare data for wandb Table
        data = [[r, p] for r, p in zip(recall, precision)]
        table = wandb.Table(data=data, columns=["Recall", "Precision"])

        # Log the PR curve as a line plot
        wandb.log(
            {
                figure_name: wandb.plot.line(
                    table, "Recall", "Precision", title=f"PR Curve Checkpoint {figure_id}"
                )
            }
        )

    # todo: python -m utility.evaluate.msmarco_passages --ranking "/path/to/msmarco.nbits=2.ranking.tsv" --qrels "/path/to/MSMARCO/qrels.dev.small.tsv"
    # todo: run my own evaluation script
    # todo: use Run().config.name to save the results in the right folder (take a look h and to wandb