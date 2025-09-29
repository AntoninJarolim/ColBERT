from matplotlib import pyplot as plt

import wandb
import numpy as np
import pandas as pd
import seaborn as sns

from colbert.infra import Run


def wandb_connect_running(run_name):
    api = wandb.Api()

    # Replace with your project name
    project_name = 'eval-' + Run().config.project_name
    entity = Run().config.wandb_entity
    run_name = 'eval_' + run_name

    # Fetch all runs in the project
    runs = api.runs(f"{entity}/{project_name}")

    # Find the run with the specified name
    for run in runs:
        if run.name == run_name:
            print(f"Resuming found run ID: {run.id}")
            wandb.init(project=project_name, id=run.id, resume="must")

    print("\n\nStarting new eval run with name:", run_name)
    wandb.init(project=project_name, name=run_name)


def wandb_log_figure(name, figure):
    wandb.log({name: wandb.Image(figure)})


def wandb_log_pr_curve(title, data, color_mapping=None, norm=None, cmap=None):
    run_name = wandb.run.name
    title = f"PR Curve {title} {run_name}"

    if len(data[0]) == 3:
        columns = ["Recall", "Precision", "Type"]
    elif len(data[0]) == 4:
        columns = ["Recall", "Precision", "Threshold", "Type"]
    else:
        raise ValueError("Data must have 3 or 4 columns")

    df = pd.DataFrame(data, columns=columns)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    if color_mapping:
        # Plot each line individually to allow custom colors (and not to label checkpoint lines)
        for t in df["Type"].unique():
            df_sub = df[df["Type"] == t]
            color = color_mapping[t] if color_mapping and t in color_mapping else None
            label = None if "Max-Values" in t else t  # Only label 'Random'
            sns.lineplot(data=df_sub, x="Recall", y="Precision", ax=ax, color=color, label=label,
                         estimator=None,  # <- skip mean aggregation
                         errorbar=None  # <- skip confidence interval bootstrapping)
                         )
    else:
        # Plot all lines with default colors (hue will do the trick)
        sns.lineplot(data=df, x="Recall", y="Precision", hue="Type", ax=ax,
                         estimator=None,  # <- skip mean aggregation
                         errorbar=None)  # <- skip confidence interval bootstrapping)


    ax.set_title(title)
    ax.set_ylim(0, 1)

    # Create colorbar as a gradient legend for Max-Values
    if norm and cmap:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Only needed for older matplotlib versions
        cbar = fig.colorbar(sm, ax=ax)

        # Force showing min and max step values
        min_step = int(norm.vmin)
        max_step = int(norm.vmax)
        cbar.set_ticks(np.linspace(min_step, max_step, num=5, dtype=int))
        cbar.set_ticklabels([str(int(x)) for x in np.linspace(min_step, max_step, num=5)])

        cbar.set_label("Checkpoint Step")

    # Add manual red marker for Random
    # ax.plot([], [], color='red', label='Random Baseline')

    # if cmap is not None, we have a colorbar and only one entry in a legend (random baseline)
    if len(df["Type"].unique()) < 6 or cmap is not None:
        legend_loc = "lower right"
        bbox = (1, 0)
    else:
        legend_loc = "upper left"
        bbox = (1, 1)

    ax.legend(loc=legend_loc, bbox_to_anchor=bbox)

    plt.tight_layout()
    plt.savefig(f"experiments/eval/fig/{title}.pdf", dpi=900)
    wandb.log({
        f"pr_curve_all_{title}": wandb.Image(fig)
    })
    plt.close(fig)


def wandb_log_extraction_accuracy(data, checkpoint_steps, collection_name):
    # Get mean of all data
    data_agg = {f"{k}-{collection_name}": np.nanmean([d[k] for d in data]) for k in data[0].keys()}

    # Log the data to wandb
    collection_cp_steps = f"checkpoint_steps-{collection_name}"
    wandb.define_metric(collection_cp_steps)
    for k in data_agg.keys():
        wandb.define_metric(k, step_metric=collection_cp_steps)

    data_agg[collection_cp_steps] = checkpoint_steps
    wandb.log(data_agg)


def wandb_log_retrieval_figs(file_datas):
    for collection_name, data in file_datas.items():
        # Log the evaluation to wandb
        data = sorted(data, key=lambda x: x['batch_steps'])
        values = [file_data.values() for file_data in data]
        tab = wandb.Table(columns=list(data[0].keys()), data=values)
        wandb.log({f"retrieval_evaluation_{collection_name}": tab})

        # Craete mrr@10 line plot and upload it to wandb
        wandb.log({f"mrr@10_steps_{collection_name}": wandb.plot.line(tab, "batch_steps", "mrr@10",
                                                                      title="MRR@10 over steps")})
        wandb.log({f"recall@50_steps_{collection_name}": wandb.plot.line(tab, "batch_steps", "recall@50",
                                                                         title="Recall@50 over steps")})


def log_wandb_table(html, name):
    wandb.log({name: wandb.Html(html)})