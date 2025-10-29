import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).parent.parent


def load_data(file_path: Path = root / "euroeval_benchmark_results.jsonl"):
    data = []
    with file_path.open("r") as f:
        for line in f:
            data.append(json.loads(line))

    for line in data:
        if "@" in line["model"]:
            line["model_name"] = line["model"].split("@")[0]
            line["model_revision"] = line["model"].split("@")[1]
    return data


def data_to_dataframe(data: list[dict]):
    rows = []
    for entry in data:
        metrics = list(entry["results"]["raw"][0].keys())

        base_row = {
            "model": entry["model_name"],
            "Training Tokens (Billions)": int(entry["model_revision"].strip("B")),
        }
        for metric in metrics:
            for i in range(len(entry["results"]["raw"])):
                row = base_row.copy()
                row["dataset"] = f"{entry['dataset']} - {metric}"
                row["value"] = entry["results"]["raw"][i][metric]
                row["step"] = i
                rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_dataset(subset: str):
    # PLOTTING
    # x-axis: Training Tokens (Billions)
    # y-axis: value (multiple point + mean)
    # hue: model
    # facet: dataset
    df_subset = df[df["dataset"] == subset]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot individual points
    for model_name, group in df_subset.groupby("model"):
        ax.scatter(
            group["Training Tokens (Billions)"]
            + (10 * (np.random.rand(len(group)) - 0.5)),  # with jitter
            group["value"],
            alpha=0.3,
        )

        # Plot mean line with 95% confidence interval
        mean_values = (
            group.groupby("Training Tokens (Billions)")["value"].mean().reset_index()
        )
        std_values = (
            group.groupby("Training Tokens (Billions)")["value"].std().reset_index()
        )
        counts = (
            group.groupby("Training Tokens (Billions)")["value"].count().reset_index()
        )
        ci95 = 1.96 * (std_values["value"] / np.sqrt(counts["value"]))
        ax.plot(
            mean_values["Training Tokens (Billions)"],
            mean_values["value"],
            label=model_name,
        )
        ax.fill_between(
            mean_values["Training Tokens (Billions)"],
            mean_values["value"] - ci95,
            mean_values["value"] + ci95,
            alpha=0.2,
            color=ax.get_lines()[-1].get_color(),
            edgecolor=None,
        )

    ax.set_xlabel("Training Tokens (Billions)")
    ax.set_ylabel("Value (Mean with 95% CI)")
    ax.set_title(f"Dataset: {subset}")
    ax.legend()
    return fig, ax


data = load_data()
df = data_to_dataframe(data)

plots_folder = root / "plots"
plots_folder.mkdir(exist_ok=True)
for dataset in df["dataset"].unique():
    fig, ax = plot_dataset(dataset)
    # save
    fig.savefig(plots_folder / f"{dataset.replace(' ', '_')}.png")
