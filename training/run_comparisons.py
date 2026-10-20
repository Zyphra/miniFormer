"""Plot grouped checkpoint results as one PNG, without running any training."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot(checkpoints, results_root="results", colors=None):
    """Accept {panel_title: {"note": description, "groups": {group: {checkpoint_name: path}}}}.

    Directories are relative to results_root and contain seed_*/metrics.json.
    Checkpoint names are used as x-axis labels.
    Each group has one color across all panels. Optional colors maps group labels
    to matplotlib colors. Repeating a directory reuses its saved results.
    """
    results_root = Path(results_root)
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")
    if not isinstance(checkpoints, dict) or not checkpoints:
        raise ValueError("Provide a nonempty dictionary of checkpoint groups")

    panels = {}
    upper = 100
    for title, panel in checkpoints.items():
        if not isinstance(panel, dict):
            raise ValueError(f"Panel {title!r} needs a note and color groups")
        note = panel.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValueError(f"Panel {title!r} needs a nonempty note describing what it represents")
        groups = panel.get("groups")
        if not isinstance(groups, dict) or not groups:
            raise ValueError(f"Panel {title!r} needs a dictionary of color groups")
        panels[title] = {}
        for group, sources in groups.items():
            if not isinstance(sources, dict) or not sources:
                raise ValueError(f"Group {group!r} needs a dictionary of checkpoint names and directories")
            bars = []
            for label, directory in sources.items():
                directory = results_root / directory
                seeds = sorted(path for path in directory.glob("seed_*") if path.is_dir())
                if len(seeds) < 2:
                    raise ValueError(f"Need at least two classifier seeds in {directory}")
                values = []
                for seed in seeds:
                    metrics = json.loads((seed / "metrics.json").read_text())
                    values.append(100 * metrics["test"]["balanced_accuracy"])
                if not all(math.isfinite(value) and 0 <= value <= 100 for value in values):
                    raise ValueError(f"Invalid test balanced accuracy in {directory}")
                average, deviation = mean(values), stdev(values)
                upper = max(upper, average + deviation + 5, max(values) + 5)
                bars.append((label, values, average, deviation))
            panels[title][group] = bars

    group_names = list(dict.fromkeys(group for groups in panels.values() for group in groups))
    palette = plt.get_cmap("tab10" if len(group_names) <= 10 else "tab20")
    group_colors = {group: palette(index % palette.N) for index, group in enumerate(group_names)}
    group_colors.update(colors or {})
    columns = min(2, len(panels))
    rows = math.ceil(len(panels) / columns)
    legend_rows = math.ceil(len(group_names) / 5)
    fig = plt.figure(figsize=(8 * columns, 3.6 * rows + 1 + .3 * legend_rows), layout="constrained")
    grid = fig.add_gridspec(rows + 1, columns, height_ratios=[.12 * legend_rows] + [1] * rows)
    fig.suptitle("Test balanced accuracy\nMean ± sample SD; dots show classifier seeds", fontsize=17)
    legend = fig.add_subplot(grid[0, :])
    legend.set_axis_off()
    legend.legend(handles=[Patch(color=group_colors[group], label=group) for group in group_names],
                  loc="center", ncol=min(5, len(group_names)), frameon=False, fontsize=12)

    for index, (title, groups) in enumerate(panels.items()):
        ax = fig.add_subplot(grid[1 + index // columns, index % columns])
        positions, labels = [], []
        x = 0
        for group, bars in groups.items():
            for label, values, average, deviation in bars:
                ax.bar(x, average, width=.7, color=group_colors[group], zorder=2)
                ax.errorbar(x, average, yerr=deviation, fmt="none", color="#283342",
                            capsize=3, linewidth=1.2, zorder=3)
                offsets = [x - .18 + .36 * i / (len(values) - 1) for i in range(len(values))]
                ax.scatter(offsets, values, s=19, color="#283342", edgecolors="white",
                           linewidths=.6, zorder=4)
                ax.text(x, max(average + deviation, *values) + 1.5, f"{average:.1f}",
                        ha="center", va="bottom", fontsize=11)
                positions.append(x)
                labels.append(label)
                x += 1
            x += .5
        rotate = len(labels) > 4
        ax.set_xticks(positions, labels, fontsize=10, rotation=35 if rotate else 0,
                      ha="right" if rotate else "center", rotation_mode="anchor")
        ax.set_ylim(0, upper)
        note = fill(checkpoints[title]["note"].strip(), width=70)
        ax.set_title(f"{title}\n{note}", loc="left", fontsize=13, pad=14)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E5E9EE", linewidth=.8)
        ax.tick_params(axis="both", length=0, pad=7)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#D1D7E0")
        if index % columns == 0:
            ax.set_ylabel("Balanced accuracy (%)")

    output = results_root / "comparison.png"
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True,
                        help="JSON mapping panel titles to notes and checkpoint groups; missing notes prompt")
    parser.add_argument("--colors", type=Path,
                        help="Optional JSON file mapping group labels to colors")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    args = parser.parse_args()
    checkpoints = json.loads(args.checkpoints.read_text())
    for title, panel in checkpoints.items():
        if not isinstance(panel, dict):
            parser.error(f"Panel {title!r} needs a note and color groups")
        while not isinstance(panel.get("note"), str) or not panel["note"].strip():
            try:
                panel["note"] = input(f"{title}: describe what this panel represents in a few words: ").strip()
            except EOFError:
                parser.error(f"Provide a nonempty 'note' for panel {title!r} in {args.checkpoints}")
    colors = json.loads(args.colors.read_text()) if args.colors else None
    print(plot(checkpoints, args.results_root, colors))
