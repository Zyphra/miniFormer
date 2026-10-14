"""Compile the saved ablation results into summary.tex and summary.pdf."""

import argparse
import json
import subprocess
from pathlib import Path


LABELS = {
    "sinusoidal": "Sinusoidal", "rotary": "RoPE", "None": "None",
    "learned": "Learned", "spatial": "Spatial", "CLS_token": "CLS token",
    "mean_pool_final": "Mean pooling", "attn_pool_final": "Attention pooling",
    "flatten_final": "Flatten",
}


def tex(value):
    return str(value).translate(str.maketrans({
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }))


def percentage(score):
    assert 0 <= score <= 1, f"Invalid balanced accuracy: {score}"
    return f"{100 * score:.1f}\\%"


def save_report(output, note=""):
    output = Path(output).resolve()
    suite = json.loads((output / "config.json").read_text())
    results = json.loads((output / "results.json").read_text())
    default = json.loads((output / suite["default_config"]).read_text())
    assert suite["normalize"] == [False, True]
    seeds = ", ".join(map(str, suite["seeds"]))
    seed_label = "seed" if len(suite["seeds"]) == 1 else "seeds"
    fixed = {
        "temporal_pos": f"temporal position: {LABELS[default['temporal_pos']]}",
        "channel_pos": f"channel position: {LABELS[str(default['channel_pos'])]}",
        "aggregation_mode": f"aggregation: {LABELS[default['aggregation_mode']]}",
        "depth": f"{default['depth']} layers",
    }
    common = (
        f"{default['n_train'] + default['n_test']:,} recordings "
        f"({default['n_train']:,} train / {default['n_test']:,} test); {default['n_classes']} classes; "
        f"{default['channels']} channels, {default['coarse_time']} patches of {default['codebook_dim']} samples; "
        f"model dimension {default['model_dim']}, {default['heads']} heads; "
        f"epochs: {default['epochs']}, batch size {default['batch_size']}, "
        f"Adam learning rate {default['learning_rate']:g}, gradient clip {default['max_grad_norm']:g}"
    )
    parts = [r"""\documentclass[11pt]{article}
\usepackage[letterpaper,margin=0.75in]{geometry}
\usepackage{booktabs,graphicx}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\renewcommand{\arraystretch}{1.15}
\begin{document}
{\Large\textbf{Architecture ablations}}\par
""", f"Scores are mean final test balanced accuracy (mean class recall), in percent, "
       f"across {len(suite['seeds'])} {seed_label} ({seeds}). Dataset generation, split, and training vary with seed.",
       r"\textit{" + tex(note) + r"}\par"]
    for parameter, title in (
        ("temporal_pos", "RoPE vs sinusoidal"),
        ("channel_pos", "Learned vs spatial"),
        ("aggregation_mode", "Aggregation mode"),
    ):
        defaults = "; ".join(value for key, value in fixed.items() if key != parameter)
        parts.extend([
            r"\medskip\textbf{" + title + r"}\par",
            r"{\small\textit{Fixed: " + tex(defaults + "; " + common + ".") + r"}\par}",
            r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrr@{}}\toprule",
            r"Variant & \texttt{normalize=False} & \texttt{normalize=True} \\\midrule",
        ])
        for value, name in suite["comparisons"][parameter].items():
            if parameter == "channel_pos" and value == "None":
                continue
            scores = [percentage(results[name][f"normalize={normalize}"]["test"]["balanced_accuracy"])
                      for normalize in (False, True)]
            parts.append(" & ".join([LABELS[value], *scores]) + r" \\")
        parts.append(r"\bottomrule\end{tabular*}\par")
    baselines = "; ".join(
        label + ": " + percentage(results["1x/baselines"][name]["test"]["balanced_accuracy"])
        for name, label in (("svm", "SVM"), ("kmeans", "K-means"), ("bag_of_tokens", "Bag-of-tokens"))
    )
    parts.append(r"\medskip{\small\textbf{Reference baselines (same 1x split):} " + baselines + r".\par}")
    parts.append(r"\newpage")
    for parameter, title in (("depth", "Increasing depth"), ("dataset_size", "Increasing dataset size")):
        parts.extend([r"\textbf{" + title + r"}\par",
                      r"\includegraphics[width=\linewidth]{" + parameter + r".pdf}\par\medskip"])
    parts.append(r"\end{document}")
    (output / "summary.tex").write_text("\n".join(parts) + "\n")
    build = output / "report_build"
    build.mkdir(exist_ok=True)
    with (build / "compile.log").open("w") as log:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape",
                        "-output-directory=report_build", "summary.tex"], cwd=output,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    (build / "summary.pdf").replace(output / "summary.pdf")
    return output / "summary.pdf"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Completed experiment directory")
    parser.add_argument("--note", default="", help="Optional report note, e.g. Validation preview")
    args = parser.parse_args()
    print(save_report(args.output, args.note))
