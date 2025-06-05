import os
import json
from typing import Dict, List

LANGUAGE_MAP = {
    "bn": "Bengali",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "ta": "Tamil",
    "te": "Telugu",
}

STRATEGY_MAP = {
    "MonoEnglish": "monoenglish",
    "MonoNative": "mononative",
    "MultiRandom": "multirandom",
    "MultiRelated": "multirelated",
}

DATASETS = ["mmlu", "hellaswag"]
MODELS = ["aya_13b", "gemma"]

METRICS = [
    "reliable_accuracy",
    "effective_reliability",
    "abstain_accuracy",
    "abstain_precision",
    "abstain_recall",
    "abstain_ece",
    "abstain_rate",
]

def load_results(dataset: str, strategy: str, model: str, lang: str) -> Dict:
    """Load a single result file if it exists."""
    filename = f"{model}_{dataset}_{lang}_{strategy}.json"
    path = os.path.join("results", dataset, strategy, filename)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_metrics(dataset: str, model: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    metrics_data: Dict[str, Dict[str, Dict[str, float]]] = {m: {} for m in METRICS}
    for lang in LANGUAGE_MAP:
        for strategy_name, strategy_dir in STRATEGY_MAP.items():
            res = load_results(dataset, strategy_dir, model, lang)
            for m in METRICS:
                metrics_data[m].setdefault(lang, {})[strategy_name] = res.get(m)
    # averages
    for m in METRICS:
        metrics_data[m]["Average"] = {}
        for strat in STRATEGY_MAP:
            vals = [metrics_data[m][lg][strat] for lg in LANGUAGE_MAP if metrics_data[m][lg][strat] is not None]
            metrics_data[m]["Average"][strat] = sum(vals) / len(vals) if vals else None
    return metrics_data

def _format_val(val: float) -> str:
    if val is None:
        return "-"
    return f"{val:.3f}"

def make_table(metric: str, data: Dict[str, Dict[str, float]]) -> str:
    lines: List[str] = []
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Language & MonoEnglish & MonoNative & MultiRandom & MultiRelated\\")
    lines.append("\\midrule")
    for lang_code in LANGUAGE_MAP:
        row = [LANGUAGE_MAP[lang_code]]
        values = [data[lang_code][s] for s in STRATEGY_MAP]
        avail = [v for v in values if v is not None]
        if metric == "abstain_ece":
            best = min(avail) if avail else None
        else:
            best = max(avail) if avail else None
        for val in values:
            if val is None:
                row.append("-")
            else:
                formatted = f"{val:.3f}"
                if best is not None and abs(val - best) < 1e-12:
                    formatted = f"\\textbf{{{formatted}}}"
                row.append(formatted)
        lines.append(" & ".join(row) + "\\")
    lines.append("\\midrule")
    avg_row = ["Average"]
    for strat in STRATEGY_MAP:
        avg_row.append(_format_val(data["Average"][strat]))
    lines.append(" & ".join(avg_row) + "\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)

def main() -> None:
    for model in MODELS:
        for dataset in DATASETS:
            metrics = collect_metrics(dataset, model)
            print(f"=== {model} - {dataset} ===")
            for metric in METRICS:
                print(f"\nMetric: {metric}")
                print(make_table(metric, metrics[metric]))
                print()

if __name__ == "__main__":
    main()
