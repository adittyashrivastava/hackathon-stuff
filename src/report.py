"""Turn results/eval_results.jsonl into a per-category accuracy report,
mirroring the shape of HydraDB's own published benchmark reports
(per-category breakdown vs. a baseline).

Usage: python src/report.py
"""

import json
import os
from collections import defaultdict

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "eval_results.jsonl")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "report.md")

VERDICT_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0, "hallucinated": 0.0, "error": 0.0}


def load_results() -> list[dict]:
    return [json.loads(l) for l in open(RESULTS_PATH) if l.strip()]


def score_of(r: dict) -> float:
    if "score" in r and isinstance(r["score"], (int, float)):
        return float(r["score"])
    return VERDICT_SCORE.get(r.get("verdict"), 0.0)


def aggregate(results: list[dict], key_fn) -> dict:
    groups = defaultdict(list)
    for r in results:
        groups[key_fn(r)].append(r)
    out = {}
    for key, items in groups.items():
        scores = [score_of(r) for r in items]
        verdict_counts = defaultdict(int)
        for r in items:
            verdict_counts[r.get("verdict", "error")] += 1
        out[key] = {
            "n": len(items),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "verdict_counts": dict(verdict_counts),
        }
    return out


def render_table(agg: dict, header: str) -> str:
    rows = [f"| {header} | N | Mean score | correct | partial | incorrect | hallucinated | error |",
            "|---|---|---|---|---|---|---|---|"]
    for key in sorted(agg.keys()):
        stats = agg[key]
        vc = stats["verdict_counts"]
        rows.append(
            f"| {key} | {stats['n']} | {stats['mean_score']:.2f} | "
            f"{vc.get('correct', 0)} | {vc.get('partially_correct', 0)} | "
            f"{vc.get('incorrect', 0)} | {vc.get('hallucinated', 0)} | {vc.get('error', 0)} |"
        )
    return "\n".join(rows)


def main():
    results = load_results()
    if not results:
        print("No results found — run src/run_eval.py first.")
        return

    by_category = aggregate(results, lambda r: r["category"])
    by_dataset = aggregate(results, lambda r: r["dataset"])

    overall_scores = [score_of(r) for r in results]
    overall_mean = sum(overall_scores) / len(overall_scores)

    flags = []
    for cat, stats in by_category.items():
        if stats["n"] >= 3 and stats["mean_score"] in (0.0, 1.0):
            flags.append(f"- `{cat}` scored a flat {stats['mean_score']:.0%} across all {stats['n']} questions — worth a manual look (could be a real pattern, or a broken judge/pipeline for that category).")

    lines = [
        "# Arjun's Memory — HydraDB Eval Report",
        "",
        f"**{len(results)} questions** across {len(by_category)} BEAM-inspired categories and "
        f"{len(by_dataset)} dataset/collection scopes. **Overall mean score: {overall_mean:.2%}**",
        "",
        "## By BEAM category",
        "",
        render_table(by_category, "category"),
        "",
        "## By dataset / relationship",
        "",
        render_table(by_dataset, "dataset"),
        "",
    ]

    if flags:
        lines += ["## Flags for manual review", ""] + flags + [""]

    errors = [r for r in results if r.get("verdict") == "error"]
    if errors:
        lines += [
            "## Errors",
            "",
            f"{len(errors)} question(s) errored during the run:",
            "",
        ]
        for r in errors[:20]:
            lines.append(f"- `{r['qid']}` ({r['category']}): {r.get('error', '')}")
        lines.append("")

    report = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(report)
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
