"""Run the full eval: for each generated question, query HydraDB + synthesize
an answer (ask.answer_question), then score it (judge.judge_answer).
Writes results/eval_results.jsonl.

Usage:
    python src/run_eval.py --limit 10          # quick sample
    python src/run_eval.py                      # full run
    python src/run_eval.py --category abstention
"""

import argparse
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from ask import answer_question
from judge import judge_answer

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "questions", "generated_questions.jsonl")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "eval_results.jsonl")


def load_questions(category: str | None = None, limit: int | None = None) -> list[dict]:
    questions = [json.loads(l) for l in open(QUESTIONS_PATH) if l.strip()]
    if category:
        questions = [q for q in questions if q["category"] == category]
    if limit:
        questions = questions[:limit]
    return questions


def run_one(question: dict) -> dict:
    try:
        result = answer_question(
            question["question"],
            collections=question.get("collections_in_scope"),
            instruction=question.get("instruction"),
        )
        verdict = judge_answer(question, result["answer"])
        return {
            "qid": question["qid"],
            "category": question["category"],
            "dataset": question["dataset"],
            "question": question["question"],
            "gold_answer": question["gold_answer"],
            "answerable": question.get("answerable", True),
            "generated_answer": result["answer"],
            "num_chunks_retrieved": result["num_chunks_retrieved"],
            **verdict,
        }
    except Exception as e:
        return {
            "qid": question["qid"],
            "category": question["category"],
            "dataset": question["dataset"],
            "question": question["question"],
            "error": str(e),
            "traceback": traceback.format_exc(),
            "verdict": "error",
            "score": 0.0,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    questions = load_questions(category=args.category, limit=args.limit)
    print(f"Running eval on {len(questions)} questions with {args.workers} workers...")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, q): q for q in questions}
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results.append(r)
            status = r.get("verdict", "?")
            print(f"[{i}/{len(questions)}] {r['qid']} ({r['category']}): {status}", flush=True)

    results.sort(key=lambda r: r["qid"])
    with open(RESULTS_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_errors = sum(1 for r in results if r.get("verdict") == "error")
    print(f"\nWrote {len(results)} results to {RESULTS_PATH} ({n_errors} errors)")


if __name__ == "__main__":
    main()
