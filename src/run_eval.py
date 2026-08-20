"""Run an eval: for each question, get an answer via a chosen backend, then
score it (judge.judge_answer). Writes results to a JSONL file.

Usage:
    python src/run_eval.py --limit 10                                    # quick sample, main question set
    python src/run_eval.py                                                # full run, main question set
    python src/run_eval.py --category abstention
    python src/run_eval.py --questions ../questions/akash_style_questions.jsonl \\
        --results ../results/akash_style_single_shot.jsonl --backend single_shot
    python src/run_eval.py --questions ../questions/akash_style_questions.jsonl \\
        --results ../results/akash_style_agentic.jsonl --backend agentic
"""

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from judge import judge_answer

DEFAULT_QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "questions", "generated_questions.jsonl")
DEFAULT_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "eval_results.jsonl")


def load_questions(path: str, category: str | None = None, limit: int | None = None) -> list[dict]:
    questions = [json.loads(l) for l in open(path) if l.strip()]
    if category:
        questions = [q for q in questions if q["category"] == category]
    if limit:
        questions = questions[:limit]
    return questions


def answer_single_shot(question: dict) -> dict:
    from ask import answer_question

    result = answer_question(
        question["question"],
        collections=question.get("collections_in_scope"),
        instruction=question.get("instruction"),
    )
    return {
        "answer": result["answer"],
        "num_chunks_retrieved": result["num_chunks_retrieved"],
        "num_raw_chunks_before_dedup": result.get("num_raw_chunks_before_dedup"),
        "retrieved_message_ids": result["retrieved_message_ids"],
    }


def answer_agentic(question: dict) -> dict:
    from agentic_ask import agentic_answer

    result = agentic_answer(question["question"], instruction=question.get("instruction"))
    return {
        "answer": result["answer"],
        "num_searches": result["num_searches"],
        "steps_used": result["steps_used"],
        "search_log": result["search_log"],
        "retrieved_message_ids": result["retrieved_message_ids"],
    }


BACKENDS = {"single_shot": answer_single_shot, "agentic": answer_agentic}


def evidence_hit_rate(question: dict, retrieved_message_ids: list) -> float | None:
    """Fraction of the question's evidence entries that look like message IDs (not
    ground-truth field names) that actually appear in what was retrieved. Returns
    None if the question's evidence has no message-ID-shaped entries to check."""
    retrieved = set(retrieved_message_ids or [])
    # message IDs in this project all look like <prefix>_<digits>, e.g. grp_01595,
    # msg_00969, ag_00001, pj_00737 — ground-truth field-name evidence (e.g.
    # "canonical_continuity", "life_events") doesn't match this shape.
    import re

    id_pattern = re.compile(r"^[a-z]{2,3}_\d{4,6}$")
    evidence_ids = [e for e in question.get("evidence", []) if id_pattern.match(e)]
    if not evidence_ids:
        return None
    hits = sum(1 for e in evidence_ids if e in retrieved)
    return hits / len(evidence_ids)


def run_one(question: dict, backend_fn) -> dict:
    try:
        t0 = time.time()
        answer_info = backend_fn(question)
        elapsed_s = time.time() - t0
        verdict = judge_answer(question, answer_info["answer"])
        retrieved_ids = answer_info.get("retrieved_message_ids", [])
        return {
            "qid": question["qid"],
            "category": question["category"],
            "dataset": question["dataset"],
            "question": question["question"],
            "gold_answer": question["gold_answer"],
            "answerable": question.get("answerable", True),
            "generated_answer": answer_info["answer"],
            "answer_latency_s": round(elapsed_s, 2),
            "evidence_hit_rate": evidence_hit_rate(question, retrieved_ids),
            **{k: v for k, v in answer_info.items() if k != "answer"},
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
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--results", default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--backend", choices=list(BACKENDS.keys()), default="single_shot")
    args = parser.parse_args()

    backend_fn = BACKENDS[args.backend]
    questions = load_questions(args.questions, category=args.category, limit=args.limit)
    print(f"Running eval on {len(questions)} questions with {args.workers} workers, backend={args.backend}...")

    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, q, backend_fn): q for q in questions}
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results.append(r)
            status = r.get("verdict", "?")
            print(f"[{i}/{len(questions)}] {r['qid']} ({r['category']}): {status}", flush=True)

    results.sort(key=lambda r: r["qid"])
    with open(args.results, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_errors = sum(1 for r in results if r.get("verdict") == "error")
    print(f"\nWrote {len(results)} results to {args.results} ({n_errors} errors)")


if __name__ == "__main__":
    main()
