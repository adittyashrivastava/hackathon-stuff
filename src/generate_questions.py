"""Generate BEAM-style synthetic eval questions grounded in each dataset's
ground-truth facts (+ a spread sample of the raw transcript), across all 10
BEAM memory-ability categories: https://github.com/mohammadtavakoli78/BEAM

Most categories are generated per-dataset (single collection in scope).
Two are structurally different:
  - instruction_following: layers a format/behavioral constraint on top of a
    grounded content question, so the judge can separately score content
    correctness and instruction compliance.
  - multi_session_reasoning: generated across a PAIR of datasets, requiring
    the answer to combine evidence from both collections.

Output: questions/generated_questions.jsonl, one question object per line.
"""

import argparse
import itertools
import json
import os
import random

from datasets import DATASETS, dataset_path, ground_truth_path
from derive_ground_truth import sample_spread, format_excerpt
from llm_client import GEN_MODEL, chat_json

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "questions", "generated_questions.jsonl")

SAMPLE_N = 50
PER_CATEGORY = 3  # questions per (dataset, category)
PER_INSTRUCTION = 3  # per dataset
PER_MULTI_SESSION = 3  # per dataset pair

HARDER_SUFFIX = """

This is an ADVERSARIAL/HARDER follow-up round — do not repeat easy or obvious \
questions. Specifically prefer questions that involve: near-miss distractors (a \
similar-but-wrong fact elsewhere in the transcript that a sloppy retriever might \
confuse with the right one), subtle multi-hop reasoning (the answer requires \
combining 2+ separate messages/facts, not one lookup), precise dates/counts rather \
than vague ones, and — where relevant to the category — deliberately tricky \
contradiction/knowledge-update traps (a fact stated, then revised or contradicted \
much later in the timeline). Skew difficulty toward "medium"/"hard"."""

CORE_CATEGORIES = {
    "abstention": "Whether a good answer appropriately withholds/hedges when the transcript lacks supporting evidence, rather than asserting an unsupported fact.",
    "contradiction_resolution": "Detecting and correctly reconciling statements that appear to conflict across distant parts of the conversation (e.g. a stated fact later superseded or contradicted).",
    "event_ordering": "Correctly reconstructing the sequence of events/statements as they evolved over time.",
    "information_extraction": "Recalling specific entities, dates, and factual details accurately from across a long history.",
    "knowledge_update": "Answering with the CURRENT/latest state of a fact that changed over time, not a stale earlier version.",
    "preference_following": "Recalling a person's stated preference or habit and answering consistently with it.",
    "summarization": "Abstracting/compressing a span of the relationship into an accurate, non-hallucinated summary.",
    "temporal_reasoning": "Correctly reasoning about explicit or implicit time relationships (durations, before/after, how long ago, etc.).",
}

GEN_SYSTEM_PROMPT = """You write evaluation questions for a long-term-memory QA benchmark, styled \
after the BEAM benchmark's memory-ability categories. You are given canonical ground-truth facts \
about one relationship-thread of a synthetic person named Arjun, plus a sample of the raw \
transcript (spread evenly across the whole time range).

Category under test: "{category}" — {definition}

Write {n} question(s) that specifically probe this memory ability, grounded ONLY in the facts and \
transcript given — never invent facts not supported by them. Respond with JSON:
{{"questions": [
  {{"question": "...",
    "gold_answer": "the correct, concise answer (or, for abstention questions, a description of the \
correct abstaining/hedging behavior)",
    "gold_rationale": "1-2 sentences on why this is correct, citing the specific fact/message",
    "evidence": ["message_id or ground-truth field name used as support"],
    "answerable": true or false,
    "difficulty": "easy" | "medium" | "hard"
  }}, ...
]}}

For "abstention" questions specifically: at least one question should ask about something that \
sounds plausible but is NOT actually stated anywhere in the facts/transcript (answerable: false), \
and gold_answer should describe the correct refusal/hedge, not a fabricated fact."""

INSTRUCTION_SYSTEM_PROMPT = """You write evaluation questions for a long-term-memory QA benchmark, \
testing "instruction_following": whether an agent maintains adherence to a user-given formatting or \
behavioral constraint while answering a content question, over a long context.

You are given canonical ground-truth facts and a transcript sample about one relationship-thread of \
a synthetic person named Arjun. Write {n} question(s), each combining (a) a real content question \
grounded in the facts/transcript, and (b) an explicit instruction about HOW to answer (format, \
length, things to avoid mentioning, style). Respond with JSON:
{{"questions": [
  {{"question": "the content question, with the instruction folded into the phrasing naturally \
(e.g. 'In exactly one sentence and without mentioning any dates, what...')",
    "instruction": "the specific constraint restated plainly, so a judge can check compliance \
mechanically (e.g. 'answer must be exactly one sentence' / 'must not contain any 4-digit year')",
    "gold_answer": "the correct content answer, already satisfying the instruction",
    "gold_rationale": "why this is correct",
    "evidence": ["message_id or ground-truth field"],
    "answerable": true,
    "difficulty": "easy" | "medium" | "hard"
  }}, ...
]}}"""

MULTI_SESSION_SYSTEM_PROMPT = """You write evaluation questions for a long-term-memory QA benchmark, \
testing "multi_session_reasoning": integrating evidence from two DIFFERENT relationship-threads \
about the same synthetic person, Arjun, that a good memory system must combine to answer correctly.

You are given ground-truth facts + transcript samples from TWO separate threads: "{name_a}" and \
"{name_b}". Write {n} question(s) that can ONLY be answered well by combining information from BOTH \
threads (e.g. comparing what Arjun told each party about the same topic/event, or noting something \
he told one party but apparently never told the other). Respond with JSON:
{{"questions": [
  {{"question": "...",
    "gold_answer": "...",
    "gold_rationale": "...",
    "evidence": ["message_id or ground-truth field, from either thread"],
    "answerable": true or false,
    "difficulty": "easy" | "medium" | "hard"
  }}, ...
]}}"""


def load_ground_truth(dataset_name: str) -> dict:
    path = ground_truth_path(dataset_name)
    if not os.path.exists(path):
        return {}
    return json.load(open(path))


def load_sample_transcript(dataset_name: str, n: int = SAMPLE_N) -> str:
    messages = [json.loads(l) for l in open(dataset_path(dataset_name)) if l.strip()]
    sample = sample_spread(messages, n)
    lines = []
    for m in sample:
        ts = m.get("timestamp") or m.get("timestamp_ist")
        lines.append(f"[{m.get('message_id')}] [{ts}] {m.get('sender')}: {m.get('text', '')}")
    return "\n".join(lines)


def context_block(dataset_name: str) -> str:
    gt = load_ground_truth(dataset_name)
    transcript = load_sample_transcript(dataset_name)
    return (
        f"--- GROUND TRUTH FACTS ({dataset_name}) ---\n{json.dumps(gt, indent=2)}\n\n"
        f"--- SAMPLED TRANSCRIPT ({dataset_name}) ---\n{transcript}"
    )


def generate_core(
    dataset_name: str, category: str, definition: str, n: int = PER_CATEGORY, harder: bool = False
) -> list[dict]:
    prompt = GEN_SYSTEM_PROMPT.format(category=category, definition=definition, n=n)
    if harder:
        prompt += HARDER_SUFFIX
    result = chat_json(
        GEN_MODEL,
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": context_block(dataset_name)},
        ],
        temperature=0.8 if harder else 0.6,
    )
    questions = result.get("questions", [])
    collection = DATASETS[dataset_name]["collection"]
    for q in questions:
        q.update({"category": category, "dataset": dataset_name, "collections_in_scope": [collection]})
    return questions


def generate_instruction_following(dataset_name: str, n: int = PER_INSTRUCTION, harder: bool = False) -> list[dict]:
    prompt = INSTRUCTION_SYSTEM_PROMPT.format(n=n)
    if harder:
        prompt += HARDER_SUFFIX
    result = chat_json(
        GEN_MODEL,
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": context_block(dataset_name)},
        ],
        temperature=0.8 if harder else 0.6,
    )
    questions = result.get("questions", [])
    collection = DATASETS[dataset_name]["collection"]
    for q in questions:
        q.update(
            {"category": "instruction_following", "dataset": dataset_name, "collections_in_scope": [collection]}
        )
    return questions


def generate_multi_session(
    dataset_a: str, dataset_b: str, n: int = PER_MULTI_SESSION, harder: bool = False
) -> list[dict]:
    prompt = MULTI_SESSION_SYSTEM_PROMPT.format(name_a=dataset_a, name_b=dataset_b, n=n)
    if harder:
        prompt += HARDER_SUFFIX
    context = context_block(dataset_a) + "\n\n" + context_block(dataset_b)
    result = chat_json(
        GEN_MODEL,
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": context},
        ],
        temperature=0.8 if harder else 0.6,
    )
    questions = result.get("questions", [])
    collections = [DATASETS[dataset_a]["collection"], DATASETS[dataset_b]["collection"]]
    for q in questions:
        q.update(
            {
                "category": "multi_session_reasoning",
                "dataset": f"{dataset_a}+{dataset_b}",
                "collections_in_scope": collections,
            }
        )
    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=PER_CATEGORY)
    parser.add_argument("--per-instruction", type=int, default=PER_INSTRUCTION)
    parser.add_argument("--per-multi-session", type=int, default=PER_MULTI_SESSION)
    parser.add_argument("--harder", action="store_true", help="Skew toward adversarial/harder questions")
    parser.add_argument("--append", action="store_true", help="Append to the existing question file instead of overwriting")
    parser.add_argument("--round-tag", default=None, help="Optional tag stamped on every question this run ('round' field)")
    args = parser.parse_args()

    random.seed(42)
    new_questions = []

    for dataset_name in DATASETS:
        for category, definition in CORE_CATEGORIES.items():
            print(f"[{dataset_name}] generating {category}...")
            qs = generate_core(dataset_name, category, definition, n=args.per_category, harder=args.harder)
            new_questions.extend(qs)
            print(f"[{dataset_name}] {category}: {len(qs)} questions")

        print(f"[{dataset_name}] generating instruction_following...")
        qs = generate_instruction_following(dataset_name, n=args.per_instruction, harder=args.harder)
        new_questions.extend(qs)
        print(f"[{dataset_name}] instruction_following: {len(qs)} questions")

    for dataset_a, dataset_b in itertools.combinations(DATASETS.keys(), 2):
        print(f"[{dataset_a}+{dataset_b}] generating multi_session_reasoning...")
        qs = generate_multi_session(dataset_a, dataset_b, n=args.per_multi_session, harder=args.harder)
        new_questions.extend(qs)
        print(f"[{dataset_a}+{dataset_b}] multi_session_reasoning: {len(qs)} questions")

    if args.round_tag:
        for q in new_questions:
            q["round"] = args.round_tag

    existing_questions = []
    if args.append and os.path.exists(QUESTIONS_PATH):
        existing_questions = [json.loads(l) for l in open(QUESTIONS_PATH) if l.strip()]

    all_questions = existing_questions + new_questions
    for i, q in enumerate(all_questions):
        q["qid"] = f"q_{i:04d}"

    os.makedirs(os.path.dirname(QUESTIONS_PATH), exist_ok=True)
    with open(QUESTIONS_PATH, "w") as f:
        for q in all_questions:
            f.write(json.dumps(q) + "\n")

    print(f"\nWrote {len(all_questions)} total questions ({len(new_questions)} new) to {QUESTIONS_PATH}")


if __name__ == "__main__":
    main()
