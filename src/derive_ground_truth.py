"""Derive a canonical-facts ground-truth JSON for datasets that didn't ship
one (mother_son, prof_jatin), by sampling messages spread across the full
time range and asking an LLM to extract structured facts — mirroring the
shape of the hand-authored ground-truth files (arjun_core5_friend_group,
arjun_gopal).

This is explicitly LLM-derived, not hand-authored — see
DATASETS[...]["ground_truth_is_authored"] in datasets.py and the README
caveat about it.
"""

import json
import os

from datasets import DATASETS, dataset_path, ground_truth_path
from llm_client import GEN_MODEL, chat_json

SAMPLE_SIZE = 220

SYSTEM_PROMPT = """You extract canonical ground-truth facts from a synthetic WhatsApp \
transcript for use as an answer key in a downstream QA eval. Read the sampled messages \
(spread evenly across the whole time range, so you see the full arc) and produce a JSON \
object with this shape:

{
  "canonical_ground_truth": { <flat key facts: names, roles, locations, stable traits> },
  "life_events": [ {"date": "YYYY-MM" or "YYYY-MM-DD", "event": "..."} ],
  "knowledge_updates": [ {"topic": "...", "before": "...", "after": "...", "changed_around": "YYYY-MM"} ],
  "evaluation_notes": [ "explicit negative constraints: things that are NEVER stated \
outright, or that a good answerer must NOT hallucinate/infer beyond what's shown" ],
  "preferences": [ {"who": "...", "preference": "...", "evidence_hint": "short quote or paraphrase"} ]
}

Only include facts you can actually support from the sampled messages. Do not invent \
facts absent from the text. If the sample doesn't cover something, omit it rather than \
guessing. Respond with JSON only."""


def sample_spread(messages: list[dict], n: int) -> list[dict]:
    if len(messages) <= n:
        return messages
    step = len(messages) / n
    idxs = [int(i * step) for i in range(n)]
    return [messages[i] for i in idxs]


def format_excerpt(msg: dict) -> str:
    ts = msg.get("timestamp") or msg.get("timestamp_ist")
    return f"[{ts}] {msg.get('sender')}: {msg.get('text', '')}"


def derive(dataset_name: str) -> dict:
    path = dataset_path(dataset_name)
    messages = [json.loads(l) for l in open(path) if l.strip()]
    sample = sample_spread(messages, SAMPLE_SIZE)
    transcript = "\n".join(format_excerpt(m) for m in sample)

    relationship = DATASETS[dataset_name]["relationship"]
    user_prompt = (
        f"Relationship context: {relationship}.\n"
        f"Full dataset spans {messages[0].get('timestamp') or messages[0].get('timestamp_ist')} "
        f"to {messages[-1].get('timestamp') or messages[-1].get('timestamp_ist')} "
        f"({len(messages)} total messages; {len(sample)} sampled below, evenly spread).\n\n"
        f"--- SAMPLED TRANSCRIPT ---\n{transcript}\n--- END TRANSCRIPT ---"
    )

    result = chat_json(
        GEN_MODEL,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    result["_meta"] = {
        "dataset": dataset_name,
        "derived_by_llm": True,
        "model": GEN_MODEL,
        "sample_size": len(sample),
        "total_messages": len(messages),
    }
    return result


def main():
    for name, cfg in DATASETS.items():
        if cfg["ground_truth_is_authored"]:
            continue
        out_path = ground_truth_path(name)
        if os.path.exists(out_path):
            print(f"[{name}] ground truth already exists at {out_path}, skipping")
            continue
        print(f"[{name}] deriving ground truth...")
        gt = derive(name)
        with open(out_path, "w") as f:
            json.dump(gt, f, indent=2)
        print(f"[{name}] wrote {out_path}")


if __name__ == "__main__":
    main()
