"""Ask a question about Arjun: query HydraDB for relevant memories, then
synthesize a final answer from the retrieved context. This is the CLI
"product" entrypoint — the same function is used by the eval harness.

Usage:
    python src/ask.py "Where did Arjun and Gopal go on their first trip?" --collections gopal
    python src/ask.py "..."   # no --collections -> searches across all relationships
"""

import argparse
import json

import hydra_client
from datasets import DATABASE, DATASETS
from llm_client import ANSWER_MODEL, chat

ALL_COLLECTIONS = [cfg["collection"] for cfg in DATASETS.values()]

ANSWER_SYSTEM_PROMPT = """You are Arjun's long-term memory assistant. You answer questions about \
Arjun using ONLY the retrieved conversation excerpts given to you below — each tagged with its \
message_id, timestamp, and sender. Rules:

1. Base your answer strictly on the excerpts. Do not use outside knowledge or invent details.
2. If the excerpts don't contain enough evidence to answer confidently, say so plainly instead of \
guessing — abstaining/hedging is the correct behavior when evidence is missing or thin.
3. If excerpts from different times conflict, prefer the most recent one and note the change if \
it's relevant to the question (e.g. a fact that was updated over time).
4. Be concise. Cite supporting message_ids in square brackets, e.g. [ag_00001], where relevant.
5. If the user's question includes an explicit formatting/behavioral instruction, follow it exactly \
in addition to answering correctly."""


def format_chunks(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        meta = c.get("metadata", {})
        lines.append(
            f"[{meta.get('message_id', c.get('id'))}] "
            f"({meta.get('timestamp', '?')}, collection={c.get('collection')}) "
            f"{c.get('chunk_content', '')}"
        )
    return "\n".join(lines) if lines else "(no relevant memories retrieved)"


def answer_question(
    question: str,
    collections: list[str] | None = None,
    instruction: str | None = None,
    max_results: int = 15,
) -> dict:
    collections = collections or ALL_COLLECTIONS
    result = hydra_client.query(DATABASE, question, collections=collections, max_results=max_results)
    chunks = result.get("data", {}).get("chunks", [])

    user_prompt = question
    if instruction:
        user_prompt = f"{question}\n\n(Formatting/behavioral instruction to follow: {instruction})"
    user_prompt += f"\n\n--- RETRIEVED MEMORIES ---\n{format_chunks(chunks)}"

    answer = chat(
        ANSWER_MODEL,
        [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return {
        "question": question,
        "answer": answer,
        "collections_queried": collections,
        "num_chunks_retrieved": len(chunks),
        "retrieved_message_ids": [c.get("metadata", {}).get("message_id") for c in chunks],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument(
        "--collections",
        nargs="*",
        default=None,
        help="Restrict to specific collections (friend_group, gopal, mother_son, prof_jatin). Default: all.",
    )
    args = parser.parse_args()

    result = answer_question(args.question, collections=args.collections)
    print(f"\n{result['answer']}\n")
    print(f"(retrieved {result['num_chunks_retrieved']} chunks from {result['collections_queried']})")


if __name__ == "__main__":
    main()
