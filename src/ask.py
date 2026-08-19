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

ANSWER_SYSTEM_PROMPT_WITH_GRAPH = """You are Arjun's long-term memory assistant. You answer questions \
about Arjun using ONLY two things given to you below: (1) retrieved conversation excerpts, each \
tagged with its message_id, timestamp, and sender, and (2) a knowledge-graph context of entity \
relationships HydraDB extracted across the conversation history (source entity --predicate--> \
target entity, with a supporting sentence). Rules:

1. Base your answer strictly on these two sources. Do not use outside knowledge or invent details.
2. The graph context is useful for connecting facts that aren't in the same chunk — e.g. relationship/
role changes, who's connected to whom, or a fact restated in different words across the timeline. \
Use it to corroborate or connect excerpts, not as a substitute for them — if the graph asserts \
something no excerpt supports, treat it as weaker evidence, not certain fact.
3. If neither source contains enough evidence to answer confidently, say so plainly instead of \
guessing — abstaining/hedging is the correct behavior when evidence is missing or thin.
4. If excerpts/graph edges from different times conflict, prefer the most recent one and note the \
change if it's relevant to the question (e.g. a fact that was updated over time).
5. Be concise. Cite supporting message_ids in square brackets, e.g. [ag_00001], where relevant.
6. If the user's question includes an explicit formatting/behavioral instruction, follow it exactly \
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


def format_graph_context(graph_context: dict, limit: int = 25) -> str:
    paths = (graph_context or {}).get("query_paths", [])
    lines = []
    seen = set()
    for path in paths:
        for t in path.get("triplets", []):
            source, relation, target = t.get("source", {}), t.get("relation", {}), t.get("target", {})
            key = (source.get("name"), relation.get("canonical_predicate"), target.get("name"))
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"{source.get('name')} --{relation.get('canonical_predicate')}--> {target.get('name')}"
                f"  (\"{relation.get('context', '')}\")"
            )
            if len(lines) >= limit:
                break
        if len(lines) >= limit:
            break
    return "\n".join(lines) if lines else "(no graph relations returned)"


def answer_question(
    question: str,
    collections: list[str] | None = None,
    instruction: str | None = None,
    max_results: int = 15,
    use_graph_context: bool = False,
) -> dict:
    """use_graph_context=False (default) matches the better-performing configuration
    from our ablation (results/comparison_graph_context.md): feeding HydraDB's raw
    graph triples into the answer prompt measured a -4.1pt regression overall across
    242 eval questions vs. chunks alone. graph_context is still always requested from
    HydraDB (mode=thinking computes it regardless) — this flag only controls whether
    it's included in the answer-synthesis prompt, so the ablation stays reproducible."""
    collections = collections or ALL_COLLECTIONS
    result = hydra_client.query(
        DATABASE, question, collections=collections, max_results=max_results, graph_context=True
    )
    data = result.get("data", {})
    chunks = data.get("chunks", [])
    graph_context = data.get("graph_context", {})

    user_prompt = question
    if instruction:
        user_prompt = f"{question}\n\n(Formatting/behavioral instruction to follow: {instruction})"
    user_prompt += f"\n\n--- RETRIEVED MEMORIES ---\n{format_chunks(chunks)}"
    if use_graph_context:
        user_prompt += f"\n\n--- GRAPH CONTEXT (entity relationships) ---\n{format_graph_context(graph_context)}"

    answer = chat(
        ANSWER_MODEL,
        [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT_WITH_GRAPH if use_graph_context else ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return {
        "question": question,
        "answer": answer,
        "collections_queried": collections,
        "num_chunks_retrieved": len(chunks),
        "num_graph_paths_retrieved": len(graph_context.get("query_paths", [])),
        "used_graph_context_in_prompt": use_graph_context,
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
    parser.add_argument(
        "--use-graph-context",
        action="store_true",
        help="Include HydraDB's graph_context triples in the answer prompt (measured worse overall in our ablation; off by default).",
    )
    args = parser.parse_args()

    result = answer_question(args.question, collections=args.collections, use_graph_context=args.use_graph_context)
    print(f"\n{result['answer']}\n")
    print(f"(retrieved {result['num_chunks_retrieved']} chunks from {result['collections_queried']})")


if __name__ == "__main__":
    main()
