"""Agentic alternative to ask.py's single-shot retrieval: instead of one HydraDB
query, an LLM agent (DeepSeek V4 Flash) is given a `search_memories` tool and
decides for itself how many searches to run and what to search for, inspecting
results between calls — e.g. a first broad search, then targeted follow-ups on
names/entities it noticed but doesn't yet have full facts on (the exact failure
mode found in ask.py: a single-shot query missed Kabir's birth message entirely
even at max_results=120; an agent that can *notice* Gopal has one known child and
go look for a second has a chance to catch it).

Usage:
    python src/agentic_ask.py "Who among Arjun's closest friends currently has children?"
"""

import argparse
import json

import ask
import hydra_client
from datasets import DATABASE, DATASETS
from llm_client import ANSWER_MODEL, chat, chat_with_tools

MAX_STEPS = 4
RESULTS_PER_SEARCH = 12

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_memories",
        "description": (
            "Search Arjun's conversation history in HydraDB for relevant memories. "
            "Returns up to a dozen relevant excerpts. Call this multiple times with "
            "different, more specific queries to build up full evidence — e.g. after "
            "a broad search surfaces a name or partial fact, run a follow-up search "
            "specifically on that name to check for more/updated information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query."},
                "collections": {
                    "type": "array",
                    "items": {"type": "string", "enum": list({c["collection"] for c in DATASETS.values()})},
                    "description": "Which relationship collections to search. Omit to search all.",
                },
            },
            "required": ["query"],
        },
    },
}

AGENT_SYSTEM_PROMPT = """You are Arjun's long-term memory research agent. You answer questions about \
Arjun by searching his conversation history with the search_memories tool — you do NOT have any \
built-in knowledge about him.

Strategy:
1. Start with a broad search on the question itself.
2. Read the results. If they mention a specific person/entity/date that seems relevant but \
incomplete (e.g. "Gopal has a child" without full details, or a name you should verify has no \
OTHER facts attached), run a targeted follow-up search specifically on that.
3. Keep searching (up to several times) until you're confident you have complete, current \
evidence — searches are cheap, a wrong or incomplete answer is not. Don't stop after just one search \
if the question has multiple parts or asks about "current"/"evolved" state.
4. IMPORTANT: this conversation history is full of repeated scripted banter lines (the same phrase \
reused dozens of times), which can bury a single, sparse, critical fact under near-duplicates when \
your query only uses abstract words like "children"/"family". If an abstract query returns thin or \
repetitive-looking results, try a follow-up using CONCRETE event language instead — e.g. "arrived", \
"born", specific names — since one-off factual messages (a birth announcement, an arrival) are \
often worded very differently from the generic term you'd naturally search for.
4. Once you have enough evidence, STOP calling tools and write your final answer using ONLY what \
the searches returned. Cite message_ids in square brackets. If you still lack evidence for part of \
the question after several searches, say so plainly rather than guessing.
5. Available collections: friend_group, gopal, mother_son, prof_jatin."""


def format_search_result(chunks: list[dict]) -> str:
    if not chunks:
        return "(no results)"
    lines = []
    for c in chunks:
        meta = c.get("metadata", {})
        lines.append(
            f"[{meta.get('message_id', c.get('id'))}] "
            f"({meta.get('timestamp', '?')}, {c.get('collection')}) "
            f"{c.get('chunk_content', '')}"
        )
    return "\n".join(lines)


def run_search(query: str, collections: list[str] | None) -> list[dict]:
    collections = collections or ask.ALL_COLLECTIONS
    result = hydra_client.query(
        DATABASE, query, collections=collections, max_results=RESULTS_PER_SEARCH * 4, graph_context=False
    )
    raw = result.get("data", {}).get("chunks", [])
    return ask.dedupe_chunks(raw, keep=RESULTS_PER_SEARCH)


def agentic_answer(question: str, instruction: str | None = None, max_steps: int = MAX_STEPS) -> dict:
    user_content = question
    if instruction:
        user_content += f"\n\n(Formatting/behavioral instruction to follow: {instruction})"

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    all_message_ids: list[str] = []
    search_log: list[dict] = []
    steps_used = 0

    for step in range(max_steps):
        steps_used = step + 1
        msg = chat_with_tools(ANSWER_MODEL, messages, tools=[SEARCH_TOOL], temperature=0.1)
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            return {
                "question": question,
                "answer": msg.get("content", ""),
                "steps_used": steps_used,
                "num_searches": len(search_log),
                "search_log": search_log,
                "retrieved_message_ids": all_message_ids,
            }

        messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": tool_calls})
        for tc in tool_calls:
            args = json.loads(tc["function"]["arguments"] or "{}")
            query = args.get("query", question)
            collections = args.get("collections")
            chunks = run_search(query, collections)
            ids = [c.get("metadata", {}).get("message_id") for c in chunks]
            all_message_ids.extend(ids)
            search_log.append({"query": query, "collections": collections, "num_results": len(chunks)})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": format_search_result(chunks),
                }
            )

    # Ran out of steps — force a final answer from whatever's accumulated. No `tools`
    # passed here at all (plain chat(), not chat_with_tools()) so it is structurally
    # impossible for the model to request another search instead of answering.
    messages.append(
        {
            "role": "user",
            "content": (
                "You've used all your allotted searches. Do not search again — write your final "
                "answer now using only the evidence already gathered above. If evidence is "
                "incomplete, say so explicitly rather than guessing."
            ),
        }
    )
    answer_text = chat(ANSWER_MODEL, messages, temperature=0.1)
    return {
        "question": question,
        "answer": answer_text,
        "steps_used": steps_used,
        "num_searches": len(search_log),
        "search_log": search_log,
        "retrieved_message_ids": all_message_ids,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()

    result = agentic_answer(args.question)
    print(f"\n{result['answer']}\n")
    print(f"(agent ran {result['num_searches']} searches over {result['steps_used']} steps)")
    for s in result["search_log"]:
        print(f"  - search: {s['query']!r} ({s['collections'] or 'all'}) -> {s['num_results']} results")


if __name__ == "__main__":
    main()
