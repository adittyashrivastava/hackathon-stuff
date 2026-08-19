"""Ingest the four Arjun relationship-thread JSONL datasets into HydraDB as memories.

Usage:
    python src/ingest.py --dataset gopal --limit 20   # smoke test
    python src/ingest.py --dataset all                # full ingest, all datasets
"""

import argparse
import json
import sys
import time

import hydra_client
from datasets import DATABASE, DATASETS, dataset_path

BATCH_SIZE = 40


def load_messages(path: str, limit: int | None = None) -> list[dict]:
    msgs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs.append(json.loads(line))
            if limit and len(msgs) >= limit:
                break
    return msgs


def to_memory_item(msg: dict, dataset: str) -> dict:
    timestamp = msg.get("timestamp") or msg.get("timestamp_ist")
    metadata = {
        "dataset": dataset,
        "message_id": msg.get("message_id"),
        "sender": msg.get("sender"),
        "sender_role": msg.get("sender_role"),
        "timestamp": timestamp,
        "conversation_id": msg.get("conversation_id"),
        "thread_id": msg.get("thread_id"),
        "topic_tags": msg.get("topic_tags", []),
    }
    text = f"[{timestamp}] {msg.get('sender')}: {msg.get('text', '')}"
    return {"text": text, "metadata": metadata}


def ingest_items_adaptive(collection: str, items: list[dict], depth: int = 0) -> list[str]:
    """Ingest a batch of memory items, halving and retrying on a 413 'batch too
    large' error, and backing off on 429 rate-limit errors."""
    try:
        return hydra_client.ingest_memories(DATABASE, collection, items)
    except hydra_client.HydraAPIError as e:
        if e.status_code == 413 and len(items) > 1:
            mid = len(items) // 2
            time.sleep(0.5)
            left = ingest_items_adaptive(collection, items[:mid], depth + 1)
            right = ingest_items_adaptive(collection, items[mid:], depth + 1)
            return left + right
        if e.status_code == 429:
            time.sleep(2.0)
            return ingest_items_adaptive(collection, items, depth + 1)
        raise


def ingest_dataset(name: str, limit: int | None = None) -> list[str]:
    """Submit all batches for a dataset without waiting for indexing. Returns
    all created source ids so the caller can wait for them in bulk."""
    cfg = DATASETS[name]
    path = dataset_path(name)
    msgs = load_messages(path, limit=limit)
    print(f"[{name}] loaded {len(msgs)} messages from {cfg['file']}", flush=True)

    all_ids = []
    n_batches = -(-len(msgs) // BATCH_SIZE)
    for i in range(0, len(msgs), BATCH_SIZE):
        batch = msgs[i : i + BATCH_SIZE]
        items = [to_memory_item(m, name) for m in batch]
        ids = ingest_items_adaptive(cfg["collection"], items)
        all_ids.extend(ids)
        print(f"[{name}] submitted batch {i // BATCH_SIZE + 1}/{n_batches} "
              f"({len(batch)} msgs, {len(ids)} ids)", flush=True)
        time.sleep(0.1)

    return all_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*DATASETS.keys(), "all"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="Limit messages per dataset (smoke test)")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for processing completion at the end")
    args = parser.parse_args()

    hydra_client.ensure_database(DATABASE)

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    ids_by_collection: dict[str, list[str]] = {}
    for name in targets:
        try:
            ids = ingest_dataset(name, limit=args.limit)
            ids_by_collection.setdefault(DATASETS[name]["collection"], []).extend(ids)
            print(f"[{name}] submitted {len(ids)} memories", flush=True)
        except Exception as e:
            print(f"[{name}] FAILED: {e}", file=sys.stderr, flush=True)
            raise

    if not args.no_wait:
        for collection, ids in ids_by_collection.items():
            print(f"[{collection}] waiting for {len(ids)} sources to finish indexing...", flush=True)
            hydra_client.wait_for_processing(DATABASE, collection, ids, timeout_s=1800, poll_s=5.0)
            print(f"[{collection}] all {len(ids)} sources processed", flush=True)


if __name__ == "__main__":
    main()
