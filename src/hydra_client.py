"""Thin REST wrapper around HydraDB's v2 API.

Base URL and API key come from the environment (see .env). We talk to the
REST API directly with `requests` rather than the official SDK, since the
SDK isn't installed in this environment and the surface we need is small:
create a database, ingest memories, poll processing status, and query.
"""

import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ["HYDRA_DB_BASE_URL"].rstrip("/")
API_KEY = os.environ["HYDRA_DB_API_KEY"]

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "API-Version": "2",
}


class HydraAPIError(RuntimeError):
    def __init__(self, resp):
        self.status_code = resp.status_code
        self.body = resp.text
        super().__init__(f"{resp.request.method} {resp.request.url} -> {resp.status_code}: {resp.text[:2000]}")


def _raise_for_status(resp):
    if not resp.ok:
        raise HydraAPIError(resp)


NETWORK_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectTimeout,
)


def _request(method: str, url: str, retries: int = 5, backoff: float = 2.0, **kwargs) -> requests.Response:
    """requests.request with retries on transient network errors (connection
    drops, timeouts) and on 5xx/429 responses. Raises HydraAPIError on a
    persistent non-2xx response, or the underlying network exception if
    retries are exhausted."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = requests.request(method, url, **kwargs)
        except NETWORK_EXCEPTIONS as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
                continue
            raise
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
                continue
        return resp
    raise last_exc


def database_status(database: str) -> dict:
    resp = _request("GET", f"{BASE_URL}/databases/status", headers=HEADERS, params={"database": database}, timeout=60)
    _raise_for_status(resp)
    return resp.json()


def wait_for_database_ready(database: str, timeout_s: int = 180, poll_s: float = 3.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            body = database_status(database)
        except HydraAPIError as e:
            if e.status_code == 404:
                # Briefly not-yet-visible right after creation; keep polling.
                time.sleep(poll_s)
                continue
            raise
        data = body.get("data", body)
        infra = data.get("infra", {})
        if infra.get("ready_for_ingestion"):
            return
        time.sleep(poll_s)
    raise TimeoutError(f"Database {database!r} did not become ready in time")


def ensure_database(database: str):
    """Create the database if it doesn't already exist, then wait for it to be
    provisioned. Idempotent."""
    resp = _request(
        "POST",
        f"{BASE_URL}/databases",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"database": database},
        timeout=30,
    )
    if resp.status_code != 409:
        _raise_for_status(resp)
    wait_for_database_ready(database)


def ingest_memories(database: str, collection: str, memories: list[dict]) -> list[str]:
    """Ingest a batch of memory items. Each item: {"text": ..., "metadata": {...}}.

    Returns the list of source IDs created.
    """
    resp = _request(
        "POST",
        f"{BASE_URL}/context/ingest",
        headers=HEADERS,
        data={
            "type": "memory",
            "database": database,
            "collection": collection,
            "memories": json.dumps(memories),
        },
        timeout=120,
    )
    _raise_for_status(resp)
    body = resp.json()
    results = body.get("data", {}).get("results", [])
    return [r["id"] for r in results]


def context_status(database: str, collection: str, ids: list[str]) -> dict:
    resp = _request(
        "GET",
        f"{BASE_URL}/context/status",
        headers=HEADERS,
        params={"database": database, "collection": collection, "ids": ids},
        timeout=30,
    )
    _raise_for_status(resp)
    return resp.json()


STATUS_CHUNK = 100


def wait_for_processing(
    database: str, collection: str, ids: list[str], timeout_s: int = 300, poll_s: float = 3.0
) -> None:
    """Poll /context/status until all given ids leave the queued/processing state.
    Checked in chunks — a GET with thousands of ids as query params blows past
    URL length limits."""
    if not ids:
        return
    remaining = set(ids)
    deadline = time.time() + timeout_s
    while time.time() < deadline and remaining:
        remaining_list = list(remaining)
        for i in range(0, len(remaining_list), STATUS_CHUNK):
            chunk = remaining_list[i : i + STATUS_CHUNK]
            body = context_status(database, collection, chunk)
            statuses = body.get("data", {}).get("statuses", [])
            for s in statuses:
                if s.get("indexing_status") in ("completed", "errored"):
                    remaining.discard(s["id"])
        if remaining:
            time.sleep(poll_s)
    if remaining:
        raise TimeoutError(f"Timed out waiting for {len(remaining)} sources to finish processing")


def query(
    database: str,
    text: str,
    collections: list[str],
    query_type: str = "memory",
    query_by: str = "hybrid",
    mode: str = "thinking",
    max_results: int = 10,
    graph_context: bool = True,
) -> dict:
    payload = {
        "database": database,
        "query": text,
        "type": query_type,
        "query_by": query_by,
        "mode": mode,
        "collections": collections,
        "max_results": max_results,
        "graph_context": graph_context,
    }
    resp = _request(
        "POST",
        f"{BASE_URL}/query",
        headers={**HEADERS, "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    _raise_for_status(resp)
    return resp.json()


def database_stats(database: str) -> dict:
    resp = _request("GET", f"{BASE_URL}/databases/stats", headers=HEADERS, params={"database": database}, timeout=30)
    _raise_for_status(resp)
    return resp.json()
