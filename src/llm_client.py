"""Thin wrapper around OpenRouter's chat-completions API.

Used for three distinct jobs in this project, each configurable via env vars
so they can be swapped independently: synthesizing final answers from
HydraDB retrieval results, generating synthetic BEAM-style questions, and
judging generated answers against ground truth.
"""

import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "deepseek/deepseek-v4-flash")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek/deepseek-v4-flash")
GEN_MODEL = os.environ.get("GEN_MODEL", "deepseek/deepseek-v4-flash")

_URL = "https://openrouter.ai/api/v1/chat/completions"


_NETWORK_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectTimeout,
)


def chat(
    model: str, messages: list[dict], temperature: float = 0.2, json_mode: bool = False, retries: int = 4
) -> str:
    body = {"model": model, "messages": messages, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                _URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=120,
            )
        except _NETWORK_EXCEPTIONS as e:
            last_exc = e
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
                continue
        if not resp.ok:
            raise RuntimeError(f"OpenRouter {resp.status_code}: {resp.text[:2000]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    raise last_exc


def chat_json(model: str, messages: list[dict], temperature: float = 0.2, retries: int = 2) -> dict:
    last_err = None
    for attempt in range(retries + 1):
        raw = chat(model, messages, temperature=temperature, json_mode=True)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    last_err = e
                    continue
            else:
                last_err = e
                continue
        if parsed:  # reject empty {} — treat as a transient bad response, retry
            return parsed
        last_err = ValueError("model returned empty JSON object")
    raise RuntimeError(f"chat_json failed after {retries + 1} attempts: {last_err}")
