# Arjun's Memory

**Hack Hydra 2026 — Track 3: Memory and Context Retrieval**

A long-term memory product for a synthetic person, "Arjun," built on **HydraDB**. It
ingests four separate relationship-thread conversation histories (spanning 2012–2026)
as HydraDB *memories*, answers natural-language questions about Arjun by retrieving
from HydraDB and synthesizing a grounded answer, and ships its own BEAM-inspired eval
harness that generates synthetic probing questions, runs them against the product, and
judges the answers against ground truth — the same shape as HydraDB's own published
benchmark reports (see `benchmarks.hydradb.com`).

## The problem

Track 3 asks for an agent memory layer with cross-session continuity that synthesizes
facts chronologically without hallucinating. We used four synthetic datasets about one
person, Arjun, each a single long-running WhatsApp/group thread:

| Dataset | Relationship | Messages | Span |
|---|---|---|---|
| `friend_group` | 5-person school-friend group | 2,800 | 2015–2026 |
| `gopal` | 1:1 best friend | 2,500 | 2016–2023 |
| `mother_son` | Arjun & his mother | 1,235 | 2025–2026 |
| `prof_jatin` | Mentor/mentee (research advisor) | 2,000 | 2012–2026 |

Two datasets (`friend_group`, `gopal`) shipped with hand-authored ground-truth files.
The other two didn't — for those, `src/derive_ground_truth.py` samples the transcript
evenly across its full time span and asks an LLM to extract a structured canonical-facts
JSON in the same shape, explicitly labeled as LLM-derived (not authored) in
`src/datasets.py` and in the output file's `_meta` block. This is a real caveat: the
`mother_son`/`prof_jatin` eval numbers are only as good as that derivation.

## Architecture

```
data/*.jsonl, *_ground_truth*.json     four datasets + ground truth (authored or derived)
        |
        v
src/ingest.py            -->  HydraDB /context/ingest (type=memory)
                               one database "arjun", one collection per relationship
        |
        v
src/generate_questions.py -->  synthetic BEAM-category questions, grounded in ground
                                truth + a sampled transcript, via an LLM (DeepSeek V4
                                Flash over OpenRouter)
        |
        v
src/ask.py                -->  the product: HydraDB /query (type=memory, hybrid,
                                graph_context) --> LLM synthesizes the final answer
                                from retrieved chunks only
        |
        v
src/judge.py + run_eval.py -->  LLM-as-judge scores each answer against gold,
                                 category-aware criteria --> results/eval_results.jsonl
        |
        v
src/report.py              -->  results/report.md, per-category + per-dataset accuracy
```

## How HydraDB is used (not just in the README)

- **Ingestion**: every message from all four threads is ingested as a HydraDB *memory*
  item (`POST /context/ingest`, `type=memory`) with metadata (sender, timestamp,
  conversation/thread id, topic tags), scoped into one of four **collections** under a
  single **database** (`arjun`) — so a query can target one relationship or blend
  several.
- **Retrieval**: the product's only source of truth is `POST /query`
  (`type=memory`, `query_by=hybrid`, `mode=thinking`, `graph_context=true`) — the LLM
  answer-synthesis step is explicitly instructed to use *only* the chunks HydraDB
  retrieves, never outside knowledge.
- **Cross-collection reasoning**: multi-relationship questions pass multiple
  `collections` in one query, letting HydraDB combine evidence across, e.g., what
  Arjun told his mother vs. what he told Gopal about the same event.

## BEAM-inspired eval

Questions are generated across all ten BEAM memory-ability categories
(github.com/mohammadtavakoli78/BEAM): abstention, contradiction resolution, event
ordering, information extraction, instruction following, knowledge update,
multi-session reasoning, preference following, summarization, temporal reasoning —
grounded in each dataset's ground truth + a spread sample of its transcript, and (for
`multi_session_reasoning`) across pairs of datasets. This is intentionally a richer,
larger question set than the bare ground-truth files provide.

See `results/report.md` for the full scored breakdown. Current run: **242 questions
(126 standard + 116 adversarial/harder — see the `round` field), 52.31% overall mean
score**. Strongest category: `abstention` (80% — the product is good at hedging
instead of fabricating when evidence is thin). Weakest: `knowledge_update` (30%),
`event_ordering` (35%), and `temporal_reasoning` (38%) — precisely locating *which*
version of a fact is current, or the exact sequence/timing of events, across thousands
of noisy chat messages is genuinely hard for a retrieval-then-synthesize pipeline. This
spread is a real, reproducible finding, not a broken harness.

**Ablation — does feeding HydraDB's `graph_context` triples to the answer LLM help?**
No: it measured a **-4.1pt regression** (52.52% → 48.41%, pre-dedup baseline), worst on
`information_extraction` (-13.2pts) and `multi_session_reasoning` (-10.3pts) — the raw
entity-relationship triples appear to dilute precise fact-recall and erode abstention
more than they add useful cross-reference signal. Full breakdown in
`results/comparison_graph_context.md`. `src/ask.py` defaults to chunks-only
accordingly (`--use-graph-context` opts back in for reproducing the comparison).

**Ablation — does over-fetching + deduping near-identical chunks help?** A teammate's
manual test case (a friend-group question about children) turned up a real retrieval
bug: this dataset reuses scripted banter lines dozens of times, and the near-duplicates
crowded out a single, sparse, critical fact from ever reaching the top-K — confirmed
down to `max_results=120`. Fixed by over-fetching and deduping before truncating
(`ask.py`, on by default). Net effect on the full suite: a wash (**52.52% → 52.31%**),
but not uniformly — helps categories needing information *diversity*
(`instruction_following` +16.5, `summarization` +11.2, `contradiction_resolution`
+10.4) and hurts categories where *repetition itself is signal*
(`preference_following` -13.3, `abstention` -9.6). Also tried an **agentic** search
alternative (`src/agentic_ask.py` — an LLM with an iterative search tool instead of one
shot), which fixed the motivating bug outright and beat single-shot by 4pts (36% → 40%)
on a 10-question hard multi-hop set, at 2-9x the searches/LLM calls per question. Full
write-up: `results/comparison_dedup.md` and `results/comparison_akash_style.md`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your own HydraDB + OpenRouter API keys
```

## Running the pipeline

```bash
# 1. derive ground truth for the two datasets that didn't ship one
python src/derive_ground_truth.py

# 2. ingest all four datasets into HydraDB
python src/ingest.py --dataset all

# 3. generate the synthetic BEAM-style question set
python src/generate_questions.py

# 4. run the eval (ask + judge every question)
python src/run_eval.py

# 5. render the report
python src/report.py

# ask a one-off question yourself (the product)
python src/ask.py "How has Arjun and Gopal's friendship changed since Gopal got married?"

# or use the agentic variant (iterative search instead of one shot — slower, sometimes better)
python src/agentic_ask.py "Who among Arjun's closest friends currently has children?"
```

`run_eval.py` also takes `--questions`, `--results`, and `--backend` (`single_shot` |
`agentic`) to run any question set through either answer pipeline — used to produce
the comparisons in `results/comparison_dedup.md` and `results/comparison_akash_style.md`.

See **[`EVAL_REPORT.md`](EVAL_REPORT.md)** for the full consolidated write-up: results,
example judged cases, the graph-context ablation, and the follow-up investigation into
a teammate's manual test case (dedup fix + agentic search comparison).

## Attribution

- **[HydraDB](https://hydradb.com)** — the memory/retrieval substrate this entire
  project is built on.
- **[BEAM](https://github.com/mohammadtavakoli78/BEAM)** — long-term memory benchmark;
  its ten memory-ability categories structure our synthetic eval questions.
- **[LongMemEval](https://github.com/xiaowu0162/LongMemEval)** — cited by Hack Hydra
  Track 3 as a reference benchmark for this problem space.
- **[DeepSeek V4 Flash](https://openrouter.ai/deepseek/deepseek-v4-flash)** via
  **[OpenRouter](https://openrouter.ai)** — powers question generation, answer
  synthesis, and LLM-as-judge scoring.
- The four conversation datasets (`data/*.jsonl`) and the two hand-authored
  ground-truth files are synthetic, provided for this hackathon.

Licensed under MIT — see `LICENSE`.

## Limitations

- `mother_son` and `prof_jatin` ground truth is LLM-derived from a transcript sample,
  not hand-authored — treat their eval numbers as noisier than `friend_group`/`gopal`.
- This is a hackathon-timeline MVP: the "product" surface is a CLI (`src/ask.py`), not
  a polished UI.
