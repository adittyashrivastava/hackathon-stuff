# Arjun's Memory — Full Evaluation Report

This is the consolidated write-up of the eval work behind this Hack Hydra Track 3
submission: what was built, how the eval was run, the full results, and a
graph-context ablation. See `README.md` for setup/architecture and
`results/report.md` / `results/comparison_graph_context.md` for the raw generated
tables this document draws from.

## 1. What this evaluates

A product ("Arjun's Memory") that ingests four synthetic relationship-thread
conversation histories about one person, Arjun, into HydraDB as memories, and answers
natural-language questions about him by retrieving from HydraDB and synthesizing a
grounded answer with an LLM. To measure how well it actually works, we built a
BEAM-inspired synthetic eval: generate probing questions across BEAM's ten memory-
ability categories, run them through the product, and score each answer against
ground truth with an LLM-as-judge — the same shape as HydraDB's own published
benchmark reports.

**Data**: four datasets, one central character (Arjun), spanning 2012–2026 across
~8,500 messages — a 5-person school-friend group, a 1:1 best-friend thread (Gopal), a
mother/son thread, and a mentor/mentee thread (Prof. Jatin). Two shipped with
hand-authored ground truth; two didn't and had ground truth derived from the
transcript by an LLM (clearly labeled `ground_truth_is_authored: false` in
`src/datasets.py`). Full details in `README.md`.

**Question generation**: 242 questions total — 126 from a standard pass (3 per
dataset × category) plus 116 from a harder, explicitly adversarial follow-up pass
(near-miss distractors, multi-hop reasoning, subtle contradiction/knowledge-update
traps), across all 10 BEAM categories: abstention, contradiction resolution, event
ordering, information extraction, instruction following, knowledge update,
multi-session reasoning, preference following, summarization, temporal reasoning.
Each question carries a gold answer, a rationale, supporting evidence (message IDs or
ground-truth fields), and an `answerable` flag. See `src/generate_questions.py`.

**Answering**: `src/ask.py` — HydraDB `/query` (`type=memory`, `query_by=hybrid`,
`mode=thinking`) over-fetches (`max_results*4`) and dedupes near-identical chunks
down to the top 15 across the relevant collection(s) — see §5 for why; an LLM
(DeepSeek V4 Flash via OpenRouter) synthesizes the final answer strictly from those
chunks, instructed to hedge/abstain rather than guess when evidence is thin.

**Judging**: `src/judge.py` — an LLM-as-judge scores each answer against the gold
answer/rationale with category-specific grading criteria (e.g. abstention questions
are graded on whether the system correctly hedges when unanswerable; knowledge_update
questions are graded on whether the answer reflects the *current* state of a fact, not
a stale one). Verdicts: `correct`, `partially_correct`, `incorrect`, `hallucinated`.

## 2. Headline result

**242 questions, 0 pipeline errors, 52.31% overall mean score** (current, with the
over-fetch+dedup retrieval fix from §5 applied — see `results/comparison_dedup.md`
for the before/after; the fix is a wash in aggregate, ±0.2pt, but shifts individual
categories meaningfully in both directions).

| category | N | mean score | correct | partial | incorrect | hallucinated |
|---|---|---|---|---|---|---|
| abstention | 24 | **80.0%** | 18 | 2 | 1 | 3 |
| contradiction_resolution | 24 | 67.5% | 15 | 2 | 7 | 0 |
| instruction_following | 20 | 59.0% | 10 | 3 | 6 | 1 |
| information_extraction | 25 | 56.8% | 12 | 3 | 9 | 1 |
| summarization | 24 | 56.7% | 10 | 3 | 7 | 3 |
| multi_session_reasoning | 30 | 52.3% | 11 | 6 | 11 | 2 |
| preference_following | 24 | 47.9% | 11 | 1 | 10 | 2 |
| temporal_reasoning | 24 | 37.9% | 8 | 0 | 13 | 3 |
| event_ordering | 23 | 35.2% | 7 | 1 | 11 | 4 |
| knowledge_update | 24 | **30.0%** | 5 | 1 | 17 | 0 |

Zero categories landed at a flat 0% or 100% — the spread itself is a signal the harness
is measuring something real, not degenerately broken.

**Reading it**: the product is genuinely good at knowing what it *doesn't* know —
abstention is by far the strongest category, meaning when evidence is thin, the system
tends to hedge rather than fabricate. It's weakest at precision tasks that require
pinpointing exactly *which* version of a fact is current (`knowledge_update`) or the
exact sequence/timing of events (`event_ordering`, `temporal_reasoning`) across
thousands of noisy, repetitive chat messages — a believable limitation of a
retrieve-then-synthesize architecture over long, informal chat history, not a broken
eval.

By dataset: `mother_son` (68%) outperforms `gopal` (53%), `prof_jatin` (52%), and
`friend_group` (35%) — plausible, since the 5-person group chat and the 10-year 1:1
thread are noisier/more repetitive (lots of near-duplicate banter, per the dataset
stats in `data/*_ground_truth*.json`) than the other two.

## 3. Example cases (judge output, verbatim from `results/eval_results.jsonl`)

**Correct** — clean negative-fact assertion with quote-level evidence:
> Q: *"Did Arjun ever express that he prefers building easy things to avoid risk?"*
> System: *"No, Arjun never expressed that... In fact, he stated 'I want to build difficult things' [pj_00737]..."*
> Judge: *"...correctly identifies that Arjun never expressed a preference for building easy things, and supports this with direct quotes..."*

**Hallucinated** — confident fabrication, not just wrongness:
> Q: *"Summarize Arjun's feedback on Gopal's parenting style..."*
> System: *"...the only direct comment from Gopal about parenting is the isolated phrase 'She's good. I'm terrified' [ag_01735]..."*
> Judge: *"...fabricates a specific quote... neither of which appear in any provided data."*

**Incorrect** — real retrieved facts, but misses the actual point of the question:
> Q: *"Summarize the mentions of Priyanka's son Rohan in the group's history."*
> Gold: notes a 2016 message about Rohan predates his 2020 birth date by ~4 years — a dataset inconsistency the question is testing for.
> Judge: *"...lists mentions of Rohan but fails to note the critical inconsistency between the 2020 birth date and the pre-2020 transcript messages..."*

**Partially correct** — right surface answer, wrong on the specific ability being tested:
> Q: *"Did Maya stop praying for Arjun and instead allocate her prayer 'budget' solely to Nisha?"*
> Judge: *"...correctly identifies there is no direct evidence... but fails to abstain as the gold answer requires. Instead it concludes evidence doesn't support X"* rather than hedging outright.

## 4. Ablation: does adding HydraDB's `graph_context` to the answer prompt help?

HydraDB's `/query` always computes `graph_context` in `mode=thinking` — a set of
entity-relationship triples (`source --predicate--> target`, with a supporting
sentence) extracted across the ingested conversations. We re-ran all 242 questions
with those triples included in the answer-synthesis prompt alongside the retrieved
chunks, to see whether giving the answering LLM the graph as well as the raw text
helps. (Measured before the over-fetch+dedup retrieval fix in §5 — against the
52.52% pre-dedup baseline, not the current 52.31%. Not re-run post-fix; the two
changes are orthogonal — one is what's *in* the prompt, the other is *which chunks*
reach the prompt — but treat the exact deltas below as relative to that earlier
baseline specifically.)

**Result: a net regression, -4.1pts overall (52.52% → 48.41%).**

| category | chunks-only | chunks+graph | delta |
|---|---|---|---|
| information_extraction | 61.2% | 48.0% | **-13.2** |
| multi_session_reasoning | 54.0% | 43.7% | **-10.3** |
| abstention | 89.6% | 79.6% | **-10.0** |
| temporal_reasoning | 37.9% | 31.7% | -6.3 |
| preference_following | 61.3% | 57.1% | -4.2 |
| knowledge_update | 32.9% | 32.5% | -0.4 |
| contradiction_resolution | 57.1% | 57.5% | +0.4 |
| event_ordering | 40.4% | 43.0% | +2.6 |
| instruction_following | 42.5% | 42.5% | +0.0 |
| summarization | 45.4% | 48.5% | +3.1 |
| **OVERALL** | **52.52%** | **48.41%** | **-4.11** |

Three findings stand out:

- **`information_extraction` (-13.2pts, the single biggest drop)**: graph triples are
  a coarse compression (entity + relation, no fine detail) — they crowd out rather
  than reinforce the precise dates/names this category actually needs.
- **`abstention` (-10.0pts)**: a confident-looking `source --predicate--> target` edge
  seems to give the model license to assert a fact it would otherwise have hedged on,
  even when the underlying textual evidence is thin. The graph is itself a lossy,
  already-confident-sounding summary — feeding it in erodes the model's willingness to
  say "I don't know."
- **`multi_session_reasoning` (-10.3pts)**: this is the category the graph should, in
  theory, help most — cross-relationship reasoning is exactly what a knowledge graph
  is supposed to be good at. It didn't. The extra relation text appears to distract
  from, rather than surface, the actual cross-collection evidence in the chunks.

This is a real, reproducible finding about *this specific integration* (raw triples
dumped into a synthesis prompt) — not a verdict on HydraDB's graph feature in general.
A different integration — e.g. using the graph to re-rank or expand which chunks get
retrieved, rather than handing the LLM raw triples to reason over directly — might
behave differently; we didn't test that here.

**Consequence**: `src/ask.py` defaults to chunks-only (`use_graph_context=False`),
matching the better-performing configuration. `--use-graph-context` opts back in to
reproduce the comparison. Full raw results for both runs are in
`results/eval_results_chunks_only.jsonl` and `results/eval_results_with_graph_context.jsonl`.

## 5. Follow-up: a teammate's manual test case, and what it led to

A teammate (Akash) sent a manual test question over Slack that the eval's automated
question set hadn't covered: "who among Arjun's closest friends currently has
children, and how did their family situations evolve?" — with a note that he wasn't
sure whether a bad answer would be the model's fault or the dataset's.

Investigating this by hand surfaced a real, previously-undetected retrieval bug: the
friend_group dataset reuses scripted banter lines dozens of times (e.g. "Easy to say
without kids." appears 10+ times), and these near-duplicates crowded out a single,
sparse, highly-informative chunk (Gopal's actual birth announcements for his two kids)
from ever reaching the top-K — confirmed by testing progressively more targeted
queries, up to `max_results=120`, none of which surfaced it. Fixed with client-side
over-fetch + dedup in `ask.py`. Also built and tested an agentic alternative
(`src/agentic_ask.py`) — an LLM given an iterative `search_memories` tool instead of
one shot — which fixed the motivating case outright and beat the fixed single-shot
pipeline by 4pts (36% → 40%) on a 10-question hard multi-hop set, at the cost of
2-9x more searches/LLM calls per question.

Along the way, an eval-authoring bug of our own was caught and fixed: a gold answer
written from a derived summary instead of the raw messages, corrected after both
systems independently surfaced the real, better-grounded explanation. Full write-up,
including the query-by-query retrieval diagnosis and the dedup ablation's per-category
trade-offs: **[`results/comparison_akash_style.md`](results/comparison_akash_style.md)**
and **[`results/comparison_dedup.md`](results/comparison_dedup.md)**.

## 6. Limitations

- `mother_son` and `prof_jatin` ground truth is LLM-derived from a transcript sample,
  not hand-authored — their eval numbers are inherently noisier than
  `friend_group`/`gopal`, which shipped with hand-authored ground truth.
- The eval questions were generated by the same tier of model (DeepSeek V4 Flash) used
  to answer and judge them — not a different, stronger model — which is a standard
  caveat for any self-generated LLM eval.
- 242 questions is a meaningful sample but still modest per category (20-30 each);
  category-level percentages should be read as directional, not precise to the point.
- The product's "product" surface is a CLI (`src/ask.py`), not a polished UI — this
  was a hackathon-timeline MVP.
