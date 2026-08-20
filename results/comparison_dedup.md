# Ablation: does deduping near-identical chunks before answering help?

Triggered by a concrete bug found manually (see `comparison_akash_style.md`): a
single-shot query for "who among Arjun's closest friends has children" never
surfaced Gopal's two kids at all, even at `max_results=120`, because the friend_group
dataset reuses scripted banter lines dozens of times (e.g. "Easy to say without
kids." appears 10+ times) — near-duplicate chunks scoring similarly crowd out a
single, sparse, highly informative chunk (a one-off birth announcement).

Fix implemented in `ask.py` (`dedupe_chunks`, `overfetch_factor=4` by default):
request `max_results * 4` raw chunks from HydraDB, dedupe by normalized content, take
the top `max_results` unique ones before answering.

**Net effect on the full 242-question suite: a wash (52.52% → 52.31%, -0.2pt), but not
uniformly** — 37 questions improved, 39 got worse, 166 unchanged:

| category | before | after | delta |
|---|---|---|---|
| instruction_following | 42.5% | 59.0% | **+16.5** |
| summarization | 45.4% | 56.7% | **+11.2** |
| contradiction_resolution | 57.1% | 67.5% | **+10.4** |
| multi_session_reasoning | 54.0% | 52.3% | -1.7 |
| temporal_reasoning | 37.9% | 37.9% | 0.0 |
| knowledge_update | 32.9% | 30.0% | -2.9 |
| information_extraction | 61.2% | 56.8% | -4.4 |
| event_ordering | 40.4% | 35.2% | -5.2 |
| abstention | 89.6% | 80.0% | -9.6 |
| preference_following | 61.3% | 47.9% | **-13.3** |
| **OVERALL** | **52.5%** | **52.3%** | **-0.2** |

**Reading it**: dedup trades repetition for diversity, and that's a real trade-off,
not a strict improvement:

- **Helps** categories that benefit from seeing more *distinct* facts per answer —
  `instruction_following` and `summarization` need breadth to cover a question fully;
  `contradiction_resolution` needs to see two genuinely different statements, not five
  copies of one side.
- **Hurts** categories where *repetition itself is signal* — a preference stated 10
  times across the transcript is stronger evidence of a stable preference than the
  same preference stated once (`preference_following`, -13.3pts, the biggest single
  loss). Similarly, `abstention` (-9.6pts) sometimes relied on a chunk set dominated by
  irrelevant repeated banter to correctly trigger "no real evidence here" — deduping
  clears space for a real-but-insufficient fact to appear instead, which can tempt a
  confident (wrong) answer instead of a hedge.

**Decision**: dedup stays on by default in `ask.py`, because it demonstrably fixes the
concrete bug that motivated it (see `comparison_akash_style.md`) and the aggregate
effect is neutral rather than a regression — but this is a judgment call, not an
unambiguous win, and the per-category trade-off above should inform anyone tuning
`overfetch_factor` or building on this further.
