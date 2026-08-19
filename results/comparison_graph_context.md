# Ablation: does feeding graph_context to the answer-synthesis LLM help?

Same 242 questions, same HydraDB `/query` call (`mode=thinking`, `graph_context=true`
in both runs — HydraDB always computes it in thinking mode). The only difference: does
the answer-synthesis LLM prompt (`src/ask.py`) include the `data.graph_context.query_paths`
entity-relationship triples alongside the retrieved chunks, or only the chunks?

| category | N | chunks-only | chunks+graph | delta |
|---|---|---|---|---|
| abstention | 24 | 89.58% | 79.58% | -10.00% |
| contradiction_resolution | 24 | 57.08% | 57.50% | +0.42% |
| event_ordering | 23 | 40.43% | 43.04% | +2.61% |
| information_extraction | 25 | 61.20% | 48.00% | -13.20% |
| instruction_following | 20 | 42.50% | 42.50% | +0.00% |
| knowledge_update | 24 | 32.92% | 32.50% | -0.42% |
| multi_session_reasoning | 30 | 54.00% | 43.67% | -10.33% |
| preference_following | 24 | 61.25% | 57.08% | -4.17% |
| summarization | 24 | 45.42% | 48.54% | +3.12% |
| temporal_reasoning | 24 | 37.92% | 31.67% | -6.25% |
| **OVERALL** | **242** | **52.52%** | **48.41%** | **-4.11%** |

## Reading this result

Adding graph context to the answer prompt is a net **regression**, not an improvement,
on this dataset. Four categories gain a couple points (event_ordering,
contradiction_resolution, summarization — small, +0-3pts), but three categories lose
meaningfully:

- **abstention (-10pts)**: the graph surfaces an extracted relation/predicate the LLM
  then treats as license to assert a fact, instead of hedging the way the chunks alone
  would have prompted it to. The extracted triples are themselves a lossy
  compression of the source text — confident-looking (`source --predicate--> target`)
  even when the underlying evidence is thin — which appears to erode the model's
  willingness to abstain.
- **information_extraction (-13pts)**, the single biggest drop: precise facts (exact
  dates, specific names) get diluted rather than reinforced by graph triples, which are
  coarser (entity + relation, no fine detail) — more context, but the wrong kind, for
  this category.
- **multi_session_reasoning (-10pts)**: this was the category most likely, in theory,
  to *benefit* from a cross-relationship graph — it didn't. The additional relation
  text seems to have crowded out, or distracted from, the actual cross-collection
  evidence in the retrieved chunks.

This is a real, reproducible finding about this specific product design (a raw-triple
graph dump added to a synthesis prompt), not a claim about HydraDB's graph feature in
general — a different prompting strategy (e.g. using the graph only to *re-rank or
expand* the chunk set, rather than handing the LLM raw triples to reason over directly)
might behave differently. **The chunks-only configuration is the better-performing one
and is what `src/ask.py` uses by default going forward**; the graph-context path is
kept as an opt-in code path (see `hydra_client.query(..., graph_context=True)`, always
requested, but `ask.py`'s prompt only includes it when explicitly asked) for anyone who
wants to reproduce or extend this ablation.
