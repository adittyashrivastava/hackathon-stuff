# Follow-up investigation: Akash's test question

Akash (teammate) sent a manual test case over Slack: *"who among Arjun's closest
friends currently has children, and if so, how did their family situations evolve?
...Not sure if the model is the issue or the dataset at the moment."*

## What we found, step by step

**1. The product's answer was wrong.** Asked directly, it named Akhil and Rahul but
completely omitted Gopal — Arjun's actual best friend per ground truth — and even
contradicted the ground truth's own claim about Rahul (said "two children, Mira and
Kavya"; ground truth says `children_2026: 1` — Kavya is Rahul's *wife*, not a second
child).

**2. The data was fine.** Both real facts exist, correctly attributed to Gopal:
`grp_01595` ("Tara arrived yesterday. 3.1 kg. Meera and baby are good.", 2021-09-08)
and `grp_02261` ("Kabir arrived yesterday morning. Everyone good.", 2024-06-03). Both
score highest (1.075, 0.704) when queried with matching phrasing.

**3. It's a retrieval-ranking failure, not a missing-data one — confirmed by testing
increasingly targeted queries:**

| query | Gopal/Tara/Kabir chunks in top 10? |
|---|---|
| Akash's original broad question | No (checked up to top 50 — none) |
| "Gopal children" | No — all 10 results are one repeated line, "Easy to say without kids." (×10) |
| "Does Gopal have kids and how old are they now" (names Gopal explicitly) | Still no — same 10 duplicates |
| "Tara arrived yesterday Meera and baby are good" (matches the actual message) | Yes, rank 1 |

The friend_group dataset reuses scripted banter lines dozens of times ("Easy to say
without kids." ×10, "Tara thinks my office is where I go to use another laptop." ×20).
These near-duplicates score similarly to each other and, in aggregate, out-rank a
single sparse birth-announcement chunk — even alpha=1 (full dense, no BM25) didn't fix
it, since the dense embedding also rates "kids" mentions as relevant regardless of
whether they're actually informative.

**4. Fix: over-fetch + dedupe (see `comparison_dedup.md`).** Requesting a larger raw
pool and deduping near-identical content before truncating to top-K fixed the
motivating case — Gopal now correctly surfaces with Tara, though a second single-shot
query still missed Kabir specifically (his announcement is short and low-overlap with
the question's wording; it didn't crack the top 60 even after dedup).

**5. Tried agentic search as a further fix.** An LLM agent (DeepSeek V4 Flash) given a
`search_memories` tool, allowed to run several targeted follow-up searches instead of
one shot (`src/agentic_ask.py`), correctly found **both** Tara and Kabir with full
detail — it noticed Gopal had one known child from an early search and went looking
for whether there was a second, running 12 searches across 4 conversation turns.

## Quantified: single-shot (dedup) vs. agentic, on 10 hand-authored hard multi-hop questions

Built a small set of Akash-style questions (`src/akash_style_questions.py`) —
identify-an-entity-then-compute-a-derived-value questions (ages as of today, trip
counts within a date range, cross-thread connections) — with gold answers computed by
hand, not by an LLM, so they're not vulnerable to LLM date-arithmetic errors.

| qid | single-shot (dedup) | agentic |
|---|---|---|
| akash_000 (Akash's exact question) | partially_correct | partially_correct |
| akash_001 (Rahul wife-vs-child) | hallucinated | hallucinated |
| akash_002 (most recent child) | incorrect | **correct** |
| akash_003 (Arjun married yet?) | incorrect | incorrect |
| akash_004 (trips during Delhi period) | incorrect | incorrect |
| akash_005 (last trip before Kabir) | incorrect | incorrect |
| akash_006 (years to first trip) | incorrect | incorrect |
| akash_007 (Maya's calendar — see note below) | partially_correct | **correct** |
| akash_008 (near-miss abstention check) | correct | correct |
| akash_009 (Prof. Jatin timeline) | partially_correct | partially_correct |
| **MEAN** | **36.0%** | **40.0%** |

Agentic search wins by 4pts on this small, deliberately hard sample — a real but
modest improvement, not a silver bullet. It still fails the same way on the
trip-registry counting/filtering questions (akash_004-006) and the Prof. Jatin
timeline question: those require aggregating across *many* scattered facts (trip
dates spanning 17 years), which more searches alone don't fully solve. It also costs
much more: 2-9 searches (and 2-9x the LLM calls) per question vs. one for single-shot
— a real latency/cost trade-off, not a free upgrade.

**akash_001 (Rahul wife-vs-child) failed in both**, and for a reason no retrieval
strategy fixes: the raw ingested transcript never explicitly states "Kavya is Rahul's
wife" — that fact exists only in the ground-truth JSON, never in an actual message.
Both single-shot and agentic default to guessing "child" from an ambiguous line
("Kavya asked why she isn't in this group.") that could plausibly read either way.
This is a corpus/product-design limitation, not a bug: a fact that was never actually
placed into any ingested conversation cannot be retrieved from it.

## A caught eval-authoring bug (worth stating plainly)

The first version of akash_007 asked about Maya "blocking out" Aug 22-Sept 2, with a
gold answer assuming this connected to Arjun's Aug 23 wedding (a different thread) —
written without checking the actual mother_son messages. Both single-shot and agentic
independently surfaced a different, real, self-contained explanation instead: Arjun
had proposed an India trip for that exact window (`msg_00969`), then cancelled
(`msg_01014`: "product launch moved to Aug 25"), so Maya deleted the calendar hold
(`msg_01017`) — nothing to do with the wedding. Verified against the raw transcript,
corrected the question and gold answer, and re-judged both systems' *existing*
answers against the fix (no need to re-run retrieval — their original answers were
already right). This is exactly the kind of gold-answer error a synthetic eval can
introduce if written from a derived summary instead of the source messages — worth
flagging rather than quietly fixing, since it's a reminder to verify gold answers
against raw text, not just against a one-level-removed ground-truth summary.
