"""Akash-style multi-hop questions: hand-authored (not LLM-generated) so gold
answers are exact, not vulnerable to LLM arithmetic errors. Modeled directly on
Akash's own example question (see Slack DM): identify an entity satisfying some
criterion among a set, pull compound facts about it, then compute a DERIVED value
(a current age, a duration, a "was this before/after X") as of TODAY.

These deliberately stress the multi-hop retrieval failure mode found by hand:
identify-then-look-up-details-then-compute, which single-shot top-K retrieval
handles worse than it looks like it should (see EVAL_REPORT.md and the Gopal/
Tara/Kabir investigation in this session).

Output: questions/akash_style_questions.jsonl
"""

import json
import os

TODAY = "2026-08-19"  # matches the session date this batch was authored on

QUESTIONS = [
    # --- friend_group: the exact class of question Akash asked about ---
    {
        "question": "Who among Arjun's closest friends currently has children, and how did their family situations evolve? Give names, birth context, and current ages as of today.",
        "gold_answer": (
            "Four of Arjun's five closest friends currently have children (Arjun himself has none as of "
            "today, 2026-08-19 — he marries Ananya on 2026-08-23, four days from now, and has no children "
            "as of that date either). Gopal (Arjun's best friend) has two: Tara, born 2021-09-07 (age 4), "
            "and Kabir, born 2024-06-02 (age 2). Rahul has one child, Mira, born 2022-03-03 (age 4) — Rahul's "
            "wife is named Kavya, which is a common mix-up point since 'Kavya' also sounds like it could be "
            "a second child. Akhil has one child, Dev, born 2023-11-12 (age 2), with wife Nidhi. Priyanka has "
            "one child, Rohan, born 2020-08-18 per the life-events record (which would make him currently 6) "
            "— note the dataset's own canonical facts separately state Rohan is 'age_in_2026: 4', a genuine "
            "internal contradiction between two ground-truth fields, not a retrieval error."
        ),
        "gold_rationale": (
            "Sourced from arjun_core5_friend_group_ground_truth.json canonical_continuity + life_events. "
            "Ages computed from birth dates relative to 2026-08-19. The Rohan age discrepancy (6 by birth "
            "date vs. 4 per canonical_continuity.Priyanka.child.age_in_2026) is a real inconsistency in the "
            "ground truth file itself, included deliberately as a contradiction-resolution stress test."
        ),
        "evidence": ["canonical_continuity", "life_events"],
        "answerable": True,
        "difficulty": "hard",
        "category": "multi_hop_akash_style",
        "dataset": "friend_group",
        "collections_in_scope": ["friend_group"],
        "source": "akash_dm_example",
    },
    {
        "question": "Is Rahul's wife also one of his children, or is that a different person? Who are Rahul's actual children, and how old are they now?",
        "gold_answer": (
            "Kavya is Rahul's WIFE (they married 2016-01-16), not a child. Rahul has exactly one child: a "
            "daughter, Mira, born 2022-03-03, currently age 4."
        ),
        "gold_rationale": "life_events: 'Rahul marriage... Rahul marries Kavya' (2016-01-16) and 'Rahul child... Rahul and Kavya have a daughter, Mira' (2022-03-03). canonical_continuity.Rahul.children_2026 = 1.",
        "evidence": ["life_events"],
        "answerable": True,
        "difficulty": "hard",
        "category": "multi_hop_akash_style",
        "dataset": "friend_group",
        "collections_in_scope": ["friend_group"],
        "source": "akash_dm_followup",
    },
    {
        "question": "Whose child was born most recently among Arjun's friend group, and how old is that child today?",
        "gold_answer": "Gopal's son Kabir, born 2024-06-02, is the most recently born child in the group. He is currently 2 years old (as of 2026-08-19).",
        "gold_rationale": "Comparing all recorded child birth dates in life_events (2020-08-18 Rohan, 2021-09-07 Tara, 2022-03-03 Mira, 2023-11-12 Dev, 2024-06-02 Kabir), Kabir's is latest.",
        "evidence": ["life_events"],
        "answerable": True,
        "difficulty": "medium",
        "category": "multi_hop_akash_style",
        "dataset": "friend_group",
        "collections_in_scope": ["friend_group"],
        "source": "akash_style_generated",
    },
    {
        "question": "Is Arjun married yet as of today? What's the most current, correct status?",
        "gold_answer": "Not yet — as of today (2026-08-19), Arjun is not married. He marries Ananya on 2026-08-23, which is four days from now, and has 0 children as of that date.",
        "gold_rationale": "life_events: 'Arjun marriage... Arjun marries Ananya' dated 2026-08-23, which is after the reference date 2026-08-19. A correct answer must reason about today's date relative to that future event, not just state the wedding as an already-completed fact.",
        "evidence": ["life_events", "canonical_continuity.Arjun"],
        "answerable": True,
        "difficulty": "hard",
        "category": "multi_hop_akash_style",
        "dataset": "friend_group",
        "collections_in_scope": ["friend_group"],
        "source": "akash_style_generated",
    },
    # --- gopal: trip_registry multi-hop filtering/counting ---
    # CAVEAT (found during the performance/retrieval investigation, 2026-08-20):
    # verified against the raw transcript that trip DESTINATION NAMES (Mysuru, Goa,
    # Manali, etc.) never appear in any ingested gopal message — grep confirms zero
    # hits for every destination in trip_registry. Only trip NUMBERS are referenced
    # in-corpus (e.g. "Fine. Trip #11. If I freeze, tell Priyanka I always loved
    # her." — matching trip11's Dec-2016 Prague date, with "freeze" hinting at the
    # cold destination without naming it). The destination-name portion of these
    # three questions is therefore unanswerable from the ingested corpus by ANY
    # method — this is the same class of issue as the Kavya-wife-vs-child case: a
    # fact that exists only in ground-truth metadata, never written into a message.
    # Left in deliberately as a documented example rather than quietly fixed.
    {
        "question": "How many trips did Arjun and Gopal take together during the roughly four years Gopal lived in Delhi (2011-2015), and which destinations were those?",
        "gold_answer": "6 trips: Manali (2011-05), Jaipur (2012-12), Sri Lanka (2013-06), Bangkok (2014-04), Singapore (2014-12), and Bali (2015-07).",
        "gold_rationale": "trip_registry entries trip04 through trip09 fall within the 'gopal_delhi_period' (roughly 2011-2015) stated in canonical_ground_truth.",
        "evidence": ["trip_registry", "canonical_ground_truth.gopal_delhi_period"],
        "answerable": True,
        "difficulty": "hard",
        "category": "multi_hop_akash_style",
        "dataset": "gopal",
        "collections_in_scope": ["gopal"],
        "source": "akash_style_generated",
    },
    {
        "question": "What was the last trip Arjun and Gopal took together before Kabir was born, and how long before Kabir's birth was it?",
        "gold_answer": "Patagonia, Argentina in March 2023 (trip23), about 14-15 months before Kabir's birth on 2024-06-02.",
        "gold_rationale": "trip_registry's last entry before 2024-06 is trip23 (2023-03, Patagonia) — trip24 (2024-10, Japan) is after Kabir's birth. ~15 months between 2023-03 and 2024-06.",
        "evidence": ["trip_registry", "friend_group life_events (Kabir born 2024-06-02)"],
        "answerable": True,
        "difficulty": "hard",
        "category": "multi_hop_akash_style",
        "dataset": "gopal",
        "collections_in_scope": ["gopal", "friend_group"],
        "source": "akash_style_generated",
    },
    {
        "question": "How many years after Arjun and Gopal's friendship began did they take their first trip together, and how many total trips have they taken since?",
        "gold_answer": "Their friendship began in 2001 (first standard); their first trip together was in May 2008 — about 7 years later. They've taken 25 trips together in total as of the most recent one (Ladakh, 2025-07).",
        "gold_rationale": "canonical_ground_truth.friendship_start = '2001, first standard'; trip_registry's trip01 = 2008-05; trip_count = 25.",
        "evidence": ["canonical_ground_truth.friendship_start", "trip_registry", "canonical_ground_truth.trip_count"],
        "answerable": True,
        "difficulty": "medium",
        "category": "multi_hop_akash_style",
        "dataset": "gopal",
        "collections_in_scope": ["gopal"],
        "source": "akash_style_generated",
    },
    # --- cross-relationship (mother_son + friend_group) ---
    {
        # NOTE: an earlier version of this question ("Maya mentioned blocking out
        # Aug 22-Sept 2...") had a gold answer authored WITHOUT checking the actual
        # raw messages, assuming a cross-thread connection to Arjun's Aug 23 wedding.
        # Both the single-shot and agentic pipelines independently surfaced a
        # different, better-grounded, purely-within-thread explanation instead
        # (msg_00969, msg_01014, msg_01015, msg_01017) — verified against the raw
        # transcript and used to correct this gold answer below.
        "question": "Maya mentioned deleting August 22 to September 2 from her calendar. Why, and what changed?",
        "gold_answer": "Arjun had originally proposed visiting India Aug 22-Sept 2 (msg_00969, 2026-06-07: 'Trying to make India happen in August. Maybe Aug 22-Sept 2.'), and Maya was holding that window open. Two weeks later he cancelled: 'August is unlikely now. Product launch moved to Aug 25.' (msg_01014, 2026-06-21). Maya asked about September instead (msg_01015), then deleted the hold: 'Fine. I am deleting Aug 22-Sept 2.' (msg_01017, same day). This is fully explained within the mother/son thread alone — there is no verified connection to Arjun's Aug 23 wedding (a different thread); that would be an unsupported guess, not a grounded answer.",
        "gold_rationale": "Verified directly against mother_son_whatsapp_1year_synthetic.jsonl message content (msg_00969, msg_01014, msg_01015, msg_01017), not just the derived life_events summary.",
        "evidence": ["msg_00969", "msg_01014", "msg_01015", "msg_01017"],
        "answerable": True,
        "difficulty": "medium",
        "category": "multi_hop_akash_style",
        "dataset": "mother_son",
        "collections_in_scope": ["mother_son"],
        "source": "akash_style_generated_corrected",
    },
    {
        "question": "Arjun asked his mother to teach him to make rajma properly. Around the same time, was anything happening in his friend group that might explain new motivation to cook well?",
        "gold_answer": "Not clearly connected based on available evidence — Arjun's rajma request was 2026-05-01, and there's no specific friend-group event recorded right around that date that obviously explains it. This should be answered with appropriate uncertainty, not a confident invented connection.",
        "gold_rationale": "This is a deliberate near-miss: it's structurally identical to the Maya-calendar question (cross-thread, 'does X connect to Y'), but here no real connection exists in the ground truth — a good answer must recognize that and not force one, i.e. an abstention-style check embedded in a multi-hop question.",
        "evidence": ["mother_son knowledge_updates (rajma, 2026-05)"],
        "answerable": False,
        "difficulty": "hard",
        "category": "multi_hop_akash_style",
        "dataset": "mother_son+friend_group",
        "collections_in_scope": ["mother_son", "friend_group"],
        "source": "akash_style_generated",
    },
    # --- prof_jatin ---
    {
        "question": "How many years after Prof. Jatin first recruited Arjun did Arjun get invited to speak to Jatin's systems class, and what had happened to Karthik in between?",
        "gold_answer": "About 10-11 years: Jatin first recruited Arjun in January 2012, and Arjun was invited to speak to the class in September 2022. In between, Karthik (mentioned in the same thread) got a job/academic offer around September 2021, about a year before that.",
        "gold_rationale": "prof_jatin life_events: 'Prof. Jatin invites Arjun to discuss a project' (2012-01-10) and 'Prof. Jatin invites Arjun to speak to his systems class' (2022-09-17); 'Karthik gets an offer' (2021-09-02) falls in between.",
        "evidence": ["prof_jatin life_events (derived)"],
        "answerable": True,
        "difficulty": "hard",
        "category": "multi_hop_akash_style",
        "dataset": "prof_jatin",
        "collections_in_scope": ["prof_jatin"],
        "source": "akash_style_generated",
    },
]


def main():
    out_path = os.path.join(os.path.dirname(__file__), "..", "questions", "akash_style_questions.jsonl")
    with open(out_path, "w") as f:
        for i, q in enumerate(QUESTIONS):
            q["qid"] = f"akash_{i:03d}"
            f.write(json.dumps(q) + "\n")
    print(f"Wrote {len(QUESTIONS)} hand-authored Akash-style questions to {out_path}")


if __name__ == "__main__":
    main()
