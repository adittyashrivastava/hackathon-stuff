"""LLM-as-judge: score a generated answer against a question's gold answer,
BEAM-category-aware (abstention and instruction_following get category-
specific judging criteria).
"""

from llm_client import JUDGE_MODEL, chat_json

JUDGE_SYSTEM_PROMPT = """You are grading one answer from a long-term-memory QA system, for a \
benchmark question in the "{category}" memory-ability category.

Grading criteria for this category: {criteria}

Score the SYSTEM ANSWER against the GOLD ANSWER / GOLD RATIONALE. Respond with JSON:
{{"verdict": "correct" | "partially_correct" | "incorrect" | "hallucinated",
  "score": 0.0 to 1.0,
  "rationale": "1-2 sentences justifying the verdict"{instruction_field}
}}

Use "hallucinated" specifically when the system answer confidently asserts a specific fact that \
is absent from or contradicted by the gold answer/rationale (not merely incomplete). Use \
"incorrect" for wrong-but-not-fabricated answers. Partial credit ("partially_correct") is for \
answers that get the core fact right but are incomplete, imprecise, or muddled."""

CATEGORY_CRITERIA = {
    "abstention": "The question may be UNANSWERABLE from the transcript (see 'answerable' flag). "
    "If unanswerable, a CORRECT system answer clearly hedges/declines rather than asserting a "
    "specific unsupported fact — grade confident fabrication here as 'hallucinated' even if it "
    "sounds plausible. If answerable, grade normally against the gold answer.",
    "contradiction_resolution": "Check whether the system answer correctly identifies/resolves the "
    "apparent contradiction per the gold rationale, rather than picking an outdated or wrong side.",
    "event_ordering": "Check whether the system answer gets the sequence of events right, not just "
    "the individual facts.",
    "information_extraction": "Check factual precision (names, dates, specific details) against the "
    "gold answer.",
    "knowledge_update": "The system answer must reflect the CURRENT/latest state of the fact, not a "
    "stale earlier version — grade a stale answer as 'incorrect' even if it was once true.",
    "preference_following": "Check whether the system answer correctly reflects the stated "
    "preference/habit.",
    "summarization": "Grade on coverage of the key facts in the gold rationale and absence of "
    "fabricated details, not exact wording.",
    "temporal_reasoning": "Check whether time relationships (durations, before/after, how long ago) "
    "are reasoned about correctly.",
    "instruction_following": "Grade BOTH content correctness against the gold answer AND whether the "
    "system answer complies with the stated instruction (see 'instruction' field) — note "
    "instruction_compliance explicitly.",
    "multi_session_reasoning": "Check whether the system answer actually combines evidence from both "
    "threads as required, not just one.",
}


def judge_answer(question: dict, generated_answer: str) -> dict:
    category = question["category"]
    criteria = CATEGORY_CRITERIA.get(category, "Grade for factual correctness against the gold answer.")

    instruction_field = ""
    if category == "instruction_following":
        instruction_field = ',\n  "instruction_compliance": true or false'

    system_prompt = JUDGE_SYSTEM_PROMPT.format(category=category, criteria=criteria, instruction_field=instruction_field)

    user_parts = [
        f"QUESTION: {question['question']}",
    ]
    if question.get("instruction"):
        user_parts.append(f"INSTRUCTION TO FOLLOW: {question['instruction']}")
    user_parts.append(f"ANSWERABLE FROM TRANSCRIPT: {question.get('answerable', True)}")
    user_parts.append(f"GOLD ANSWER: {question['gold_answer']}")
    user_parts.append(f"GOLD RATIONALE: {question.get('gold_rationale', '')}")
    user_parts.append(f"SYSTEM ANSWER: {generated_answer}")

    verdict = chat_json(
        JUDGE_MODEL,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ],
        temperature=0.0,
    )
    return verdict
