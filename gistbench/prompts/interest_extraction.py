"""Prompt construction for interest extraction (Step 1).

Key design: closed-loop verification — the exact evidence thresholds used
to verify interests (IG) are encoded into the generation prompt.
"""

from __future__ import annotations

from gistbench.schema import DatasetConfig


def build_extraction_prompt(config: DatasetConfig) -> str:
    """Build the system prompt for interest extraction.

    The prompt adapts based on which signal types the dataset has,
    encoding the exact IG verification thresholds as instructions.
    """
    instructions = [
        f"Identify common interests and themes between {config.object_name}s. "
        f'Avoid overly generic interests like "Food", "Sports".\n',
    ]

    # Positive evidence requirements (matches IG verification predicates)
    grounding = "Each interest must be grounded in one of the following conditions:"
    if config.has_implicit_positive:
        grounding += "\n    * at least 3 implicit_positive engagements"
    if config.has_explicit_positive:
        grounding += "\n    * at least 2 explicit_positive engagements"
    if config.has_implicit_positive and config.has_explicit_positive:
        grounding += (
            "\n    * at least 1 explicit_positive engagement "
            "and 2 implicit_positive engagements"
        )
    instructions.append(grounding + "\n")

    # Negative constraints
    if config.has_implicit_negative:
        instructions.append(
            "Each interest must not have more than 3 implicit_negative engagements.\n"
        )
    if config.has_explicit_negative:
        instructions.append(
            "Each interest must not have more than 2 explicit_negative engagements.\n"
        )

    instructions.append(
        f"Each {config.object_name} can be associated with exactly two or less interests.\n"
    )
    instructions.append(
        "For each interest return:\n"
        "    - interest (specific and descriptive, between 2 and 5 words)\n"
        "    - item_ids (item ids from the engagement_history provided)\n"
        "    - evidence_excerpt (<=50 words from the interest description and entities used)\n"
        "    - brief_rationale (<=50 words)\n"
    )

    instruction_str = ""
    for i, instruction in enumerate(instructions):
        instruction_str += f"{i + 1}. {instruction}"

    return (
        f"Task: A user engaged with the following items ({config.object_name}) "
        f"on an online platform, you should find commonalities between the groups of "
        f"items by constructing a knowledge graph to identify strong interests. "
        f"Use the instructions to guide you in how to identify an interest.\n\n"
        f"<engagement_history>\n{{engagement_history}}\n</engagement_history>\n\n"
        f"Instructions:\n{instruction_str}\n"
        f"Output only valid JSON matching this structure:\n"
        f"```json\n"
        f'[ {{ "interest":"...", "item_ids": ["...", ...], '
        f'"evidence_excerpt":"...", "brief_rationale":"..." }}, ...]\n'
        f"```\n\n"
        f"Rules:\n"
        f"- If no clear evidence, do not include in your output.\n"
        f"- Prioritize strong interests as opposed to mild or weak. "
        f"Several mild interests can create an interest.\n"
        f"- Output only JSON, no extra text.\n"
    )


def format_engagement_history(
    engagements: list[dict[str, str]],
) -> str:
    """Format a list of engagements into numbered text for the prompt.

    Each engagement is formatted as: "[object_id] <interaction_type> - <object_text>"
    """
    lines = []
    for eng in engagements:
        obj_id = eng.get("object_id", "")
        interaction = eng.get("interaction_type", "unknown")
        text = eng.get("object_text", "")
        lines.append(f"[{obj_id}] {interaction} - {text}")
    return "\n".join(lines)


def build_extraction_messages(
    config: DatasetConfig,
    engagements: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Build chat messages for interest extraction.

    Returns chat format: [{"role": "user", ...}]
    """
    system_prompt = build_extraction_prompt(config)
    assert system_prompt.count("{engagement_history}") == 1, (
        "Prompt must contain exactly one {engagement_history} placeholder"
    )
    engagement_text = format_engagement_history(engagements)
    user_prompt = system_prompt.replace("{engagement_history}", engagement_text)

    return [
        {"role": "user", "content": user_prompt},
    ]
