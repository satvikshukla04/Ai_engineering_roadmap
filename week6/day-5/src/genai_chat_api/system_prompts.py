"""A small, hardcoded catalog of selectable system prompts.

In a bigger app these would live in the DB and be editable by admins, but
for this project a fixed dict is enough to demonstrate "system prompt
selection" as a feature: the client picks an id when creating a session,
and that id decides how the assistant behaves for every message in it.
"""

from __future__ import annotations

from genai_chat_api.models import SystemPromptRead

SYSTEM_PROMPTS: dict[str, str] = {
    "default": "You are a helpful, concise assistant.",
    "coding_tutor": (
        "You are a patient coding tutor. Explain concepts step by step, "
        "and prefer small examples over long theory."
    ),
    "pirate": "You are a pirate. Respond to everything in pirate speak, arr!",
}

DEFAULT_SYSTEM_PROMPT_ID = "default"


def list_system_prompts() -> list[SystemPromptRead]:
    return [
        SystemPromptRead(id=prompt_id, name=prompt_id.replace("_", " ").title(), prompt=prompt)
        for prompt_id, prompt in SYSTEM_PROMPTS.items()
    ]


def get_system_prompt(prompt_id: str) -> str:
    return SYSTEM_PROMPTS.get(prompt_id, SYSTEM_PROMPTS[DEFAULT_SYSTEM_PROMPT_ID])
