"""genai_chat_api

A small authenticated, streaming chat API built on FastAPI.

Feature checklist (per the day's task):
- auth              -> src/genai_chat_api/auth.py
- session creation   -> POST /sessions          (src/genai_chat_api/main.py)
- message streaming   -> POST /sessions/{id}/messages  (Server-Sent Events)
- history persistence -> SQLite via SQLModel     (src/genai_chat_api/db.py)
- system prompt selection -> src/genai_chat_api/system_prompts.py
"""

__version__ = "0.1.0"
