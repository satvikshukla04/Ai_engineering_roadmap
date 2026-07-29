"""
chat_session.py

Daily task: Build a ChatSession class with:
  - Sliding window memory (last 10 turns)
  - Token-aware truncation
  - LLM-based summarization of old turns
  - JSON persistence to disk
  - Verified to work correctly across session restarts
"""

import json
import os
from google import genai
from google.genai import types

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class ChatSession:
    """
    Manages conversational memory with a sliding window, token-aware
    truncation, LLM-based summarization, and JSON persistence to disk.
    """

    def __init__(
        self,
        session_id: str,
        api_key: str = None,
        max_turns: int = 10,
        token_limit: int = 8000,
        model_name: str = "gemini-3.5-flash",
    ):
        self.session_id = session_id
        # A turn consists of 1 user message + 1 model message
        self.max_messages = max_turns * 2
        self.token_limit = token_limit
        self.model_name = model_name
        self.filepath = f"session_{session_id}.json"

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing API Key: Ensure GEMINI_API_KEY is set in your .env "
                "file or passed to ChatSession."
            )

        self.client = genai.Client(api_key=self.api_key)

        self.summary = ""
        self.history = []
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self):
        """Loads session history and summary from JSON storage on disk."""
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.summary = data.get("summary", "")
                self.history = data.get("history", [])
            print(
                f"[*] Loaded session '{self.session_id}': "
                f"{len(self.history) // 2} turns, "
                f"Summary: {'Yes' if self.summary else 'No'}"
            )
        else:
            self.summary = ""
            self.history = []
            print(f"[*] Started new session '{self.session_id}'")

    def save(self):
        """Persists the current summary and raw history to disk."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump({"summary": self.summary, "history": self.history}, f, indent=2)

    # ------------------------------------------------------------------
    # Token-aware truncation
    # ------------------------------------------------------------------
    def _count_tokens(self, text: str) -> int:
        """Counts tokens for a piece of text using the Gemini API."""
        if not text.strip():
            return 0
        response = self.client.models.count_tokens(model=self.model_name, contents=text)
        return response.total_tokens

    # ------------------------------------------------------------------
    # LLM-based summarization
    # ------------------------------------------------------------------
    def _summarize_oldest(self, num_messages_to_summarize: int = 4):
        """Summarizes the oldest N messages via the LLM and drops them from history."""
        old_msgs = self.history[:num_messages_to_summarize]
        transcript = "\n".join(
            f"{m['role'].upper()}: {m['parts'][0]['text']}" for m in old_msgs
        )

        prompt = (
            "You are a memory management assistant. Summarize the following "
            "conversation snippet to retain key facts, user preferences, and "
            "context. Be concise.\n\n"
        )
        if self.summary:
            prompt += f"Incorporate this existing summary of prior turns:\n{self.summary}\n\n"
        prompt += f"New conversation to summarize:\n{transcript}"

        print(f"[-] Compressing oldest {num_messages_to_summarize // 2} turns into summary...")
        response = self.client.models.generate_content(model=self.model_name, contents=prompt)

        self.summary = response.text
        self.history = self.history[num_messages_to_summarize:]

    # ------------------------------------------------------------------
    # Sliding window + token budget enforcement
    # ------------------------------------------------------------------
    def _manage_memory(self):
        """Enforces the sliding window (last N turns) and token limit before each request."""
        # 1. Sliding window: keep only the most recent `max_turns` turns
        while len(self.history) > self.max_messages:
            self._summarize_oldest(4)

        # 2. Token-aware truncation: stay under 80% of the context budget
        full_text = self.summary + " " + " ".join(
            m["parts"][0]["text"] for m in self.history
        )

        if full_text.strip():
            current_tokens = self._count_tokens(full_text)
            budget_threshold = self.token_limit * 0.8

            while current_tokens > budget_threshold and len(self.history) >= 2:
                print(
                    f"[!] Token warning: {current_tokens} > {budget_threshold}. "
                    "Summarizing to save space."
                )
                self._summarize_oldest(2)
                full_text = self.summary + " " + " ".join(
                    m["parts"][0]["text"] for m in self.history
                )
                current_tokens = self._count_tokens(full_text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def send_message(self, user_text: str) -> str:
        """Processes a new user message, manages memory, calls the LLM, and persists state."""
        self._manage_memory()

        config = types.GenerateContentConfig()
        if self.summary:
            config.system_instruction = f"Context from earlier in the conversation:\n{self.summary}"

        formatted_history = [
            types.Content(role=msg["role"], parts=[types.Part.from_text(text=msg["parts"][0]["text"])])
            for msg in self.history
        ]
        formatted_history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_text)])
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=formatted_history,
            config=config,
        )

        self.history.append({"role": "user", "parts": [{"text": user_text}]})
        self.history.append({"role": "model", "parts": [{"text": response.text}]})

        self.save()
        return response.text


if __name__ == "__main__":
    session_id = "test_user_001"

    print("=== FIRST EXECUTION (simulating a fresh session) ===")
    chat1 = ChatSession(session_id=session_id, max_turns=2)

    print(f"Assistant: {chat1.send_message('Hi, my name is Alex and I love Python.')}\n")
    print(f"Assistant: {chat1.send_message('I am building an LLM app today.')}\n")
    print(f"Assistant: {chat1.send_message('What did I say my name was?')}\n")

    print("\n=== SECOND EXECUTION (simulating a process restart) ===")
    chat2 = ChatSession(session_id=session_id, max_turns=2)
    print(f"Assistant: {chat2.send_message('Can you remind me what language I love?')}\n")

    print("=== FINAL STATE ===")
    print("Summary Content:")
    print(chat2.summary)