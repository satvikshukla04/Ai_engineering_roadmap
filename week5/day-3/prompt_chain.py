from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, ValidationError

# ============================================================
# Load API Key
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

# ============================================================
# Gemini Client
# ============================================================

client = genai.Client(api_key=API_KEY)

# ============================================================
# Pydantic Models
# ============================================================


class Fact(BaseModel):
    subject: str
    fact: str


class FactExtraction(BaseModel):
    facts: List[Fact]


class SectionSentiment(BaseModel):
    section: str
    sentiment: str
    reason: str


class SentimentAnalysis(BaseModel):
    sentiments: List[SectionSentiment]


class ExecutiveSummary(BaseModel):
    summary: str
    cited_facts: List[str]


# ============================================================
# Helper Function
# ============================================================


def ask(prompt: str) -> str:
    """
    Sends a prompt to Gemini and returns JSON text.
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        },
    )

    return response.text


# ============================================================
# Prompt Templates
# ============================================================

EXTRACT_PROMPT = """
You are an information extraction assistant.

Extract ONLY factual statements from the document.

Return exactly this schema:

{{
  "facts":[
    {{
      "subject":"...",
      "fact":"..."
    }}
  ]
}}

Respond ONLY with valid JSON.

Document:
{document}
"""

SENTIMENT_PROMPT = """
...

Return exactly:

{{
  "sentiments":[
    {{
      "section":"...",
      "sentiment":"...",
      "reason":"..."
    }}
  ]
}}

...

Document:

{document}
"""

SUMMARY_PROMPT = """
...

Generate:

{{
  "summary":"...",
  "cited_facts":[
      "...",
      "..."
  ]
}}

...


The summary must reference the extracted facts.

Respond ONLY with valid JSON.
"""


# ============================================================
# Prompt Chain
# ============================================================


def run_document(document: str):

    print("=" * 70)
    print("DOCUMENT")
    print(document)

    # -------------------------
    # STEP 1
    # -------------------------

    print("\nSTEP 1 : FACT EXTRACTION\n")

    facts_json = ask(
        EXTRACT_PROMPT.format(document=document)
    )

    try:
        facts = FactExtraction.model_validate_json(facts_json)
    except ValidationError as e:
        print("Fact Extraction Validation Failed")
        print(e)
        return

    print(facts.model_dump_json(indent=4))

    # -------------------------
    # STEP 2
    # -------------------------

    print("\nSTEP 2 : SENTIMENT ANALYSIS\n")

    sentiment_json = ask(
        SENTIMENT_PROMPT.format(document=document)
    )

    try:
        sentiments = SentimentAnalysis.model_validate_json(sentiment_json)
    except ValidationError as e:
        print("Sentiment Validation Failed")
        print(e)
        return

    print(sentiments.model_dump_json(indent=4))

    # -------------------------
    # STEP 3
    # -------------------------

    print("\nSTEP 3 : EXECUTIVE SUMMARY\n")

    summary_json = ask(
        SUMMARY_PROMPT.format(
            facts=facts.model_dump_json(indent=2),
            sentiments=sentiments.model_dump_json(indent=2),
        )
    )

    try:
        summary = ExecutiveSummary.model_validate_json(summary_json)
    except ValidationError as e:
        print("Summary Validation Failed")
        print(e)
        return

    print(summary.model_dump_json(indent=4))


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    documents_dir = Path("documents")

    document_files = [
        documents_dir / "doc1.txt",
        documents_dir / "doc2.txt",
    ]

    for file_path in document_files:

        print(f"\nProcessing: {file_path.name}")

        document = file_path.read_text(encoding="utf-8")

        run_document(document)

        #comment
        