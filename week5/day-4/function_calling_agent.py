"""
function_calling_agent.py

Manual Gemini function-calling agent with:
- .env API key loading
- Manual tool execution loop
- Logging
- Three tools:
    1. get_current_date()
    2. calculator(expression)
    3. search_knowledge_base(query)

"""

from __future__ import annotations

import ast
import logging
import operator
import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ----------------------------- Configuration -----------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

MODEL = "gemini-3.6-flash"
MAX_TOOL_CALLS = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("agent.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GeminiAgent")

client = genai.Client(api_key=API_KEY)

# ----------------------------- Knowledge Base -----------------------------

KNOWLEDGE_BASE = {
    "python": "Python is a high-level, general-purpose programming language.",
    "gemini": "Gemini is Google's family of multimodal AI models.",
    "llm": "Large Language Models predict tokens and can use external tools.",
}

# ----------------------------- Tools -----------------------------

def get_current_date() -> str:
    """Return today's date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


_ALLOWED = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def _eval(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants allowed.")
        return node.value
    if isinstance(node, ast.BinOp):
        return _ALLOWED[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _ALLOWED[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression.")

def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree))
    except Exception as e:
        return f"Error: {e}"

def search_knowledge_base(query: str) -> str:
    """Search the local knowledge base."""
    q = query.lower()
    for k, v in KNOWLEDGE_BASE.items():
        if k in q:
            return v
    return "No relevant information found."

TOOLS = [get_current_date, calculator, search_knowledge_base]
FUNCTION_MAP = {f.__name__: f for f in TOOLS}

# ----------------------------- Agent -----------------------------

def run_agent(prompt: str) -> None:
    history: list[Any] = [
        types.Content(role="user", parts=[types.Part(text=prompt)])
    ]

    tool_calls = 0

    while tool_calls < MAX_TOOL_CALLS:
        response = client.models.generate_content(
            model=MODEL,
            contents=history,
            config=types.GenerateContentConfig(tools=TOOLS),
        )

        if not response.candidates:
            print("No response from model.")
            return

        candidate = response.candidates[0]
        history.append(candidate.content)

        tool_used = False

        for part in candidate.content.parts:
            if getattr(part, "text", None):
                print("\nAssistant:", part.text)

            if getattr(part, "function_call", None):
                tool_used = True
                tool_calls += 1

                fc = part.function_call
                fn_name = fc.name
                args = dict(fc.args)

                logger.info("Tool: %s", fn_name)
                logger.info("Args: %s", args)

                try:
                    result = FUNCTION_MAP[fn_name](**args)
                except Exception as e:
                    result = f"Tool execution failed: {e}"

                logger.info("Result: %s", result)

                history.append(
                    types.Content(
                        role="tool",
                        parts=[
                            types.Part.from_function_response(
                                name=fn_name,
                                response={"result": result},
                            )
                        ],
                    )
                )

        if not tool_used:
            return

    print("Maximum tool call limit reached.")

def main():
    print("Gemini Function Calling Agent")
    print("Type 'exit' to quit.\n")

    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break
        if prompt:
            run_agent(prompt)

if __name__ == "__main__":
    main()