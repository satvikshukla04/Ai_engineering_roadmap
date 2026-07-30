"""
eval_harness.py

Daily task: Build a 10-question Q&A eval set. Run GeminiClient on all 10.
Build an LLM-as-judge evaluator scoring correctness + relevance (1-5).
Report mean scores and flag failures.

Usage:
    1. Place a 10-item eval_dataset.json (list of {id, question, reference_answer})
       in this directory.
    2. Run: python eval_harness.py
    3. Results (per-question scores + aggregate summary) are written to
       eval_results.json.
"""

import json
import logging
import time
import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from google import genai
from google.genai import types

# ==========================================
# 0. Initialization & Config
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("EvalHarness")

load_dotenv()

if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("API key not found. Please set GEMINI_API_KEY in your .env file.")


# ==========================================
# 1. Data Models
# ==========================================
class TestCase(BaseModel):
    id: str
    question: str
    reference_answer: str


class EvalScore(BaseModel):
    correctness: int = Field(description="Score from 1 to 5 indicating factual accuracy compared to the reference.")
    relevance: int = Field(description="Score from 1 to 5 indicating how directly the answer addresses the question.")
    reasoning: str = Field(description="Short explanation justifying the scores.")


class EvalResult(BaseModel):
    test_id: str
    question: str
    generated_answer: str
    scores: EvalScore
    passed: bool
    latency_ms: float


# ==========================================
# 2. Resiliency & Clients
# ==========================================
def call_with_retry(func, *args, max_retries=4, **kwargs):
    """Handles 429 rate limits with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = 15 * (attempt + 1)
                logger.warning(f"Rate limited (429)! Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                # Structural errors (e.g. 404) should surface immediately
                raise e
    raise Exception("Max retries exceeded due to rate limits.")


class GeminiClient:
    """Application client that generates answers to be evaluated."""

    def __init__(self, model_name: str = "gemini-3.5-flash"):
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.model_name = model_name

    def generate_answer(self, question: str) -> str:
        response = self.client.models.generate_content(model=self.model_name, contents=question)
        return response.text.strip()


class JudgeClient:
    """LLM-as-judge client that scores generated answers for correctness and relevance."""

    def __init__(self, model_name: str = "gemini-3.5-flash"):
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.model_name = model_name
        self.prompt_template = """
        You are an impartial expert evaluator. Evaluate the generated answer against the question and the reference answer.

        Question: {question}
        Reference Answer: {reference}
        Generated Answer: {generated}

        Score Correctness (1-5): 1 is completely wrong, 5 is perfectly accurate to the reference.
        Score Relevance (1-5): 1 is completely off-topic, 5 directly answers the question with no fluff.
        Provide brief reasoning for your scores.
        """

    def evaluate(self, question: str, reference: str, generated: str) -> EvalScore:
        prompt = self.prompt_template.format(question=question, reference=reference, generated=generated)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EvalScore,
                temperature=0.0,  # Zero temperature for deterministic judging
            ),
        )
        return response.parsed


# ==========================================
# 3. The Harness
# ==========================================
def run_eval_harness(test_cases: List[TestCase], pass_threshold: int = 3) -> Dict[str, Any]:
    app_client = GeminiClient()
    judge = JudgeClient()

    results: List[EvalResult] = []
    logger.info(f"Starting evaluation of {len(test_cases)} test cases...")

    for tc in test_cases:
        logger.info(f"Processing Test ID: {tc.id}")

        # 1. Generate answer (wrapped in retry)
        start_time = time.time()
        try:
            generated_ans = call_with_retry(app_client.generate_answer, tc.question)
        except Exception as e:
            logger.error(f"Generation failed for {tc.id}: {e}")
            generated_ans = "ERROR: Generation Failed"

        latency = round((time.time() - start_time) * 1000, 2)

        # 2. Evaluate answer via LLM-as-judge (wrapped in retry)
        try:
            scores = call_with_retry(judge.evaluate, tc.question, tc.reference_answer, generated_ans)
        except Exception as e:
            logger.error(f"Evaluation failed for {tc.id}: {e}")
            scores = EvalScore(correctness=0, relevance=0, reasoning=f"Judge failure: {str(e)}")

        # 3. Flag failures
        passed = scores.correctness >= pass_threshold and scores.relevance >= pass_threshold
        if not passed:
            logger.warning(f"Test {tc.id} FAILED. Correctness: {scores.correctness}, Relevance: {scores.relevance}")

        results.append(
            EvalResult(
                test_id=tc.id,
                question=tc.question,
                generated_answer=generated_ans,
                scores=scores,
                passed=passed,
                latency_ms=latency,
            )
        )

        # 4. Strict pacing: ~15 RPM on the free tier
        time.sleep(4)

    # 5. Aggregate metrics
    total_correctness = sum(r.scores.correctness for r in results)
    total_relevance = sum(r.scores.relevance for r in results)
    fail_count = sum(1 for r in results if not r.passed)

    return {
        "timestamp": time.time(),
        "total_tests": len(test_cases),
        "mean_correctness": round(total_correctness / len(test_cases), 2),
        "mean_relevance": round(total_relevance / len(test_cases), 2),
        "fail_rate_percentage": round((fail_count / len(test_cases)) * 100, 2),
        "failed_test_ids": [r.test_id for r in results if not r.passed],
        "detailed_results": [r.model_dump() for r in results],
    }


if __name__ == "__main__":
    if not os.path.exists("eval_dataset.json"):
        raise FileNotFoundError("eval_dataset.json missing. Save a 10-item JSON array in this directory before running.")

    with open("eval_dataset.json", "r") as f:
        raw_cases = json.load(f)
        dataset = [TestCase(**tc) for tc in raw_cases]

    final_report = run_eval_harness(dataset)

    with open("eval_results.json", "w") as f:
        json.dump(final_report, f, indent=2)

    logger.info("Evaluation complete. Results written to eval_results.json.")