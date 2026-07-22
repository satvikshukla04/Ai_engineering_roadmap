import logging
import mimetypes
import time
from pathlib import Path
 
from google import genai
from google.genai import types
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
 
 
class GeminiClient:
    """A thin wrapper around the Gemini SDK with budget checks and retries."""
 
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        system_instruction: str = "You are a helpful AI assistant.",
        token_budget: int = 8000,
        max_retries: int = 3,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.token_budget = token_budget
        self.max_retries = max_retries
        self.config = types.GenerateContentConfig(
            system_instruction=system_instruction
        )
 
    # ------------------------------------------------------------------
    # 1. Token counting
    # ------------------------------------------------------------------
    def count_tokens(self, contents) -> int:
        """Return the number of tokens `contents` would use."""
        response = self.client.models.count_tokens(
            model=self.model,
            contents=contents,
        )
        return response.total_tokens
 
    # ------------------------------------------------------------------
    # 2. Budget enforcement
    # ------------------------------------------------------------------
    def _enforce_budget(self, contents) -> int:
        """Count tokens and raise if the request would exceed the budget."""
        tokens = self.count_tokens(contents)
        logger.info("Prompt tokens: %s", tokens)
 
        if tokens > self.token_budget:
            raise ValueError(
                f"Token budget exceeded ({tokens} > {self.token_budget})"
            )
        return tokens
 
    # ------------------------------------------------------------------
    # 3. Retry on 429 with exponential backoff
    # ------------------------------------------------------------------
    def _generate_with_retry(self, contents):
        delay = 1
 
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=self.config,
                )
            except Exception as e:
                if "429" not in str(e):
                    raise
 
                logger.warning(
                    "Rate limited (429). Retry %s/%s in %ss",
                    attempt,
                    self.max_retries,
                    delay,
                )
                time.sleep(delay)
                delay *= 2
 
        raise RuntimeError("Maximum retries exceeded.")
 
    # ------------------------------------------------------------------
    # 4. Token cost logging
    # ------------------------------------------------------------------
    @staticmethod
    def _log_usage(response) -> None:
        usage = response.usage_metadata
        logger.info(
            "Token usage — prompt: %s, output: %s, total: %s",
            usage.prompt_token_count,
            usage.candidates_token_count,
            usage.total_token_count,
        )
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, prompt: str) -> str:
        """Generate text from a plain prompt, with budget check + retry + logging."""
        self._enforce_budget(prompt)
        response = self._generate_with_retry(prompt)
        self._log_usage(response)
        return response.text
 
    def generate_from_image(self, image_path: str, question: str) -> str:
        """Generate text from an image + question (multimodal)."""
        path = Path(image_path)
        mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"
        image_bytes = path.read_bytes()
 
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        contents = [image_part, question]
 
        self._enforce_budget(contents)
        response = self._generate_with_retry(contents)
        self._log_usage(response)
        return response.text