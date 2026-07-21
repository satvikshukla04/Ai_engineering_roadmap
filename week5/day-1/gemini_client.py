import os
import csv
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
 
load_dotenv()
 
 
class GeminiClient:
    def __init__(self, model="gemini-3.1-flash-lite", system_instruction=None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY in your .env file or environment.")
 
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.system_instruction = system_instruction
        self.log = []  # every call gets appended here
 
    def generate(self, prompt, temperature=0.7, top_p=0.95, top_k=None, system_instruction=None):
        sys_prompt = system_instruction or self.system_instruction
 
        config = types.GenerateContentConfig(
            system_instruction=sys_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
 
        start = time.perf_counter()
        # Free tier = 5 requests/minute. Retry with backoff if we hit the quota.
        for attempt in range(5):
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                break
            except Exception as e:
                if ("429" in str(e) or "503" in str(e)) and attempt < 4:
                    wait = 20 * (attempt + 1)
                    print(f"Temporary error, waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    raise
        latency = round(time.perf_counter() - start, 2)
 
        result = {
            "system_prompt": sys_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "prompt": prompt,
            "output": response.text,
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
            "latency_sec": latency,
        }
        self.log.append(result)
        print(f"[temp={temperature} top_p={top_p} top_k={top_k}] -> {response.text[:80]}...")
        return response.text
 
    def save_log(self, path="gemini_call_log.csv"):
        if not self.log:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.log[0].keys())
            writer.writeheader()
            writer.writerows(self.log)
        print(f"Saved {len(self.log)} runs to {path}")
 
 
if __name__ == "__main__":
    # ---- Daily task: 15 runs, varying temperature + system prompt ----
    question = "Describe what happens when you leave a glass of milk out of the fridge overnight."
 
    system_prompts = {
        "neutral": "You are a helpful assistant.",
        "creative": "You are a whimsical creative writer who loves vivid imagery.",
        "strict_formal": "You are a precise technical assistant. Be factual and concise.",
    }
    temperatures = [0.0, 0.3, 0.7, 1.0]
 
    client = GeminiClient()
 
    # Free tier = 5 requests/minute -> pause ~13s between calls to stay under it.
    PAUSE_SECONDS = 13
 
    try:
        # 3 system prompts x 4 temperatures = 12 runs
        for label, sys_prompt in system_prompts.items():
            for temp in temperatures:
                client.generate(question, temperature=temp, system_instruction=sys_prompt)
                time.sleep(PAUSE_SECONDS)
 
        # 3 extra edge-case runs to reach 15
        client.generate(question, temperature=1.0, top_p=0.5, system_instruction=system_prompts["neutral"])  # narrow top_p
        time.sleep(PAUSE_SECONDS)
        client.generate(question, temperature=0.7, top_k=1, system_instruction=system_prompts["neutral"])    # greedy via top_k
        time.sleep(PAUSE_SECONDS)
        client.generate(question, temperature=1.5, system_instruction=system_prompts["creative"])            # extreme temp
    finally:
        # Save whatever succeeded, even if a later call errors out for good.
        client.save_log("gemini_call_log.csv")