# End of Selection
"""
OpenAI Client - The "Discriminator" in our adversarial system
Responsible for critiquing Claude's solutions and finding flaws
"""

import os
from typing import List, Dict, Optional
from openai import OpenAI

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class OpenAIClient:
    """OpenAI client for critiquing and challenging code solutions."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        # Initialize the SDK client
        self.client = OpenAI(api_key=self.api_key)

        # Recommended models (2025)
        self.available_models = {
            "gpt-5": "gpt-5",
            "gpt-5-codex": "gpt-5-codex",
            "o3": "o3",
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
        }

        # Default to GPT-5-Codex
        self.model = model or self.available_models["gpt-5-codex"]

    def critique_solution(
        self,
        original_query: str,
        claude_solution: str,
        claude_reasoning: str,
    ) -> Dict[str, str]:
        """Critique Claude's solution and reasoning."""
        prompt = self._build_critique_prompt(
            original_query, claude_solution, claude_reasoning
        )

        # ✅ Correct call for openai>=1.0
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )

        content = response.choices[0].message.content
        return {
            "critique": content,
            "model_used": self.model,
        }

    def final_assessment(
        self, original_query: str, final_claude_solution: str
    ) -> Dict[str, str]:
        """Provide final assessment of Claude's solution."""
        prompt = self._build_assessment_prompt(original_query, final_claude_solution)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        return {
            "final_assessment": response.choices[0].message.content,
            "model_used": self.model,
        }

    # ---------- Helpers ----------
    def _build_critique_prompt(self, query: str, solution: str, reasoning: str) -> str:
        return f"""You are an expert code reviewer.

ORIGINAL PROBLEM:
{query}

PROPOSED SOLUTION:
{solution}

AUTHOR'S REASONING:
{reasoning}

Provide:
## ISSUES FOUND
## SUGGESTIONS
## OVERALL ASSESSMENT
"""

    def _build_assessment_prompt(self, query: str, solution: str) -> str:
        return f"""Final review of the solution:

ORIGINAL PROBLEM:
{query}

FINAL SOLUTION:
{solution}

Give a brief overall assessment and quality rating (1–10).
"""


def test_openai_client():
    client = OpenAIClient()
    sample_query = "Write a Python function to find the maximum element in a list"
    sample_solution = """def find_max(lst):
    max_val = lst[0]
    for item in lst:
        if item > max_val:
            max_val = item
    return max_val"""
    sample_reasoning = "Iterates through list comparing each element."
    result = client.critique_solution(sample_query, sample_solution, sample_reasoning)
    print(result["critique"][:500])


if __name__ == "__main__":
    test_openai_client()
