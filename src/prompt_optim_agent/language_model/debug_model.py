"""Fake LLM for debugging — no API calls, deterministic responses.

Usage in config:
    base_model_setting:
      model_type: debug
      model_name: debug-base
      temperature: 0.0
      max_tokens: 100
      latency: 0.5          # seconds per call (default 0)

    optim_model_setting:
      model_type: debug
      model_name: debug-optim
      temperature: 1.0
      max_tokens: 100
      latency: 1.0

The model detects its role from prompt content and returns
plausible responses so the full MCTS pipeline runs end-to-end.
"""

import hashlib
import logging
import re
import time
from typing import Dict, List, Tuple, Union

from .base_model import BaseLanguageModel

logger = logging.getLogger(__name__)


class DebugModel(BaseLanguageModel):
    """Deterministic fake LLM for pipeline debugging."""

    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        **kwargs,
    ):
        super().__init__(model_name, temperature, max_tokens, **kwargs)
        self.latency = kwargs.get("latency", 0.0)
        self._call_count = 0

    def generate(self, input: str) -> Tuple[str, Dict[str, Union[int, float]]]:
        self._call_count += 1
        if self.latency:
            time.sleep(self.latency)

        response = self._route(input)
        info = self._token_info(input, response)
        return response, info

    def batch_forward_func(
        self, batch_prompts: List[str]
    ) -> Tuple[List[str], Dict[str, Union[int, float]]]:
        results = [self.generate(p) for p in batch_prompts]
        responses = [r for r, _ in results]
        infos = [info for _, info in results]
        aggregated = {key: sum(d[key] for d in infos) for key in infos[0]}
        return responses, aggregated

    # ------------------------------------------------------------------
    # Response routing
    # ------------------------------------------------------------------
    def _route(self, prompt: str) -> str:
        lower = prompt.lower()

        # Optimize prompt — asks for <START>...<END> wrapped output
        if "<start>" in lower and "<end>" in lower:
            return self._optimize_response()

        # Gradient summarization
        if "you are an expert summarizer" in lower:
            return self._summarize_response()

        # Gradient prompt — asks for reasons why prompt leads to wrong/correct answer
        if "reasons why the prompt leads to" in lower:
            return self._gradient_response()

        # Default: forward/answer prompt
        return self._answer_response(prompt)

    # ------------------------------------------------------------------
    # Response generators
    # ------------------------------------------------------------------
    def _answer_response(self, prompt: str) -> str:
        """Pick an answer from options found in the prompt.

        Extracts answer options from the prompt text (e.g. "- Entailment"
        or "(A) Not similar") and deterministically selects one using a
        hash of the prompt content. This gives a stable mix of correct
        and incorrect answers to exercise both gradient paths.
        """
        options = self._extract_options(prompt)
        h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
        choice = options[h % len(options)]
        return (
            f"After careful analysis, I believe the answer is {choice}.\n"
            f"<answer>{choice}</answer>"
        )

    def _gradient_response(self) -> str:
        n = self._call_count
        return (
            f"[debug gradient #{n}] The prompt could be improved by: "
            "1) Adding clearer evaluation criteria, "
            "2) Being more specific about the comparison dimensions, "
            "3) Providing explicit output format instructions."
        )

    def _optimize_response(self) -> str:
        n = self._call_count
        return (
            f"<START>Debug optimized prompt v{n}: "
            "Carefully analyze the given input. Compare the key aspects "
            "mentioned in both items. Consider similarities and differences. "
            f"Provide a clear, well-reasoned answer. (iteration {n})<END>"
        )

    def _summarize_response(self) -> str:
        return (
            "The key themes across all feedback are: "
            "1) Need for clearer evaluation criteria, "
            "2) More specific comparison dimensions, "
            "3) Better output format guidance."
        )

    # ------------------------------------------------------------------
    # Option extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_options(prompt: str) -> List[str]:
        """Extract answer options from the prompt text.

        Handles common formats:
          - Bullet lists:  "- Entailment\\n- Contradiction\\n- Neutral"
          - Letter lists:  "(A) Not similar  (B) Somewhat similar  (C) Similar"
          - Fallback:      ["A", "B", "C", "D"]
        """
        # Try bullet-list format: lines starting with "- "
        bullets = re.findall(r"^- (.+)$", prompt, re.MULTILINE)
        if len(bullets) >= 2:
            return [b.strip() for b in bullets]

        # Try parenthesized-letter format: (A) text (B) text ...
        letters = re.findall(r"\(([A-Z])\)\s*([^(]+)", prompt)
        if len(letters) >= 2:
            return [f"({letter})" for letter, _ in letters]

        # Fallback: generic letters
        return ["A", "B", "C", "D"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _token_info(self, prompt: str, response: str) -> Dict[str, Union[int, float]]:
        prompt_tokens = len(prompt.split())
        gen_tokens = len(response.split())
        return {
            "generated_tokens": gen_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens + gen_tokens,
            "latency": self.latency,
        }
