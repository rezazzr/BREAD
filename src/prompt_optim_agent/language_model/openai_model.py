import logging
import time
from typing import Dict, Tuple

from joblib import Parallel, delayed
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam

from .base_model import BaseLanguageModel

logger = logging.getLogger(__name__)


class OpenAIModel(BaseLanguageModel):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float,
        max_tokens: int,
        **kwargs,
    ):
        super().__init__(model_name, temperature, max_tokens, **kwargs)

        if api_key is None:
            raise ValueError(f"api_key error: {api_key}")
        try:
            self.model = OpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"Init openai client error: {e}")
            raise RuntimeError("Failed to initialize OpenAI client") from e

        self.max_parallel_requests = kwargs.get("max_parallel_requests", 4)
        self.max_retries = kwargs.get("max_retries", 10)
        self.initial_backoff = kwargs.get("initial_backoff", 1.0)
        self.backoff_multiplier = kwargs.get("backoff_multiplier", 1.5)

    def batch_forward_func(self, batch_prompts):
        """
        Process a batch of prompts and return responses and aggregated logging info.
        Returns:
            Tuple[List[str], Dict[str, int]]
        """

        def process_batch():
            results = Parallel(
                n_jobs=min(self.max_parallel_requests, len(batch_prompts)),
                backend="threading",
            )(delayed(self.generate)(prompt) for prompt in batch_prompts)
            return results

        results, batch_latency = self.timed_call(process_batch)
        responses = [result_tuple[0] for result_tuple in results]  # type: ignore
        # Aggregate logging info
        total_prompt_tokens = sum(
            result_tuple[1]["prompt_tokens"] for result_tuple in results  # type: ignore
        )
        total_generated_tokens = sum(
            result_tuple[1]["generated_tokens"] for result_tuple in results  # type: ignore
        )
        total_tokens = sum(
            result_tuple[1]["total_tokens"] for result_tuple in results  # type: ignore
        )
        logging_dict = {
            "generated_tokens": total_generated_tokens,
            "prompt_tokens": total_prompt_tokens,
            "total_tokens": total_tokens,
            "latency": batch_latency,
        }
        return responses, logging_dict

    def generate(self, input: str) -> Tuple[str, Dict[str, int]]:
        """
        Generate a response for a single input prompt.
        Returns:
            Tuple[str, Dict[str, int]]
        """
        messages = [
            ChatCompletionUserMessageParam(role="user", content=input),
        ]
        backoff_time = self.initial_backoff
        for attempt in range(1, self.max_retries + 1):
            try:

                def call_openai():
                    return self.model.chat.completions.create(
                        messages=messages,
                        model=self.model_name,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )

                response, latency = self.timed_call(call_openai)
                choice = response.choices[0]
                content = choice.message.content.strip()  # type: ignore
                usage = getattr(response, "usage", None)
                if usage is None:
                    raise RuntimeError("OpenAI response missing usage information.")
                logging_dict = {
                    "generated_tokens": usage.completion_tokens,
                    "prompt_tokens": usage.prompt_tokens,
                    "total_tokens": usage.total_tokens,
                    "latency": latency,
                }
                return content, logging_dict
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt}/{self.max_retries} failed: {e}. "
                    f"Sleeping {backoff_time:.1f}s..."
                )
                time.sleep(backoff_time)
                backoff_time *= self.backoff_multiplier

        raise RuntimeError(
            f"OpenAI API call failed after {self.max_retries} retries for model '{self.model_name}'."
        )
