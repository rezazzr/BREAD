import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union


class BaseLanguageModel(ABC):
    """Base class for all language models.

    To add a new model backend:
    1. Subclass this and implement batch_forward_func() and generate()
    2. Register in language_model/__init__.py
    """

    def __init__(self, model_name: str, temperature: float, max_tokens: int, **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def should_sample(self) -> bool:
        """Determine if sampling should be used based on temperature."""
        return self.temperature != 0

    @abstractmethod
    def batch_forward_func(
        self, batch_prompts: List[str]
    ) -> Tuple[List[str], Dict[str, Union[int, float]]]:
        """
        Process a batch of prompts and return responses.

        Args:
            batch_prompts: List of input prompts

        Returns:
            Tuple containing a list of generated responses and a single
            dictionary of additional information describing the batch, such as token usage.
        """
        pass

    @abstractmethod
    def generate(self, input: str) -> Tuple[str, Dict[str, Union[int, float]]]:
        """
        Generate a response for a single input prompt.

        Args:
            input: Input prompt string

        Returns:
            Tuple containing the generated response and a dictionary of additional information such as token usage.
        """
        pass

    def timed_call(self, func, *args, **kwargs):
        """
        Utility to time a function call.
        Returns:
            Tuple[result, latency]
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        latency = time.time() - start_time
        return result, latency
