import logging
import random
import time
import traceback
from logging import Logger
from typing import Any

from .adapters import AnthropicAdapter, LocalAdapter, OpenAIAdapter

# Logger + Tokenizer
log: Logger = logging.getLogger(__name__)


class SimpleRetry:
    """
    Class to handle Retry Errors.
    """

    RETRYABLE_ERRORS = (
        "RateLimit",
        "ServiceUnavailable",
        "Timeout",
        "APIConnection",
        "InternalServer",
        "APIError",
    )

    def __init__(self, max_attempts: int = 5, base: int = 60):
        self.max_attempts: int = max_attempts
        self.base: int = base
        self.attempt: int = 0

    def should_retry(self, err: Exception) -> bool:
        """Check if we need to retry"""
        is_retryable: bool = (
            any(name in err.__class__.__name__ for name in self.RETRYABLE_ERRORS)
            and self.attempt < self.max_attempts
        )
        return is_retryable

    def sleep(self) -> None:
        """Wait base_wait seconds + up to jitter extra."""
        self.attempt += 1
        wait: float = self.base + 10 * random.uniform(0, 0.5)
        log.warning(
            msg=f"Retrying in ({wait:.2f}s) ({self.attempt}/{self.max_attempts})"
        )
        time.sleep(wait)


class Agent:
    """
    Provider-agnostic agent for different adapters
    The adapter just needs a .complete(...) method (see adapters.py).
    """

    def __init__(
        self,
        adapter: OpenAIAdapter | LocalAdapter | AnthropicAdapter,
        retry: SimpleRetry,
        max_output_tokens: int = 4096,
    ):
        self.adapter = adapter
        self.retry = retry
        self.max_output_tokens = max_output_tokens

    def ask(
        self,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> str | None:

        retry: bool = True
        while retry:
            try:
                return self.adapter.complete(
                    messages, max_output_tokens=self.max_output_tokens, **kwargs
                )
            except Exception as e:
                retry = self.retry.should_retry(e)
                if not retry:
                    log.error("Agent error: %r", e)
                    traceback.print_exc()
                    return
                self.retry.sleep()
