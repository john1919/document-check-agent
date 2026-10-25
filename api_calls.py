"""Bounded timeout-only retry shared by chat and embedding requests."""
from openai import APITimeoutError
import config


def call_with_timeout_retry(operation, *args, **kwargs):
    for attempt in range(config.TIMEOUT_RETRIES + 1):
        try:
            return operation(*args, **kwargs)
        except (APITimeoutError, TimeoutError):
            if attempt == config.TIMEOUT_RETRIES:
                raise
