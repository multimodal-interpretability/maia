import os

from .adapters import AnthropicAdapter, LocalAdapter, OpenAIAdapter
from .agent import Agent, SimpleRetry


def infer_provider(model: str) -> str:
    """Infer the provider name based on the model prefix."""
    if model.startswith("gpt"):
        return "openai"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("local-"):
        return "local"
    raise ValueError(f"Unsupported model prefix: {model}")


def create_agent(model: str, **kwargs) -> Agent:
    """
    Public entry point for creating an Agent.
    (api_key and organization can be setup as env. vars)
    Examples:
        create_agent("gpt-4o", api_key="...", organization="...")
        create_agent("claude-3-5-sonnet", api_key="...")
        create_agent("local-google/gemma-3-27b-it", base_url="http://localhost:11434/v1")
    """
    provider: str = infer_provider(model)
    retry = SimpleRetry(
        max_attempts=kwargs.pop("max_attempts", 5), base=kwargs.pop("base", 60)
    )
    max_output_tokens: int = kwargs.pop("max_output_tokens", 4096)

    if provider == "openai":
        adapter = OpenAIAdapter(
            api_key=kwargs.pop("api_key", os.getenv("OPENAI_API_KEY")),
            organization=kwargs.pop("organization", os.getenv("OPENAI_ORGANIZATION")),
            model="gpt-4o",
        )
    elif provider == "anthropic":
        adapter = AnthropicAdapter(
            api_key=kwargs.pop("api_key", os.getenv("ANTHROPIC_API_KEY")),
            model="claude-4-sonnet-20250514",
        )
    elif provider == "local":
        adapter = LocalAdapter(
            base_url=kwargs.pop("base_url", "http://localhost:11434/v1"),
            model=model.removeprefix("local-"),
            api_key=kwargs.pop("api_key", "123"),
        )
    else:
        raise ValueError(f"Unrecognized provider: {provider}")

    return Agent(adapter=adapter, retry=retry, max_output_tokens=max_output_tokens)
