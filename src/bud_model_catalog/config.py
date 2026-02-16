from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogConfig:
    litellm_url: str = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
    ai_models_url: str = "https://github.com/truefoundry/models/archive/refs/heads/main.zip"
    timeout: int = 30
    include_deprecated: bool = False
    max_retries: int = 2
    cache: bool = True

    def __post_init__(self) -> None:
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        if not self.litellm_url.startswith(("http://", "https://")):
            raise ValueError(f"litellm_url must be an HTTP(S) URL, got {self.litellm_url!r}")
        if not self.ai_models_url.startswith(("http://", "https://")):
            raise ValueError(f"ai_models_url must be an HTTP(S) URL, got {self.ai_models_url!r}")
        if self.max_retries < 1:
            raise ValueError(f"max_retries must be at least 1, got {self.max_retries}")
