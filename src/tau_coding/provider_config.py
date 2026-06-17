"""Durable provider configuration for Tau coding sessions."""

from dataclasses import dataclass, field
from json import dumps, loads
from os import environ
from pathlib import Path
from typing import Any

from tau_ai import OpenAICompatibleConfig, openai_compatible_config_from_env
from tau_ai.env import DEFAULT_OPENAI_COMPATIBLE_BASE_URL
from tau_coding.paths import TauPaths

DEFAULT_PROVIDER_NAME = "openai"
DEFAULT_MODEL = "gpt-4.1-mini"


class ProviderConfigError(ValueError):
    """Raised when Tau provider configuration is invalid."""


@dataclass(frozen=True, slots=True)
class OpenAICompatibleProviderConfig:
    """Durable settings for one OpenAI-compatible provider."""

    name: str
    base_url: str = DEFAULT_OPENAI_COMPATIBLE_BASE_URL
    api_key_env: str = "OPENAI_API_KEY"
    models: tuple[str, ...] = (DEFAULT_MODEL,)
    default_model: str = DEFAULT_MODEL

    def to_json(self) -> dict[str, Any]:
        """Serialize this provider config to JSON-compatible data."""
        return {
            "name": self.name,
            "type": "openai-compatible",
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "models": list(self.models),
            "default_model": self.default_model,
        }


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    """Tau provider settings loaded from Tau home."""

    default_provider: str = DEFAULT_PROVIDER_NAME
    providers: tuple[OpenAICompatibleProviderConfig, ...] = field(
        default_factory=lambda: (default_openai_provider_config(),)
    )

    def get_provider(self, name: str | None = None) -> OpenAICompatibleProviderConfig:
        """Return a configured provider by name."""
        target = name or self.default_provider
        for provider in self.providers:
            if provider.name == target:
                return provider
        raise ProviderConfigError(f"Unknown provider: {target}")

    def to_json(self) -> dict[str, Any]:
        """Serialize these settings to JSON-compatible data."""
        return {
            "default_provider": self.default_provider,
            "providers": [provider.to_json() for provider in self.providers],
        }


@dataclass(frozen=True, slots=True)
class ProviderSelection:
    """Resolved provider/model selection for a Tau run."""

    provider: OpenAICompatibleProviderConfig
    model: str


def default_openai_provider_config() -> OpenAICompatibleProviderConfig:
    """Return Tau's default OpenAI-compatible provider entry."""
    return OpenAICompatibleProviderConfig(name=DEFAULT_PROVIDER_NAME)


def provider_settings_path(paths: TauPaths | None = None) -> Path:
    """Return the durable provider settings path."""
    return (paths or TauPaths()).home / "providers.json"


def load_provider_settings(paths: TauPaths | None = None) -> ProviderSettings:
    """Load durable provider settings, falling back to env-compatible defaults."""
    path = provider_settings_path(paths)
    if not path.exists():
        return ProviderSettings()
    raw = loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ProviderConfigError("Provider settings must be a JSON object")
    return provider_settings_from_json(raw)


def save_provider_settings(
    settings: ProviderSettings, paths: TauPaths | None = None
) -> Path:
    """Write durable provider settings and return the path."""
    path = provider_settings_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(settings.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def provider_settings_from_json(data: dict[str, Any]) -> ProviderSettings:
    """Parse provider settings from JSON-compatible data."""
    default_provider = _string(data.get("default_provider"), "default_provider")
    providers_data = data.get("providers")
    if not isinstance(providers_data, list) or not providers_data:
        raise ProviderConfigError("Provider settings must include at least one provider")
    providers = tuple(_provider_from_json(item) for item in providers_data)
    names = [provider.name for provider in providers]
    if len(set(names)) != len(names):
        raise ProviderConfigError("Provider names must be unique")
    settings = ProviderSettings(default_provider=default_provider, providers=providers)
    settings.get_provider(default_provider)
    return settings


def resolve_provider_selection(
    settings: ProviderSettings,
    *,
    provider_name: str | None = None,
    model: str | None = None,
) -> ProviderSelection:
    """Resolve the provider and model for a run."""
    provider = settings.get_provider(provider_name)
    selected_model = model or provider.default_model
    if not selected_model:
        raise ProviderConfigError(f"Provider {provider.name} does not define a default model")
    return ProviderSelection(provider=provider, model=selected_model)


def openai_compatible_config_from_provider(
    provider: OpenAICompatibleProviderConfig,
) -> OpenAICompatibleConfig:
    """Build runtime provider config from durable settings and environment."""
    if (
        provider.name == DEFAULT_PROVIDER_NAME
        and provider.api_key_env == "OPENAI_API_KEY"
        and provider.base_url == DEFAULT_OPENAI_COMPATIBLE_BASE_URL
    ):
        return openai_compatible_config_from_env(base_url_var="OPENAI_BASE_URL")
    api_key = environ.get(provider.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing required environment variable: {provider.api_key_env}")
    return OpenAICompatibleConfig(api_key=api_key, base_url=provider.base_url.rstrip("/"))


def _provider_from_json(data: object) -> OpenAICompatibleProviderConfig:
    if not isinstance(data, dict):
        raise ProviderConfigError("Provider entries must be JSON objects")
    provider_type = _string(data.get("type"), "providers[].type")
    if provider_type != "openai-compatible":
        raise ProviderConfigError(f"Unsupported provider type: {provider_type}")
    name = _string(data.get("name"), "providers[].name")
    base_url = _string(data.get("base_url"), f"providers[{name}].base_url").rstrip("/")
    api_key_env = _string(data.get("api_key_env"), f"providers[{name}].api_key_env")
    models = _string_tuple(data.get("models"), f"providers[{name}].models")
    default_model = _string(data.get("default_model"), f"providers[{name}].default_model")
    if default_model not in models:
        models = (*models, default_model)
    return OpenAICompatibleProviderConfig(
        name=name,
        base_url=base_url,
        api_key_env=api_key_env,
        models=models,
        default_model=default_model,
    )


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProviderConfigError(f"Provider field must be a non-empty string: {field_name}")
    return value.strip()


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ProviderConfigError(f"Provider field must be a non-empty string list: {field_name}")
    items = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    if len(items) != len(value):
        raise ProviderConfigError(f"Provider field must be a string list: {field_name}")
    return items
