# Phase 18: Provider Configuration Foundation

This phase starts Tau's durable provider configuration work without adding an
extension system.

The implementation lives in:

```text
src/tau_coding/provider_config.py
src/tau_coding/cli.py
src/tau_coding/tui/app.py
src/tau_coding/commands.py
src/tau_coding/session.py
```

## What was added

Tau now has a provider settings model under `tau_coding`:

```python
ProviderSettings
OpenAICompatibleProviderConfig
ProviderSelection
```

Settings are stored at:

```text
~/.tau/providers.json
```

If that file does not exist, Tau uses an OpenAI-compatible default:

```text
provider: openai
model: gpt-4.1-mini
api key env var: OPENAI_API_KEY
base URL env var: OPENAI_BASE_URL
```

API keys are not stored in the config file. Provider entries name the
environment variable that should hold the key.

## Example config

```json
{
  "default_provider": "local",
  "providers": [
    {
      "name": "local",
      "type": "openai-compatible",
      "base_url": "http://localhost:11434/v1",
      "api_key_env": "LOCAL_API_KEY",
      "models": ["qwen", "llama"],
      "default_model": "qwen"
    }
  ]
}
```

## Runtime resolution

Print mode and TUI startup now resolve provider/model selection from durable
settings:

```text
tau --provider local --model qwen
tau -p "review this" --provider local
```

When `--model` is omitted, Tau uses the configured provider's default model.
When `--provider` is omitted, Tau uses `default_provider`.

## Commands

Slash commands now expose the active provider/model configuration:

```text
/provider
/model
/model <name>
```

`/model <name>` switches the active model for future turns in the running
process when the model is known for the active provider. Provider switching is
still a startup concern and is done with `--provider <name>`.

## Boundary

Provider settings belong to `tau_coding`, not `tau_agent`.

The reusable harness still receives only a ready `ModelProvider` and a model
name. It does not know about Tau home, JSON config files, environment variables,
or CLI/TUI setup behavior.

## Remaining Phase 18 work

This is the foundation, not the full setup UX. Remaining work includes:

- a first-run setup command or interactive flow
- editing provider config through commands/TUI
- richer provider switching in a running TUI
- docs for config migration and troubleshooting

## Tests

The phase is covered by:

```text
tests/test_provider_config.py
tests/test_cli.py
tests/test_commands.py
tests/test_tui_app.py
```

The tests verify:

- missing config falls back to OpenAI-compatible defaults
- provider settings round-trip through `~/.tau/providers.json`
- default provider/model selection
- configured API key environment variables
- CLI provider/model forwarding
- TUI startup selection
- `/provider` and `/model` command behavior
