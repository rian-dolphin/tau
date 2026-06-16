# 00 — Roadmap

Tau is being built as a Python implementation of Pi's minimalist coding-agent harness architecture.
The goal is not a line-by-line port; the goal is to preserve the same boundaries while using Python-native tools.

## Package layers

```text
tau_ai       provider/model streaming layer
tau_agent    portable agent harness, loop, tools, events, sessions
tau_coding   CLI app, resources, skills, extensions, commands, UI integration
```

## Phase plan

0. Project foundation and design docs.
1. Core message, tool, and event types.
2. Provider interface with fake and real providers.
3. Pure agent loop.
4. Reusable `AgentHarness`.
5. Built-in coding tools.
6. Non-interactive print-mode CLI.
7. Append-only session tree persistence.
8. Coding session wrapper with commands.
9. Skills, prompt templates, and system prompt assembly.
10. Rich renderers.
11. Textual TUI behind an adapter boundary.
12. Extensions.
13. Compaction and context management.
14. Packaging, docs, and examples.

## Phase 0 deliverables

Phase 0 creates the docs, package scaffold, development checks, and a basic `tau --version` command.
