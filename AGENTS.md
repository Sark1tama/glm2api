# Repository Guidelines

## Project Structure & Module Organization

Application code lives in `src/glm2api/`. HTTP startup and routing are handled by `app.py`, `server.py`, and `__main__.py`; configuration and logging have dedicated modules. Protocol translation and upstream integration belong under `src/glm2api/services/`, while reusable parsing helpers live in `src/glm2api/utils/`. Keep the root `main.py` as a thin launcher. Tests are in `tests/` and generally mirror a capability, such as `test_translator.py` or `test_video.py`. Runtime state belongs in `data/`; protocol notes belong in `docs/`.

## Build, Test, and Development Commands

- `uv sync` creates the Python 3.14 environment from `pyproject.toml` and `uv.lock`.
- `uv run python main.py` starts the local API server (port `8000` by default).
- `uv run pytest` runs the complete test suite; add a path or `-k expression` for a focused run.
- `uv build` creates source and wheel distributions.
- `docker compose up --build` builds and runs the containerized service. The compose file expects the external `shared-net` Docker network.

After startup, verify the service with `curl http://127.0.0.1:8000/health`.

## Coding Style & Naming Conventions

Use four-space indentation and standard Python conventions: `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Prefer type hints on public boundaries and small, focused adapters over cross-module abstractions. Match the existing import grouping and keep protocol-specific logic in the relevant service module. No formatter or linter is currently configured, so avoid unrelated formatting churn.

## Testing Guidelines

Tests use `pytest` with `src` added to `PYTHONPATH` by project configuration. Name files `test_<feature>.py` and functions `test_<behavior>()`. Add regression tests for protocol conversion, streaming envelopes, parser edge cases, and error handling. Tests should be deterministic and should mock upstream ChatGLM traffic rather than require credentials or network access.

## Commit & Pull Request Guidelines

Recent history uses concise Conventional Commit-style subjects, for example `feat: add ChatGLM 5.3 media and web search support`. Use an imperative subject with an appropriate prefix such as `feat:`, `fix:`, `test:`, or `docs:`. Pull requests should explain the behavior change, list verification commands, link related issues, and include sample request/response payloads for API changes. Update README or protocol documentation when public behavior changes.

## Security & Configuration

Copy `.env.example` to `.env`; never commit `.env`, `token.txt`, refresh tokens, API keys, or generated logs. Keep secrets out of test fixtures and issue reports.
