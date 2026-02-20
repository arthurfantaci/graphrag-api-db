# Contributing

Thank you for your interest in contributing to the GraphRAG Knowledge Graph Pipeline.

## Development Setup

### Prerequisites

- **Python 3.13+**
- **[UV](https://docs.astral.sh/uv/)** (recommended package manager)
- **Neo4j 5.x** with APOC plugin (for integration tests)

### Getting Started

```bash
# Clone the repository
git clone https://github.com/arthurfantaci/graphrag-api-db.git
cd graphrag-api-db

# Install all dependencies including dev tools
uv sync --group dev

# Install pre-commit hooks
pre-commit install

# Copy and configure environment variables
cp .env.example .env
```

## Code Quality Standards

This project enforces strict quality gates via pre-commit hooks and CI.

### Linting & Formatting

[Ruff](https://docs.astral.sh/ruff/) handles both linting and formatting with 19 rule categories enabled, including security checks (Bandit) and documentation enforcement (pydocstyle).

```bash
uv run ruff check .          # Lint
uv run ruff check . --fix    # Lint with auto-fix
uv run ruff format .         # Format
```

### Type Checking

[ty](https://github.com/astral-sh/ty) provides type checking for the codebase.

```bash
uv run ty check src/
```

### Docstrings

All public functions require [Google-style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstrings with `Args`, `Returns`, and `Raises` sections where applicable.

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run with coverage report
uv run pytest --cov=graphrag_kg_pipeline --cov-report=term-missing

# Run a specific test file
uv run pytest tests/test_extraction.py
```

## Git Workflow

This project follows a structured branching workflow:

1. **Create an issue** describing the change
2. **Create a feature branch** from `main`:
   - `feat/<description>` for new features
   - `fix/<description>` for bug fixes
   - `chore/<description>` for maintenance tasks
3. **Commit with conventional commits** (e.g., `feat:`, `fix:`, `chore:`)
4. **Open a pull request** against `main`
5. **Ensure CI passes** — linting, type checking, and tests must all succeed
6. **Request review** — all PRs require review before merge

### Commit Message Format

```
<type>: <short description>

[Optional body with additional context]
```

Types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`, `perf`

## Project Architecture

See [CLAUDE.md](CLAUDE.md) for a detailed architecture overview including the 5-stage pipeline, module descriptions, and key design decisions.

### Source Layout

The project uses the [src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) recommended by PyPA:

```
src/graphrag_kg_pipeline/    # Main package
tests/                       # Test suite
examples/                    # Usage demonstrations
```

## Reporting Issues

Use the [issue templates](https://github.com/arthurfantaci/graphrag-api-db/issues/new/choose) for bug reports and feature requests.
