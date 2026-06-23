# Contributing to AeroOpti

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -e . -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- Follow PEP 8.
- Use type hints for function signatures.
- Keep imports sorted: stdlib, third-party, local.

## Pull Requests

1. Create a feature branch from `main`.
2. Make your changes and ensure all tests pass.
3. Open a pull request with a clear description of what changed and why.
