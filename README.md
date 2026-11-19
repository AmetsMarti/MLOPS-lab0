# MLOps Lab 0

Simple data preprocessing logic with a Click CLI, plus linting, formatting, and tests.

## Setup
```bash
uv sync
```

## CLI usage
Run commands with:
```bash
uv run python -m src.cli --help
```
Examples:
- `uv run python -m src.cli clean remove-missing '[1, null, "", "a"]'`
- `uv run python -m src.cli clean fill-missing '[1, null, ""]' --value 0`
- `uv run python -m src.cli numeric normalize '[1,2,3]' --min 0 --max 1`
- `uv run python -m src.cli text remove-stop-words "the quick brown fox" --stop-words '["the","fox"]'`
- `uv run python -m src.cli struct unique-values '[1,2,2,3]'`

## Quality checks
```bash
uv run python -m pylint src tests
uv run black src tests
uv run python -m pytest -v --cov=src
```
