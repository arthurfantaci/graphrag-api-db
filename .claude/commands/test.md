# Test Command

Run tests for: $ARGUMENTS

## Instructions

1. Run the full test suite:
   ```bash
   uv run pytest tests/ -v --tb=short
   ```

2. If a specific area is mentioned, run targeted tests:
   ```bash
   uv run pytest tests/test_<area>.py -v
   ```

3. Check coverage:
   ```bash
   uv run pytest --cov=graphrag_kg_pipeline --cov-report=term-missing
   ```

## Output

Provide:
1. Test results (pass/fail counts)
2. Any failures with root cause analysis
3. Coverage summary if requested
