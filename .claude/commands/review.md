# Review Command

Review the code for: $ARGUMENTS

## Instructions

1. Run static analysis:
   ```bash
   uv run ruff check src/
   uv run ty check src/
   ```

2. Check for:
   - Type errors and missing annotations
   - Security issues (hardcoded secrets, injection vectors)
   - Dead code and unused imports
   - Missing or inaccurate docstrings
   - Test coverage gaps

3. Rate each finding: P0 (must fix) / P1 (should fix) / P2 (nice to have)

## Output

Provide a structured review with:
1. Summary of findings by severity
2. Specific file paths and line numbers
3. Suggested fixes for P0 and P1 items
