# PR Workflow

Guided pull request creation and merge following professional git workflow best practices.

Use when the user says `/pr-workflow`, "create a PR", "open a pull request", or "merge a PR".

## Determine workflow type

Ask: "Is this a simple PR (one branch, one issue) or part of a stacked set?"

---

## Simple PR (one branch → one issue)

### 1. Identify the associated issue(s)

```bash
git log --oneline origin/main..HEAD
gh issue list --state open --limit 10
```

Ask: "Which GitHub issue(s) does this PR close?"

### 2. Gather PR context

Run in parallel:
```bash
git status
git diff origin/main --stat
git log --oneline origin/main..HEAD
```

### 3. Create the PR

**Title:** `<type>: brief description` (under 70 chars)
Types: `fix:`, `feat:`, `refactor:`, `docs:`, `test:`, `chore:`

```bash
gh pr create --title "<type>: brief description" --body "$(cat <<'EOF'
Closes #<issue_number>

## Summary
- Bullet point changes (2-4 bullets)

## Test plan
- [ ] Tests pass (`uv run pytest`)
- [ ] Lint clean (`uv run ruff check`)
- [ ] Type check clean (`uv run ty check src/`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

### 4. Post-creation verify

```bash
gh pr view --json title,body,url
```

### 5. Merge — choose the right strategy

Recommend based on situation:

| Strategy | Command | When to use |
|---|---|---|
| **Squash** | `gh pr merge <n> --squash` | Most simple PRs — clean 1-commit history |
| **Merge commit** | `gh pr merge <n> --merge` | When commit-by-commit history matters |
| **Rebase** | `gh pr merge <n> --rebase` | Linear history preference, small PRs |

**Default recommendation for this project**: `--squash` for simple PRs.

### 6. Post-merge verification (ALWAYS do this)

```bash
# Verify the issue auto-closed
gh issue view <N> --json state --jq .state

# If NOT closed (shows "OPEN"), close manually
gh issue close <N> --comment "Delivered in PR #<pr_number>"

# Clean up local branch
git checkout main && git pull
git branch -D <branch-name>
```

---

## Stacked PRs (multiple dependent branches)

Use when work spans multiple issues and each phase depends on the previous.

### 1. Create branches in sequence

```bash
git checkout main && git pull
git checkout -b phase-1
# ... work, commit ...
git checkout -b phase-2   # branches from phase-1
# ... work, commit ...
```

### 2. Create PRs — each with its own Closes keyword

Each PR body MUST close its own issue(s):

```bash
# PR for phase 1
gh pr create --title "refactor: phase 1 work" --body "$(cat <<'EOF'
Closes #10

## Summary
...
EOF
)"

# PR for phase 2 (targets phase-1 branch, NOT main)
gh pr create --base phase-1 --title "refactor: phase 2 work" --body "$(cat <<'EOF'
Closes #11

## Summary
...
EOF
)"
```

### 3. Choose merge strategy for the stack

#### Option A: Squash merge (clean history, manual rebase between each)

After squash-merging PR #1, rebase downstream:

```bash
git fetch origin main
git checkout -B phase-2 origin/main
git cherry-pick <phase-2-specific-commit-sha>
git push --force-with-lease origin phase-2
gh pr edit <pr-number> --base main
```

#### Option B: Merge commits (preserves branch topology, no rebase needed)

```bash
gh pr merge <n> --merge
```

#### Option C: Collapse the stack into one PR

```bash
gh pr close <intermediate-pr> --comment "Collapsing into single PR #<final>"
gh pr edit <final-pr> --body "$(cat <<'EOF'
Closes #10, Closes #11, Closes #12

## Summary
...
EOF
)"
gh pr merge <final-pr> --squash
```

### 4. Post-merge verification (ALWAYS do this for every PR in the stack)

```bash
for issue in 10 11 12; do
  echo "#$issue: $(gh issue view $issue --json state --jq .state)"
done

# Clean up local branches
git checkout main && git pull
git branch -D phase-1 phase-2 phase-3
```

---

## Closing keyword rules (both workflows)

**CRITICAL** — these auto-close issues when the PR merges:
- `Closes #N`, `Fixes #N`, or `Resolves #N`
- Multiple: `Closes #10, Closes #11`
- Must be in the PR **body** (not title, not comments)

**What does NOT work:**
- `(#42)` in title — just a mention/hyperlink
- Closing keyword in a PR comment added later
- Closing keyword in commit messages on non-default branches

No associated issue? Write "No associated issue" in the body (bypasses the hook).

---

## CI/CD awareness

Before merging, always verify CI status:
```bash
gh pr view <n> --json statusCheckRollup --jq '[.statusCheckRollup[] | {name, status, conclusion}]'
```

If CI fails, diagnose before merging. Never skip failing checks.
