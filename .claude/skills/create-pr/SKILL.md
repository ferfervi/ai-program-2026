---
name: create-pr
description: Open a pull request for the current branch. Defaults to targeting `main`, but accepts a different base branch when the user specifies one (e.g. "open a PR against branch pre-session-04"). Runs tests locally first, pushes the branch if needed, derives the PR title/body from the diff vs. the base, then opens the PR with `gh pr create`. Use when the user asks to "open a PR", "create a pull request", or "ship this branch".
---

# create-pr

Open a pull request for the current branch against a chosen base branch.

## Resolving the base branch

- Default base: `main`.
- If the user mentions a target branch (phrases like "against branch X", "into X", "targeting X", "base X"), use that as the base. Examples:
  - "open a PR against branch pre-session-04" → base = `pre-session-04`
  - "create a PR into develop" → base = `develop`
- Before continuing, verify the base branch exists on the remote: `git ls-remote --exit-code --heads origin <base>`. If it does not exist, stop and ask the user to confirm the name.
- Throughout the rest of this skill, `<base>` refers to the resolved base branch.

## Preconditions to check (fail fast — do not auto-fix without asking)

1. Current branch is not `<base>` — you cannot PR a branch into itself.
2. Working tree has no uncommitted changes — if dirty, stop and ask the user how to proceed (commit / stash / abort).
3. At least one commit ahead of `origin/<base>` — otherwise there is nothing to PR.

## Steps

Run in parallel where independent:

1. `git status` (with no `-uall`), `git log --oneline origin/<base>..HEAD`, and `git diff origin/<base>...HEAD --stat` to understand the branch state and the full set of changes vs. the base.
2. Run the test suite locally: `uv run pytest -v`. If tests fail, stop and report — do not open a PR with red tests.
3. If the branch has no upstream, push with `git push -u origin HEAD`. Otherwise `git push`.
4. Draft the PR title (≤ 70 chars) and body from the **full commit range**, not just the latest commit. Body should follow the template below.
5. Open the PR with `gh pr create --base <base>` using a HEREDOC for the body. Return the PR URL to the user.

## PR body template

```
## Summary
- <1–3 bullets describing the change and why>

## Test plan
- [ ] `uv run pytest -v` passes locally
- [ ] CI (`.github/workflows/test.yml`) green
- [ ] <feature-specific manual checks, if any>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

## Guardrails

- Never force-push.
- Never push to `main` directly, regardless of the chosen `<base>`.
- Never skip hooks (`--no-verify`) or bypass signing.
- If the PR already exists for this branch (`gh pr view` succeeds), do not open a duplicate — surface the existing URL and ask whether to update it. Note `gh pr view` returns the PR for the current branch regardless of base, so an existing PR against a different base still counts.
