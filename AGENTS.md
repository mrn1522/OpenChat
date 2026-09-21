# Agent Rules

## Rule: CodeRabbit PR Feedback Loop

### ACTIVATION CONDITION

Apply the following instructions ONLY when ALL of the following are true:

1. You are actively working on an open GitHub Pull Request.
2. You observe comments from `coderabbitai[bot]` or the repository root contains `.coderabbit.yaml`.
3. You are explicitly tasked with addressing PR review feedback.

Otherwise, completely ignore this rule.

### BEHAVIOR

- Apply this behavior only during Devin sessions.
- Read inline suggestions and warnings posted by @coderabbitai on the PR diff.
- Address legitimate runtime bugs, missing null checks, security flaws, and runtime regressions in a new commit. Ignore stylistic nitpicks and subjective refactoring suggestions.
- Never engage in conversational chat with @coderabbitai. Only use commands:
  - If you need to stop review noise during multi-step edits, comment `@coderabbitai pause`.
  - When code is ready, comment `@coderabbitai resume` and `@coderabbitai review` (incremental review of new changes).
  - After a force-push or major rewrite, comment `@coderabbitai full review` to re-review the entire diff from scratch.
  - If the PR description changed, comment `@coderabbitai summary` to regenerate CodeRabbit's PR summary.
  - To permanently disable auto-reviews on a PR, put `@coderabbitai ignore` in the PR description (not a comment). Prefer `pause` for temporary silence.
  - When unsure which command applies, comment `@coderabbitai help`.
  - To approve the PR via CodeRabbit, comment `@coderabbitai approve` — requires `reviews.request_changes_workflow: true` in `.coderabbit.yaml`, otherwise it is ignored.
  - Once all actionable feedback has been resolved, comment `@coderabbitai resolve` (top-level comment; resolves all review threads).
- Do not trigger CodeRabbit content generation (`@coderabbitai generate unit tests`, `generate docstrings`, or the finishing-touches checkboxes) unless the user asks.
- Once resolved, stop execution and hand control back to the user.

## Rule: Branch & Review Flow (develop -> main)

### ACTIVATION CONDITION

Apply when creating or updating a pull request in this repository.

### BEHAVIOR

- Feature/improvement PRs target `develop`. CodeRabbit and Devin Review auto-review every push; fix findings in follow-up commits — the review/fix loop on develop is expected and free (no CI build runs).
- `develop` -> `main` merge PRs are **single-pass** — neither bot supports per-base-branch policies, so enforce it per PR:
  1. Open the PR as a **draft** and include `@coderabbitai ignore` in the description so CodeRabbit does not auto-review each push.
  2. Resolve outstanding feedback while the PR is a draft; the desktop build never runs in draft.
  3. When the diff is settled, comment `@coderabbitai review` for the one CodeRabbit pass (optional if develop-side reviews already covered it).
  4. Mark the PR **ready for review** — this fires the desktop build once and triggers Devin Review's auto-review.
  5. Do not push fixup commits after ready unless required — every push re-triggers the build and re-review. If large changes are needed, convert back to draft first.
  6. Human approval is the final merge gate.
