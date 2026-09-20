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
