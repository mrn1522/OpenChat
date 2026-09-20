# Review Guidelines

Treat every PR as a deep audit, not a surface pass. Beyond Bug Catcher defaults, actively hunt for security issues, edge cases, performance regressions, and correctness violations. Label low-confidence findings as flags or questions — do not assert breakage without evidence.

## Security

- Flag hardcoded secrets, tokens, API keys, or credentials in any file (including `.env.example` — it must only contain placeholders).
- Flag unsafe DOM/HTML injection: `dangerouslySetInnerHTML`, or unsanitized data interpolated into markup. Model outputs are untrusted input — check how `react-markdown` and remark plugins handle raw HTML and links (`javascript:` URLs, `target="_blank"` without `rel="noopener"`).
- Flag `eval`, `new Function`, and other dynamic code execution.
- External input (URL params, user prompts, `localStorage`, network responses, streamed model output) must be validated or sanitized before use.
- Flag weakened security controls: disabled linters, skipped checks, relaxed TypeScript strictness, broadened `CORS_ALLOW_ORIGINS` (must never default to `*` with credentials).
- New dependencies: flag unused, unscoped, or unpinned packages.

## Edge Cases & Correctness

- Per-model failure isolation: one source model erroring or timing out must not block or corrupt the other streams, the judge, or the synthesizer.
- Flag unhandled promise rejections, missing error handling on fetch/IO, and races on rapid state changes (double Send, model reselection mid-stream, abort during streaming).
- Check boundary conditions: empty/no models selected, `OPENCHAT_MAX_PARALLEL_SOURCES` cap, `OPENCHAT_TIMEOUT_SECONDS` expiry, empty or truncated model responses.
- Stream lifecycle: abort/cancel must actually stop upstream requests and update UI state; judge/synthesizer must not run on an empty or failed source set without user-visible signaling.
- Optimize/personas gate: the ✓ / ✕ confirm flow must not send stale prompts, personas, or optimized text after edits.

## Performance

- Streaming: per-token SSE/fetch updates must not cause unbounded React re-renders — flag missing batching/throttling and per-chunk object allocation in the hot path.
- Long transcripts must stay virtualized; flag unbounded list growth or dropping virtualization.
- Fan-out must respect the parallel-sources limit; flag unbounded concurrency or missing timeouts/cancellation on hung requests.
- Event listeners, timers, `AbortController`s, and stream readers must have teardown; flag leaks (added but never removed).

## Code Quality

- TypeScript strict: flag `any`, unsafe casts, and missing null checks on DOM/element lookups.
- Python: flag untyped or loosely-typed changes to request/response models — new fields and endpoints should use Pydantic models.
- The repo has no lint/test tooling yet; do not assume `npm run lint` or `pytest` exist — flag clearly-broken syntax/types directly instead of deferring to tooling.

## Scope

- Ignore generated output (`dist/`, `*.min.*`, `__pycache__/`) and lockfile-only formatting or ordering churn. Review dependency version, resolution, and integrity changes under the deep-audit and security rules above.
- Markdown-only changes (`docs/`, `*.md`) need only a light pass, except `AGENTS.md` and `REVIEW.md`, which require the deep audit above.
