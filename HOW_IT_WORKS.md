# OpenChat — How It Works

OpenChat is a **Mixture-of-Agents (MoA)** chat app. One user prompt is sent to several independent LLM "source" models, their answers are stress-tested in a debate round, and a final synthesis model fuses everything into one answer. The whole pipeline streams to the UI in stages.

---

## The Flow (end to end)

```
User prompt
   │
   ▼
1. (Optional) Auto-Optimize  ── rewrite the prompt via gpt-oss-20b, user confirms ✓/✕
   │
   ▼
2. (Optional) Persona assignment ── gemma-4-31b-it gives each source agent a unique
   │                              persona + temperature; user reviews ✓/✕
   ▼
3. Source models answer IN PARALLEL  ── each with its own persona/temperature
   │                                  (independent, no awareness of each other)
   ▼
4. Debate / critique round ── each source answer is reviewed against its peers
   │                          (reviewer drops its persona, acts as a fair judge)
   ▼
5. Synthesis (fusion) model ── rebuilds one best answer from sources + debate notes
   │
   ▼
Final Markdown answer  +  saved to SQLite history
```

---

## Step-by-step description

### 1. Auto-Optimize (optional, pre-send)
Before anything runs, the prompt can be rewritten by `openai/gpt-oss-20b` (reasoning disabled) for clarity. The rewritten prompt is shown to the user, who either accepts (✓ → sends immediately) or rejects (✕ → restores original and waits for Send).

### 2. Persona assignment (optional)
If Personas are enabled, OpenChat calls `google/gemma-4-31b-it` (low reasoning, no tools) to assign **a unique persona to each selected source model**, based on the current question. Each persona carries:
- a **title** and **guidance description** (the perspective the model should adopt), and
- a **temperature** (0.5–1.2) used for that model's source run.

The generated personas appear in editable text boxes (one per source model) and are gated by the same ✓/✕ review controls:
- ✓ → sends the reviewed prompt **and** reviewed personas,
- ✕ → sends the original prompt with personas disabled for that run.

If personas are off, every source model uses the run's global temperature and a plain "helpful assistant" system prompt.

### 3. Source models answer in parallel
The prompt is fanned out to all selected source models **in parallel and independently**. Each model:
- gets its own system prompt (with its persona block appended, if enabled),
- uses its persona temperature (or the global temperature if no persona),
- may use web search / reasoning per the run settings,
- streams back a `source_result` (status `ok` or `error`).

At this stage the models have **no knowledge of each other** — they each answer the original question on their own. Failed models are isolated and surfaced individually without breaking the run.

### 4. Debate / critique round
Once source answers are in, OpenChat pairs them up for review. The pairing depends on `debate_mode`:

- **off** — skip debate entirely, go straight to fusion.
- **partial** (default) — a round-robin: each answer is reviewed by the *next* agent (agent N reviews agent N-1, wrapping around).
- **full** — every agent reviews every other agent (all non-self pairs).

If there's only one source agent, the **fusion model** itself acts as the reviewer.

For each pair, a **debate prompt** is built containing:
- the original question,
- the **target response** (the one being reviewed), wrapped as untrusted data,
- the **peer responses** (all other source answers), also wrapped as untrusted data.

The reviewer is given the `DEBATE_SYSTEM_PROMPT`, which makes it an **elite logical evaluator** that must:
1. **Steelman** the target — restate its argument in the strongest, most charitable form before being allowed to criticize.
2. **Critique** — then systematically dismantle that steelman: attack core logic, expose hidden assumptions, find catastrophic edge cases.

Key point about personas here: **the reviewer does not retain its source persona during debate.** The debate system prompt replaces the persona — the reviewer acts as a neutral, fair (but ruthless) judge, not as its assigned character. It reviews the target answer *with the peers' initial responses in mind*, using them only to cross-check quality, consistency, and missing points.

Each review is streamed as a `debate_result` and all reviews are compiled into a single **Debate Report** (markdown), emitted as the `critique_ready` event.

### 5. Synthesis (fusion)
Finally the **fusion/synthesis model** takes over. It receives:
- the original question,
- all successful source responses (as untrusted data),
- the full debate report (as untrusted data).

Guided by `FUSION_SYSTEM_PROMPT`, the synthesizer must **not** just copy or average the sources. Instead it:
- reconstructs a single superior answer **from first principles**,
- patches the flaws the debate uncovered,
- **definitively resolves** every disagreement (no "it depends" / weak compromises),
- applies a **burden of proof** to critiques — rejecting weak attacks and upholding solid original claims,
- covers blind spots and partially-covered points,
- never mentions the orchestration/debate/synthesis process.

The result is streamed as `fusion_ready` and the run closes with a `completed` event. Everything (prompt, source outputs, debate critique, fusion answer, personas) is persisted to a local SQLite history DB so any chat can be reopened later.

---

## One-line summary
> Independent source models (optionally persona'd) answer in parallel → a persona-free debate round steelmans and critiques each answer against its peers → a synthesis model rebuilds one definitive answer from first principles.
