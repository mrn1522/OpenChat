import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from openai import AsyncOpenAI

from app.config import settings
from app.models import (
    AttachmentInput,
    DirectChatMessage,
    OpenRouterModel,
    OpenRouterModelsResponse,
    PersonaAssignment,
    SourceResult,
)
from app.prompting import (
    build_persona_generation_prompt,
    build_source_system_prompt,
)

WEB_SEARCH_TOOLS = [
    {
        "type": "openrouter:web_search",
        "parameters": {
            "engine": "auto",
            "max_results": 5,
            "max_total_results": 15,
        },
    },
    {
        "type": "openrouter:web_fetch",
        "parameters": {
            "max_content_tokens": 10000,
        },
    },
]

OPENROUTER_TOKEN_LIMIT = 65000
SOURCE_EMPTY_RETRY_COUNT = 2
OPENROUTER_TOOL_LOOP_MAX_STEPS = 8
PROMPT_OPTIMIZER_MODEL = "openai/gpt-oss-20b"
PERSONA_GENERATOR_MODEL = "qwen/qwen3.7-max"
PERSONA_MIN_TEMPERATURE = 0.5
PERSONA_MAX_TEMPERATURE = 1.2

PROMPT_OPTIMIZER_SYSTEM_PROMPT = """You are a prompt optimization assistant.

You will receive one user prompt as plain text data. Improve it for clarity, structure, and downstream model performance while preserving the original intent and constraints.

Requirements:
- Keep the same language as the input unless explicitly asked otherwise.
- Do not add new facts, requirements, or assumptions.
- Do not remove explicit constraints.
- Improve specificity and organization.
- Return strict JSON only (no markdown, no prose, no code fences).

Return exactly this schema:
{
  "optimized_prompt": "<improved prompt text>"
}
"""

PERSONA_GENERATOR_SYSTEM_PROMPT = """You are a cognitive architecture assistant for a multi-agent LLM workflow.

You will receive:
1) the user's question, and
2) a list of source models.

Your job is to assign a unique combination of a Domain Persona (the "Who"), a dynamically derived Cognitive Reasoning Framework (the "How"), and an optimal configuration Temperature to each model. This ensures each model approaches the question using completely different underlying logical paths, resulting in independent errors rather than shared blind spots.

Requirements:
- CRITICAL: Do not just change the vocabulary or job titles (e.g., creating a "Manager" and a "Developer" who use the same underlying approach). You must force completely different structural logical rules for each agent.
- Dynamic Cognitive Frameworks: Tailor the cognitive styles directly to the problem domain of the user's prompt. Dynamically invent or apply distinct, non-overlapping reasoning frameworks. 
  * For analytical/logical tasks, generate styles focused on baseline axioms, empirical evidence collection, adversarial dismantling, or reverse-engineering failure states.
  * For creative, strategic, or human-centric tasks, generate styles focused on unrestricted lateral associations, psychological/empathetic dynamics, or brutal operational efficiency.
- Ensure no two models are assigned overlapping cognitive frameworks; their logical paths must diverge entirely to prevent correlated errors.
- Optimal Temperature Allocation: Assign an execution temperature bound strictly between 0.5 and 1.2 for each model. Scale this dynamically based on the chosen cognitive framework:
  * Low (0.5 - 0.7): For analytical, deductive, fact-checking, or highly structured frameworks requiring precise, deterministic execution.
  * Moderate (0.7 - 0.9): For evaluative, adversarial, stress-testing, operational, or human-centric reasoning where balanced divergence is required.
  * High (0.9 - 1.2): For generative, brainstorming, or radical lateral association frameworks where maximum novelty and exploration are necessary.
- Output strict JSON only (no markdown, no prose, no code fences).

Return exactly this schema:
{
  "personas": [
    {
      "model": "<model id from input>",
      "title": "<Format as: 'Domain Role [Cognitive Framework]', e.g., 'Financial Analyst [Inversion Thinker]'>",
      "temperature": <float between 0.5 and 1.2>,
      "description": "<Must explicitly dictate the logical framework constraints. Example: 'You must approach this strictly using a framework of [Cognitive Framework Name]. Completely disregard industry consensus; instead, you must...' >"
    }
  ]
}
"""


def build_client() -> AsyncOpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in server environment")

    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.openchat_timeout_seconds,
    )


async def fetch_openrouter_models() -> OpenRouterModelsResponse:
    base_url = settings.openai_base_url.rstrip("/")
    models_url = f"{base_url}/models"

    async with httpx.AsyncClient(timeout=settings.openchat_timeout_seconds) as client:
        response = await client.get(models_url)
        response.raise_for_status()

    payload = response.json()
    models = payload.get("data", []) if isinstance(payload, dict) else []
    parsed = [OpenRouterModel.model_validate(item) for item in models if isinstance(item, dict)]
    return OpenRouterModelsResponse(data=parsed)


def _build_prompt_with_attachments(prompt: str, attachments: list[AttachmentInput]) -> str:
    if not attachments:
        return prompt

    blocks: list[str] = [prompt.strip(), "", "Attached context files:"]
    for attachment in attachments:
        blocks.append(f"\n---\nFile: {attachment.name} ({attachment.content_type}, {attachment.size} bytes)\n")
        blocks.append(attachment.content.strip())
    return "\n".join(blocks).strip()


def _build_openrouter_extra_body(
    *,
    web_search_enabled: bool,
    reasoning_effort: str,
    reasoning_exclude: bool,
    allow_tools: bool = True,
) -> dict[str, Any]:
    extra_body: dict[str, Any] = {
        "reasoning": {
            "effort": reasoning_effort,
            "exclude": reasoning_exclude,
        },
    }

    if allow_tools:
        extra_body["max_tool_calls"] = 25
        extra_body["parallel_tool_calls"] = True

    if web_search_enabled and allow_tools:
        extra_body["tools"] = WEB_SEARCH_TOOLS

    if not allow_tools:
        extra_body["tools"] = []
        extra_body["max_tool_calls"] = 0
        extra_body["parallel_tool_calls"] = False

    return extra_body


def _extract_optimized_prompt(raw_content: str) -> str | None:
    cleaned = raw_content.strip()
    if not cleaned:
        return None

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    candidates = [cleaned]
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(cleaned[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        for key in ("optimized_prompt", "improved_prompt", "prompt"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def _extract_persona_assignments(raw_content: str, source_models: list[str]) -> list[PersonaAssignment] | None:
    cleaned = raw_content.strip()
    if not cleaned:
        return None

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    candidates = [cleaned]
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(cleaned[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if not isinstance(payload, dict):
            continue

        persona_items = payload.get("personas")
        if not isinstance(persona_items, list):
            continue

        parsed: list[PersonaAssignment] = []
        for item in persona_items:
            if not isinstance(item, dict):
                continue

            model = str(item.get("model", "")).strip()
            title = str(item.get("title", "")).strip()
            raw_temperature = item.get("temperature")
            description = str(item.get("description", "")).strip()
            try:
                temperature = float(raw_temperature)
            except (TypeError, ValueError):
                continue

            if not (
                PERSONA_MIN_TEMPERATURE <= temperature <= PERSONA_MAX_TEMPERATURE
            ):
                continue

            if not model or not title or not description:
                continue

            try:
                parsed.append(
                    PersonaAssignment(
                        model=model,
                        title=title[:120],
                        temperature=temperature,
                        description=description[:1200],
                    )
                )
            except Exception:  # noqa: BLE001
                continue

        if not parsed:
            continue

        parsed_by_model = {assignment.model: assignment for assignment in parsed}
        if not all(model in parsed_by_model for model in source_models):
            continue

        return [parsed_by_model[model] for model in source_models]

    return None


def _normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if not isinstance(item, dict):
                continue

            text = item.get("text")
            if isinstance(text, str) and text:
                chunks.append(text)

        return "\n".join(chunks).strip()

    return ""


def _extract_first_choice_message_content(response: Any, *, context: str) -> str:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"{context} returned no choices.")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise RuntimeError(f"{context} returned a choice without a message payload.")

    return _normalize_message_content(getattr(message, "content", ""))


def _normalize_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
    if not raw_tool_calls:
        return []

    normalized: list[dict[str, Any]] = []
    for raw_call in raw_tool_calls:
        call_payload: dict[str, Any]
        if hasattr(raw_call, "model_dump"):
            call_payload = raw_call.model_dump()
        elif isinstance(raw_call, dict):
            call_payload = raw_call
        else:
            raise RuntimeError(f"Unsupported tool call payload type: {type(raw_call)}")

        tool_call_id = call_payload.get("id")
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise RuntimeError("Tool call is missing required 'id'.")

        raw_function = call_payload.get("function")
        if hasattr(raw_function, "model_dump"):
            raw_function = raw_function.model_dump()
        if not isinstance(raw_function, dict):
            raise RuntimeError(f"Tool call '{tool_call_id}' is missing function payload.")

        tool_name = raw_function.get("name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise RuntimeError(f"Tool call '{tool_call_id}' is missing function name.")

        arguments = raw_function.get("arguments", "{}")
        if isinstance(arguments, (dict, list)):
            arguments = json.dumps(arguments)
        elif arguments is None:
            arguments = "{}"
        elif not isinstance(arguments, str):
            arguments = str(arguments)

        normalized.append(
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": arguments,
                },
                "_raw": call_payload,
            }
        )

    return normalized


def _extract_tool_result_content(tool_call: dict[str, Any]) -> str:
    tool_call_id = tool_call["id"]
    raw_payload = tool_call.get("_raw", {})

    candidate_values = [
        raw_payload.get("result"),
        raw_payload.get("output"),
        raw_payload.get("content"),
    ]

    function_payload = raw_payload.get("function")
    if isinstance(function_payload, dict):
        candidate_values.extend(
            [
                function_payload.get("result"),
                function_payload.get("output"),
                function_payload.get("content"),
            ]
        )

    for value in candidate_values:
        if value is None:
            continue
        if isinstance(value, str):
            return value
        return json.dumps(value)

    raise RuntimeError(
        f"Tool call '{tool_call_id}' did not include a result payload to forward back to the model."
    )


async def _run_chat_completion_with_tool_loop(
    *,
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    messages: list[dict[str, Any]],
    extra_body: dict[str, Any],
    max_steps: int = OPENROUTER_TOOL_LOOP_MAX_STEPS,
) -> str:
    history: list[dict[str, Any]] = [dict(message) for message in messages]

    for _ in range(max_steps):
        response = await client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=OPENROUTER_TOKEN_LIMIT,
            max_completion_tokens=OPENROUTER_TOKEN_LIMIT,
            messages=history,
            extra_body=extra_body,
        )

        message = response.choices[0].message
        content = _normalize_message_content(getattr(message, "content", ""))
        raw_tool_calls = getattr(message, "tool_calls", None)
        tool_calls = _normalize_tool_calls(raw_tool_calls)

        if not tool_calls:
            return content

        history.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tool_call["id"],
                        "type": tool_call["type"],
                        "function": tool_call["function"],
                    }
                    for tool_call in tool_calls
                ],
            }
        )

        for tool_call in tool_calls:
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": _extract_tool_result_content(tool_call),
                }
            )

    raise RuntimeError(
        f"Tool-calling loop exceeded {max_steps} steps for model '{model}' without final text content."
    )


async def run_single_model(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    temperature: float,
    web_search_enabled: bool,
    reasoning_effort: str,
    reasoning_exclude: bool,
    attachments: list[AttachmentInput],
    system_prompt: str | None = None,
) -> SourceResult:
    """Run a single source model with retry-on-empty and return a SourceResult.

    Always returns a SourceResult; never returns None.
    """
    start = time.perf_counter()
    try:
        composed_prompt = _build_prompt_with_attachments(prompt, attachments)
        extra_body = _build_openrouter_extra_body(
            web_search_enabled=web_search_enabled,
            reasoning_effort=reasoning_effort,
            reasoning_exclude=reasoning_exclude,
        )

        messages: list[dict[str, Any]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": composed_prompt})

        total_attempts = 1 + SOURCE_EMPTY_RETRY_COUNT

        for attempt_index in range(total_attempts):
            content = await _run_chat_completion_with_tool_loop(
                client=client,
                model=model,
                temperature=temperature,
                messages=messages,
                extra_body=extra_body,
            )
            if content.strip():
                elapsed = int((time.perf_counter() - start) * 1000)
                return SourceResult(model=model, content=content, status="ok", latency_ms=elapsed)

            if attempt_index == total_attempts - 1:
                elapsed = int((time.perf_counter() - start) * 1000)
                return SourceResult(
                    model=model,
                    content="",
                    status="error",
                    error=(
                        "Model returned empty content after "
                        f"{total_attempts} attempts."
                    ),
                    latency_ms=elapsed,
                )
    except Exception as exc:  # noqa: BLE001
        elapsed = int((time.perf_counter() - start) * 1000)
        return SourceResult(model=model, content="", status="error", error=str(exc), latency_ms=elapsed)

    # Defensive fallback: should be unreachable given the logic above, but
    # guarantees the function always returns a SourceResult (never None).
    elapsed = int((time.perf_counter() - start) * 1000)
    return SourceResult(
        model=model,
        content="",
        status="error",
        error="Model execution ended without a result.",
        latency_ms=elapsed,
    )


async def run_source_models(
    client: AsyncOpenAI,
    models: list[str],
    prompt: str,
    temperature: float,
    web_search_enabled: bool,
    reasoning_effort: str,
    reasoning_exclude: bool,
    attachments: list[AttachmentInput],
    temperature_by_model: dict[str, float] | None = None,
    system_prompt: str | None = None,
    system_prompt_by_model: dict[str, str] | None = None,
) -> AsyncGenerator[SourceResult, None]:
    semaphore = asyncio.Semaphore(settings.openchat_max_parallel_sources)

    async def runner(model_name: str) -> SourceResult:
        async with semaphore:
            resolved_system_prompt = (
                system_prompt_by_model.get(model_name)
                if system_prompt_by_model is not None
                else system_prompt
            )
            resolved_temperature = (
                temperature_by_model.get(model_name, temperature)
                if temperature_by_model is not None
                else temperature
            )
            return await run_single_model(
                client=client,
                model=model_name,
                prompt=prompt,
                temperature=resolved_temperature,
                web_search_enabled=web_search_enabled,
                reasoning_effort=reasoning_effort,
                reasoning_exclude=reasoning_exclude,
                attachments=attachments,
                system_prompt=resolved_system_prompt,
            )

    tasks = [asyncio.create_task(runner(model_name)) for model_name in models]

    for completed in asyncio.as_completed(tasks):
        yield await completed


async def run_markdown_model(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    web_search_enabled: bool,
    reasoning_effort: str,
    reasoning_exclude: bool,
) -> str:
    extra_body = _build_openrouter_extra_body(
        web_search_enabled=web_search_enabled,
        reasoning_effort=reasoning_effort,
        reasoning_exclude=reasoning_exclude,
    )

    messages: list[dict[str, Any]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt})

    return await _run_chat_completion_with_tool_loop(
        client=client,
        model=model,
        temperature=temperature,
        messages=messages,
        extra_body=extra_body,
    )


async def run_direct_chat_model(
    client: AsyncOpenAI,
    model: str,
    messages: list[DirectChatMessage],
    system_prompt: str,
    temperature: float,
    web_search_enabled: bool,
    reasoning_effort: str,
    reasoning_exclude: bool,
    attachments: list[AttachmentInput],
) -> str:
    normalized_messages: list[dict[str, Any]] = _build_direct_chat_messages(
        messages=messages,
        attachments=attachments,
        system_prompt=system_prompt,
    )

    extra_body = _build_openrouter_extra_body(
        web_search_enabled=web_search_enabled,
        reasoning_effort=reasoning_effort,
        reasoning_exclude=reasoning_exclude,
    )

    return await _run_chat_completion_with_tool_loop(
        client=client,
        model=model,
        temperature=temperature,
        messages=normalized_messages,
        extra_body=extra_body,
    )


def _build_direct_chat_messages(
    *,
    messages: list[DirectChatMessage],
    attachments: list[AttachmentInput],
    system_prompt: str | None,
) -> list[dict[str, Any]]:
    resolved_system_prompt = (system_prompt or "").strip() or build_source_system_prompt(None)

    normalized_messages: list[dict[str, Any]] = [{"role": "system", "content": resolved_system_prompt}]
    last_user_index = -1
    for message in messages:
        normalized_messages.append({"role": message.role, "content": message.content})
        if message.role == "user":
            last_user_index = len(normalized_messages) - 1

    if attachments and last_user_index >= 0:
        original = str(normalized_messages[last_user_index]["content"])
        normalized_messages[last_user_index]["content"] = _build_prompt_with_attachments(original, attachments)

    return normalized_messages


async def optimize_prompt_text(client: AsyncOpenAI, prompt: str) -> str:
    response = await client.chat.completions.create(
        model=PROMPT_OPTIMIZER_MODEL,
        temperature=0.1,
        max_tokens=OPENROUTER_TOKEN_LIMIT,
        max_completion_tokens=OPENROUTER_TOKEN_LIMIT,
        messages=[
            {"role": "system", "content": PROMPT_OPTIMIZER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        extra_body={
            "reasoning": {
                "effort": "low",
                "exclude": True,
            }
        },
    )

    content = _extract_first_choice_message_content(response, context="Prompt optimizer")
    optimized = _extract_optimized_prompt(content)
    return optimized or prompt.strip()


async def generate_persona_assignments(
    client: AsyncOpenAI,
    user_prompt: str,
    source_models: list[str],
) -> list[PersonaAssignment]:
    if not source_models:
        return []

    user_content = build_persona_generation_prompt(user_prompt, source_models)

    response = await client.chat.completions.create(
        model=PERSONA_GENERATOR_MODEL,
        temperature=0.1,
        max_tokens=OPENROUTER_TOKEN_LIMIT,
        max_completion_tokens=OPENROUTER_TOKEN_LIMIT,
        messages=[
            {"role": "system", "content": PERSONA_GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        extra_body=_build_openrouter_extra_body(
            web_search_enabled=False,
            reasoning_effort="high",
            reasoning_exclude=True,
            allow_tools=False,
        ),
    )

    content = _extract_first_choice_message_content(response, context="Persona generator")
    assignments = _extract_persona_assignments(content, source_models)
    if assignments is None:
        raise RuntimeError("Persona generator returned invalid or incomplete persona assignments.")

    return assignments
