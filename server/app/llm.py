import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, TypeVar

import httpx
from openai import AsyncOpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from app.config import settings
from app.models import (
    AttachmentInput,
    DirectChatMessage,
    OpenRouterModel,
    OpenRouterModelsResponse,
    PersonaAssignment,
    SourceAgentSpec,
    SourceResult,
)
from app.prompting import (
    build_persona_generation_prompt,
    build_source_system_prompt,
)

T = TypeVar("T")

WEB_SEARCH_TOOLS = [
    {
        "type": "openrouter:web_search",
        "parameters": {
            "engine": "auto",
            "max_results": 5,
            "max_total_results": 10,
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

# Bounded silent retry for synthesis-style markdown calls (fusion + debate).
# A single retry absorbs transient OpenRouter empty-choice / transport failures
# before surfacing an actionable, SSE-safe error to the client.
MARKDOWN_MODEL_RETRY_COUNT = 1
MARKDOWN_MODEL_RETRY_BACKOFF_SECONDS = 1.5

logger = logging.getLogger(__name__)

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
2) a list of source agents, each identified by a unique agent_id and a model.

Your job is to assign a unique combination of a Domain Persona (the "Who"), a dynamically derived Cognitive Reasoning Framework (the "How"), and an optimal configuration Temperature to each agent. This ensures each agent approaches the question using completely different underlying logical paths, resulting in independent errors rather than shared blind spots.

CRITICAL: Multiple agents may share the same underlying model. You MUST still produce exactly one persona per agent, and you MUST key each persona by its agent_id (NOT by model). Never collapse or merge agents that share a model.

Requirements:
- CRITICAL: Do not just change the vocabulary or job titles (e.g., creating a "Manager" and a "Developer" who use the same underlying approach). You must force completely different structural logical rules for each agent.
- Dynamic Cognitive Frameworks: Tailor the cognitive styles directly to the problem domain of the user's prompt. Dynamically invent or apply distinct, non-overlapping reasoning frameworks.
  * For analytical/logical tasks, generate styles focused on baseline axioms, empirical evidence collection, adversarial dismantling, or reverse-engineering failure states.
  * For creative, strategic, or human-centric tasks, generate styles focused on unrestricted lateral associations, psychological/empathetic dynamics, or brutal operational efficiency.
- Ensure no two agents are assigned overlapping cognitive frameworks; their logical paths must diverge entirely to prevent correlated errors.
- Optimal Temperature Allocation: Assign an execution temperature bound strictly between 0.5 and 1.2 for each agent. Scale this dynamically based on the chosen cognitive framework:
  * Low (0.5 - 0.7): For analytical, deductive, fact-checking, or highly structured frameworks requiring precise, deterministic execution.
  * Moderate (0.7 - 0.9): For evaluative, adversarial, stress-testing, operational, or human-centric reasoning where balanced divergence is required.
  * High (0.9 - 1.2): For generative, brainstorming, or radical lateral association frameworks where maximum novelty and exploration are necessary.
- Output strict JSON only (no markdown, no prose, no code fences).
- Return exactly one persona entry per input agent, in the same order as the input agents.

Return exactly this schema:
{
  "personas": [
    {
      "agent_id": "<agent_id from input>",
      "model": "<model id from input>",
      "title": "<Format as: 'Domain Role [Cognitive Framework]', e.g., 'Financial Analyst [Inversion Thinker]'>",
      "temperature": <float between 0.5 and 1.2>,
      "description": "<Must explicitly dictate the logical framework constraints. Example: 'You must approach this strictly using a framework of [Cognitive Framework Name]. Completely disregard industry consensus; instead, you must...' >"
    }
  ]
}
"""


# One long-lived connection pool is shared by the OpenAI SDK and the models
# endpoint: previously every request built a fresh AsyncOpenAI/httpx client,
# paying TCP+TLS setup per call and leaking an unclosed pool each time.
_shared_http_client: httpx.AsyncClient | None = None
_openai_client: AsyncOpenAI | None = None
_openai_client_key: tuple[str, str, float] | None = None

_MODELS_CACHE_TTL_SECONDS = 300.0
# (base_url, fetched_at, response): the catalog belongs to the provider it was
# fetched from — after a base_url change it is neither fresh nor a valid
# stale fallback.
_models_cache: tuple[str, float, OpenRouterModelsResponse] | None = None
_models_cache_lock = asyncio.Lock()

_SERVICE_TIERS_CACHE_TTL_SECONDS = 300.0
# (base_url, model_id) -> (fetched_at, tiers). Endpoint rosters change rarely;
# on a refresh failure the entry is served stale instead of erroring.
_service_tiers_cache: dict[tuple[str, str], tuple[float, list[str]]] = {}
_service_tiers_lock = asyncio.Lock()

# Tier suffixes on provider endpoint tags (`openai/flex`,
# `google-vertex/global/priority`). "fast" is OpenAI's rename of the priority
# tier and is reported back as "priority", so it normalizes here.
_ENDPOINT_TAG_TIER_BY_SUFFIX = {"flex": "flex", "fast": "priority", "priority": "priority"}


def _shared_http() -> httpx.AsyncClient:
    global _shared_http_client
    if _shared_http_client is None:
        # Match the SDK's DefaultAsyncHttpxClient (follow_redirects + its
        # connection limits) so injecting this client changes pooling only,
        # not redirect or connection-limit behavior.
        _shared_http_client = httpx.AsyncClient(
            timeout=settings.openchat_timeout_seconds,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100),
        )
    return _shared_http_client


def build_client() -> AsyncOpenAI:
    """Return the shared AsyncOpenAI client, rebuilding when settings change.

    Reuses one connection pool across all requests. A settings change rebuilds
    only the thin SDK wrapper; the shared pool persists until app shutdown
    (``aclose_clients``).
    """
    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in server environment")

    global _openai_client, _openai_client_key
    key = (
        settings.openai_api_key,
        settings.openai_base_url,
        settings.openchat_timeout_seconds,
    )
    if _openai_client is not None and _openai_client_key == key:
        return _openai_client

    _openai_client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.openchat_timeout_seconds,
        http_client=_shared_http(),
    )
    _openai_client_key = key
    return _openai_client


async def aclose_clients() -> None:
    """Close the shared connection pool (app shutdown)."""
    global _shared_http_client, _openai_client, _openai_client_key
    client = _shared_http_client
    _shared_http_client = None
    _openai_client = None
    _openai_client_key = None
    if client is not None:
        await client.aclose()


async def fetch_openrouter_models() -> OpenRouterModelsResponse:
    """Fetch the model catalog, cached for ``_MODELS_CACHE_TTL_SECONDS``.

    The catalog is large (hundreds of KB) and changes rarely; on a refresh
    failure a stale cached response is served instead of erroring.
    """
    base_url = settings.openai_base_url.rstrip("/")
    models_url = f"{base_url}/models"

    global _models_cache

    def _stale() -> OpenRouterModelsResponse | None:
        """The cached catalog, only when it came from the current base_url."""
        if _models_cache is not None and _models_cache[0] == base_url:
            return _models_cache[2]
        return None

    now = time.monotonic()
    cached = _stale()
    if cached is not None and now - _models_cache[1] < _MODELS_CACHE_TTL_SECONDS:
        return cached

    async with _models_cache_lock:
        # Re-check inside the lock: a concurrent caller may have refreshed.
        now = time.monotonic()
        cached = _stale()
        if cached is not None and now - _models_cache[1] < _MODELS_CACHE_TTL_SECONDS:
            return cached

        try:
            response = await _shared_http().get(
                models_url,
                timeout=settings.openchat_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            # ValueError covers JSON decoding, shape mismatches, and pydantic
            # model validation: a malformed or empty 200 triggers the same
            # stale fallback as a 5xx instead of replacing the good cache
            # with an empty catalog.
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, list):
                raise ValueError("Models catalog missing 'data' list")
            result = OpenRouterModelsResponse(
                data=[
                    OpenRouterModel.model_validate(item)
                    for item in data
                    if isinstance(item, dict)
                ]
            )
            if not result.data:
                raise ValueError("Models catalog contains no usable models")
        except (httpx.HTTPError, ValueError):
            if cached is not None:
                logger.warning(
                    "OpenRouter models refresh failed; serving stale catalog.",
                    exc_info=True,
                )
                return cached
            raise

        _models_cache = (base_url, time.monotonic(), result)
        return result


def _extract_service_tiers(payload: Any) -> list[str]:
    """Collect non-default service tiers from a model's endpoint roster.

    Tier-capable provider endpoints carry a tag suffix (``openai/flex``,
    ``openai/fast``, ``google-vertex/global/priority``); regional and
    quantization suffixes (``/us-east5``, ``/fp8``) are not tiers.
    """
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    endpoints = data.get("endpoints") if isinstance(data, dict) else None
    if not isinstance(endpoints, list):
        return []

    tiers: set[str] = set()
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        tag = endpoint.get("tag")
        if not isinstance(tag, str) or "/" not in tag:
            continue
        tier = _ENDPOINT_TAG_TIER_BY_SUFFIX.get(tag.rsplit("/", 1)[1])
        if tier is not None:
            tiers.add(tier)

    return [tier for tier in ("flex", "priority") if tier in tiers]


async def fetch_model_service_tiers(model_id: str) -> list[str]:
    """Fetch the non-default service tiers available for ``model_id``.

    Reads the model's provider endpoint list (same host as the model
    catalog). Cached per (base_url, model_id) for
    ``_SERVICE_TIERS_CACHE_TTL_SECONDS``; a failed refresh serves the stale
    entry when one exists. An empty list is a valid cached result — the model
    simply has no tier endpoints.
    """
    base_url = settings.openai_base_url.rstrip("/")
    endpoints_url = f"{base_url}/models/{model_id}/endpoints"
    cache_key = (base_url, model_id)

    now = time.monotonic()
    cached = _service_tiers_cache.get(cache_key)
    if cached is not None and now - cached[0] < _SERVICE_TIERS_CACHE_TTL_SECONDS:
        return list(cached[1])

    async with _service_tiers_lock:
        now = time.monotonic()
        cached = _service_tiers_cache.get(cache_key)
        if cached is not None and now - cached[0] < _SERVICE_TIERS_CACHE_TTL_SECONDS:
            return list(cached[1])

        try:
            response = await _shared_http().get(
                endpoints_url,
                timeout=settings.openchat_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            # A 200 without an endpoint list is malformed, not "no tiers" —
            # treat it like a transport failure so a stale entry survives.
            if not (
                isinstance(payload, dict)
                and isinstance(payload.get("data"), dict)
                and isinstance(payload["data"].get("endpoints"), list)
            ):
                raise ValueError("Malformed model endpoints response")
            tiers = _extract_service_tiers(payload)
        except (httpx.HTTPError, ValueError):
            if cached is not None:
                logger.warning(
                    "OpenRouter service-tier refresh failed for '%s'; serving stale tiers.",
                    model_id,
                    exc_info=True,
                )
                return list(cached[1])
            raise

        _service_tiers_cache[cache_key] = (time.monotonic(), tiers)
        return list(tiers)


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
    service_tier: str | None = None,
) -> dict[str, Any]:
    extra_body: dict[str, Any] = {
        "reasoning": {
            "effort": reasoning_effort,
            "exclude": reasoning_exclude,
        },
    }

    # "default" is the standard tier — omitting the parameter routes the same
    # way, so only named tiers are sent upstream.
    if service_tier and service_tier != "default":
        extra_body["service_tier"] = service_tier

    # Only attach tool-control parameters when an actual tools array is
    # present. Sending `max_tool_calls`/`parallel_tool_calls` without a
    # `tools` array is a malformed OpenRouter request that causes the
    # provider to return an error response (choices: null), which surfaces
    # as a TypeError when indexing `response.choices`.
    tools = WEB_SEARCH_TOOLS if (web_search_enabled and allow_tools) else []

    if not allow_tools:
        # Explicitly disable tools for providers that default them on.
        extra_body["tools"] = []
        extra_body["max_tool_calls"] = 0
        extra_body["parallel_tool_calls"] = False
    elif tools:
        extra_body["tools"] = tools
        extra_body["max_tool_calls"] = 10
        extra_body["parallel_tool_calls"] = True

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


def _extract_persona_assignments(raw_content: str, agents: list[SourceAgentSpec]) -> list[PersonaAssignment] | None:
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

    agent_ids = [agent.id for agent in agents]
    agent_id_by_model: dict[str, str] = {}
    for agent in agents:
        agent_id_by_model.setdefault(agent.model, agent.id)

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
            agent_id = str(item.get("agent_id", "")).strip()
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

            # Prefer an explicit agent_id from the generator; otherwise infer
            # one from the model when that mapping is unambiguous.
            if not agent_id:
                agent_id = agent_id_by_model.get(model, "")

            try:
                parsed.append(
                    PersonaAssignment(
                        model=model,
                        agent_id=agent_id,
                        title=title[:120],
                        temperature=temperature,
                        description=description[:1200],
                    )
                )
            except Exception:  # noqa: BLE001
                continue

        if not parsed:
            continue

        # Strategy 1: match by explicit agent_id returned by the generator.
        parsed_by_agent = {assignment.agent_id: assignment for assignment in parsed if assignment.agent_id}
        if agent_ids and all(aid in parsed_by_agent for aid in agent_ids):
            return [parsed_by_agent[aid] for aid in agent_ids]

        # Strategy 2: legacy generators that only keyed by model. Only usable
        # when every agent maps to a unique model.
        if len(agents) == len(agent_id_by_model):
            parsed_by_model = {assignment.model: assignment for assignment in parsed}
            if all(agent.model in parsed_by_model for agent in agents):
                return [
                    parsed_by_model[agent.model].model_copy(update={"agent_id": agent.id})
                    for agent in agents
                ]

        # Strategy 3: positional fallback. When multiple agents share the
        # same model (so model-keying collapses them) and the generator did
        # not return usable agent_ids, fall back to matching personas to
        # agents in the order they were returned. This handles the common
        # case where the generator emits one persona per agent but only
        # keyed by model.
        if len(parsed) == len(agents):
            return [
                assignment.model_copy(update={"agent_id": agent.id, "model": agent.model})
                for agent, assignment in zip(agents, parsed, strict=True)
            ]

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
        raise CompletionFailure(f"{context} returned no choices.")

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


class CompletionFailure(RuntimeError):
    """Raised when a chat completion returns no usable choices.

    Carries sanitized diagnostics (no prompt/answer/API-key data) so callers
    can surface actionable errors and bound retries. The ``diagnostics`` dict
    is safe to serialize into an SSE error event.
    """

    def __init__(self, message: str, *, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics: dict[str, Any] = diagnostics or {}


# OpenRouter response headers that are safe to expose for debugging. None of
# these carry prompt content, completions, or credentials.
_OPENROUTER_DIAGNOSTIC_HEADERS = frozenset(
    {
        "x-openrouter-request-id",
        "x-openrouter-version",
        "x-openrouter-processed-at",
        "x-openrouter-credits-used",
        "x-openrouter-credits-left",
        "x-openrouter-credits",
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
        "cf-ray",
    }
)


def _safe_getattr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name, None)
    except Exception:  # noqa: BLE001
        return None


def _sanitize_value(value: Any) -> Any:
    """Convert SDK model objects / mappings into JSON-safe primitives."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:  # noqa: BLE001
            return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    return str(value)


def _extract_response_diagnostics(response: Any, headers: Any) -> dict[str, Any]:
    """Build a sanitized diagnostic dict from a parsed completion + headers.

    Intentionally excludes message content, tool call arguments, and the
    authorization header. Only metadata useful for identifying provider
    rejections (ids, usage, error payloads, rate-limit headers) is kept.
    """
    diagnostics: dict[str, Any] = {}

    if response is not None:
        for field in ("id", "model", "object", "created", "system_fingerprint"):
            value = _safe_getattr(response, field)
            if value is not None:
                diagnostics[field] = _sanitize_value(value)

        usage = _safe_getattr(response, "usage")
        if usage is not None:
            diagnostics["usage"] = _sanitize_value(usage)

        # OpenRouter surfaces provider errors at the top level of the body.
        for field in ("error", "provider_error", "user_id"):
            value = _safe_getattr(response, field)
            if value is not None:
                diagnostics[field] = _sanitize_value(value)

    if headers is not None:
        try:
            header_items: list[tuple[str, str]] = []
            if hasattr(headers, "items"):
                header_items = list(headers.items())
            elif isinstance(headers, dict):
                header_items = list(headers.items())
            for key, value in header_items:
                lowered = key.lower()
                if lowered in _OPENROUTER_DIAGNOSTIC_HEADERS:
                    diagnostics[lowered] = value
        except Exception:  # noqa: BLE001
            pass

    return diagnostics


def _is_retryable_exception(exc: BaseException) -> bool:
    """True for transient OpenRouter / transport failures worth one retry."""
    if isinstance(exc, CompletionFailure):
        return True
    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)):
        return True
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = getattr(exc.response, "status_code", None)
        if isinstance(status, int) and (status == 429 or status >= 500):
            return True
    return False


async def _with_transient_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    context: str,
    model: str,
    max_retries: int = MARKDOWN_MODEL_RETRY_COUNT,
) -> T:
    """Run ``operation`` with bounded retry on transient failures.

    Retries only errors classified by ``_is_retryable_exception`` (empty
    ``choices``, timeouts, connection errors, 429/5xx). Non-retryable errors
    propagate immediately; on final failure the last exception (with any
    sanitized ``CompletionFailure`` diagnostics) is re-raised.
    """
    last_exc: BaseException | None = None
    total_attempts = 1 + max(0, max_retries)

    for attempt_index in range(total_attempts):
        try:
            return await operation()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt_index == total_attempts - 1 or not _is_retryable_exception(exc):
                raise
            logger.warning(
                "%s for model '%s' failed on attempt %d/%d with a retryable error; "
                "retrying after backoff. Error: %s",
                context,
                model,
                attempt_index + 1,
                total_attempts,
                exc,
            )
            await asyncio.sleep(MARKDOWN_MODEL_RETRY_BACKOFF_SECONDS)

    # Unreachable: the loop either returns or raises. Kept for type safety.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{context} for model '{model}' failed without a captured exception.")


async def _run_chat_completion_with_tool_loop(
    *,
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    messages: list[dict[str, Any]],
    extra_body: dict[str, Any],
    max_steps: int = OPENROUTER_TOOL_LOOP_MAX_STEPS,
    context: str = "Chat completion",
) -> str:
    history: list[dict[str, Any]] = [dict(message) for message in messages]

    for _ in range(max_steps):
        # Use with_raw_response so OpenRouter diagnostic headers (request id,
        # rate-limit state, provider error fields) remain inspectable when a
        # parsed completion has no choices.
        raw = await client.chat.completions.with_raw_response.create(
            model=model,
            temperature=temperature,
            max_tokens=OPENROUTER_TOKEN_LIMIT,
            max_completion_tokens=OPENROUTER_TOKEN_LIMIT,
            messages=history,
            extra_body=extra_body,
        )
        response = raw.parse()
        headers = getattr(raw, "headers", None)

        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            diagnostics = _extract_response_diagnostics(response, headers)
            raise CompletionFailure(
                f"{context} for model '{model}' returned no choices.",
                diagnostics=diagnostics,
            )

        message = choices[0].message
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
    agent_id: str = "",
    service_tier: str | None = None,
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
            service_tier=service_tier,
        )

        messages: list[dict[str, Any]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": composed_prompt})

        total_attempts = 1 + SOURCE_EMPTY_RETRY_COUNT
        content = ""

        for attempt_index in range(total_attempts):
            try:
                content = await _run_chat_completion_with_tool_loop(
                    client=client,
                    model=model,
                    temperature=temperature,
                    messages=messages,
                    extra_body=extra_body,
                )
            except Exception as exc:  # noqa: BLE001
                # Transient failures (timeout, 429/5xx, no-choices) retry within
                # the same bounded budget as empty-content retries; anything
                # else propagates to the error SourceResult below.
                if attempt_index < total_attempts - 1 and _is_retryable_exception(exc):
                    logger.warning(
                        "Source model '%s' failed on attempt %d/%d with a retryable "
                        "error; retrying after backoff. Error: %s",
                        model,
                        attempt_index + 1,
                        total_attempts,
                        exc,
                    )
                    await asyncio.sleep(MARKDOWN_MODEL_RETRY_BACKOFF_SECONDS)
                    continue
                raise
            if content.strip():
                elapsed = int((time.perf_counter() - start) * 1000)
                return SourceResult(
                    model=model,
                    agent_id=agent_id or model,
                    content=content,
                    status="ok",
                    latency_ms=elapsed,
                )

        elapsed = int((time.perf_counter() - start) * 1000)
        return SourceResult(
            model=model,
            agent_id=agent_id or model,
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
        return SourceResult(
            model=model,
            agent_id=agent_id or model,
            content="",
            status="error",
            error=str(exc),
            latency_ms=elapsed,
        )

    # Defensive fallback: should be unreachable given the logic above, but
    # guarantees the function always returns a SourceResult (never None).
    elapsed = int((time.perf_counter() - start) * 1000)
    return SourceResult(
        model=model,
        agent_id=agent_id or model,
        content="",
        status="error",
        error="Model execution ended without a result.",
        latency_ms=elapsed,
    )


async def run_source_models(
    client: AsyncOpenAI,
    agents: list[SourceAgentSpec],
    prompt: str,
    temperature: float,
    web_search_enabled: bool,
    reasoning_effort: str,
    reasoning_exclude: bool,
    attachments: list[AttachmentInput],
    temperature_by_agent: dict[str, float] | None = None,
    system_prompt: str | None = None,
    system_prompt_by_agent: dict[str, str] | None = None,
    service_tier_by_model: dict[str, str] | None = None,
) -> AsyncGenerator[SourceResult, None]:
    semaphore = asyncio.Semaphore(max(1, settings.openchat_max_parallel_sources))

    async def runner(agent: SourceAgentSpec) -> SourceResult:
        async with semaphore:
            resolved_system_prompt = (
                system_prompt_by_agent.get(agent.id)
                if system_prompt_by_agent is not None
                else system_prompt
            )
            resolved_temperature = (
                temperature_by_agent.get(agent.id, temperature)
                if temperature_by_agent is not None
                else temperature
            )
            return await run_single_model(
                client=client,
                model=agent.model,
                prompt=prompt,
                temperature=resolved_temperature,
                web_search_enabled=web_search_enabled,
                reasoning_effort=reasoning_effort,
                reasoning_exclude=reasoning_exclude,
                attachments=attachments,
                system_prompt=resolved_system_prompt,
                agent_id=agent.id,
                service_tier=(service_tier_by_model or {}).get(agent.model),
            )

    tasks = [asyncio.create_task(runner(agent)) for agent in agents]
    try:
        for completed in asyncio.as_completed(tasks):
            yield await completed
    finally:
        # Ensure no orphaned tasks if the SSE consumer disconnects mid-stream.
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


async def run_markdown_model(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    web_search_enabled: bool,
    reasoning_effort: str,
    reasoning_exclude: bool,
    *,
    context: str = "Synthesis",
    max_retries: int = MARKDOWN_MODEL_RETRY_COUNT,
    service_tier: str | None = None,
) -> str:
    """Run a synthesis-style markdown completion with bounded silent retry.

    Transient OpenRouter failures (empty ``choices``, timeouts, 429/5xx) are
    retried once after a short backoff. Non-retryable errors propagate
    immediately. On final failure the last ``CompletionFailure`` (with
    sanitized diagnostics) is re-raised so callers can surface an actionable
    SSE error instead of crashing the ASGI stream.
    """
    extra_body = _build_openrouter_extra_body(
        web_search_enabled=web_search_enabled,
        reasoning_effort=reasoning_effort,
        reasoning_exclude=reasoning_exclude,
        service_tier=service_tier,
    )

    messages: list[dict[str, Any]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt})

    return await _with_transient_retry(
        lambda: _run_chat_completion_with_tool_loop(
            client=client,
            model=model,
            temperature=temperature,
            messages=messages,
            extra_body=extra_body,
            context=context,
        ),
        context=context,
        model=model,
        max_retries=max_retries,
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
    service_tier: str | None = None,
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
        service_tier=service_tier,
    )

    return await _with_transient_retry(
        lambda: _run_chat_completion_with_tool_loop(
            client=client,
            model=model,
            temperature=temperature,
            messages=normalized_messages,
            extra_body=extra_body,
            context="Direct chat",
        ),
        context="Direct chat",
        model=model,
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
    async def _call() -> str:
        # Extraction runs inside the retry boundary so an empty-choices
        # completion (CompletionFailure) retries like a transport failure.
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
        return _extract_first_choice_message_content(response, context="Prompt optimizer")

    content = await _with_transient_retry(
        _call,
        context="Prompt optimizer",
        model=PROMPT_OPTIMIZER_MODEL,
    )

    optimized = _extract_optimized_prompt(content)
    return optimized or prompt.strip()


async def generate_persona_assignments(
    client: AsyncOpenAI,
    user_prompt: str,
    agents: list[SourceAgentSpec],
) -> list[PersonaAssignment]:
    if not agents:
        return []

    user_content = build_persona_generation_prompt(user_prompt, agents)

    async def _call() -> str:
        # Extraction runs inside the retry boundary so an empty-choices
        # completion (CompletionFailure) retries like a transport failure.
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
        return _extract_first_choice_message_content(response, context="Persona generator")

    content = await _with_transient_retry(
        _call,
        context="Persona generator",
        model=PERSONA_GENERATOR_MODEL,
    )

    assignments = _extract_persona_assignments(content, agents)
    if assignments is None:
        raise RuntimeError("Persona generator returned invalid or incomplete persona assignments.")

    return assignments
