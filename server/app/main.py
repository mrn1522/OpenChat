import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.chat_store import (
    WorkflowNameExistsError,
    delete_chat_record,
    delete_workflow_record,
    get_chat_record,
    init_chat_store,
    list_chat_records,
    list_workflow_records,
    save_chat_record,
    save_workflow_record,
)
from app.config import settings
from app.llm import (
    build_client,
    fetch_openrouter_models,
    generate_persona_assignments,
    optimize_prompt_text,
    run_direct_chat_model,
    run_markdown_model,
    run_source_models,
)
from app.models import (
    ChatHistoryDetail,
    ChatHistoryListResponse,
    DirectChatRequest,
    FusionRegenerateRequest,
    OpenRouterModelsResponse,
    PersonaAssignment,
    PersonaPreviewRequest,
    PersonaPreviewResponse,
    PromptOptimizeRequest,
    PromptOptimizeResponse,
    RunRequest,
    SourceAgentSpec,
    SynthesisResult,
    SourceResult,
    WorkflowCreateRequest,
    WorkflowListResponse,
    WorkflowSummary,
)
from app.prompting import (
    DEBATE_SYSTEM_PROMPT,
    FUSION_SYSTEM_PROMPT,
    build_debate_markdown,
    build_debate_prompt,
    build_source_system_prompt,
    build_system_prompt_with_current_aest,
    build_synth_prompt,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    init_chat_store(settings.openchat_history_db_path)
    yield


app = FastAPI(title="OpenChat MoA Proxy", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def build_debate_pairs(
    agent_ids: list[str],
    debate_mode: Literal["off", "partial", "full"],
    fusion_model: str,
) -> list[tuple[str, str]]:
    """Return (reviewer_agent_id, target_agent_id) assignments."""
    if len(agent_ids) == 1:
        return [(fusion_model, agent_ids[0])]

    if debate_mode == "full":
        pairs: list[tuple[str, str]] = []
        for target_id in agent_ids:
            for reviewer_id in agent_ids:
                if reviewer_id == target_id:
                    continue
                pairs.append((reviewer_id, target_id))
        return pairs

    return [
        (agent_ids[(index + 1) % len(agent_ids)], agent_ids[index])
        for index in range(len(agent_ids))
    ]


@dataclass(frozen=True)
class DebateJob:
    """One independent reviewer→target debate assignment."""

    pair_index: int
    reviewer_id: str
    reviewer_model: str
    target_id: str
    target_model: str
    prompt: str


@dataclass(frozen=True)
class DebateJobSuccess:
    pair_index: int
    target_model: str
    target_agent_id: str
    reviewer_model: str
    reviewer_agent_id: str
    content: str


@dataclass(frozen=True)
class DebateJobFailure:
    pair_index: int
    target_model: str
    target_agent_id: str
    reviewer_model: str
    reviewer_agent_id: str
    error: str


DebateJobOutcome = DebateJobSuccess | DebateJobFailure


def build_debate_jobs(
    *,
    debate_pairs: list[tuple[str, str]],
    source_by_agent: dict[str, SourceResult],
    agent_by_id: dict[str, SourceAgentSpec],
    user_prompt: str,
) -> list[DebateJob]:
    """Materialize independent debate jobs in stable pair order."""
    jobs: list[DebateJob] = []
    for pair_index, (reviewer_id, target_id) in enumerate(debate_pairs):
        target_result = source_by_agent.get(target_id)
        if target_result is None:
            continue

        peer_results = [
            result
            for aid, result in source_by_agent.items()
            if aid != target_id
        ]
        debate_prompt = build_debate_prompt(
            user_prompt=user_prompt,
            target_result=target_result,
            peer_results=peer_results,
        )
        reviewer_agent = agent_by_id.get(reviewer_id)
        reviewer_model = reviewer_agent.model if reviewer_agent else reviewer_id
        jobs.append(
            DebateJob(
                pair_index=pair_index,
                reviewer_id=reviewer_id,
                reviewer_model=reviewer_model,
                target_id=target_id,
                target_model=target_result.model,
                prompt=debate_prompt,
            )
        )
    return jobs


async def run_debate_jobs(
    *,
    jobs: list[DebateJob],
    client: Any,
    debate_system_prompt: str,
    temperature: float,
    reasoning_effort: str,
    reasoning_exclude: bool,
    run_model=run_markdown_model,
) -> AsyncIterator[DebateJobOutcome]:
    """Run all debate jobs concurrently with no concurrency cap.

    Yields outcomes in completion order so callers can stream ``debate_result``
    SSE events as soon as each review finishes. The final Debate Report should
    still be assembled from successful outcomes sorted by ``pair_index``.
    """
    if not jobs:
        return

    async def _run_one(job: DebateJob) -> DebateJobOutcome:
        try:
            content = await run_model(
                client=client,
                model=job.reviewer_model,
                prompt=job.prompt,
                system_prompt=debate_system_prompt,
                temperature=temperature,
                web_search_enabled=False,
                reasoning_effort=reasoning_effort,
                reasoning_exclude=reasoning_exclude,
                context="Debate",
            )
        except Exception as exc:  # noqa: BLE001
            return DebateJobFailure(
                pair_index=job.pair_index,
                target_model=job.target_model,
                target_agent_id=job.target_id,
                reviewer_model=job.reviewer_model,
                reviewer_agent_id=job.reviewer_id,
                error=str(exc),
            )

        return DebateJobSuccess(
            pair_index=job.pair_index,
            target_model=job.target_model,
            target_agent_id=job.target_id,
            reviewer_model=job.reviewer_model,
            reviewer_agent_id=job.reviewer_id,
            content=content,
        )

    tasks = [asyncio.create_task(_run_one(job)) for job in jobs]
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


def resolve_agents(request: RunRequest) -> list[SourceAgentSpec]:
    """Build the effective runtime agent list for a run.

    Prefers explicit ``source_agents`` (which may contain duplicate model
    instances, each with its own stable id). Falls back to one agent per entry
    in ``source_models`` for backward compatibility with older clients.
    """
    if request.source_agents:
        return request.source_agents

    return [SourceAgentSpec(id=model_name, model=model_name) for model_name in request.source_models]


def _build_synthesis_error_message(exc: BaseException, *, stage: str, model: str) -> str:
    """Build an actionable, SSE-safe error string from a synthesis failure.

    Uses the sanitized ``diagnostics`` carried by ``CompletionFailure`` (no
    prompt/answer/API-key data) so the client can report the OpenRouter
    request id and provider error without leaking sensitive content.
    """
    diagnostics = getattr(exc, "diagnostics", None) or {}
    request_id = (
        diagnostics.get("x-openrouter-request-id")
        or diagnostics.get("id")
        or ""
    )
    parts = [f"{stage} model '{model}' failed to produce a final answer: {exc}"]
    if request_id:
        parts.append(f"OpenRouter request id: {request_id}")
    if diagnostics:
        parts.append(f"Diagnostics: {json.dumps(diagnostics, default=str)}")
    return " | ".join(parts)


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "env": settings.app_env}


@app.post("/api/run/stream")
async def run_stream(request: RunRequest):
    try:
        client = build_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    async def event_stream():
        run_id = str(uuid.uuid4())
        chat_id = run_id
        started = time.perf_counter()
        agents = resolve_agents(request)
        source_results: list[SourceResult] = []
        debate_results: list[dict] = []
        persona_by_agent: dict[str, PersonaAssignment] = {}
        source_system_prompt_by_agent: dict[str, str] = {}
        debate_system_prompt = build_system_prompt_with_current_aest(DEBATE_SYSTEM_PROMPT)
        fusion_system_prompt = build_system_prompt_with_current_aest(FUSION_SYSTEM_PROMPT)

        yield sse("run_started", {"run_id": run_id})
        yield sse(
            "orchestration_step",
            {
                "step": "route",
                "status": "active",
                "detail": "Understanding your query and selecting best agents.",
            },
        )

        if request.persona_enabled:
            override_by_agent = {
                assignment.agent_id: assignment
                for assignment in request.persona_assignments_override
                if assignment.agent_id
            }

            if override_by_agent:
                persona_by_agent = {
                    agent.id: override_by_agent[agent.id]
                    for agent in agents
                    if agent.id in override_by_agent
                }
                yield sse(
                    "persona_assignments",
                    {
                        "items": [assignment.model_dump() for assignment in persona_by_agent.values()],
                    },
                )
            else:
                yield sse(
                    "orchestration_step",
                    {
                        "step": "route",
                        "status": "active",
                        "detail": "Assigning unique personas for each source agent.",
                    },
                )

                try:
                    persona_assignments = await generate_persona_assignments(
                        client=client,
                        user_prompt=request.prompt,
                        agents=agents,
                    )
                    persona_by_agent = {
                        assignment.agent_id: assignment for assignment in persona_assignments
                    }
                    yield sse(
                        "persona_assignments",
                        {
                            "items": [assignment.model_dump() for assignment in persona_assignments],
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    yield sse(
                        "orchestration_step",
                        {
                            "step": "route",
                            "status": "active",
                            "detail": f"Persona generation failed; continuing without personas: {exc}",
                        },
                    )
                    yield sse("persona_assignments", {"items": [], "error": str(exc)})

        source_system_prompt_by_agent = {
            agent.id: build_source_system_prompt(persona_by_agent.get(agent.id))
            for agent in agents
        }
        source_temperature_by_agent = {
            agent.id: persona_by_agent.get(agent.id).temperature
            if persona_by_agent.get(agent.id) is not None
            else request.temperature
            for agent in agents
        }

        yield sse(
            "orchestration_step",
            {
                "step": "route",
                "status": "done",
                "detail": "Query routed.",
            },
        )
        yield sse(
            "orchestration_step",
            {
                "step": "search",
                "status": "active",
                "detail": "Agents are searching knowledge and web context.",
            },
        )
        yield sse(
            "orchestration_step",
            {
                "step": "parallel",
                "status": "active",
                "detail": "Source agents are reasoning in parallel.",
            },
        )

        async for result in run_source_models(
            client=client,
            agents=agents,
            prompt=request.prompt,
            temperature=request.temperature,
            web_search_enabled=request.web_search_enabled,
            reasoning_effort=request.reasoning.effort,
            reasoning_exclude=request.reasoning.exclude,
            attachments=request.attachments,
            temperature_by_agent=source_temperature_by_agent,
            system_prompt_by_agent=source_system_prompt_by_agent,
        ):
            result_persona = persona_by_agent.get(result.agent_id or result.model)
            if result_persona is not None and result.persona is None:
                result = result.model_copy(update={"persona": result_persona})
            source_results.append(result)
            yield sse("source_result", result.model_dump())

        successful_source_results = [result for result in source_results if result.status == "ok"]
        successful_agent_ids = [
            agent.id
            for agent in agents
            if any((result.agent_id or result.model) == agent.id for result in successful_source_results)
        ]

        yield sse(
            "orchestration_step",
            {
                "step": "search",
                "status": "done",
                "detail": "Search complete.",
            },
        )
        yield sse(
            "orchestration_step",
            {
                "step": "parallel",
                "status": "done",
                "detail": "Parallel reasoning complete.",
            },
        )
        judge_output = ""
        if request.debate_mode == "off":
            yield sse(
                "orchestration_step",
                {
                    "step": "critique",
                    "status": "done",
                    "detail": "Debate mode is off. Skipping critique and proceeding to fusion.",
                },
            )
            yield sse("critique_ready", {"model": "none", "content": ""})
        else:
            yield sse(
                "orchestration_step",
                {
                    "step": "critique",
                    "status": "active",
                    "detail": "Running debate reviews across agent answers.",
                },
            )

            source_by_agent = {result.agent_id or result.model: result for result in successful_source_results}
            agent_by_id = {agent.id: agent for agent in agents}
            debate_pairs = build_debate_pairs(
                agent_ids=successful_agent_ids,
                debate_mode=request.debate_mode,
                fusion_model=request.fusion_model,
            )
            debate_jobs = build_debate_jobs(
                debate_pairs=debate_pairs,
                source_by_agent=source_by_agent,
                agent_by_id=agent_by_id,
                user_prompt=request.prompt,
            )

            # Collect successes keyed by original pair order so the final
            # Debate Report stays deterministic even though SSE events are
            # emitted in completion order as each concurrent review finishes.
            successful_by_pair: dict[int, DebateJobSuccess] = {}
            async for outcome in run_debate_jobs(
                jobs=debate_jobs,
                client=client,
                debate_system_prompt=debate_system_prompt,
                temperature=request.temperature,
                reasoning_effort=request.reasoning.effort,
                reasoning_exclude=request.reasoning.exclude,
            ):
                if isinstance(outcome, DebateJobFailure):
                    yield sse(
                        "orchestration_step",
                        {
                            "step": "critique",
                            "status": "active",
                            "detail": (
                                f"Skipped one debate review after {outcome.reviewer_model} "
                                f"failed on {outcome.target_model}: {outcome.error}"
                            ),
                        },
                    )
                    continue

                successful_by_pair[outcome.pair_index] = outcome
                debate_result_payload = {
                    "target_model": outcome.target_model,
                    "target_agent_id": outcome.target_agent_id,
                    "reviewer_model": outcome.reviewer_model,
                    "reviewer_agent_id": outcome.reviewer_agent_id,
                    "content": outcome.content,
                }
                debate_results.append(debate_result_payload)
                yield sse("debate_result", debate_result_payload)

            debate_outputs = [
                (
                    successful_by_pair[index].target_model,
                    successful_by_pair[index].target_agent_id,
                    successful_by_pair[index].reviewer_model,
                    successful_by_pair[index].reviewer_agent_id,
                    successful_by_pair[index].content,
                )
                for index in sorted(successful_by_pair)
            ]
            judge_output = build_debate_markdown(debate_outputs)
            yield sse("critique_ready", {"model": "multi-debate", "content": judge_output})
            yield sse(
                "orchestration_step",
                {
                    "step": "critique",
                    "status": "done",
                    "detail": "Debate reviews complete.",
                },
            )
        yield sse(
            "orchestration_step",
            {
                "step": "fusion",
                "status": "active",
                "detail": "Fusing insights into the best final answer.",
            },
        )

        synth_prompt = build_synth_prompt(request.prompt, successful_source_results, judge_output)
        try:
            final_output = await run_markdown_model(
                client=client,
                model=request.fusion_model,
                prompt=synth_prompt,
                system_prompt=fusion_system_prompt,
                temperature=request.temperature,
                web_search_enabled=False,
                reasoning_effort=request.reasoning.effort,
                reasoning_exclude=request.reasoning.exclude,
                context="Fusion",
            )
        except Exception as exc:  # noqa: BLE001
            # Fusion is the terminal stage. Never let a provider failure tear
            # down the ASGI stream: emit an actionable SSE error, persist the
            # partial run (sources + debate are still valuable), and close the
            # stream cleanly so the client can recover / regenerate.
            error_message = _build_synthesis_error_message(
                exc, stage="Fusion", model=request.fusion_model
            )
            yield sse(
                "orchestration_step",
                {
                    "step": "fusion",
                    "status": "done",
                    "detail": f"Fusion failed; source and debate results are preserved. {exc}",
                },
            )
            yield sse("error", {"message": error_message})

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            try:
                save_chat_record(
                    settings.openchat_history_db_path,
                    chat_id=chat_id,
                    run_id=run_id,
                    status="failed_fusion",
                    elapsed_ms=elapsed_ms,
                    request_payload=request.model_dump(),
                    source_results=[result.model_dump() for result in source_results],
                    debate_results=debate_results,
                    critique_output=judge_output,
                    fusion_output="",
                )
            except Exception:
                # History persistence must not mask the fusion error.
                pass

            yield sse("completed", {"elapsed_ms": elapsed_ms})
            return
        yield sse("fusion_ready", {"model": request.fusion_model, "content": final_output})
        yield sse(
            "orchestration_step",
            {
                "step": "fusion",
                "status": "done",
                "detail": "Fusion complete.",
            },
        )

        elapsed_ms = int((time.perf_counter() - started) * 1000)

        try:
            save_chat_record(
                settings.openchat_history_db_path,
                chat_id=chat_id,
                run_id=run_id,
                status="completed",
                elapsed_ms=elapsed_ms,
                request_payload=request.model_dump(),
                source_results=[result.model_dump() for result in source_results],
                debate_results=debate_results,
                critique_output=judge_output,
                fusion_output=final_output,
            )
        except Exception:
            # Do not break streaming completion when history persistence fails.
            pass

        yield sse("completed", {"elapsed_ms": elapsed_ms})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/direct-chat/stream")
async def direct_chat_stream(request: DirectChatRequest):
    try:
        client = build_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    async def event_stream():
        run_id = str(uuid.uuid4())
        chat_id = run_id
        started = time.perf_counter()
        direct_system_prompt = build_source_system_prompt(None)

        yield sse("run_started", {"run_id": run_id})

        try:
            assistant_output = await run_direct_chat_model(
                client=client,
                model=request.model,
                messages=request.messages,
                system_prompt=direct_system_prompt,
                temperature=request.temperature,
                web_search_enabled=request.web_search_enabled,
                reasoning_effort=request.reasoning.effort,
                reasoning_exclude=request.reasoning.exclude,
                attachments=request.attachments,
            )
        except Exception as exc:  # noqa: BLE001
            yield sse("error", {"message": str(exc)})
            return

        yield sse("assistant_ready", {"model": request.model, "content": assistant_output})
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        latest_user_prompt = next(
            (message.content for message in reversed(request.messages) if message.role == "user"),
            "",
        )

        try:
            save_chat_record(
                settings.openchat_history_db_path,
                chat_id=chat_id,
                run_id=run_id,
                status="direct_completed",
                elapsed_ms=elapsed_ms,
                request_payload={
                    "prompt": latest_user_prompt,
                    "source_models": [request.model],
                    "fusion_model": request.model,
                    "debate_mode": "off",
                    "temperature": request.temperature,
                    "max_output_tokens": request.max_output_tokens,
                    "web_search_enabled": request.web_search_enabled,
                    "persona_enabled": False,
                    "persona_assignments_override": [],
                    "reasoning": request.reasoning.model_dump(),
                    "attachments": [attachment.model_dump() for attachment in request.attachments],
                },
                source_results=[
                    {
                        "model": request.model,
                        "content": assistant_output,
                        "status": "ok",
                        "persona": None,
                        "error": None,
                        "latency_ms": elapsed_ms,
                    }
                ],
                debate_results=[],
                critique_output="",
                fusion_output=assistant_output,
            )
        except Exception:
            pass

        yield sse("completed", {"elapsed_ms": elapsed_ms})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/models", response_model=OpenRouterModelsResponse)
async def get_models() -> OpenRouterModelsResponse:
    try:
        return await fetch_openrouter_models()
    except httpx.HTTPStatusError as exc:
        detail = f"OpenRouter models request failed: {exc.response.status_code}"
        raise HTTPException(status_code=502, detail=detail) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"OpenRouter models request failed: {exc}") from exc


@app.get("/api/chats", response_model=ChatHistoryListResponse)
async def list_chats() -> ChatHistoryListResponse:
    try:
        records = list_chat_records(settings.openchat_history_db_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to load chat history: {exc}") from exc

    return ChatHistoryListResponse(data=records)


@app.get("/api/workflows", response_model=WorkflowListResponse)
async def list_workflows() -> WorkflowListResponse:
    try:
        records = list_workflow_records(settings.openchat_history_db_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to load workflows: {exc}") from exc

    return WorkflowListResponse(data=[WorkflowSummary(**record) for record in records])


@app.post("/api/workflows", response_model=WorkflowSummary, status_code=201)
async def create_workflow(request: WorkflowCreateRequest) -> WorkflowSummary:
    workflow_name = request.name.strip()
    if not workflow_name:
        raise HTTPException(status_code=422, detail="Workflow name cannot be empty")

    workflow_id = str(uuid.uuid4())

    try:
        record = save_workflow_record(
            settings.openchat_history_db_path,
            workflow_id=workflow_id,
            name=workflow_name,
            config_payload=request.config.model_dump(),
        )
    except WorkflowNameExistsError:
        raise HTTPException(status_code=409, detail="Workflow name already exists. Choose a different name.")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to save workflow: {exc}") from exc

    return WorkflowSummary(**record)


@app.delete("/api/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str) -> dict:
    try:
        deleted = delete_workflow_record(settings.openchat_history_db_path, workflow_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow: {exc}") from exc

    if not deleted:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return {"ok": True}


@app.get("/api/chats/{chat_id}", response_model=ChatHistoryDetail)
async def get_chat(chat_id: str) -> ChatHistoryDetail:
    try:
        record = get_chat_record(settings.openchat_history_db_path, chat_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to load chat: {exc}") from exc

    if record is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    return ChatHistoryDetail(**record)


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str) -> dict:
    try:
        deleted = delete_chat_record(settings.openchat_history_db_path, chat_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to delete chat: {exc}") from exc

    if not deleted:
        raise HTTPException(status_code=404, detail="Chat not found")

    return {"ok": True}


@app.post("/api/prompt/optimize", response_model=PromptOptimizeResponse)
async def optimize_prompt(request: PromptOptimizeRequest) -> PromptOptimizeResponse:
    try:
        client = build_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        optimized_prompt = await optimize_prompt_text(client=client, prompt=request.prompt)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Prompt optimization failed: {exc}") from exc

    return PromptOptimizeResponse(
        original_prompt=request.prompt,
        optimized_prompt=optimized_prompt,
        model="openai/gpt-oss-20b",
    )


@app.post("/api/personas/preview", response_model=PersonaPreviewResponse)
async def preview_personas(request: PersonaPreviewRequest) -> PersonaPreviewResponse:
    try:
        client = build_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    agents = request.source_agents or [
        SourceAgentSpec(id=model_name, model=model_name) for model_name in request.source_models
    ]
    if not agents:
        raise HTTPException(status_code=422, detail="At least one source agent is required.")

    try:
        assignments = await generate_persona_assignments(
            client=client,
            user_prompt=request.prompt,
            agents=agents,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Persona generation failed: {exc}") from exc

    return PersonaPreviewResponse(items=assignments, model="google/gemma-4-31b-it")


@app.post("/api/fusion/regenerate", response_model=SynthesisResult)
async def regenerate_fusion(request: FusionRegenerateRequest) -> SynthesisResult:
    try:
        client = build_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    successful_source_results = [
        result
        for result in request.source_results
        if result.status == "ok" and result.content.strip()
    ]
    if not successful_source_results:
        raise HTTPException(status_code=422, detail="At least one successful source result is required.")

    fusion_system_prompt = build_system_prompt_with_current_aest(FUSION_SYSTEM_PROMPT)
    synth_prompt = build_synth_prompt(request.prompt, successful_source_results, request.critique_output)

    try:
        final_output = await run_markdown_model(
            client=client,
            model=request.fusion_model,
            prompt=synth_prompt,
            system_prompt=fusion_system_prompt,
            temperature=request.temperature,
            web_search_enabled=False,
            reasoning_effort=request.reasoning.effort,
            reasoning_exclude=request.reasoning.exclude,
            context="Fusion regeneration",
        )
    except Exception as exc:  # noqa: BLE001
        detail = _build_synthesis_error_message(
            exc, stage="Fusion regeneration", model=request.fusion_model
        )
        raise HTTPException(status_code=502, detail=detail) from exc

    return SynthesisResult(model=request.fusion_model, content=final_output)
