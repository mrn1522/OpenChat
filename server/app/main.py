import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

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
    source_models: list[str],
    debate_mode: Literal["off", "partial", "full"],
    fusion_model: str,
) -> list[tuple[str, str]]:
    """Return (reviewer_model, target_model) assignments."""
    if len(source_models) == 1:
        return [(fusion_model, source_models[0])]

    if debate_mode == "full":
        pairs: list[tuple[str, str]] = []
        for target_model in source_models:
            for reviewer_model in source_models:
                if reviewer_model == target_model:
                    continue
                pairs.append((reviewer_model, target_model))
        return pairs

    return [
        (source_models[(index + 1) % len(source_models)], source_models[index])
        for index in range(len(source_models))
    ]


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
        source_results: list[SourceResult] = []
        debate_results: list[dict] = []
        persona_by_model: dict[str, PersonaAssignment] = {}
        source_system_prompt_by_model: dict[str, str] = {}
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
            override_by_model = {
                assignment.model: assignment for assignment in request.persona_assignments_override
            }

            if override_by_model:
                persona_by_model = {
                    model_name: override_by_model[model_name]
                    for model_name in request.source_models
                    if model_name in override_by_model
                }
                yield sse(
                    "persona_assignments",
                    {
                        "items": [assignment.model_dump() for assignment in persona_by_model.values()],
                    },
                )
            else:
                yield sse(
                    "orchestration_step",
                    {
                        "step": "route",
                        "status": "active",
                        "detail": "Assigning unique personas for each source model.",
                    },
                )

                try:
                    persona_assignments = await generate_persona_assignments(
                        client=client,
                        user_prompt=request.prompt,
                        source_models=request.source_models,
                    )
                    persona_by_model = {
                        assignment.model: assignment for assignment in persona_assignments
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

        source_system_prompt_by_model = {
            model_name: build_source_system_prompt(persona_by_model.get(model_name))
            for model_name in request.source_models
        }
        source_temperature_by_model = {
            model_name: persona_by_model.get(model_name).temperature
            if persona_by_model.get(model_name) is not None
            else request.temperature
            for model_name in request.source_models
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
            models=request.source_models,
            prompt=request.prompt,
            temperature=request.temperature,
            web_search_enabled=request.web_search_enabled,
            reasoning_effort=request.reasoning.effort,
            reasoning_exclude=request.reasoning.exclude,
            attachments=request.attachments,
            temperature_by_model=source_temperature_by_model,
            system_prompt_by_model=source_system_prompt_by_model,
        ):
            result_persona = persona_by_model.get(result.model)
            if result_persona is not None and result.persona is None:
                result = result.model_copy(update={"persona": result_persona})
            source_results.append(result)
            yield sse("source_result", result.model_dump())

        successful_source_results = [result for result in source_results if result.status == "ok"]
        successful_source_models = [
            model_name
            for model_name in request.source_models
            if any(result.model == model_name for result in successful_source_results)
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

            source_by_model = {result.model: result for result in successful_source_results}
            debate_pairs = build_debate_pairs(
                source_models=successful_source_models,
                debate_mode=request.debate_mode,
                fusion_model=request.fusion_model,
            )

            debate_outputs: list[tuple[str, str, str]] = []
            for reviewer_model, target_model in debate_pairs:
                target_result = source_by_model.get(target_model)
                if target_result is None:
                    continue

                peer_results = [result for model_name, result in source_by_model.items() if model_name != target_model]
                debate_prompt = build_debate_prompt(
                    user_prompt=request.prompt,
                    target_result=target_result,
                    peer_results=peer_results,
                )
                try:
                    debate_output = await run_markdown_model(
                        client=client,
                        model=reviewer_model,
                        prompt=debate_prompt,
                        system_prompt=debate_system_prompt,
                        temperature=request.temperature,
                        web_search_enabled=False,
                        reasoning_effort=request.reasoning.effort,
                        reasoning_exclude=request.reasoning.exclude,
                    )
                except Exception as exc:  # noqa: BLE001
                    yield sse(
                        "orchestration_step",
                        {
                            "step": "critique",
                            "status": "active",
                            "detail": (
                                f"Skipped one debate review after {reviewer_model} failed on {target_model}: {exc}"
                            ),
                        },
                    )
                    continue

                debate_outputs.append((target_model, reviewer_model, debate_output))
                debate_result_payload = {
                    "target_model": target_model,
                    "reviewer_model": reviewer_model,
                    "content": debate_output,
                }
                debate_results.append(debate_result_payload)
                yield sse(
                    "debate_result",
                    debate_result_payload,
                )

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
        final_output = await run_markdown_model(
            client=client,
            model=request.fusion_model,
            prompt=synth_prompt,
            system_prompt=fusion_system_prompt,
            temperature=request.temperature,
            web_search_enabled=False,
            reasoning_effort=request.reasoning.effort,
            reasoning_exclude=request.reasoning.exclude,
        )
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

    try:
        assignments = await generate_persona_assignments(
            client=client,
            user_prompt=request.prompt,
            source_models=request.source_models,
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
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Fusion regeneration failed: {exc}") from exc

    return SynthesisResult(model=request.fusion_model, content=final_output)
