from typing import Any, Literal

from pydantic import BaseModel, Field


class ReasoningConfig(BaseModel):
    effort: Literal["xhigh", "high", "medium", "low", "minimal", "none"] = "medium"
    exclude: bool = False


class AttachmentInput(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    size: int = Field(ge=1, le=262144)
    content_type: str = Field(default="application/octet-stream", max_length=128)
    content: str = Field(min_length=1, max_length=100000)


class SourceAgentSpec(BaseModel):
    id: str = Field(min_length=1)
    model: str = Field(min_length=1)


class PersonaAssignment(BaseModel):
    model: str = Field(min_length=1)
    agent_id: str = ""
    title: str = Field(min_length=1, max_length=120)
    temperature: float = Field(default=0.7, ge=0.5, le=1.2)
    description: str = Field(min_length=1, max_length=1200)


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1)
    source_models: list[str] = Field(min_length=1)
    source_agents: list[SourceAgentSpec] = Field(default_factory=list)
    fusion_model: str = Field(min_length=1)
    debate_mode: Literal["off", "partial", "full"] = "partial"
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_output_tokens: int = Field(default=1000, ge=128, le=10000)
    web_search_enabled: bool = False
    persona_enabled: bool = False
    persona_assignments_override: list[PersonaAssignment] = Field(default_factory=list, max_length=48)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    attachments: list[AttachmentInput] = Field(default_factory=list, max_length=5)


class DirectChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=100000)


class DirectChatRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[DirectChatMessage] = Field(min_length=1, max_length=200)
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_output_tokens: int = Field(default=1000, ge=128, le=10000)
    web_search_enabled: bool = False
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    attachments: list[AttachmentInput] = Field(default_factory=list, max_length=5)


class PromptOptimizeRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=100000)


class PromptOptimizeResponse(BaseModel):
    original_prompt: str
    optimized_prompt: str
    model: str


class FusionRegenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    fusion_model: str = Field(min_length=1)
    temperature: float = Field(default=0.2, ge=0, le=2)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    source_results: list["SourceResult"] = Field(min_length=1)
    critique_output: str = ""


class PersonaPreviewRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=100000)
    source_models: list[str] = Field(default_factory=list)
    source_agents: list[SourceAgentSpec] = Field(default_factory=list)


class PersonaPreviewResponse(BaseModel):
    items: list[PersonaAssignment]
    model: str


class SourceResult(BaseModel):
    model: str
    agent_id: str = ""
    content: str
    status: Literal["ok", "error"]
    persona: PersonaAssignment | None = None
    error: str | None = None
    latency_ms: int | None = None


class DebateResult(BaseModel):
    target_model: str
    reviewer_model: str
    target_agent_id: str = ""
    reviewer_agent_id: str = ""
    content: str


class JudgeResult(BaseModel):
    model: str
    content: str


class SynthesisResult(BaseModel):
    model: str
    content: str


class OpenRouterModel(BaseModel):
    id: str
    created: int | None = None
    created_at: str | None = None
    createdAt: str | None = None
    canonical_slug: str | None = None
    name: str | None = None
    description: str | None = None
    context_length: int | None = None
    architecture: dict[str, Any] | None = None
    pricing: dict[str, Any] | None = None
    top_provider: dict[str, Any] | None = None
    supported_parameters: list[str] | None = None
    default_parameters: dict[str, Any] | None = None
    links: dict[str, Any] | None = None


class OpenRouterModelsResponse(BaseModel):
    data: list[OpenRouterModel]


class ChatHistorySummary(BaseModel):
    chat_id: str
    created_at: str
    status: str
    prompt_preview: str
    source_models: list[str]
    fusion_model: str
    has_fusion_output: bool


class ChatHistoryListResponse(BaseModel):
    data: list[ChatHistorySummary]


class ChatHistoryDetail(BaseModel):
    chat_id: str
    created_at: str
    run_id: str
    status: str
    elapsed_ms: int | None = None
    request: RunRequest
    source_results: list[SourceResult]
    debate_results: list[DebateResult]
    critique_output: str
    fusion_output: str


class WorkflowAttachmentMeta(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    size: int = Field(ge=0, le=262144)
    content_type: str = Field(default="application/octet-stream", max_length=128)


class WorkflowConfig(BaseModel):
    source_models: list[str] = Field(min_length=1)
    fusion_model: str = Field(min_length=1)
    temperature: float = Field(default=0.2, ge=0, le=2)
    reasoning_effort: Literal["xhigh", "high", "medium", "low", "minimal", "none"] = "medium"
    debate_mode: Literal["off", "partial", "full"] = "partial"
    web_search_enabled: bool = False
    persona_enabled: bool = False
    attachments: list[WorkflowAttachmentMeta] = Field(default_factory=list, max_length=5)


class WorkflowCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=96)
    config: WorkflowConfig


class WorkflowSummary(BaseModel):
    workflow_id: str
    name: str
    created_at: str
    config: WorkflowConfig


class WorkflowListResponse(BaseModel):
    data: list[WorkflowSummary]
