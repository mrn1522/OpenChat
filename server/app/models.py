import base64
import binascii
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

MAX_TEXT_ATTACHMENT_BYTES = 262_144
MAX_IMAGE_ATTACHMENT_BYTES = 5 * 1024 * 1024
# Bound on the total decoded image payload one direct-chat request can carry
# (current attachments plus prior-turn re-embedded pixels).
MAX_TOTAL_IMAGE_BYTES = 20 * 1024 * 1024
MAX_TEXT_ATTACHMENT_CHARS = 100_000
# ceil(5MiB / 3) * 4 base64 chars, with padding headroom.
MAX_IMAGE_BASE64_CHARS = 7_000_000
# Formats every major vision provider accepts via OpenRouter data URLs.
SUPPORTED_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})


def base64_decoded_len(content: str) -> int:
    """Decoded byte length of a base64 string, excluding trailing `=` padding."""
    return 3 * (len(content) // 4) - (len(content) - len(content.rstrip("=")))

# OpenRouter service tiers. "fast" is an upstream alias for "priority" and is
# normalized to "priority" at the boundary, so only these three are valid here.
ServiceTier = Literal["default", "flex", "priority"]


class ReasoningConfig(BaseModel):
    effort: Literal["max", "xhigh", "high", "medium", "low", "minimal", "none"] = "medium"
    exclude: bool = False


class SettingsResponse(BaseModel):
    api_key_configured: bool
    api_key_hint: str | None
    base_url: str


class SettingsUpdateRequest(BaseModel):
    api_key: str | None = None
    base_url: str | None = None


class AttachmentInput(BaseModel):
    """A composer attachment: either an inlined text file or an image.

    Image attachments carry raw base64 in ``content`` (no ``data:`` prefix)
    and are rendered as ``image_url`` parts in direct chat requests.
    """

    name: str = Field(min_length=1, max_length=256)
    size: int = Field(ge=1)
    content_type: str = Field(default="application/octet-stream", max_length=128)
    content: str = Field(min_length=1)

    @property
    def is_image(self) -> bool:
        return self.content_type.startswith("image/")

    @model_validator(mode="after")
    def _validate_payload_size(self) -> "AttachmentInput":
        if self.is_image:
            if self.content_type not in SUPPORTED_IMAGE_TYPES:
                raise ValueError(f"unsupported image type: {self.content_type}")
            if self.size > MAX_IMAGE_ATTACHMENT_BYTES:
                raise ValueError(
                    f"image attachment exceeds {MAX_IMAGE_ATTACHMENT_BYTES} bytes"
                )
            # Check the encoded length before decoding so oversized payloads
            # are rejected without allocating for them.
            if len(self.content) > MAX_IMAGE_BASE64_CHARS:
                raise ValueError("image attachment exceeds maximum base64 length")
            try:
                decoded = base64.b64decode(self.content, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("image attachment content is not valid base64") from exc
            if len(decoded) > MAX_IMAGE_ATTACHMENT_BYTES:
                raise ValueError(
                    f"image attachment exceeds {MAX_IMAGE_ATTACHMENT_BYTES} bytes"
                )
        else:
            if self.size > MAX_TEXT_ATTACHMENT_BYTES:
                raise ValueError(
                    f"attachment exceeds {MAX_TEXT_ATTACHMENT_BYTES} bytes"
                )
            if len(self.content) > MAX_TEXT_ATTACHMENT_CHARS:
                raise ValueError("attachment exceeds maximum content length")
        return self


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
    # Per-model service tier map keyed by model id (source agents, debate
    # reviewers, and the fusion model all resolve through this map).
    service_tiers: dict[str, ServiceTier] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _reject_image_attachments(self) -> "RunRequest":
        # Source/fusion/debate prompts are text-only; images dropped silently
        # is worse than a clear rejection — direct chat is the image path.
        if any(attachment.is_image for attachment in self.attachments):
            raise ValueError("image attachments are only supported in direct chat")
        return self


class DirectChatImageMeta(BaseModel):
    """Provenance for an image attached to a direct-chat turn.

    The current turn's pixels travel in ``DirectChatRequest.attachments``;
    prior turns re-embed their pixels here as base64 ``content`` so follow-up
    requests still carry earlier images. History persists only name and
    content_type — never the payload.
    """

    name: str = Field(min_length=1, max_length=256)
    content_type: str = Field(default="image/png", max_length=128)
    content: str | None = Field(default=None, max_length=MAX_IMAGE_BASE64_CHARS)

    @model_validator(mode="after")
    def _validate_content(self) -> "DirectChatImageMeta":
        if self.content_type not in SUPPORTED_IMAGE_TYPES:
            raise ValueError(f"unsupported image type: {self.content_type}")
        if self.content is None:
            return self
        try:
            decoded = base64.b64decode(self.content, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image content is not valid base64") from exc
        if len(decoded) > MAX_IMAGE_ATTACHMENT_BYTES:
            raise ValueError(
                f"image content exceeds {MAX_IMAGE_ATTACHMENT_BYTES} bytes"
            )
        return self


class DirectChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=100000)
    images: list[DirectChatImageMeta] = Field(default_factory=list, max_length=5)


class DirectChatRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[DirectChatMessage] = Field(min_length=1, max_length=200)
    conversation_id: str = Field(default="", max_length=64)
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_output_tokens: int = Field(default=1000, ge=128, le=10000)
    web_search_enabled: bool = False
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    attachments: list[AttachmentInput] = Field(default_factory=list, max_length=5)
    service_tier: ServiceTier | None = None

    @model_validator(mode="after")
    def _cap_total_image_bytes(self) -> "DirectChatRequest":
        # Per-image caps alone let 200 messages x 5 images exceed what a
        # single upstream call should carry — bound the cumulative payload.
        total_bytes = sum(
            base64_decoded_len(image.content)
            for message in self.messages
            for image in message.images
            if image.content
        ) + sum(
            base64_decoded_len(attachment.content)
            for attachment in self.attachments
            if attachment.is_image
        )
        if total_bytes > MAX_TOTAL_IMAGE_BYTES:
            raise ValueError(
                f"total image payload exceeds {MAX_TOTAL_IMAGE_BYTES} bytes"
            )
        return self


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
    service_tier: ServiceTier | None = None


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


class ServiceTiersResponse(BaseModel):
    """Non-default service tiers discovered for one model.

    Populated from the model's provider endpoint list: tier-capable endpoints
    carry a slug suffix (``openai/flex``, ``openai/fast``,
    ``google-vertex/global/priority``). ``fast`` is an alias for ``priority``.
    """

    model: str
    tiers: list[ServiceTier]


class ChatHistorySummary(BaseModel):
    chat_id: str
    created_at: str
    updated_at: str
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
    messages: list[DirectChatMessage] = []


class WorkflowAttachmentMeta(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    size: int = Field(ge=0, le=262144)
    content_type: str = Field(default="application/octet-stream", max_length=128)


class WorkflowConfig(BaseModel):
    source_models: list[str] = Field(min_length=1)
    fusion_model: str = Field(min_length=1)
    temperature: float = Field(default=0.2, ge=0, le=2)
    reasoning_effort: Literal["max", "xhigh", "high", "medium", "low", "minimal", "none"] = "medium"
    debate_mode: Literal["off", "partial", "full"] = "partial"
    web_search_enabled: bool = False
    persona_enabled: bool = False
    attachments: list[WorkflowAttachmentMeta] = Field(default_factory=list, max_length=5)
    service_tiers: dict[str, ServiceTier] = Field(default_factory=dict)


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
