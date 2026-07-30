export type SourceAgentSpec = {
  id: string;
  model: string;
};

export type SourceModelResult = {
  model: string;
  agent_id?: string;
  content: string;
  status: "ok" | "error";
  persona?: PersonaAssignment;
  error?: string;
  latency_ms?: number;
};

export type PersonaAssignment = {
  model: string;
  agent_id?: string;
  title: string;
  temperature: number;
  description: string;
};

export type DebateResult = {
  target_model: string;
  reviewer_model: string;
  target_agent_id?: string;
  reviewer_agent_id?: string;
  content: string;
};

export type ReasoningEffort = "xhigh" | "high" | "medium" | "low" | "minimal" | "none";

export type ReasoningConfig = {
  effort: ReasoningEffort;
  exclude: boolean;
};

export type DebateMode = "off" | "partial" | "full";

export type AttachmentInput = {
  name: string;
  size: number;
  content_type: string;
  content: string;
};

export type DirectChatRole = "user" | "assistant";

export type DirectChatMessage = {
  role: DirectChatRole;
  content: string;
};

export type StreamEvent =
  | { type: "run_started"; run_id: string }
  | {
      type: "orchestration_step";
      data: {
        step: "route" | "search" | "parallel" | "critique" | "fusion";
        status: "pending" | "active" | "done";
        detail?: string;
      };
    }
  | { type: "source_result"; data: SourceModelResult }
  | { type: "persona_assignments"; data: { items: PersonaAssignment[]; error?: string } }
  | { type: "debate_result"; data: DebateResult }
  | { type: "critique_ready"; data: { model: string; content: string } }
  | { type: "fusion_ready"; data: { model: string; content: string } }
  | { type: "completed"; data: { elapsed_ms: number } }
  | { type: "error"; data: { message: string } };

export type DirectChatStreamEvent =
  | { type: "run_started"; run_id: string }
  | { type: "assistant_chunk"; data: { content: string } }
  | { type: "assistant_ready"; data: { model: string; content: string } }
  | { type: "completed"; data: { elapsed_ms: number } }
  | { type: "error"; data: { message: string } };

export type RunRequest = {
  prompt: string;
  source_models: string[];
  source_agents?: SourceAgentSpec[];
  fusion_model: string;
  debate_mode?: DebateMode;
  temperature?: number;
  max_output_tokens?: number;
  web_search_enabled?: boolean;
  persona_enabled?: boolean;
  persona_assignments_override?: PersonaAssignment[];
  reasoning?: ReasoningConfig;
  attachments?: AttachmentInput[];
};

export type DirectChatRequest = {
  model: string;
  messages: DirectChatMessage[];
  temperature?: number;
  max_output_tokens?: number;
  web_search_enabled?: boolean;
  reasoning?: ReasoningConfig;
  attachments?: AttachmentInput[];
};

export type FusionRegenerateRequest = {
  prompt: string;
  fusion_model: string;
  temperature?: number;
  reasoning?: ReasoningConfig;
  source_results: SourceModelResult[];
  critique_output: string;
};

export type FusionRegenerateResponse = {
  model: string;
  content: string;
};

export type PersonaPreviewRequest = {
  prompt: string;
  source_models?: string[];
  source_agents?: SourceAgentSpec[];
};

export type PersonaPreviewResponse = {
  items: PersonaAssignment[];
  model: string;
};

export type PromptOptimizeRequest = {
  prompt: string;
};

export type PromptOptimizeResponse = {
  original_prompt: string;
  optimized_prompt: string;
  model: string;
};

export type OpenRouterModel = {
  id: string;
  created?: number | null;
  created_at?: string | null;
  createdAt?: string | null;
  canonical_slug?: string | null;
  name?: string | null;
  description?: string | null;
  context_length?: number | null;
  architecture?: {
    modality?: string;
    input_modalities?: string[];
    output_modalities?: string[];
    tokenizer?: string;
    instruct_type?: string | null;
  } | null;
  pricing?: Record<string, string | number | null> | null;
  top_provider?: Record<string, string | number | boolean | null> | null;
  supported_parameters?: string[] | null;
  default_parameters?: Record<string, string | number | boolean | null> | null;
  links?: Record<string, string | null> | null;
};

export type OpenRouterModelsResponse = {
  data: OpenRouterModel[];
};

export type ChatHistorySummary = {
  chat_id: string;
  created_at: string;
  status: string;
  prompt_preview: string;
  source_models: string[];
  fusion_model: string;
  has_fusion_output: boolean;
};

export type ChatHistoryListResponse = {
  data: ChatHistorySummary[];
};

export type ChatHistoryDetail = {
  chat_id: string;
  created_at: string;
  run_id: string;
  status: string;
  elapsed_ms: number | null;
  request: RunRequest;
  source_results: SourceModelResult[];
  debate_results: DebateResult[];
  critique_output: string;
  fusion_output: string;
};

export type WorkflowAttachmentMeta = {
  name: string;
  size: number;
  content_type: string;
};

export type SavedWorkflowConfig = {
  source_models: string[];
  fusion_model: string;
  temperature: number;
  reasoning_effort: ReasoningEffort;
  debate_mode: DebateMode;
  web_search_enabled: boolean;
  persona_enabled: boolean;
  attachments: WorkflowAttachmentMeta[];
};

export type SavedWorkflow = {
  workflow_id: string;
  name: string;
  created_at: string;
  config: SavedWorkflowConfig;
};

export type WorkflowListResponse = {
  data: SavedWorkflow[];
};

export type WorkflowCreateRequest = {
  name: string;
  config: SavedWorkflowConfig;
};
