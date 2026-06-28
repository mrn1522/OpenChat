import type {
  ChatHistoryDetail,
  ChatHistoryListResponse,
  DirectChatRequest,
  DirectChatStreamEvent,
  FusionRegenerateRequest,
  FusionRegenerateResponse,
  OpenRouterModel,
  OpenRouterModelsResponse,
  PersonaPreviewRequest,
  PersonaPreviewResponse,
  PromptOptimizeRequest,
  PromptOptimizeResponse,
  RunRequest,
  StreamEvent,
  WorkflowCreateRequest,
  WorkflowListResponse,
  SavedWorkflow,
} from "./types";

const API_BASE =
  import.meta.env.VITE_OPENCHAT_API_BASE ?? "http://localhost:8000";

const extractErrorMessage = async (response: Response): Promise<string> => {
  const text = await response.text();
  if (!text) return `Request failed with status ${response.status}`;

  try {
    const parsed = JSON.parse(text) as { detail?: string };
    if (parsed.detail) return parsed.detail;
  } catch {
    // Plain text error payload.
  }

  return text;
};

export async function streamRun(
  payload: RunRequest,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/run/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(await extractErrorMessage(response));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      const lines = chunk.split("\n");
      const eventLine = lines.find((line) => line.startsWith("event:"));
      const dataLine = lines.find((line) => line.startsWith("data:"));
      if (!eventLine || !dataLine) continue;

      const eventType = eventLine.replace("event:", "").trim();
      const json = dataLine.replace("data:", "").trim();

      try {
        const data = JSON.parse(json);
        onEvent(
          {
            type: eventType,
            ...(eventType === "run_started"
              ? { run_id: data.run_id }
              : { data }),
          } as StreamEvent
        );
      } catch {
        // Ignore malformed payloads.
      }
    }
  }
}

export async function streamDirectChat(
  payload: DirectChatRequest,
  onEvent: (event: DirectChatStreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/direct-chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(await extractErrorMessage(response));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      const lines = chunk.split("\n");
      const eventLine = lines.find((line) => line.startsWith("event:"));
      const dataLine = lines.find((line) => line.startsWith("data:"));
      if (!eventLine || !dataLine) continue;

      const eventType = eventLine.replace("event:", "").trim();
      const json = dataLine.replace("data:", "").trim();

      try {
        const data = JSON.parse(json);
        onEvent(
          {
            type: eventType,
            ...(eventType === "run_started"
              ? { run_id: data.run_id }
              : { data }),
          } as DirectChatStreamEvent
        );
      } catch {
        // Ignore malformed payloads.
      }
    }
  }
}

export async function regenerateFusion(
  payload: FusionRegenerateRequest,
  signal?: AbortSignal
): Promise<FusionRegenerateResponse> {
  const response = await fetch(`${API_BASE}/api/fusion/regenerate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  return (await response.json()) as FusionRegenerateResponse;
}

export async function fetchModels(signal?: AbortSignal): Promise<OpenRouterModel[]> {
  const response = await fetch(`${API_BASE}/api/models`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
    signal,
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  const payload = (await response.json()) as OpenRouterModelsResponse;
  return payload.data;
}

export async function optimizePrompt(
  payload: PromptOptimizeRequest,
  signal?: AbortSignal
): Promise<PromptOptimizeResponse> {
  const response = await fetch(`${API_BASE}/api/prompt/optimize`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  return (await response.json()) as PromptOptimizeResponse;
}

export async function previewPersonas(
  payload: PersonaPreviewRequest,
  signal?: AbortSignal
): Promise<PersonaPreviewResponse> {
  const response = await fetch(`${API_BASE}/api/personas/preview`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  return (await response.json()) as PersonaPreviewResponse;
}

export async function fetchChatHistory(signal?: AbortSignal): Promise<ChatHistoryListResponse> {
  const response = await fetch(`${API_BASE}/api/chats`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
    signal,
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  return (await response.json()) as ChatHistoryListResponse;
}

export async function fetchChatHistoryDetail(chatId: string, signal?: AbortSignal): Promise<ChatHistoryDetail> {
  const response = await fetch(`${API_BASE}/api/chats/${encodeURIComponent(chatId)}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
    signal,
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  return (await response.json()) as ChatHistoryDetail;
}

export async function deleteChatHistory(chatId: string, signal?: AbortSignal): Promise<void> {
  const response = await fetch(`${API_BASE}/api/chats/${encodeURIComponent(chatId)}`, {
    method: "DELETE",
    headers: {
      "Content-Type": "application/json",
    },
    signal,
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }
}

export async function fetchWorkflows(signal?: AbortSignal): Promise<WorkflowListResponse> {
  const response = await fetch(`${API_BASE}/api/workflows`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    },
    signal,
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  return (await response.json()) as WorkflowListResponse;
}

export async function createWorkflow(payload: WorkflowCreateRequest, signal?: AbortSignal): Promise<SavedWorkflow> {
  const response = await fetch(`${API_BASE}/api/workflows`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  return (await response.json()) as SavedWorkflow;
}

export async function deleteWorkflow(workflowId: string, signal?: AbortSignal): Promise<void> {
  const response = await fetch(`${API_BASE}/api/workflows/${encodeURIComponent(workflowId)}`, {
    method: "DELETE",
    headers: {
      "Content-Type": "application/json",
    },
    signal,
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }
}
