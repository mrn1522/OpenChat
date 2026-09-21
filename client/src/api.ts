import type {
  ChatHistoryDetail,
  ChatHistoryListResponse,
  AppSettings,
  AppSettingsUpdate,
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
  UpdateCheckResult,
  UpdateInstaller,
} from "./types";

import pkg from "../package.json";

export const isDesktopApp = (): boolean =>
  "window" in globalThis && "__TAURI_INTERNALS__" in window;

let apiBasePromise: Promise<string> | undefined;

const getApiBase = (): Promise<string> => {
  if (!apiBasePromise) {
    apiBasePromise = isDesktopApp()
      ? import("@tauri-apps/api/core").then(({ invoke }) => invoke<string>("api_base"))
      : Promise.resolve(import.meta.env.VITE_OPENCHAT_API_BASE ?? "http://localhost:8000");
  }
  return apiBasePromise;
};

let appVersionPromise: Promise<string> | undefined;

export const getAppVersion = (): Promise<string> => {
  if (!appVersionPromise) {
    appVersionPromise = isDesktopApp()
      ? import("@tauri-apps/api/app").then(({ getVersion }) => getVersion())
      : Promise.resolve(pkg.version);
  }
  return appVersionPromise;
};

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
  const base = await getApiBase();
  const response = await fetch(`${base}/api/run/stream`, {
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
  const base = await getApiBase();
  const response = await fetch(`${base}/api/direct-chat/stream`, {
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
              ? { run_id: data.run_id, conversation_id: data.conversation_id }
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
  const base = await getApiBase();
  const response = await fetch(`${base}/api/fusion/regenerate`, {
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
  const base = await getApiBase();
  const response = await fetch(`${base}/api/models`, {
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
  const base = await getApiBase();
  const response = await fetch(`${base}/api/prompt/optimize`, {
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
  const base = await getApiBase();
  const response = await fetch(`${base}/api/personas/preview`, {
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
  const base = await getApiBase();
  const response = await fetch(`${base}/api/chats`, {
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
  const base = await getApiBase();
  const response = await fetch(`${base}/api/chats/${encodeURIComponent(chatId)}`, {
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
  const base = await getApiBase();
  const response = await fetch(`${base}/api/chats/${encodeURIComponent(chatId)}`, {
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
  const base = await getApiBase();
  const response = await fetch(`${base}/api/workflows`, {
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
  const base = await getApiBase();
  const response = await fetch(`${base}/api/workflows`, {
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
  const base = await getApiBase();
  const response = await fetch(`${base}/api/workflows/${encodeURIComponent(workflowId)}`, {
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

export async function getSettings(signal?: AbortSignal): Promise<AppSettings> {
  const base = await getApiBase();
  const response = await fetch(`${base}/api/settings`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
    signal,
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  return (await response.json()) as AppSettings;
}

export async function updateSettings(
  payload: AppSettingsUpdate,
  signal?: AbortSignal
): Promise<AppSettings> {
  const base = await getApiBase();
  const response = await fetch(`${base}/api/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  return (await response.json()) as AppSettings;
}

const LATEST_RELEASE_URL =
  "https://api.github.com/repos/mrn1522/OpenChat/releases/latest";

type GitHubReleaseAsset = {
  name: string;
  browser_download_url: string;
  size: number;
  digest?: string | null;
};

type GitHubRelease = {
  tag_name: string;
  html_url: string;
  published_at: string | null;
  assets: GitHubReleaseAsset[];
};

// Strict semver: three core identifiers without leading zeroes, optional
// -prerelease (numeric identifiers also reject leading zeroes) and +build.
const VERSION_RE =
  /^((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$/;

const parseVersion = (
  version: string
): { core: string[]; prerelease: boolean } | null => {
  const match = VERSION_RE.exec(version.trim().replace(/^v/i, ""));
  if (!match) return null;
  return { core: match[1].split("."), prerelease: match[2] !== undefined };
};

// Exact ordering for digit strings of any length — no Number precision loss.
const compareNumericSegments = (a: string, b: string): number => {
  const left = a.replace(/^0+/, "") || "0";
  const right = b.replace(/^0+/, "") || "0";
  if (left.length !== right.length) return left.length - right.length;
  return left < right ? -1 : left > right ? 1 : 0;
};

export const isNewerVersion = (latest: string, current: string): boolean | null => {
  const latestParts = parseVersion(latest);
  const currentParts = parseVersion(current);
  if (!latestParts || !currentParts) return null;
  const length = Math.max(latestParts.core.length, currentParts.core.length);
  for (let i = 0; i < length; i += 1) {
    const diff = compareNumericSegments(latestParts.core[i] ?? "0", currentParts.core[i] ?? "0");
    if (diff !== 0) return diff > 0;
  }
  // Same core: a stable release outranks a prerelease.
  return !latestParts.prerelease && currentParts.prerelease;
};

const pickInstallerAsset = (assets: GitHubReleaseAsset[]): UpdateInstaller | null => {
  const executables = assets.filter((asset) => /\.exe$/i.test(asset.name));
  if (executables.length === 0) return null;
  const installer =
    executables.find((asset) => /setup|nsis/i.test(asset.name)) ?? executables[0];
  const digest = installer.digest ?? "";
  return {
    name: installer.name,
    url: installer.browser_download_url,
    size: installer.size,
    sha256: digest.toLowerCase().startsWith("sha256:") ? digest.slice(7) : null,
  };
};

export async function checkForUpdate(signal?: AbortSignal): Promise<UpdateCheckResult> {
  const currentVersion = await getAppVersion();
  const response = await fetch(LATEST_RELEASE_URL, {
    headers: { Accept: "application/vnd.github+json" },
    signal,
  });
  if (response.status === 404) {
    throw new Error("No published GitHub release found.");
  }
  if (response.status === 403 || response.status === 429) {
    throw new Error("GitHub rate limit reached — try again later.");
  }
  if (!response.ok) {
    throw new Error(`GitHub release check failed (HTTP ${response.status}).`);
  }

  const release = (await response.json()) as GitHubRelease;
  const latestVersion = release.tag_name.replace(/^v/i, "");
  const comparison = isNewerVersion(latestVersion, currentVersion);
  if (comparison === null) {
    throw new Error(
      `Cannot compare release tag "${release.tag_name}" against version ${currentVersion}.`
    );
  }
  return {
    status: comparison ? "available" : "up-to-date",
    currentVersion,
    latestVersion,
    tagName: release.tag_name,
    releaseUrl: release.html_url,
    publishedAt: release.published_at,
    installer: pickInstallerAsset(release.assets),
  };
}

export async function installDesktopUpdate(installer: UpdateInstaller): Promise<void> {
  if (!installer.sha256) {
    throw new Error("Release asset has no integrity digest — download it manually instead.");
  }
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("install_update", {
    downloadUrl: installer.url,
    sha256: installer.sha256,
    fileName: installer.name,
  });
}
