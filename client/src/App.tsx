import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, ClipboardEvent } from "react";
import type { ComponentType } from "react";
import type { CSSProperties } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import DirectChatTranscript from "./DirectChatTranscript";
import {
  Anthropic,
  Aws,
  Cohere,
  DeepSeek,
  Google,
  Meta,
  Microsoft,
  Mistral,
  Moonshot,
  Nvidia,
  OpenAI,
  Perplexity,
  Qwen,
  XAI,
  Zhipu,
} from "@lobehub/icons";
import brainCogIcon from "./assets/icons/brain-cog.svg";
import bookmarkIcon from "./assets/icons/bookmark.svg";
import copyIcon from "./assets/icons/copy.svg";
import downloadIcon from "./assets/icons/download.svg";
import gemIcon from "./assets/icons/gem.svg";
import globeIcon from "./assets/icons/globe.svg";
import historyIcon from "./assets/icons/history.svg";
import messageSquareIcon from "./assets/icons/message-square.svg";
import moonIcon from "./assets/icons/moon.svg";
import paperclipIcon from "./assets/icons/paperclip.svg";
import rotateCcwIcon from "./assets/icons/rotate-ccw.svg";
import searchIcon from "./assets/icons/search.svg";
import sendIcon from "./assets/icons/send.svg";
import settingsIcon from "./assets/icons/settings.svg";
import shieldIcon from "./assets/icons/shield.svg";
import slidersHorizontalIcon from "./assets/icons/sliders-horizontal.svg";
import sparklesIcon from "./assets/icons/sparkles.svg";
import squarePenIcon from "./assets/icons/square-pen.svg";
import sunIcon from "./assets/icons/sun.svg";
import waypointsIcon from "./assets/icons/waypoints.svg";
import zapIcon from "./assets/icons/zap.svg";
import {
  checkForUpdate,
  createWorkflow,
  deleteWorkflow,
  deleteChatHistory,
  fetchChatHistory,
  fetchChatHistoryDetail,
  fetchModels,
  fetchServiceTiers,
  fetchWorkflows,
  getAppVersion,
  getSettings,
  installDesktopUpdate,
  isDesktopApp,
  isWindowsDesktop,
  optimizePrompt,
  previewPersonas,
  regenerateFusion,
  streamDirectChat,
  streamRun,
  updateSettings,
} from "./api";
import type {
  AppSettings,
  AppSettingsUpdate,
  AttachmentInput,
  ChatHistoryDetail,
  ChatHistorySummary,
  DebateMode,
  DebateResult,
  DirectChatMessage,
  DirectChatStreamEvent,
  OpenRouterModel,
  PersonaAssignment,
  ReasoningEffort,
  RunRequest,
  SavedWorkflow,
  SavedWorkflowConfig,
  ServiceTier,
  SourceAgentSpec,
  SourceModelResult,
  StreamEvent,
  UpdateInstaller,
  UpdateState,
} from "./types";

type PickerKind = "source" | "fusion" | "direct";
type OrchestrationStep = "route" | "search" | "parallel" | "critique" | "fusion";
type StepStatus = "pending" | "active" | "done";
type ResultTab = "fusion" | "summaries" | "sources";
type AppPage = "fusion" | "direct" | "history";
type ThemeMode = "dark" | "light";

const readStoredTheme = (): ThemeMode => {
  try {
    return window.localStorage.getItem("openchat-theme") === "light" ? "light" : "dark";
  } catch {
    return "dark";
  }
};

const visibleSettingsButton = (): HTMLElement | null => {
  const railButton = document.querySelector<HTMLElement>(".rail-settings-btn");
  if (railButton && railButton.getClientRects().length > 0) return railButton;
  return document.querySelector<HTMLElement>(".mobile-settings-btn");
};
type ExperienceModeTab = {
  id: string;
  title: string;
  sub: string;
  icon: string;
  workflowId: string | null;
};

type ComposerAttachment = AttachmentInput & {
  id: string;
  // Object URL for thumbnails — built from the source File so no extra
  // base64 copy is retained per turn. Client-only; stripped before transport.
  thumb_url?: string;
};

type DisplayModel = {
  id: string;
  name: string;
  provider: string;
  createdTimestamp: number | null;
  catalogRank: number | null;
  description: string | null;
  contextLength: number | null;
  modality: string | null;
  promptPrice: string | null;
  completionPrice: string | null;
};

type ProviderIconProps = {
  size?: string | number;
  className?: string;
};

type ProviderIconVariant = ComponentType<ProviderIconProps>;

type ProviderIconSet = ProviderIconVariant &
  Partial<{
    Color: ProviderIconVariant;
    BrandColor: ProviderIconVariant;
    colorPrimary: string;
    colorGradient: string;
  }>;

type ProviderIconRender = {
  Icon: ProviderIconVariant;
  tint: string | null;
};

const PROVIDER_ICON_BY_KEY = {
  anthropic: Anthropic,
  aws: Aws,
  cohere: Cohere,
  deepseek: DeepSeek,
  google: Google,
  meta: Meta,
  microsoft: Microsoft,
  mistral: Mistral,
  moonshot: Moonshot,
  nvidia: Nvidia,
  openai: OpenAI,
  perplexity: Perplexity,
  qwen: Qwen,
  xai: XAI,
  zhipu: Zhipu,
} satisfies Record<string, ProviderIconSet>;

type ProviderIconKey = keyof typeof PROVIDER_ICON_BY_KEY;

const PROVIDER_KEY_ALIASES: Record<string, ProviderIconKey> = {
  anthropic: "anthropic",
  claude: "anthropic",
  amazon: "aws",
  aws: "aws",
  bedrock: "aws",
  cohere: "cohere",
  commanda: "cohere",
  deepseek: "deepseek",
  gemini: "google",
  google: "google",
  meta: "meta",
  metallama: "meta",
  llama: "meta",
  microsoft: "microsoft",
  azure: "microsoft",
  mistral: "mistral",
  moonshot: "moonshot",
  kimi: "moonshot",
  nvidia: "nvidia",
  openai: "openai",
  chatgpt: "openai",
  perplexity: "perplexity",
  qwen: "qwen",
  tongyi: "qwen",
  alibaba: "qwen",
  xai: "xai",
  grok: "xai",
  zhipu: "zhipu",
  chatglm: "zhipu",
  glm: "zhipu",
};

const normalizeProviderToken = (value: string): string => value.toLowerCase().replace(/[^a-z0-9]+/g, "");

const resolveProviderIconVariant = (icon: ProviderIconSet): ProviderIconRender => {
  const Icon = icon.Color ?? icon.BrandColor ?? icon;
  const tint = icon.colorPrimary ?? icon.colorGradient ?? null;
  return { Icon, tint };
};

const resolveProviderKey = (candidate: string): ProviderIconKey | null => {
  const normalized = normalizeProviderToken(candidate);
  if (!normalized) return null;

  if (normalized in PROVIDER_ICON_BY_KEY) {
    return normalized as ProviderIconKey;
  }

  if (normalized in PROVIDER_KEY_ALIASES) {
    return PROVIDER_KEY_ALIASES[normalized];
  }

  return null;
};

const resolveProviderIcon = (model: DisplayModel): ProviderIconRender | null => {
  const providerFromId = model.id.includes("/") ? model.id.split("/")[0] : model.provider;
  const rawCandidates = [providerFromId, model.provider, model.name, model.id];

  for (const candidate of rawCandidates) {
    const key = resolveProviderKey(candidate);
    if (key) return resolveProviderIconVariant(PROVIDER_ICON_BY_KEY[key]);
  }

  const tokenized = rawCandidates
    .join(" ")
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);

  for (const token of tokenized) {
    const key = resolveProviderKey(token);
    if (key) return resolveProviderIconVariant(PROVIDER_ICON_BY_KEY[key]);
  }

  const compact = normalizeProviderToken(rawCandidates.join(""));
  for (const [alias, key] of Object.entries(PROVIDER_KEY_ALIASES)) {
    if (compact.includes(alias)) return resolveProviderIconVariant(PROVIDER_ICON_BY_KEY[key]);
  }

  return null;
};

const EXPERIENCE_MODES = [
  { id: "quality", title: "Quality", sub: "Deepest answers", icon: gemIcon },
  { id: "speed", title: "Speed", sub: "Fastest results", icon: zapIcon },
  { id: "custom", title: "Custom", sub: "Your workflow", icon: slidersHorizontalIcon },
] as const;

const EXPERIENCE_MODE_PRESETS: Partial<
  Record<(typeof EXPERIENCE_MODES)[number]["id"], { sourceModels: string[]; fusionModel: string }>
> = {
  speed: {
    sourceModels: ["z-ai/glm-4.7", "openai/gpt-oss-120b"],
    fusionModel: "openai/gpt-oss-120b",
  },
  quality: {
    sourceModels: ["openai/gpt-5.5", "google/gemini-3.1-pro-preview"],
    fusionModel: "openai/gpt-5.5",
  },
};

const workflowModeId = (workflowId: string) => `workflow:${workflowId}`;

const ORCHESTRATION_FLOW: Array<{ id: OrchestrationStep; label: string; detail: string; icon: string }> = [
  { id: "route", label: "Route", detail: "Understanding your query and selecting best agents.", icon: waypointsIcon },
  { id: "search", label: "Search", detail: "Agents are searching the web and knowledge bases.", icon: searchIcon },
  { id: "parallel", label: "Parallel Reasoning", detail: "Selected agents reason in parallel.", icon: brainCogIcon },
  { id: "critique", label: "Critique", detail: "Agents debate and critique each response.", icon: shieldIcon },
  { id: "fusion", label: "Fusion", detail: "Fusion model synthesizes the final answer.", icon: sparklesIcon },
];

const REASONING_EFFORT_OPTIONS: ReasoningEffort[] = ["max", "xhigh", "high", "medium", "low", "minimal", "none"];
const DEBATE_MODE_OPTIONS: DebateMode[] = ["off", "partial", "full"];
const SERVICE_TIER_LABELS: Record<ServiceTier, string> = {
  default: "Default",
  flex: "Flex",
  priority: "Priority",
};
const SERVICE_TIER_HINTS: Record<ServiceTier, string> = {
  default: "",
  flex: "lower cost",
  priority: "faster",
};
const SERVICE_TIER_TOOLTIPS: Record<ServiceTier, string> = {
  default: "Default routing",
  flex: "Flex — lower cost, slower",
  priority: "Priority — faster, higher cost",
};
const SERVICE_TIER_RETRY_MS = 60_000;
const OPENROUTER_TOKEN_LIMIT = 10_000;
const MAX_ATTACHMENTS = 5;
const MAX_ATTACHMENT_SIZE_BYTES = 262_144;
const MAX_ATTACHMENT_CONTENT_CHARS = 100_000;
// 5MB — the tightest image cap across common vision providers (Anthropic).
const MAX_IMAGE_ATTACHMENT_SIZE_BYTES = 5_242_880;
const IMAGE_CONTENT_TYPE_PREFIX = "image/";
// Formats every major vision provider accepts via OpenRouter data URLs.
const SUPPORTED_IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/gif", "image/webp"]);
const SUPPORTED_ATTACHMENT_TYPES = new Set([
  "txt",
  "md",
  "json",
  "csv",
  "py",
  "js",
  "ts",
  "tsx",
  "jsx",
  "html",
  "css",
  "yaml",
  "yml",
  "xml",
  "log",
]);

const AGENT_COLORS = ["orange", "green", "cyan", "blue", "purple"] as const;

const initialStepState = (): Record<OrchestrationStep, StepStatus> => ({
  route: "pending",
  search: "pending",
  parallel: "pending",
  critique: "pending",
  fusion: "pending",
});

const initialStepDetail = (): Record<OrchestrationStep, string> => ({
  route: ORCHESTRATION_FLOW[0].detail,
  search: ORCHESTRATION_FLOW[1].detail,
  parallel: ORCHESTRATION_FLOW[2].detail,
  critique: ORCHESTRATION_FLOW[3].detail,
  fusion: ORCHESTRATION_FLOW[4].detail,
});

const toTitle = (value: string) =>
  value
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");

const fallbackNameFromId = (id: string) => {
  const [, modelSlug = id] = id.split("/");
  return modelSlug
    .split("-")
    .filter(Boolean)
    .map((part) => {
      if (/^\d/.test(part)) return part;
      if (part.length <= 3) return part.toUpperCase();
      return part.charAt(0).toUpperCase() + part.slice(1);
    })
    .join(" ");
};

const toPrice = (value: unknown): string | null => {
  if (value === null || value === undefined || value === "") return null;
  const numeric = typeof value === "number" ? value : Number.parseFloat(String(value));
  if (!Number.isFinite(numeric)) return null;
  const perMillion = numeric * 1_000_000;
  return `$${perMillion.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })} / M tokens`;
};

const normalizeEpochMillis = (value: number): number | null => {
  if (!Number.isFinite(value)) return null;
  const absValue = Math.abs(value);
  if (absValue < 1e11) return Math.trunc(value * 1000);
  if (absValue < 1e14) return Math.trunc(value);
  if (absValue < 1e17) return Math.trunc(value / 1000);
  return null;
};

const toCreatedTimestamp = (model: OpenRouterModel | undefined): number | null => {
  if (!model) return null;
  if (typeof model.created === "number" && Number.isFinite(model.created)) {
    return normalizeEpochMillis(model.created);
  }
  const createdAtRaw = model.created_at ?? model.createdAt;
  if (typeof createdAtRaw === "number" && Number.isFinite(createdAtRaw)) {
    return normalizeEpochMillis(createdAtRaw);
  }
  if (typeof createdAtRaw === "string") {
    const maybeNumeric = Number.parseFloat(createdAtRaw);
    if (Number.isFinite(maybeNumeric)) {
      const normalized = normalizeEpochMillis(maybeNumeric);
      if (normalized !== null) return normalized;
    }
    const parsed = Date.parse(createdAtRaw);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
};

const toDisplayModel = (
  id: string,
  model: OpenRouterModel | undefined,
  catalogRank: number | null
): DisplayModel => {
  const provider = id.includes("/") ? id.split("/")[0] : "unknown";

  return {
    id,
    name: (model?.name && model.name.trim()) || fallbackNameFromId(id),
    provider: toTitle(provider),
    createdTimestamp: toCreatedTimestamp(model),
    catalogRank,
    description: model?.description ?? null,
    contextLength: model?.context_length ?? null,
    modality:
      model?.architecture?.modality ??
      model?.architecture?.input_modalities?.join("+") ??
      null,
    promptPrice: toPrice(model?.pricing?.prompt),
    completionPrice: toPrice(model?.pricing?.completion),
  };
};

const isAbortError = (err: unknown): boolean => {
  if (err instanceof DOMException && err.name === "AbortError") return true;
  if (err instanceof Error && err.name === "AbortError") return true;
  return false;
};

const formatAgentStatus = (status: "ready" | "running" | "completed" | "error") => {
  if (status === "completed") return "Completed";
  if (status === "running") return "Running";
  if (status === "error") return "Error";
  return "Ready";
};

const agentKey = (result: { agent_id?: string; model: string }): string =>
  result.agent_id?.trim() || result.model;

const MAX_AGENT_COUNT = 8;

const buildRuntimeAgents = (
  sourceModels: string[],
  counts: Record<string, number>
): SourceAgentSpec[] => {
  const agents: SourceAgentSpec[] = [];
  for (const modelId of sourceModels) {
    const count = Math.max(1, Math.min(MAX_AGENT_COUNT, counts[modelId] ?? 1));
    for (let index = 1; index <= count; index += 1) {
      agents.push({ id: `${modelId}#${index}`, model: modelId });
    }
  }
  return agents;
};

const getChatType = (status: string): "fusion" | "direct" =>
  status.startsWith("direct") ? "direct" : "fusion";

const toChatTypeLabel = (status: string): string =>
  getChatType(status) === "direct" ? "Direct" : "Fusion";

const isImageFile = (file: File): boolean => file.type.startsWith(IMAGE_CONTENT_TYPE_PREFIX);

const isImageAttachment = (attachment: Pick<AttachmentInput, "content_type">): boolean =>
  attachment.content_type.startsWith(IMAGE_CONTENT_TYPE_PREFIX);

const readAttachmentFile = async (file: File): Promise<ComposerAttachment | null> => {
  const extension = file.name.includes(".") ? file.name.split(".").pop()?.toLowerCase() ?? "" : "";
  const looksLikeText = file.type.startsWith("text/") || SUPPORTED_ATTACHMENT_TYPES.has(extension);
  if (!looksLikeText) return null;

  const rawContent = await file.text();
  const content = rawContent.slice(0, MAX_ATTACHMENT_CONTENT_CHARS);
  if (!content.trim()) return null;

  return {
    id: `${file.name}-${file.size}-${file.lastModified}`,
    name: file.name,
    size: file.size,
    content_type: file.type || "text/plain",
    content,
  };
};

const readImageAttachmentFile = (file: File): Promise<ComposerAttachment | null> =>
  new Promise((resolve) => {
    if (!SUPPORTED_IMAGE_TYPES.has(file.type) || file.size === 0) {
      resolve(null);
      return;
    }
    const reader = new FileReader();
    reader.onerror = () => resolve(null);
    reader.onload = () => {
      const result = typeof reader.result === "string" ? reader.result : "";
      const match = /^data:(image\/[\w.+-]+);base64,(.+)$/.exec(result);
      if (!match) {
        resolve(null);
        return;
      }
      const [, contentType, base64] = match;
      const extension = contentType.split("/")[1]?.split("+")[0] ?? "png";
      resolve({
        id: `image-${crypto.randomUUID()}`,
        name: file.name || `pasted-image.${extension}`,
        size: file.size,
        content_type: contentType,
        content: base64,
        thumb_url: URL.createObjectURL(file),
      });
    };
    reader.readAsDataURL(file);
  });

// Re-encodes a blob: object URL back to base64 so earlier turns can re-send
// their pixels on follow-ups without retaining a second base64 copy in state.
const blobUrlToBase64 = async (url: string): Promise<string | null> => {
  try {
    const blob = await fetch(url).then((response) => response.blob());
    const bytes = new Uint8Array(await blob.arrayBuffer());
    let binary = "";
    const chunkSize = 0x8000;
    for (let index = 0; index < bytes.length; index += chunkSize) {
      binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
    }
    return btoa(binary);
  } catch {
    return null;
  }
};

const SettingsUpdateSection = ({ appVersion }: { appVersion: string | null }) => {
  const [updateState, setUpdateState] = useState<UpdateState>({ kind: "idle" });
  const checkAbortRef = useRef<AbortController | null>(null);

  useEffect(
    () => () => {
      checkAbortRef.current?.abort();
    },
    []
  );

  const runUpdateCheck = async () => {
    if (updateState.kind === "checking" || updateState.kind === "installing") return;
    checkAbortRef.current?.abort();
    const controller = new AbortController();
    checkAbortRef.current = controller;
    setUpdateState({ kind: "checking" });
    try {
      const result = await checkForUpdate(controller.signal);
      if (controller.signal.aborted) return;
      setUpdateState(
        result.status === "available"
          ? { kind: "available", result }
          : { kind: "up-to-date", latestVersion: result.latestVersion }
      );
    } catch (err) {
      if (controller.signal.aborted) return;
      setUpdateState({
        kind: "error",
        message: err instanceof Error ? err.message : "Update check failed.",
      });
    }
  };

  const runDesktopUpdate = async (installer: UpdateInstaller) => {
    setUpdateState({ kind: "installing" });
    try {
      await installDesktopUpdate(installer);
      setUpdateState({ kind: "launched" });
    } catch (err) {
      setUpdateState({
        kind: "error",
        message: err instanceof Error ? err.message : "Update install failed.",
      });
    }
  };

  const available = updateState.kind === "available" ? updateState.result : null;
  const installer = available?.installer ?? null;

  return (
    <div className="settings-section">
      <h3 className="settings-section-title">App updates</h3>
      <p className="api-settings-copy">
        {appVersion ? `You're running v${appVersion}. ` : ""}
        Check GitHub for a newer release and install it without leaving the app.
      </p>
      <div className="update-controls">
        <button
          type="button"
          className="subtle-btn"
          onClick={() => void runUpdateCheck()}
          disabled={updateState.kind === "checking" || updateState.kind === "installing"}
        >
          {updateState.kind === "checking" ? "Checking..." : "Check for updates"}
        </button>
        {updateState.kind === "up-to-date" && (
          <span className="update-note ok">Latest version installed (v{updateState.latestVersion}).</span>
        )}
        {updateState.kind === "installing" && (
          <span className="update-note">Downloading and launching the installer...</span>
        )}
        {updateState.kind === "launched" && (
          <span className="update-note ok">Installer launched — OpenChat will close and reopen updated.</span>
        )}
        {updateState.kind === "error" && (
          <span className="update-note error">{updateState.message}</span>
        )}
      </div>
      {available && (
        <div className="update-available">
          <p className="update-note">
            v{available.latestVersion} is available
            {available.publishedAt &&
              ` (released ${new Date(available.publishedAt).toLocaleDateString()})`}
            {installer && ` — ${(installer.size / (1024 * 1024)).toFixed(0)} MB download`}
          </p>
          <div className="update-controls">
            {isWindowsDesktop() && installer?.sha256 && (
              <button
                type="button"
                className="send-btn"
                onClick={() => void runDesktopUpdate(installer)}
              >
                Update now
              </button>
            )}
            {isWindowsDesktop() && installer && !installer.sha256 && (
              <span className="update-note">
                Release asset has no integrity digest — get it from the GitHub release page instead.
              </span>
            )}
            {!isDesktopApp() && installer && (
              <a className="update-link" href={installer.url}>
                Download installer
              </a>
            )}
            {(!isWindowsDesktop() || !installer?.sha256) && (
              <a className="update-link" href={available.releaseUrl} target="_blank" rel="noreferrer">
                View release
              </a>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

function App() {
  const [activePage, setActivePage] = useState<AppPage>("fusion");
  const [prompt, setPrompt] = useState("");
  const [sourceModels, setSourceModels] = useState<string[]>([]);
  const [agentCounts, setAgentCounts] = useState<Record<string, number>>({});
  const [activeRuntimeAgents, setActiveRuntimeAgents] = useState<SourceAgentSpec[]>([]);
  const [fusionModel, setFusionModel] = useState("");
  const [temperature, setTemperature] = useState(1.0);
  const [reasoningEffort, setReasoningEffort] = useState<ReasoningEffort>("medium");
  const [debateMode, setDebateMode] = useState<DebateMode>("partial");
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);
  const [personaEnabled, setPersonaEnabled] = useState(false);
  const [attachments, setAttachments] = useState<ComposerAttachment[]>([]);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [appSettings, setAppSettings] = useState<AppSettings | null>(null);
  const [isAppSettingsOpen, setIsAppSettingsOpen] = useState(false);
  const appSettingsOpenerRef = useRef<HTMLElement | null>(null);
  const [themeMode, setThemeMode] = useState<ThemeMode>(readStoredTheme);
  const [apiKeyInput, setApiKeyInput] = useState("");
  const [baseUrlInput, setBaseUrlInput] = useState("");
  const [isSavingApiSettings, setIsSavingApiSettings] = useState(false);
  const [apiSettingsError, setApiSettingsError] = useState<string | null>(null);
  const [activeMode, setActiveMode] = useState<string>("custom");
  const [autoOptimizeEnabled, setAutoOptimizeEnabled] = useState(false);
  const [isOptimizingPrompt, setIsOptimizingPrompt] = useState(false);
  const [pendingOptimizedPrompt, setPendingOptimizedPrompt] = useState<string | null>(null);
  const [pendingOriginalPrompt, setPendingOriginalPrompt] = useState<string | null>(null);
  const [pendingPersonaAssignments, setPendingPersonaAssignments] = useState<PersonaAssignment[]>([]);
  const [savedWorkflows, setSavedWorkflows] = useState<SavedWorkflow[]>([]);
  const [workflowsError, setWorkflowsError] = useState<string | null>(null);

  const [isSaveWorkflowEditing, setIsSaveWorkflowEditing] = useState(false);
  const [workflowNameInput, setWorkflowNameInput] = useState("");
  const [isSavingWorkflow, setIsSavingWorkflow] = useState(false);
  const [workflowSaveError, setWorkflowSaveError] = useState<string | null>(null);

  const [isRunning, setIsRunning] = useState(false);
  const [hasRunStarted, setHasRunStarted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [sourceResults, setSourceResults] = useState<SourceModelResult[]>([]);
  const [personaAssignments, setPersonaAssignments] = useState<PersonaAssignment[]>([]);
  const [debateResults, setDebateResults] = useState<DebateResult[]>([]);
  const [critiqueOutput, setCritiqueOutput] = useState("");
  const [fusionOutput, setFusionOutput] = useState("");
  const [activeResultTab, setActiveResultTab] = useState<ResultTab>("fusion");

  const [stepState, setStepState] = useState<Record<OrchestrationStep, StepStatus>>(initialStepState);
  const [stepDetail, setStepDetail] = useState<Record<OrchestrationStep, string>>(initialStepDetail);

  const [catalog, setCatalog] = useState<OpenRouterModel[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [isCatalogLoading, setIsCatalogLoading] = useState(false);

  // Per-model service tiers: `serviceTiersByModel` caches which non-default
  // tiers each model exposes (absent = not yet fetched, [] = none);
  // `serviceTierByModel` holds the user's selection per model.
  const [serviceTiersByModel, setServiceTiersByModel] = useState<Record<string, ServiceTier[]>>({});
  const [serviceTierByModel, setServiceTierByModel] = useState<Record<string, ServiceTier>>({});
  const serviceTierFetchInFlightRef = useRef<Set<string>>(new Set());
  // modelId -> last failed discovery time; failed models retry once the
  // cooldown elapses via the retry timer below.
  const serviceTierFailedAtRef = useRef<Record<string, number>>({});
  const serviceTierRetryTimerRef = useRef<Record<string, number>>({});
  // Bumped when the provider base URL changes so stale in-flight lookups
  // from the previous provider can never commit into the fresh caches.
  const serviceTierGenerationRef = useRef(0);
  const [tierRetryTick, bumpTierRetryTick] = useState(0);
  // Gates async tier-lookup handlers so nothing commits state or schedules
  // a retry after the component unmounts.
  const mountedRef = useRef(true);

  const [activePicker, setActivePicker] = useState<PickerKind | null>(null);
  const [pickerQuery, setPickerQuery] = useState("");
  const [hoveredModelId, setHoveredModelId] = useState<string | null>(null);
  const [historyItems, setHistoryItems] = useState<ChatHistorySummary[]>([]);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [historyBusyChatId, setHistoryBusyChatId] = useState<string | null>(null);
  const [appVersion, setAppVersion] = useState<string | null>(null);

  const [directPrompt, setDirectPrompt] = useState("");
  const [directModel, setDirectModel] = useState("");
  const [directMessages, setDirectMessages] = useState<DirectChatMessage[]>([]);
  const [directConversationId, setDirectConversationId] = useState<string | null>(null);
  const [isDirectRunning, setIsDirectRunning] = useState(false);
  const [directError, setDirectError] = useState<string | null>(null);
  const [directRunId, setDirectRunId] = useState<string | null>(null);
  const [directWebSearchEnabled, setDirectWebSearchEnabled] = useState(false);
  const [directTemperature, setDirectTemperature] = useState(1.0);
  const [directReasoningEffort, setDirectReasoningEffort] = useState<ReasoningEffort>("medium");
  const [directAttachments, setDirectAttachments] = useState<ComposerAttachment[]>([]);
  const [isDirectSettingsOpen, setIsDirectSettingsOpen] = useState(false);

  const pickerRef = useRef<HTMLDivElement | null>(null);
  const pickerSearchRef = useRef<HTMLInputElement | null>(null);
  const previousActivePickerRef = useRef<PickerKind | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const settingsRef = useRef<HTMLDivElement | null>(null);
  const directFileInputRef = useRef<HTMLInputElement | null>(null);
  const directSettingsRef = useRef<HTMLDivElement | null>(null);
  const directStreamControllerRef = useRef<AbortController | null>(null);
  const directRequestIdRef = useRef(0);
  // Bumped whenever the direct session resets/hydrates so file reads started
  // against the old session cannot commit attachments into the new one.
  const directAttachmentGenerationRef = useRef(0);
  // Mirror of directAttachments for reads that must not go through stale
  // closures (sendDirectMessage awaits in-flight reads, then snapshots this).
  const directAttachmentsRef = useRef<ComposerAttachment[]>([]);
  const directAttachmentReadsRef = useRef<Promise<void>[]>([]);
  // Every object URL minted for composer/transcript thumbnails, revoked when
  // nothing references it (see releaseDirectThumbs) or the session resets.
  const directThumbUrlsRef = useRef<Set<string>>(new Set());
  // Holds the session generation of the in-flight send (null = idle). Scoped
  // to the generation so a reset unblocks Send in the new session and a stale
  // send's finally cannot clear a newer send's guard.
  const directSendGenerationRef = useRef<number | null>(null);
  // Set on unmount so late image reads revoke their object URLs instead of
  // committing state to a dead component.
  const directUnmountedRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      Object.values(serviceTierRetryTimerRef.current).forEach(clearTimeout);
      serviceTierRetryTimerRef.current = {};
    };
  }, []);

  useEffect(() => {
    let isActive = true;
    getAppVersion()
      .then((version) => {
        if (isActive) setAppVersion(version);
      })
      .catch(() => {});
    return () => {
      isActive = false;
    };
  }, []);

  const openAppSettings = () => {
    const opener = document.activeElement;
    appSettingsOpenerRef.current = opener instanceof HTMLElement ? opener : null;
    setApiKeyInput("");
    setBaseUrlInput(appSettings?.base_url ?? "https://openrouter.ai/api/v1");
    setApiSettingsError(null);
    setIsAppSettingsOpen(true);
  };

  useEffect(() => {
    document.documentElement.dataset.theme = themeMode;
    try {
      window.localStorage.setItem("openchat-theme", themeMode);
    } catch {
      // Storage may be blocked; the theme still applies for this session.
    }
  }, [themeMode]);

  useEffect(() => {
    if (!isAppSettingsOpen) return;
    const canDismiss = Boolean(appSettings?.api_key_configured);
    const panel = document.querySelector<HTMLElement>(".api-settings-panel");
    const handleKeydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (canDismiss) setIsAppSettingsOpen(false);
        return;
      }
      if (event.key !== "Tab" || !panel) return;
      const focusables = Array.from(
        panel.querySelectorAll<HTMLElement>(
          'button, input, select, textarea, a[href], [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((el) => !el.hasAttribute("disabled"));
      if (focusables.length === 0) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      const active = document.activeElement;
      if (!panel.contains(active)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      } else if (event.shiftKey && active === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", handleKeydown);
    return () => window.removeEventListener("keydown", handleKeydown);
  }, [isAppSettingsOpen, appSettings?.api_key_configured]);

  useEffect(() => {
    if (!isAppSettingsOpen) return;
    return () => {
      const stored = appSettingsOpenerRef.current;
      appSettingsOpenerRef.current = null;
      const target =
        stored && stored.getClientRects().length > 0 ? stored : visibleSettingsButton();
      target?.focus();
    };
  }, [isAppSettingsOpen]);

  const saveApiSettings = async () => {
    if (isSavingApiSettings) return;
    if (!apiKeyInput.trim() && !appSettings?.api_key_configured) {
      setApiSettingsError("Enter an OpenRouter API key to connect.");
      return;
    }

    setIsSavingApiSettings(true);
    setApiSettingsError(null);
    try {
      const payload: AppSettingsUpdate = {
        api_key: apiKeyInput.trim() || null,
        base_url: baseUrlInput.trim() || null,
      };
      const updated = await updateSettings(payload);
      if (updated.base_url !== appSettings?.base_url) {
        // Tier availability is per-provider: bump the generation so pending
        // lookups from the old provider can't commit, drop cached lookups,
        // failure marks, pending retries, and selections.
        serviceTierGenerationRef.current += 1;
        setServiceTiersByModel({});
        setServiceTierByModel({});
        serviceTierFetchInFlightRef.current.clear();
        serviceTierFailedAtRef.current = {};
        Object.values(serviceTierRetryTimerRef.current).forEach(clearTimeout);
        serviceTierRetryTimerRef.current = {};
      }
      setAppSettings(updated);
      setIsAppSettingsOpen(false);
      setApiKeyInput("");
    } catch (err) {
      setApiSettingsError(err instanceof Error ? err.message : "Failed to save API settings.");
    } finally {
      setIsSavingApiSettings(false);
    }
  };

  useEffect(() => {
    let isActive = true;
    let retryTimer: number | undefined;

    const loadSettings = async (attempt: number) => {
      try {
        const loaded = await getSettings();
        if (!isActive) return;
        setAppSettings(loaded);
        setBaseUrlInput(loaded.base_url);
        if (!loaded.api_key_configured) {
          appSettingsOpenerRef.current = visibleSettingsButton();
          setIsAppSettingsOpen(true);
        }
      } catch (err) {
        if (!isActive) return;
        if (attempt < 10) {
          retryTimer = window.setTimeout(() => void loadSettings(attempt + 1), 500);
        } else {
          setApiSettingsError(err instanceof Error ? err.message : "Unable to reach the OpenChat server.");
        }
      }
    };

    void loadSettings(1);
    return () => {
      isActive = false;
      if (retryTimer !== undefined) window.clearTimeout(retryTimer);
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let isActive = true;

    const loadWorkflows = async () => {
      if (isActive) setWorkflowsError(null);
      try {
        const payload = await fetchWorkflows(controller.signal);
        if (!isActive) return;
        setSavedWorkflows(payload.data);
      } catch (err) {
        if (!isActive || isAbortError(err)) return;
        setWorkflowsError(err instanceof Error ? err.message : "Failed to load workflows.");
      }
    };

    loadWorkflows();

    return () => {
      isActive = false;
      controller.abort();
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let isActive = true;

    const loadCatalog = async () => {
      if (isActive) {
        setCatalogError(null);
        setIsCatalogLoading(true);
      }
      try {
        const models = await fetchModels(controller.signal);
        if (!isActive) return;
        setCatalog(models);
      } catch (err) {
        if (!isActive || isAbortError(err)) return;
        setCatalogError(err instanceof Error ? err.message : "Failed to load model catalog");
      } finally {
        if (isActive) setIsCatalogLoading(false);
      }
    };

    loadCatalog();

    return () => {
      isActive = false;
      controller.abort();
    };
  }, []);

  useEffect(() => {
    if (activePage !== "history") return;

    const controller = new AbortController();
    let isActive = true;

    const loadHistory = async () => {
      if (isActive) {
        setHistoryError(null);
        setIsHistoryLoading(true);
      }
      try {
        const payload = await fetchChatHistory(controller.signal);
        if (!isActive) return;
        setHistoryItems(
          [...payload.data].sort((a, b) => {
            const aActivity = a.updated_at ?? a.created_at;
            const bActivity = b.updated_at ?? b.created_at;
            const aTs = Date.parse(aActivity);
            const bTs = Date.parse(bActivity);
            if (Number.isFinite(aTs) && Number.isFinite(bTs) && aTs !== bTs) {
              return bTs - aTs;
            }
            return bActivity.localeCompare(aActivity);
          })
        );
      } catch (err) {
        if (!isActive || isAbortError(err)) return;
        setHistoryError(err instanceof Error ? err.message : "Failed to load chat history");
      } finally {
        if (isActive) setIsHistoryLoading(false);
      }
    };

    loadHistory();

    return () => {
      isActive = false;
      controller.abort();
    };
  }, [activePage]);

  const catalogById = useMemo(() => {
    const map = new Map<string, OpenRouterModel>();
    for (const model of catalog) map.set(model.id, model);
    return map;
  }, [catalog]);

  const catalogIndexById = useMemo(() => {
    const map = new Map<string, number>();
    for (let index = 0; index < catalog.length; index += 1) {
      map.set(catalog[index].id, index);
    }
    return map;
  }, [catalog]);

  const allModelIds = useMemo(() => {
    const all = new Set<string>([...catalog.map((model) => model.id), ...sourceModels, fusionModel]);
    all.delete("");
    return Array.from(all);
  }, [catalog, sourceModels, fusionModel]);

  const modelCatalog = useMemo(() => {
    const models = allModelIds.map((id) =>
      toDisplayModel(id, catalogById.get(id), catalogIndexById.get(id) ?? null)
    );

    return models.sort((a, b) => {
      if (a.createdTimestamp !== null && b.createdTimestamp !== null && a.createdTimestamp !== b.createdTimestamp) {
        return b.createdTimestamp - a.createdTimestamp;
      }
      if (a.createdTimestamp !== null && b.createdTimestamp === null) return -1;
      if (a.createdTimestamp === null && b.createdTimestamp !== null) return 1;

      if (a.catalogRank !== null && b.catalogRank !== null && a.catalogRank !== b.catalogRank) {
        return a.catalogRank - b.catalogRank;
      }
      if (a.catalogRank !== null && b.catalogRank === null) return -1;
      if (a.catalogRank === null && b.catalogRank !== null) return 1;

      const byName = a.name.localeCompare(b.name);
      if (byName !== 0) return byName;
      return a.id.localeCompare(b.id);
    });
  }, [allModelIds, catalogById, catalogIndexById]);

  const orderedModelIds = useMemo(() => modelCatalog.map((item) => item.id), [modelCatalog]);

  useEffect(() => {
    if (orderedModelIds.length === 0) return;
    if (fusionModel && !orderedModelIds.includes(fusionModel)) setFusionModel("");
    setSourceModels((prev) => {
      const next = prev.filter((id) => orderedModelIds.includes(id));
      if (next.length === prev.length && next.every((id, index) => id === prev[index])) {
        return prev;
      }
      return next;
    });
  }, [fusionModel, orderedModelIds]);

  useEffect(() => {
    if (!fusionModel && sourceModels.length > 0) {
      setFusionModel(sourceModels[0]);
    }
  }, [fusionModel, sourceModels]);

  const modeTabs = useMemo<ExperienceModeTab[]>(() => {
    const builtInModes: ExperienceModeTab[] = EXPERIENCE_MODES.map((mode) => ({
      id: mode.id,
      title: mode.title,
      sub: mode.sub,
      icon: mode.icon,
      workflowId: null,
    }));

    const workflowModes: ExperienceModeTab[] = savedWorkflows.map((workflow) => ({
      id: workflowModeId(workflow.workflow_id),
      title: workflow.name,
      sub: "Saved workflow",
      icon: bookmarkIcon,
      workflowId: workflow.workflow_id,
    }));

    return [...builtInModes, ...workflowModes];
  }, [savedWorkflows]);

  const selectedSourceModels = useMemo(
    () =>
      sourceModels.map((id) =>
        toDisplayModel(id, catalogById.get(id), catalogIndexById.get(id) ?? null)
      ),
    [catalogById, catalogIndexById, sourceModels]
  );

  const runtimeAgents = useMemo(
    () => buildRuntimeAgents(sourceModels, agentCounts),
    [sourceModels, agentCounts]
  );

  useEffect(() => {
    setAgentCounts((prev) => {
      const next: Record<string, number> = {};
      let changed = false;
      for (const modelId of sourceModels) {
        const existing = prev[modelId];
        if (typeof existing === "number" && existing >= 1) {
          next[modelId] = Math.min(MAX_AGENT_COUNT, existing);
        } else {
          next[modelId] = 1;
          changed = true;
        }
      }
      if (!changed && Object.keys(prev).length === Object.keys(next).length) {
        const sameEntries = Object.entries(next).every(
          ([key, value]) => prev[key] === value
        );
        if (sameEntries) return prev;
      }
      return next;
    });
  }, [sourceModels]);

  const setAgentCount = (modelId: string, count: number) => {
    const clamped = Math.max(1, Math.min(MAX_AGENT_COUNT, Math.trunc(count) || 1));
    setAgentCounts((prev) => ({ ...prev, [modelId]: clamped }));
  };

  const incrementAgentCount = (modelId: string) => {
    setAgentCounts((prev) => {
      const current = prev[modelId] ?? 1;
      return { ...prev, [modelId]: Math.min(MAX_AGENT_COUNT, current + 1) };
    });
  };

  const decrementAgentCount = (modelId: string) => {
    setAgentCounts((prev) => {
      const current = prev[modelId] ?? 1;
      return { ...prev, [modelId]: Math.max(1, current - 1) };
    });
  };

  const sourceResultsByAgent = useMemo(() => {
    const map = new Map<string, SourceModelResult>();
    for (const result of sourceResults) map.set(agentKey(result), result);
    return map;
  }, [sourceResults]);

  const personaByAgent = useMemo(() => {
    const map = new Map<string, PersonaAssignment>();
    for (const assignment of personaAssignments) map.set(assignment.agent_id?.trim() || assignment.model, assignment);
    for (const result of sourceResults) {
      if (result.persona) map.set(agentKey(result), result.persona);
    }
    return map;
  }, [personaAssignments, sourceResults]);

  const summaryAgents = useMemo(
    () =>
      activeRuntimeAgents.filter((agent) => {
        const result = sourceResultsByAgent.get(agent.id);
        return !result || result.status === "ok";
      }),
    [activeRuntimeAgents, sourceResultsByAgent]
  );

  const filteredModels = useMemo(() => {
    const term = pickerQuery.trim().toLowerCase();
    if (!term) return modelCatalog;
    return modelCatalog.filter((model) => {
      const haystack = `${model.name} ${model.id} ${model.provider}`.toLowerCase();
      return haystack.includes(term);
    });
  }, [modelCatalog, pickerQuery]);

  const focusedModelId =
    hoveredModelId && filteredModels.some((model) => model.id === hoveredModelId)
      ? hoveredModelId
      : filteredModels[0]?.id ?? null;

  const focusedModel = focusedModelId
    ? filteredModels.find((model) => model.id === focusedModelId) ?? null
    : null;

  const modelsNeedingTierLookup = useMemo(() => {
    const ids = new Set<string>(sourceModels);
    if (fusionModel) ids.add(fusionModel);
    if (directModel) ids.add(directModel);
    if (focusedModelId) ids.add(focusedModelId);
    return [...ids];
  }, [sourceModels, fusionModel, directModel, focusedModelId]);

  // Lazily discover non-default service tiers for every model in view;
  // results (including "none") are cached so each model is fetched once.
  // Each lookup commits its own outcome — earlier passes must not be
  // invalidated when a sibling lookup resolves and reruns this effect.
  useEffect(() => {
    const generation = serviceTierGenerationRef.current;
    for (const modelId of modelsNeedingTierLookup) {
      if (modelId in serviceTiersByModel || serviceTierFetchInFlightRef.current.has(modelId)) {
        continue;
      }
      const failedAt = serviceTierFailedAtRef.current[modelId];
      if (failedAt !== undefined && Date.now() - failedAt < SERVICE_TIER_RETRY_MS) {
        continue;
      }
      serviceTierFetchInFlightRef.current.add(modelId);
      fetchServiceTiers(modelId)
        .then((tiers) => {
          if (!mountedRef.current || serviceTierGenerationRef.current !== generation) return;
          delete serviceTierFailedAtRef.current[modelId];
          setServiceTiersByModel((prev) => ({ ...prev, [modelId]: tiers }));
        })
        .catch(() => {
          if (!mountedRef.current || serviceTierGenerationRef.current !== generation) return;
          // Record the failure and schedule a retry once the cooldown ends;
          // nothing is cached as "no tiers", so a recovered provider is
          // re-probed even when the selection is otherwise unchanged.
          serviceTierFailedAtRef.current[modelId] = Date.now();
          if (serviceTierRetryTimerRef.current[modelId] === undefined) {
            serviceTierRetryTimerRef.current[modelId] = window.setTimeout(() => {
              delete serviceTierRetryTimerRef.current[modelId];
              bumpTierRetryTick((tick) => tick + 1);
            }, SERVICE_TIER_RETRY_MS);
          }
        })
        .finally(() => {
          if (serviceTierGenerationRef.current !== generation) return;
          serviceTierFetchInFlightRef.current.delete(modelId);
        });
    }
  }, [modelsNeedingTierLookup, serviceTiersByModel, tierRetryTick]);

  // A hydrated/saved tier only survives when discovery hasn't ruled it out:
  // pending or failed lookups (undefined) keep the tier, a completed roster
  // that excludes it drops it.
  const effectiveTier = useCallback(
    (modelId: string): ServiceTier | undefined => {
      const tier = serviceTierByModel[modelId];
      if (!tier || tier === "default") return undefined;
      const discoveredTiers = serviceTiersByModel[modelId];
      if (discoveredTiers !== undefined && !discoveredTiers.includes(tier)) {
        return undefined;
      }
      return tier;
    },
    [serviceTierByModel, serviceTiersByModel]
  );

  const activeServiceTierSelection = useMemo(() => {
    const relevant = new Set<string>(sourceModels);
    if (fusionModel) relevant.add(fusionModel);
    const entries = Object.keys(serviceTierByModel)
      .filter((model) => relevant.has(model))
      .map((model) => [model, effectiveTier(model)] as const)
      .filter((entry): entry is readonly [string, ServiceTier] => entry[1] !== undefined);
    return Object.fromEntries(entries);
  }, [effectiveTier, serviceTierByModel, sourceModels, fusionModel]);

  const renderServiceTierSelect = (modelId: string, disabled: boolean) => {
    const tiers = serviceTiersByModel[modelId];
    if (!tiers || tiers.length === 0) return null;
    return (
      <label className="service-tier-control">
        <span className="service-tier-label">Tier</span>
        <select
          className="service-tier-select"
          aria-label={`Service tier for ${modelId}`}
          title={SERVICE_TIER_TOOLTIPS[effectiveTier(modelId) ?? "default"]}
          value={effectiveTier(modelId) ?? "default"}
          disabled={disabled}
          onChange={(event) =>
            setServiceTierByModel((prev) => ({
              ...prev,
              [modelId]: event.target.value as ServiceTier,
            }))
          }
        >
          <option value="default">{SERVICE_TIER_LABELS.default}</option>
          {tiers.map((tier) => (
            <option key={tier} value={tier} title={SERVICE_TIER_TOOLTIPS[tier]}>
              {SERVICE_TIER_LABELS[tier] ?? tier}
            </option>
          ))}
        </select>
        {SERVICE_TIER_HINTS[effectiveTier(modelId) ?? "default"] && (
          <span className="service-tier-hint">
            {SERVICE_TIER_HINTS[effectiveTier(modelId) ?? "default"]}
          </span>
        )}
      </label>
    );
  };

  const isOptimizationReviewPending = pendingOriginalPrompt !== null && (
    pendingOptimizedPrompt !== null || pendingPersonaAssignments.length > 0
  );

  const canRun = useMemo(
    () =>
      prompt.trim().length > 0 &&
      sourceModels.length > 0 &&
      fusionModel.length > 0 &&
      !isRunning &&
      !isOptimizingPrompt &&
      !isOptimizationReviewPending,
    [fusionModel.length, isOptimizingPrompt, isOptimizationReviewPending, isRunning, prompt, sourceModels.length]
  );

  const canRegenerateFusion = useMemo(
    () =>
      !isRunning &&
      prompt.trim().length > 0 &&
      fusionModel.length > 0 &&
      sourceResults.some((result) => result.status === "ok" && result.content.trim().length > 0),
    [fusionModel.length, isRunning, prompt, sourceResults]
  );

  const canSendDirect = useMemo(
    () =>
      (directPrompt.trim().length > 0 || directAttachments.some(isImageAttachment)) &&
      directModel.length > 0 &&
      !isDirectRunning,
    [directAttachments, directModel.length, directPrompt, isDirectRunning]
  );

  const hasSourceResults = sourceResults.length > 0;
  const hasCritiqueOutput = critiqueOutput.trim().length > 0;
  const hasFusionOutput = fusionOutput.trim().length > 0;
  const hasOrchestrationProgress = ORCHESTRATION_FLOW.some((item) => stepState[item.id] !== "pending");
  const showResultsPanel = hasRunStarted && (isRunning || hasOrchestrationProgress || hasSourceResults || hasCritiqueOutput || hasFusionOutput);

  useEffect(() => {
    if (activeResultTab === "sources" && !hasSourceResults) setActiveResultTab("fusion");
  }, [activeResultTab, hasSourceResults]);

  useEffect(() => {
    if (!activePicker) {
      previousActivePickerRef.current = null;
      return;
    }

    if (previousActivePickerRef.current !== activePicker) {
      setPickerQuery("");
      setHoveredModelId(
        activePicker === "fusion"
          ? fusionModel || null
          : activePicker === "direct"
            ? directModel || null
            : sourceModels[0] || null
      );
      previousActivePickerRef.current = activePicker;
    }
  }, [activePicker, directModel, fusionModel, sourceModels]);

  useEffect(() => {
    if (!activePicker) return;
    pickerSearchRef.current?.focus();
  }, [activePicker]);

  useEffect(() => {
    if (!isSettingsOpen) return;

    const onMouseDown = (event: MouseEvent) => {
      if (!settingsRef.current) return;
      if (!settingsRef.current.contains(event.target as Node)) setIsSettingsOpen(false);
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setIsSettingsOpen(false);
    };

    document.addEventListener("mousedown", onMouseDown);
    document.addEventListener("keydown", onKeyDown);

    return () => {
      document.removeEventListener("mousedown", onMouseDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [isSettingsOpen]);

  useEffect(() => {
    if (!isDirectSettingsOpen) return;

    const onMouseDown = (event: MouseEvent) => {
      if (!directSettingsRef.current) return;
      if (!directSettingsRef.current.contains(event.target as Node)) setIsDirectSettingsOpen(false);
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setIsDirectSettingsOpen(false);
    };

    document.addEventListener("mousedown", onMouseDown);
    document.addEventListener("keydown", onKeyDown);

    return () => {
      document.removeEventListener("mousedown", onMouseDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [isDirectSettingsOpen]);

  useEffect(() => {
    if (activePage === "direct") return;
    if (!isDirectRunning) return;
    directStreamControllerRef.current?.abort();
    directStreamControllerRef.current = null;
    directRequestIdRef.current += 1;
    setIsDirectRunning(false);
  }, [activePage, isDirectRunning]);

  useEffect(() => {
    if (!activePicker) return;

    const onMouseDown = (event: MouseEvent) => {
      if (!pickerRef.current) return;
      if (!pickerRef.current.contains(event.target as Node)) setActivePicker(null);
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setActivePicker(null);
    };

    document.addEventListener("mousedown", onMouseDown);
    document.addEventListener("keydown", onKeyDown);

    return () => {
      document.removeEventListener("mousedown", onMouseDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [activePicker]);

  const onEvent = (event: StreamEvent) => {
    if (event.type === "run_started") {
      setRunId(event.run_id);
      return;
    }

    if (event.type === "orchestration_step") {
      setStepState((prev) => ({ ...prev, [event.data.step]: event.data.status }));
      if (event.data.detail) {
        setStepDetail((prev) => ({ ...prev, [event.data.step]: event.data.detail ?? prev[event.data.step] }));
      }
      return;
    }

    if (event.type === "source_result") {
      setSourceResults((prev) => {
        const key = agentKey(event.data);
        const without = prev.filter((item) => agentKey(item) !== key);
        return [...without, event.data];
      });
      return;
    }

    if (event.type === "persona_assignments") {
      setPersonaAssignments(event.data.items);
      if (event.data.error) setError(event.data.error);
      return;
    }

    if (event.type === "debate_result") {
      setDebateResults((prev) => [...prev, event.data]);
      return;
    }

    if (event.type === "critique_ready") {
      setCritiqueOutput(event.data.content);
      return;
    }

    if (event.type === "fusion_ready") {
      setFusionOutput(event.data.content);
      return;
    }

    if (event.type === "error") {
      setError(event.data.message);
      setIsRunning(false);
      return;
    }

    if (event.type === "completed") {
      setIsRunning(false);
    }
  };

  const onDirectEvent = (event: DirectChatStreamEvent, requestId: number) => {
    if (requestId !== directRequestIdRef.current) return;

    if (event.type === "run_started") {
      setDirectRunId(event.run_id);
      // The server may remap a supplied id that collides with a
      // non-direct conversation — adopt the effective id.
      if (event.conversation_id) setDirectConversationId(event.conversation_id);
      return;
    }

    if (event.type === "assistant_ready") {
      const content = event.data.content.trim();
      setDirectMessages((prev) => [...prev, { role: "assistant", content: content || "(No response returned.)" }]);
      return;
    }

    if (event.type === "error") {
      setDirectError(event.data.message);
      setIsDirectRunning(false);
      return;
    }

    if (event.type === "completed") {
      setIsDirectRunning(false);
      directStreamControllerRef.current = null;
    }
  };

  const resetDirectSession = () => {
    directStreamControllerRef.current?.abort();
    directStreamControllerRef.current = null;
    directRequestIdRef.current += 1;
    setIsDirectRunning(false);
    setDirectError(null);
    setDirectRunId(null);
    setDirectPrompt("");
    setDirectModel("");
    setDirectMessages([]);
    setDirectConversationId(null);
    setDirectAttachments([]);
    directAttachmentsRef.current = [];
    releaseDirectThumbs([...directThumbUrlsRef.current]);
    directAttachmentGenerationRef.current += 1;
    setIsDirectSettingsOpen(false);
  };

  const clearOptimizationReview = () => {
    setPendingOptimizedPrompt(null);
    setPendingOriginalPrompt(null);
    setPendingPersonaAssignments([]);
  };

  const handleStartNewChat = () => {
    setActivePage("fusion");
    setPrompt("");
    setSourceModels([]);
    setAgentCounts({});
    setActiveRuntimeAgents([]);
    setFusionModel("");
    setServiceTierByModel({});
    setTemperature(1.0);
    setReasoningEffort("medium");
    setDebateMode("partial");
    setWebSearchEnabled(false);
    setPersonaEnabled(false);
    setAttachments([]);
    setIsSettingsOpen(false);
    setActiveMode("custom");
    setAutoOptimizeEnabled(false);
    setIsOptimizingPrompt(false);
    clearOptimizationReview();

    setIsRunning(false);
    setHasRunStarted(false);
    setError(null);
    setRunId(null);
    setSourceResults([]);
    setPersonaAssignments([]);
    setDebateResults([]);
    setCritiqueOutput("");
    setFusionOutput("");
    setActiveResultTab("fusion");

    setStepState(initialStepState());
    setStepDetail(initialStepDetail());

    setActivePicker(null);
    setPickerQuery("");
    setHoveredModelId(null);
  };

  const hydrateFromHistory = (chat: ChatHistoryDetail) => {
    setActivePage("fusion");
    setPrompt(chat.request.prompt);
    setSourceModels(chat.request.source_models);
    setAgentCounts({});
    const hydratedAgents: SourceAgentSpec[] =
      chat.request.source_agents && chat.request.source_agents.length > 0
        ? chat.request.source_agents
        : chat.source_results.map((result) => ({
            id: result.agent_id?.trim() || result.model,
            model: result.model,
          }));
    setActiveRuntimeAgents(hydratedAgents);
    setFusionModel(chat.request.fusion_model);
    setServiceTierByModel(chat.request.service_tiers ?? {});
    setDebateMode(chat.request.debate_mode ?? "partial");
    setTemperature(chat.request.temperature ?? 1.0);
    setWebSearchEnabled(chat.request.web_search_enabled ?? false);
    setPersonaEnabled(chat.request.persona_enabled ?? false);
    setReasoningEffort(chat.request.reasoning?.effort ?? "medium");
    setAttachments(
      (chat.request.attachments ?? []).map((attachment, index) => ({
        ...attachment,
        id: `${chat.chat_id}-${attachment.name}-${index}`,
      }))
    );

    setRunId(chat.run_id);
    setIsRunning(false);
    setHasRunStarted(true);
    setSourceResults(chat.source_results);
    setPersonaAssignments(
      chat.source_results
        .map((result) => result.persona)
        .filter((persona): persona is PersonaAssignment => Boolean(persona))
    );
    setDebateResults(chat.debate_results);
    setCritiqueOutput(chat.critique_output);
    setFusionOutput(chat.fusion_output);
    setStepState({ route: "done", search: "done", parallel: "done", critique: "done", fusion: "done" });
    setStepDetail(initialStepDetail());
    setError(null);
    setActiveResultTab("fusion");
    setActiveMode("custom");
    setIsSettingsOpen(false);
    setActivePicker(null);
  };

  const hydrateDirectFromHistory = (chat: ChatHistoryDetail) => {
    setActivePage("direct");
    const hydratedDirectModel = chat.request.fusion_model || chat.request.source_models[0] || "";
    setDirectModel(hydratedDirectModel);
    const hydratedDirectTier = chat.request.service_tiers?.[hydratedDirectModel];
    setServiceTierByModel((prev) => {
      const next = { ...prev };
      if (hydratedDirectModel) {
        // A missing saved tier means default routing — clear any stale
        // selection for this model rather than inheriting it.
        if (hydratedDirectTier) {
          next[hydratedDirectModel] = hydratedDirectTier;
        } else {
          delete next[hydratedDirectModel];
        }
      }
      return next;
    });
    setDirectPrompt("");
    setDirectAttachments([]);
    directAttachmentsRef.current = [];
    releaseDirectThumbs([...directThumbUrlsRef.current]);
    directAttachmentGenerationRef.current += 1;
    setIsDirectSettingsOpen(false);
    setDirectError(null);
    setDirectRunId(chat.run_id);
    setIsDirectRunning(false);
    setDirectConversationId(chat.chat_id);

    if (chat.messages && chat.messages.length > 0) {
      setDirectMessages(chat.messages);
    } else {
      const latestUserPrompt = chat.request.prompt.trim();
      const assistantReply = (chat.fusion_output || chat.source_results[0]?.content || "").trim();
      const rebuiltMessages: DirectChatMessage[] = [];
      if (latestUserPrompt) rebuiltMessages.push({ role: "user", content: latestUserPrompt });
      if (assistantReply) rebuiltMessages.push({ role: "assistant", content: assistantReply });
      setDirectMessages(rebuiltMessages);
    }
  };

  const handleOpenHistoryChat = async (chatId: string) => {
    setHistoryError(null);
    setHistoryBusyChatId(chatId);
    try {
      const detail = await fetchChatHistoryDetail(chatId);
      if (detail.status.startsWith("direct")) {
        hydrateDirectFromHistory(detail);
      } else {
        hydrateFromHistory(detail);
      }
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : "Failed to load selected chat.");
    } finally {
      setHistoryBusyChatId(null);
    }
  };

  const handleDeleteHistoryChat = async (chatId: string) => {
    setHistoryError(null);
    setHistoryBusyChatId(chatId);
    try {
      await deleteChatHistory(chatId);
      setHistoryItems((prev) => prev.filter((item) => item.chat_id !== chatId));
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : "Failed to delete selected chat.");
    } finally {
      setHistoryBusyChatId(null);
    }
  };

  const run = async (promptToRun: string, personaConfig?: { enabled: boolean; assignments?: PersonaAssignment[] }) => {
    setError(null);
    setRunId(null);
    setHasRunStarted(true);
    setActiveResultTab("fusion");
    setSourceResults([]);
    setDebateResults([]);
    setPersonaAssignments([]);
    setCritiqueOutput("");
    setFusionOutput("");
    setIsRunning(true);
    setStepState({ route: "active", search: "pending", parallel: "pending", critique: "pending", fusion: "pending" });
    setStepDetail(initialStepDetail());

    const agents = runtimeAgents;
    setActiveRuntimeAgents(agents);

    const payload: RunRequest = {
      prompt: promptToRun,
      source_models: sourceModels,
      source_agents: agents,
      fusion_model: fusionModel,
      debate_mode: debateMode,
      temperature,
      max_output_tokens: OPENROUTER_TOKEN_LIMIT,
      web_search_enabled: webSearchEnabled,
      persona_enabled: personaConfig?.enabled ?? personaEnabled,
      persona_assignments_override: personaConfig?.assignments ?? [],
      reasoning: {
        effort: reasoningEffort,
        exclude: false,
      },
      attachments: attachments.map(({ id: _id, ...attachment }) => attachment),
      service_tiers: activeServiceTierSelection,
    };

    try {
      await streamRun(payload, onEvent);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setIsRunning(false);
    }
  };

  const handleSend = async () => {
    if (isRunning || isOptimizingPrompt) return;

    const trimmedPrompt = prompt.trim();
    if (!trimmedPrompt || sourceModels.length === 0 || !fusionModel) return;

    if (!autoOptimizeEnabled && !personaEnabled) {
      clearOptimizationReview();
      await run(trimmedPrompt, { enabled: false });
      return;
    }

    setError(null);
    setIsOptimizingPrompt(true);

    try {
      let optimized = trimmedPrompt;
      if (autoOptimizeEnabled) {
        const optimizeResult = await optimizePrompt({ prompt: trimmedPrompt });
        optimized = optimizeResult.optimized_prompt.trim() || trimmedPrompt;
      }

      let generatedPersonas: PersonaAssignment[] = [];
      if (personaEnabled) {
        const personaResult = await previewPersonas({
          prompt: optimized,
          source_agents: runtimeAgents,
        });
        generatedPersonas = personaResult.items;
      }

      setPendingOriginalPrompt(trimmedPrompt);
      setPendingOptimizedPrompt(autoOptimizeEnabled ? optimized : null);
      setPendingPersonaAssignments(generatedPersonas);
      setPrompt(optimized);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prompt optimization failed.");
    } finally {
      setIsOptimizingPrompt(false);
    }
  };

  const handleApproveOptimizedPrompt = async () => {
    if (!pendingOriginalPrompt || isRunning || isOptimizingPrompt) return;
    const approvedPrompt = (pendingOptimizedPrompt ?? prompt.trim()) || pendingOriginalPrompt;
    const approvedPersonas = pendingPersonaAssignments;
    clearOptimizationReview();
    await run(approvedPrompt, {
      enabled: approvedPersonas.length > 0,
      assignments: approvedPersonas,
    });
  };

  const handleRejectOptimizedPrompt = async () => {
    if (!pendingOriginalPrompt) return;
    const original = pendingOriginalPrompt;
    setPrompt(original);
    clearOptimizationReview();
    setPersonaAssignments([]);
    setError(null);
    await run(original, { enabled: false, assignments: [] });
  };

  const updatePendingPersona = (
    agentId: string,
    field: "title" | "description" | "temperature",
    value: string
  ) => {
    setPendingPersonaAssignments((prev) =>
      prev.map((item) => {
        const itemAgentId = item.agent_id?.trim() || item.model;
        if (itemAgentId !== agentId) return item;
        if (field !== "temperature") return { ...item, [field]: value };

        const parsed = Number(value);
        if (!Number.isFinite(parsed)) return item;
        const normalized = Math.min(1.2, Math.max(0.5, parsed));
        return { ...item, temperature: Number(normalized.toFixed(2)) };
      })
    );
  };

  const removeSourceModel = (model: string) => {
    setSourceModels((prev) => prev.filter((entry) => entry !== model));
    setAgentCounts((prev) => {
      if (!(model in prev)) return prev;
      const next = { ...prev };
      delete next[model];
      return next;
    });
  };

  const removeAttachment = (attachmentId: string) => {
    setAttachments((prev) => prev.filter((item) => item.id !== attachmentId));
  };

  const applySavedWorkflow = (config: SavedWorkflowConfig) => {
    setSourceModels(config.source_models);
    setAgentCounts({});
    setActiveRuntimeAgents([]);
    setFusionModel(config.fusion_model);
    setTemperature(config.temperature);
    setReasoningEffort(config.reasoning_effort);
    setDebateMode(config.debate_mode);
    setWebSearchEnabled(config.web_search_enabled);
    setPersonaEnabled(config.persona_enabled ?? false);
    setServiceTierByModel(config.service_tiers ?? {});
    setAttachments([]);
  };

  const handleSelectMode = (modeId: string) => {
    const preset = EXPERIENCE_MODE_PRESETS[modeId as (typeof EXPERIENCE_MODES)[number]["id"]];
    if (preset) {
      setSourceModels(preset.sourceModels);
      setAgentCounts({});
      setActiveRuntimeAgents([]);
      setFusionModel(preset.fusionModel);
      setActiveMode(modeId);
      setWorkflowSaveError(null);
      return;
    }

    if (modeId === "custom") {
      setActiveMode(modeId);
      setWorkflowSaveError(null);
      return;
    }

    const workflow = savedWorkflows.find((item) => workflowModeId(item.workflow_id) === modeId);
    if (!workflow) return;

    applySavedWorkflow(workflow.config);
    setActiveMode(modeId);
    setWorkflowSaveError(null);
  };

  const saveCurrentWorkflow = async (workflowNameRaw: string) => {
    if (isSavingWorkflow) return;

    const workflowName = workflowNameRaw.trim();
    if (!workflowName) {
      setWorkflowSaveError("Workflow name is required.");
      setIsSaveWorkflowEditing(false);
      setWorkflowNameInput("");
      return;
    }

    if (sourceModels.length === 0 || !fusionModel) {
      setWorkflowSaveError("Select at least one source model and a fusion model before saving a workflow.");
      setIsSaveWorkflowEditing(false);
      setWorkflowNameInput("");
      return;
    }

    setIsSavingWorkflow(true);
    try {
      const saved = await createWorkflow({
        name: workflowName,
        config: {
          source_models: sourceModels,
          fusion_model: fusionModel,
          temperature,
          reasoning_effort: reasoningEffort,
          debate_mode: debateMode,
          web_search_enabled: webSearchEnabled,
          persona_enabled: personaEnabled,
          service_tiers: activeServiceTierSelection,
          attachments: attachments.map((attachment) => ({
            name: attachment.name,
            size: attachment.size,
            content_type: attachment.content_type,
          })),
        },
      });

      setSavedWorkflows((prev) => [saved, ...prev]);
      setWorkflowsError(null);
      setWorkflowSaveError(null);
      setActiveMode(workflowModeId(saved.workflow_id));
    } catch (err) {
      setWorkflowSaveError(err instanceof Error ? err.message : "Failed to save workflow.");
    } finally {
      setIsSavingWorkflow(false);
      setIsSaveWorkflowEditing(false);
      setWorkflowNameInput("");
    }
  };

  const beginWorkflowSave = () => {
    setWorkflowSaveError(null);
    setIsSaveWorkflowEditing(true);
    setWorkflowNameInput("");
  };

  const handleDeleteSavedWorkflow = async (workflowId: string) => {
    try {
      await deleteWorkflow(workflowId);
      setSavedWorkflows((prev) => prev.filter((item) => item.workflow_id !== workflowId));
      if (activeMode === workflowModeId(workflowId)) {
        setActiveMode("custom");
      }
      setWorkflowSaveError(null);
    } catch (err) {
      setWorkflowSaveError(err instanceof Error ? err.message : "Failed to delete workflow.");
    }
  };

  const triggerAttachmentPicker = () => {
    fileInputRef.current?.click();
  };

  const onAttachmentChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (selected.length === 0) return;

    const slotsRemaining = MAX_ATTACHMENTS - attachments.length;
    if (slotsRemaining <= 0) {
      setError(`Attachment limit reached (${MAX_ATTACHMENTS}). Remove one to add another.`);
      return;
    }

    const nextFiles = selected.slice(0, slotsRemaining);
    const oversize = nextFiles.find((file) => file.size > MAX_ATTACHMENT_SIZE_BYTES);
    if (oversize) {
      setError(`Attachment ${oversize.name} exceeds 256KB and cannot be added.`);
      return;
    }

    const parsed = await Promise.all(nextFiles.map((file) => readAttachmentFile(file)));
    const rejectedCount = parsed.filter((item) => !item).length;
    const accepted = parsed.filter((item): item is ComposerAttachment => item !== null);

    if (accepted.length === 0) {
      setError("No supported text attachments found. Use text-based files like .txt, .md, .json, .csv, or code files.");
      return;
    }

    setAttachments((prev) => {
      const merged = [...prev, ...accepted].filter(
        (item, index, array) => array.findIndex((entry) => entry.id === item.id) === index
      );
      return merged.slice(0, MAX_ATTACHMENTS);
    });

    if (rejectedCount > 0) {
      setError(`${rejectedCount} file(s) were skipped because they are unsupported or empty.`);
    } else {
      setError(null);
    }
  };

  const triggerDirectAttachmentPicker = () => {
    directFileInputRef.current?.click();
  };

  const addDirectFiles = async (files: File[]) => {
    if (files.length === 0) return;

    const slotsRemaining = MAX_ATTACHMENTS - directAttachmentsRef.current.length;
    if (slotsRemaining <= 0) {
      setDirectError(`Attachment limit reached (${MAX_ATTACHMENTS}). Remove one to add another.`);
      return;
    }

    const nextFiles = files.slice(0, slotsRemaining);
    const oversize = nextFiles.find(
      (file) => file.size > (isImageFile(file) ? MAX_IMAGE_ATTACHMENT_SIZE_BYTES : MAX_ATTACHMENT_SIZE_BYTES)
    );
    if (oversize) {
      const limitLabel = isImageFile(oversize) ? "5MB" : "256KB";
      setDirectError(`Attachment ${oversize.name} exceeds ${limitLabel} and cannot be added.`);
      return;
    }

    const generation = directAttachmentGenerationRef.current;
    const parsed = await Promise.all(
      nextFiles.map((file) => (isImageFile(file) ? readImageAttachmentFile(file) : readAttachmentFile(file)))
    );
    if (generation !== directAttachmentGenerationRef.current || directUnmountedRef.current) {
      // Session moved on or the app unmounted while reading — release
      // thumbnails we won't show instead of committing stale state.
      releaseDirectThumbs(
        parsed.flatMap((item) => (item?.thumb_url ? [item.thumb_url] : []))
      );
      return;
    }
    const rejectedCount = parsed.filter((item) => !item).length;
    const accepted = parsed.filter((item): item is ComposerAttachment => item !== null);

    if (accepted.length === 0) {
      setDirectError("No supported attachments found. Use text files like .txt, .md, .json, .csv, or images.");
      return;
    }

    // Commit to the ref synchronously — a send awaiting this read must see
    // the new attachments — then mirror to React state for rendering.
    // Capacity is recomputed here: another batch may have committed while
    // this read was in flight, so overflow is dropped with an explicit count.
    const capacityLeft = Math.max(0, MAX_ATTACHMENTS - directAttachmentsRef.current.length);
    const merged = [...directAttachmentsRef.current, ...accepted.slice(0, capacityLeft)].filter(
      (item, index, array) => array.findIndex((entry) => entry.id === item.id) === index
    );
    const next = merged.slice(0, MAX_ATTACHMENTS);
    const droppedCount = accepted.length - next.filter((item) => accepted.includes(item)).length;
    directAttachmentsRef.current = next;
    for (const item of next) {
      if (item.thumb_url) directThumbUrlsRef.current.add(item.thumb_url);
    }
    setDirectAttachments(next);
    releaseDirectThumbs(
      accepted
        .filter((item) => !next.some((entry) => entry.id === item.id))
        .flatMap((item) => (item.thumb_url ? [item.thumb_url] : []))
    );

    if (droppedCount > 0) {
      setDirectError(`${droppedCount} file(s) were skipped — attachment limit (${MAX_ATTACHMENTS}) reached.`);
    } else if (rejectedCount > 0) {
      setDirectError(`${rejectedCount} file(s) were skipped because they are unsupported or empty.`);
    } else {
      setDirectError(null);
    }
  };

  const releaseDirectThumbs = (urls: string[]) => {
    for (const url of urls) {
      URL.revokeObjectURL(url);
      directThumbUrlsRef.current.delete(url);
    }
  };

  // Release any remaining object URLs when the app unmounts.
  useEffect(
    () => () => {
      directUnmountedRef.current = true;
      for (const url of directThumbUrlsRef.current) {
        URL.revokeObjectURL(url);
      }
      directThumbUrlsRef.current.clear();
    },
    []
  );

  // Tracks each addDirectFiles call so sendDirectMessage can wait for the
  // image read to finish instead of sending a turn without its attachment.
  const queueDirectFiles = (files: File[]) => {
    const task = addDirectFiles(files).catch(() => undefined);
    directAttachmentReadsRef.current.push(task);
    void task.finally(() => {
      directAttachmentReadsRef.current = directAttachmentReadsRef.current.filter(
        (entry) => entry !== task
      );
    });
  };

  const onDirectAttachmentChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(event.target.files ?? []);
    event.target.value = "";
    queueDirectFiles(selected);
  };

  const onDirectComposerPaste = (event: ClipboardEvent<HTMLTextAreaElement>) => {
    const imageFiles = Array.from(event.clipboardData?.items ?? [])
      .filter((item) => item.kind === "file" && item.type.startsWith(IMAGE_CONTENT_TYPE_PREFIX))
      .map((item) => item.getAsFile())
      .filter((file): file is File => file !== null);
    if (imageFiles.length === 0) return;
    event.preventDefault();

    // A clipboard payload can carry both files and text — keep the text by
    // inserting it at the caret ourselves since the default paste is suppressed.
    const pastedText = event.clipboardData?.getData("text/plain") ?? "";
    if (pastedText) {
      const textarea = event.currentTarget;
      const nextValue =
        textarea.value.slice(0, textarea.selectionStart) +
        pastedText +
        textarea.value.slice(textarea.selectionEnd);
      const caret = textarea.selectionStart + pastedText.length;
      setDirectPrompt(nextValue);
      requestAnimationFrame(() => textarea.setSelectionRange(caret, caret));
    }

    if (imageFiles.every((file) => !SUPPORTED_IMAGE_TYPES.has(file.type))) {
      setDirectError("Unsupported image type — use PNG, JPEG, GIF, or WebP.");
      return;
    }
    queueDirectFiles(imageFiles);
  };

  const removeDirectAttachment = (attachmentId: string) => {
    const removed = directAttachmentsRef.current.find((item) => item.id === attachmentId);
    const next = directAttachmentsRef.current.filter((item) => item.id !== attachmentId);
    directAttachmentsRef.current = next;
    setDirectAttachments(next);
    if (
      removed?.thumb_url &&
      !directMessages.some((message) =>
        message.images?.some((image) => image.data_url === removed.thumb_url)
      )
    ) {
      releaseDirectThumbs([removed.thumb_url]);
    }
  };

  const toggleFromPicker = (id: string) => {
    if (activePicker === "source") {
      setSourceModels((prev) => {
        if (prev.includes(id)) return prev.filter((entry) => entry !== id);
        if (prev.length === 0 && !fusionModel) setFusionModel(id);
        return [...prev, id];
      });
      setHoveredModelId(id);
      return;
    }
    if (activePicker === "direct") {
      setDirectModel(id);
      setActivePicker(null);
      return;
    }
    setFusionModel(id);
    setActivePicker(null);
  };

  const sendDirectMessage = async () => {
    const sendGeneration = directAttachmentGenerationRef.current;
    if (directSendGenerationRef.current === sendGeneration) return;
    directSendGenerationRef.current = sendGeneration;
    try {
      const trimmedPrompt = directPrompt.trim();
      // A pasted image can still be reading when Send fires — wait for it so
      // the outgoing turn includes it instead of it landing in the next turn.
      if (directAttachmentReadsRef.current.length > 0) {
        await Promise.allSettled([...directAttachmentReadsRef.current]);
      }
      // The session may have been reset or rehydrated while reads settled;
      // the prompt/model/directMessages captured above belong to it, so bail.
      if (sendGeneration !== directAttachmentGenerationRef.current) return;
      const attachments = directAttachmentsRef.current;
      const promptText =
        trimmedPrompt || (attachments.some(isImageAttachment) ? "What's in this image?" : "");
      if (!promptText || !directModel || isDirectRunning) return;

      const requestId = directRequestIdRef.current + 1;
      directRequestIdRef.current = requestId;
      setDirectError(null);
      setIsDirectRunning(true);
      setDirectRunId(null);

      const conversationId = directConversationId ?? crypto.randomUUID();
      setDirectConversationId(conversationId);
      const imageAttachments = attachments.filter(isImageAttachment);
      const nextMessages: DirectChatMessage[] = [
        ...directMessages,
        {
          role: "user",
          content: promptText,
          ...(imageAttachments.length > 0
            ? {
                images: imageAttachments.map((attachment) => ({
                  name: attachment.name,
                  content_type: attachment.content_type,
                  data_url: attachment.thumb_url,
                })),
              }
            : {}),
        },
      ];
      setDirectMessages(nextMessages);
      setDirectPrompt("");

      const controller = new AbortController();
      directStreamControllerRef.current = controller;

      try {
        await streamDirectChat(
          {
            model: directModel,
            messages: await Promise.all(
              nextMessages.map(async (message, index) => {
                if (!message.images?.length) return message;
                // The latest user turn's pixels travel via `attachments`; for
                // earlier image turns we re-encode the blob thumbnail so the
                // model still sees them on follow-ups. data_url is stripped
                // regardless — it is display-only.
                const isLatestMessage = index === nextMessages.length - 1;
                const images = await Promise.all(
                  message.images.map(async ({ name, content_type, data_url }) => {
                    const content =
                      !isLatestMessage && data_url ? await blobUrlToBase64(data_url) : null;
                    return content ? { name, content_type, content } : { name, content_type };
                  })
                );
                return { role: message.role, content: message.content, images };
              })
            ),
            conversation_id: conversationId,
            temperature: directTemperature,
            max_output_tokens: OPENROUTER_TOKEN_LIMIT,
            web_search_enabled: directWebSearchEnabled,
            reasoning: {
              effort: directReasoningEffort,
              exclude: false,
            },
            attachments: attachments.map(
              ({ id: _id, thumb_url: _thumb, ...attachment }) => attachment
            ),
            service_tier: effectiveTier(directModel),
          },
          (event) => onDirectEvent(event, requestId),
          controller.signal
        );
      } catch (err) {
        if (requestId !== directRequestIdRef.current) return;
        if (isAbortError(err)) return;
        setDirectError(err instanceof Error ? err.message : "Direct chat failed.");
        setIsDirectRunning(false);
        directStreamControllerRef.current = null;
      }
    } finally {
      if (directSendGenerationRef.current === sendGeneration) {
        directSendGenerationRef.current = null;
      }
    }
  };

  const fusionModelDisplay = fusionModel
    ? toDisplayModel(fusionModel, catalogById.get(fusionModel), catalogIndexById.get(fusionModel) ?? null)
    : null;
  const directModelDisplay = directModel
    ? toDisplayModel(directModel, catalogById.get(directModel), catalogIndexById.get(directModel) ?? null)
    : null;
  const FusionProviderIcon = fusionModelDisplay ? resolveProviderIcon(fusionModelDisplay) : null;
  const FusionIcon = FusionProviderIcon?.Icon ?? null;
  const fusionProviderTintStyle = FusionProviderIcon?.tint
    ? ({ "--provider-tint": FusionProviderIcon.tint } as CSSProperties)
    : undefined;
  const DirectProviderIcon = directModelDisplay ? resolveProviderIcon(directModelDisplay) : null;
  const DirectIcon = DirectProviderIcon?.Icon ?? null;
  const directProviderTintStyle = DirectProviderIcon?.tint
    ? ({ "--provider-tint": DirectProviderIcon.tint } as CSSProperties)
    : undefined;

  const getAgentStatus = (agentId: string): "ready" | "running" | "completed" | "error" => {
    const result = sourceResultsByAgent.get(agentId);
    if (result) return result.status === "ok" ? "completed" : "error";
    if (isRunning && (stepState.search === "active" || stepState.parallel === "active" || stepState.parallel === "done")) {
      return "running";
    }
    return "ready";
  };

  const modelRuntimeAgents = (modelId: string) =>
    activeRuntimeAgents.filter((agent) => agent.model === modelId);

  const getCardStatus = (modelId: string): "ready" | "running" | "completed" | "error" => {
    const agents = modelRuntimeAgents(modelId);
    if (agents.length === 0) {
      if (isRunning && (stepState.search === "active" || stepState.parallel === "active" || stepState.parallel === "done")) {
        return "running";
      }
      return "ready";
    }
    const results = agents
      .map((agent) => sourceResultsByAgent.get(agent.id))
      .filter((result): result is SourceModelResult => Boolean(result));
    if (results.length === 0) {
      if (isRunning && (stepState.search === "active" || stepState.parallel === "active" || stepState.parallel === "done")) {
        return "running";
      }
      return "ready";
    }
    if (results.every((result) => result.status === "ok")) return "completed";
    if (results.every((result) => result.status === "error")) return "error";
    return "completed";
  };

  const personaForModel = (modelId: string): PersonaAssignment | null => {
    const agents = modelRuntimeAgents(modelId);
    for (const agent of agents) {
      const persona = personaByAgent.get(agent.id);
      if (persona) return persona;
    }
    return null;
  };

  const agentDisplayModel = (modelId: string): DisplayModel =>
    toDisplayModel(modelId, catalogById.get(modelId), catalogIndexById.get(modelId) ?? null);

  const agentDisplayName = (agent: SourceAgentSpec): string => {
    const display = agentDisplayModel(agent.model);
    const siblings = activeRuntimeAgents.filter((entry) => entry.model === agent.model);
    if (siblings.length > 1) {
      const instanceIndex = siblings.findIndex((entry) => entry.id === agent.id) + 1;
      return `${display.name} #${instanceIndex}`;
    }
    return display.name;
  };

  const liveAgentRows = activeRuntimeAgents.map((agent) => {
    const result = sourceResultsByAgent.get(agent.id);
    const status = getAgentStatus(agent.id);
    const persona = personaByAgent.get(agent.id) ?? null;
    return {
      agent,
      model: agentDisplayModel(agent.model),
      displayName: agentDisplayName(agent),
      status,
      latency: result?.latency_ms ?? null,
      persona,
    } as const;
  });

  const orchestrationStatusLabel = useMemo(() => {
    const statuses = ORCHESTRATION_FLOW.map((item) => stepState[item.id]);
    if (statuses.some((status) => status === "active")) return "Live";
    if (statuses.every((status) => status === "done")) return "Complete";
    return "Pending";
  }, [stepState]);

  const getActiveTabText = (): string => {
    if (activeResultTab === "fusion") return fusionOutput || critiqueOutput;
    if (activeResultTab === "sources") {
      return sourceResults
        .map((result) => {
          const name = agentDisplayName({ id: agentKey(result), model: result.model });
          return `## ${name}\n\n${result.status === "ok" ? result.content : result.error || "Error"}`;
        })
        .join("\n\n");
    }

    const summaries = summaryAgents
      .map((agent) => {
        const result = sourceResultsByAgent.get(agent.id);
        const initialResponse = result
          ? result.status === "ok"
            ? result.content
            : result.error || "Error"
          : "No response yet.";

        return `## ${agentDisplayName(agent)}\n\nStatus: ${formatAgentStatus(getAgentStatus(agent.id))}\n\n### Initial Response\n${initialResponse}`;
      })
      .join("\n\n");

    return `${summaries}\n\n### Critique / Debate\n${critiqueOutput || "Not available yet."}`;
  };

  const handleCopyResult = async () => {
    const text = getActiveTabText().trim();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      setError("Failed to copy result to clipboard.");
    }
  };

  const handleExportResult = () => {
    const text = getActiveTabText().trim();
    if (!text) return;

    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const href = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = href;
    anchor.download = `openchat-${activeResultTab}-result.txt`;
    anchor.click();
    URL.revokeObjectURL(href);
  };

  const handleRegenerateFusion = async () => {
    if (!canRegenerateFusion) return;

    setError(null);
    setHasRunStarted(true);
    setIsRunning(true);
    setActiveResultTab("fusion");
    setFusionOutput("");
    setStepState((prev) => ({ ...prev, fusion: "active" }));
    setStepDetail((prev) => ({
      ...prev,
      fusion: "Regenerating fusion response from existing source results.",
    }));

    try {
      const result = await regenerateFusion({
        prompt,
        fusion_model: fusionModel,
        temperature,
        reasoning: {
          effort: reasoningEffort,
          exclude: false,
        },
        source_results: sourceResults,
        critique_output: critiqueOutput,
        service_tier: effectiveTier(fusionModel),
      });

      setFusionOutput(result.content);
      setStepState((prev) => ({ ...prev, fusion: "done" }));
      setStepDetail((prev) => ({ ...prev, fusion: "Fusion regeneration complete." }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Fusion regeneration failed.");
      setStepState((prev) => ({ ...prev, fusion: "pending" }));
    } finally {
      setIsRunning(false);
    }
  };

  const renderPicker = () => {
    if (!activePicker) return null;

    const focusedTiers = focusedModel ? serviceTiersByModel[focusedModel.id] ?? [] : [];

    return (
      <div className="model-picker-popover" role="dialog" aria-label="Model picker" ref={pickerRef}>
        <div className="picker-search-row">
          <input
            ref={pickerSearchRef}
            value={pickerQuery}
            onChange={(event) => setPickerQuery(event.target.value)}
            placeholder="Search models"
            aria-label="Search models"
          />
        </div>

        <div className="picker-content-grid">
          <div className="picker-list" role="listbox" aria-label="Available models">
            {filteredModels.map((model) => {
              const isFocused = focusedModelId === model.id;
              const isSelected =
                activePicker === "source" ? sourceModels.includes(model.id) : model.id === fusionModel;

              return (
                <button
                  key={model.id}
                  type="button"
                  className={`picker-row${isFocused ? " focused" : ""}${isSelected ? " selected" : ""}`}
                  onMouseEnter={() => setHoveredModelId(model.id)}
                  onFocus={() => setHoveredModelId(model.id)}
                  onClick={() => toggleFromPicker(model.id)}
                >
                  <span className="picker-row-main">
                    <span className="picker-row-name">{model.name}</span>
                    <span className="picker-row-id">{model.id}</span>
                  </span>
                  {activePicker === "source" && <span className="picker-row-check">{isSelected ? "✓" : ""}</span>}
                </button>
              );
            })}
            {filteredModels.length === 0 && <p className="muted picker-empty">No models match your search.</p>}
          </div>

          <aside className="picker-detail" aria-live="polite">
            {focusedModel ? (
              <>
                <h4>{focusedModel.name}</h4>
                <p className="muted picker-provider">{focusedModel.provider}</p>
                <p className="picker-model-id">{focusedModel.id}</p>
                <p className="muted">
                  {focusedModel.description ?? "No model description is available for this entry."}
                </p>

                <dl className="picker-metadata-grid">
                  <div>
                    <dt>Context</dt>
                    <dd>{focusedModel.contextLength ? `${focusedModel.contextLength.toLocaleString()} tokens` : "—"}</dd>
                  </div>
                  <div>
                    <dt>Modality</dt>
                    <dd>{focusedModel.modality ?? "—"}</dd>
                  </div>
                  <div>
                    <dt>Input</dt>
                    <dd>{focusedModel.promptPrice ?? "—"}</dd>
                  </div>
                  <div>
                    <dt>Output</dt>
                    <dd>{focusedModel.completionPrice ?? "—"}</dd>
                  </div>
                  <div>
                    <dt>Tiers</dt>
                    <dd>{focusedTiers.length > 0 ? focusedTiers.join(", ") : "—"}</dd>
                  </div>
                </dl>
              </>
            ) : (
              <p className="muted">Select a model to preview details.</p>
            )}
          </aside>
        </div>
      </div>
    );
  };

  return (
    <div className="app-shell">
      <div className="bg-lights" aria-hidden="true" />

      {isAppSettingsOpen && (
        <div className="api-settings-overlay" role="presentation">
          <section className="api-settings-panel" role="dialog" aria-modal="true" aria-labelledby="app-settings-title">
            <div className="api-settings-header">
              <div>
                <p className="eyebrow">OpenChat</p>
                <h2 id="app-settings-title">Settings</h2>
              </div>
              {appSettings?.api_key_configured && (
                <button type="button" className="api-settings-close" onClick={() => setIsAppSettingsOpen(false)} aria-label="Close">
                  ×
                </button>
              )}
            </div>

            <div className="settings-section">
              <h3 className="settings-section-title">Appearance</h3>
              <div className="theme-toggle" role="group" aria-label="Theme">
                <button
                  type="button"
                  className={`theme-toggle-btn${themeMode === "dark" ? " active" : ""}`}
                  aria-pressed={themeMode === "dark"}
                  onClick={() => setThemeMode("dark")}
                >
                  <img src={moonIcon} alt="" aria-hidden="true" className="ui-icon" />
                  Dark
                </button>
                <button
                  type="button"
                  className={`theme-toggle-btn${themeMode === "light" ? " active" : ""}`}
                  aria-pressed={themeMode === "light"}
                  onClick={() => setThemeMode("light")}
                >
                  <img src={sunIcon} alt="" aria-hidden="true" className="ui-icon" />
                  Light
                </button>
              </div>
            </div>

            <div className="settings-section">
              <h3 className="settings-section-title">Connect OpenRouter</h3>
              <p className="api-settings-copy">
                Add your OpenRouter API key to use model fusion. It is stored locally in the app data folder.
              </p>
              <label className="api-settings-field">
                <span>API key</span>
                <input
                  type="password"
                  value={apiKeyInput}
                  onChange={(event) => setApiKeyInput(event.target.value)}
                  placeholder={appSettings?.api_key_hint ?? "sk-or-..."}
                  autoFocus
                />
              </label>
              <label className="api-settings-field">
                <span>Base URL <small>(optional)</small></span>
                <input
                  type="url"
                  value={baseUrlInput}
                  onChange={(event) => setBaseUrlInput(event.target.value)}
                  placeholder="https://openrouter.ai/api/v1"
                />
              </label>
            </div>

            <SettingsUpdateSection appVersion={appVersion} />

            {apiSettingsError && <p className="error">{apiSettingsError}</p>}
            <div className="api-settings-actions">
              {appSettings?.api_key_configured && (
                <button type="button" className="subtle-btn" onClick={() => setIsAppSettingsOpen(false)}>
                  Cancel
                </button>
              )}
              <button type="button" className="send-btn" onClick={() => void saveApiSettings()} disabled={isSavingApiSettings}>
                {isSavingApiSettings ? "Saving..." : "Save"}
              </button>
            </div>
          </section>
        </div>
      )}

      <aside className="sidebar-rail">
        <button type="button" className="rail-new-chat-btn" onClick={handleStartNewChat} aria-label="Start new chat">
          <img src={squarePenIcon} alt="" aria-hidden="true" className="ui-icon rail-new-chat-icon" />
        </button>
        <button
          type="button"
          className={`rail-direct-btn${activePage === "direct" ? " active" : ""}`}
          onClick={() => {
            resetDirectSession();
            setActivePage("direct");
          }}
          aria-label="Open direct chat"
        >
          <img src={messageSquareIcon} alt="" aria-hidden="true" className="ui-icon rail-direct-icon" />
        </button>
        <button
          type="button"
          className={`rail-history-btn${activePage === "history" ? " active" : ""}`}
          onClick={() => setActivePage("history")}
          aria-label="Open chat history"
        >
          <img src={historyIcon} alt="" aria-hidden="true" className="ui-icon rail-history-icon" />
        </button>
        <button
          type="button"
          className={`rail-settings-btn${isAppSettingsOpen ? " active" : ""}`}
          onClick={openAppSettings}
          aria-label="Open settings"
        >
          <img src={settingsIcon} alt="" aria-hidden="true" className="ui-icon rail-settings-icon" />
        </button>
      </aside>
      <button
        type="button"
        className={`rail-settings-btn mobile-settings-btn${isAppSettingsOpen ? " active" : ""}`}
        onClick={openAppSettings}
        aria-label="Open settings"
      >
        <img src={settingsIcon} alt="" aria-hidden="true" className="ui-icon rail-settings-icon" />
      </button>

      {activePage === "fusion" && (
        <div className="page-top">
          <header className="hero">
            <h1>
              <span className="hero-title-glow">Model Fusion</span> <span className="beta">BETA</span>
            </h1>
            <p>Multiple models think, search, and synthesize into one answer.</p>
          </header>

          <section className="top-controls panel">
            <div className="mode-tabs">
              {modeTabs.map((mode) => (
                <div className="mode-tab-wrap" key={mode.id}>
                  <button
                    type="button"
                    className={`mode-tab${activeMode === mode.id ? " active" : ""}`}
                    onClick={() => handleSelectMode(mode.id)}
                  >
                    <img src={mode.icon} alt="" aria-hidden="true" className="ui-icon mode-icon" />
                    <span>
                      <strong>{mode.title}</strong>
                      <small>{mode.sub}</small>
                    </span>
                  </button>
                  {mode.workflowId && (
                    <button
                      type="button"
                      className="mode-tab-delete"
                      aria-label={`Delete workflow ${mode.title}`}
                      title="Delete workflow"
                      onClick={(event) => {
                        event.stopPropagation();
                        void handleDeleteSavedWorkflow(mode.workflowId as string);
                      }}
                    >
                      ×
                    </button>
                  )}
                </div>
              ))}
            </div>

            {isSaveWorkflowEditing ? (
              <div className={`save-btn save-btn-editing${isSavingWorkflow ? " saving" : ""}`}>
                <img src={bookmarkIcon} alt="" aria-hidden="true" className="ui-icon" />
                <input
                  className="save-btn-input"
                  value={workflowNameInput}
                  onChange={(event) => setWorkflowNameInput(event.target.value)}
                  onBlur={() => {
                    void saveCurrentWorkflow(workflowNameInput);
                  }}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault();
                      void saveCurrentWorkflow(workflowNameInput);
                    }
                    if (event.key === "Escape") {
                      setIsSaveWorkflowEditing(false);
                      setWorkflowNameInput("");
                    }
                  }}
                  placeholder="Workflow name"
                  disabled={isSavingWorkflow}
                  autoFocus
                />
              </div>
            ) : (
              <button type="button" className="save-btn" onClick={beginWorkflowSave}>
                <img src={bookmarkIcon} alt="" aria-hidden="true" className="ui-icon" />
                Save Workflow
              </button>
            )}
          </section>

          {(workflowSaveError || workflowsError) && <p className="error workflow-save-error">{workflowSaveError ?? workflowsError}</p>}
        </div>
      )}

      {activePage === "direct" && (
        <div className="page-top">
          <header className="hero">
            <h1>
              <span className="hero-title-glow">Direct Chat</span> <span className="beta">BETA</span>
            </h1>
            <p>Single-model chat with persistent browser-session transcript.</p>
          </header>

          <section className="panel direct-model-panel">
            <div className="panel-heading">
              <h2>MODEL</h2>
              <button
                type="button"
                className="subtle-btn"
                disabled={isCatalogLoading || modelCatalog.length === 0}
                onClick={() => setActivePicker("direct")}
              >
                Select Model
              </button>
            </div>

            <button
              type="button"
              className="fusion-model-row"
              onClick={() => setActivePicker("direct")}
              disabled={isCatalogLoading || modelCatalog.length === 0}
            >
              {directModelDisplay ? (
                <div className="fusion-left">
                  <span className={`fusion-star${DirectIcon ? " has-icon" : ""}`} style={directProviderTintStyle}>
                    {DirectIcon ? <DirectIcon size={18} className="model-company-icon" /> : "✧"}
                  </span>
                  <span>
                    <strong>{directModelDisplay.name}</strong>
                    <small>{directModelDisplay.provider}</small>
                  </span>
                </div>
              ) : (
                <span className="muted">Select direct chat model</span>
              )}
            </button>
            {directModel && renderServiceTierSelect(directModel, isDirectRunning)}
            <div className="picker-anchor">{activePicker === "direct" && renderPicker()}</div>
          </section>
        </div>
      )}

      <main className="main-column">
        {activePage === "history" && (
          <section className="panel history-panel">
            <div className="history-head">
              <h2>Chat History</h2>
            </div>

            {historyError && <p className="error">{historyError}</p>}

            {isHistoryLoading ? (
              <p className="muted">Loading saved chats...</p>
            ) : historyItems.length === 0 ? (
              <div className="history-empty">No saved chats yet. Run a chat and it will appear here automatically.</div>
            ) : (
              <div className="history-list" role="list" aria-label="Saved chats">
                {historyItems.map((chat) => (
                  <article className="history-item" role="listitem" key={chat.chat_id}>
                    <button
                      type="button"
                      className="history-item-open"
                      onClick={() => handleOpenHistoryChat(chat.chat_id)}
                      disabled={historyBusyChatId === chat.chat_id}
                    >
                      <div className="history-title-row">
                        <h3>{chat.prompt_preview || "Untitled chat"}</h3>
                        <span className={`history-type-badge ${getChatType(chat.status)}`}>
                          {toChatTypeLabel(chat.status)}
                        </span>
                      </div>
                      <p className="muted">{new Date(chat.created_at).toLocaleString()}</p>
                      <p className="history-models">
                        {getChatType(chat.status) === "direct"
                          ? `Model: ${chat.fusion_model || chat.source_models[0] || "—"}`
                          : `Sources: ${chat.source_models.join(", ")} · Fusion: ${chat.fusion_model}`}
                      </p>
                    </button>
                    <button
                      type="button"
                      className="history-item-delete"
                      onClick={() => handleDeleteHistoryChat(chat.chat_id)}
                      disabled={historyBusyChatId === chat.chat_id}
                    >
                      Delete
                    </button>
                  </article>
                ))}
              </div>
            )}
          </section>
        )}

        {activePage === "fusion" && (
          <>
        <section className="panel agents-panel">
          <div className="panel-heading">
            <h2>AGENTS <span>(Parallel Reasoning)</span></h2>
            <button
              type="button"
              className="subtle-btn"
              disabled={isCatalogLoading || modelCatalog.length === 0}
              onClick={() => setActivePicker("source")}
            >
              + Add Agent
            </button>
          </div>

          {catalogError && <p className="error">{catalogError}</p>}

          <div className="agent-card-grid">
            {(() => {
              const firstAgentIdByModel = new Map<string, string>();
              for (const agent of runtimeAgents) {
                if (!firstAgentIdByModel.has(agent.model)) firstAgentIdByModel.set(agent.model, agent.id);
              }

              if (runtimeAgents.length === 0) {
                return (
                  <div className="empty-card">Select agents from + Add Agent to begin parallel reasoning.</div>
                );
              }

              return runtimeAgents.map((agent) => {
                const modelDisplay = agentDisplayModel(agent.model);
                const modelIndex = Math.max(0, sourceModels.indexOf(agent.model));
                const colorName = AGENT_COLORS[modelIndex % AGENT_COLORS.length];
                const providerIcon = resolveProviderIcon(modelDisplay);
                const ProviderIcon = providerIcon?.Icon ?? null;
                const providerTintStyle = providerIcon?.tint
                  ? ({ "--provider-tint": providerIcon.tint } as CSSProperties)
                  : undefined;
                const agentStatus = getAgentStatus(agent.id);
                const count = agentCounts[agent.model] ?? 1;
                const isFirstInstance = firstAgentIdByModel.get(agent.model) === agent.id;
                const persona = personaByAgent.get(agent.id) ?? null;

                const handleClose = () => {
                  if (isFirstInstance && count <= 1) {
                    removeSourceModel(agent.model);
                  } else {
                    decrementAgentCount(agent.model);
                  }
                };

                return (
                  <article key={agent.id} className={`agent-card tone-${colorName}`}>
                    <button type="button" className="card-close" onClick={handleClose} aria-label="Remove agent instance">
                      ×
                    </button>
                    <div className="agent-avatar" aria-hidden="true" style={providerTintStyle}>
                      {ProviderIcon ? (
                        <ProviderIcon size={20} className="model-company-icon" />
                      ) : (
                        modelDisplay.name.slice(0, 2).toUpperCase()
                      )}
                    </div>
                    <h3>{agentDisplayName(agent)}</h3>
                    <p>{modelDisplay.provider}</p>
                    {persona && (
                      <div className="persona-card-block">
                        <p className="persona-chip">{persona.title}</p>
                        <p className="persona-chip">Temp {persona.temperature.toFixed(2)}</p>
                        <p className="persona-description">{persona.description}</p>
                      </div>
                    )}
                    <p className={`agent-status status-${agentStatus}`}>Status: {formatAgentStatus(agentStatus)}</p>
                    {isFirstInstance ? (
                      <>
                        <div className="agent-count-control" role="group" aria-label={`Instance count for ${modelDisplay.name}`}>
                          <button
                            type="button"
                            className="agent-count-btn"
                            onClick={() => decrementAgentCount(agent.model)}
                            disabled={count <= 1 || isRunning}
                            aria-label="Decrease instance count"
                          >
                            −
                          </button>
                          <span className="agent-count-value" aria-live="polite">{count}</span>
                          <button
                            type="button"
                            className="agent-count-btn"
                            onClick={() => incrementAgentCount(agent.model)}
                            disabled={count >= MAX_AGENT_COUNT || isRunning}
                            aria-label="Increase instance count"
                          >
                            +
                          </button>
                        </div>
                        {renderServiceTierSelect(agent.model, isRunning)}
                      </>
                    ) : (
                      <p className="agent-instance-label">Instance #{agent.id.split("#")[1] ?? ""}</p>
                    )}
                  </article>
                );
              });
            })()}
          </div>

          <div className="picker-anchor">{activePicker === "source" && renderPicker()}</div>

          <div className="fusion-row-wrap">
            <div className="panel-heading">
              <h2>FUSION MODEL <span>(Final Synthesis)</span></h2>
            </div>

            <button
              type="button"
              className="fusion-model-row"
              onClick={() => setActivePicker("fusion")}
              disabled={isCatalogLoading || modelCatalog.length === 0}
            >
              {fusionModelDisplay ? (
                <>
                  <div className="fusion-left">
                    <span className={`fusion-star${FusionIcon ? " has-icon" : ""}`} style={fusionProviderTintStyle}>
                      {FusionIcon ? <FusionIcon size={18} className="model-company-icon" /> : "✧"}
                    </span>
                  <span>
                    <strong>{fusionModelDisplay.name}</strong>
                    <small>{fusionModelDisplay.provider}</small>
                  </span>
                </div>
                </>
              ) : (
                <span className="muted">Select fusion model</span>
              )}
            </button>
            {fusionModel && renderServiceTierSelect(fusionModel, isRunning)}
            <div className="picker-anchor">{activePicker === "fusion" && renderPicker()}</div>
          </div>
        </section>

        {!hasRunStarted && (
          <section
            className={`panel composer-panel${isOptimizationReviewPending ? " optimization-pending" : ""}${isOptimizingPrompt ? " optimizing" : ""}`}
          >
            <textarea
              value={prompt}
              onChange={(event) => {
                const nextPrompt = event.target.value;
                setPrompt(nextPrompt);
                if (isOptimizationReviewPending) {
                  clearOptimizationReview();
                }
              }}
              placeholder="Ask anything... agents will research, debate, and fuse the best answer."
              rows={4}
            />

            {isOptimizingPrompt && <p className="composer-inline-note">Optimizing prompt with openai/gpt-oss-20b...</p>}
            {isOptimizationReviewPending && (
              <p className="composer-inline-note pending-review">
                Review pending changes: ✓ accepts prompt/personas and runs. ✕ runs immediately with your original prompt and no personas.
              </p>
            )}

            {isOptimizationReviewPending && pendingPersonaAssignments.length > 0 && (
              <div className="persona-review-grid" role="group" aria-label="Persona review">
                {pendingPersonaAssignments.map((assignment) => {
                  const assignmentKey = assignment.agent_id?.trim() || assignment.model;
                  const agentSpec = runtimeAgents.find((agent) => agent.id === assignmentKey);
                  const label = agentSpec
                    ? agentDisplayName(agentSpec)
                    : assignment.model;
                  return (
                    <article key={assignmentKey} className="persona-review-card">
                      <h4>{label}</h4>
                      <label>
                        <span>Persona title</span>
                        <input
                          value={assignment.title}
                          onChange={(event) => updatePendingPersona(assignmentKey, "title", event.target.value)}
                        />
                      </label>
                      <label>
                        <span>Persona guidance</span>
                        <textarea
                          value={assignment.description}
                          rows={3}
                          onChange={(event) => updatePendingPersona(assignmentKey, "description", event.target.value)}
                        />
                      </label>
                      <label>
                        <span>Temperature (0.50–1.20)</span>
                        <input
                          type="number"
                          min={0.5}
                          max={1.2}
                          step={0.05}
                          value={assignment.temperature}
                          onChange={(event) => updatePendingPersona(assignmentKey, "temperature", event.target.value)}
                        />
                      </label>
                    </article>
                  );
                })}
              </div>
            )}

            <input
              ref={fileInputRef}
              type="file"
              className="composer-file-input"
              onChange={onAttachmentChange}
              multiple
              accept=".txt,.md,.json,.csv,.py,.js,.ts,.tsx,.jsx,.html,.css,.yaml,.yml,.xml,.log,text/*"
            />

            {attachments.length > 0 && (
              <div className="attachment-chip-row" role="list" aria-label="Attached files">
                {attachments.map((attachment) => (
                  <div key={attachment.id} className="attachment-chip" role="listitem">
                    <span>{attachment.name}</span>
                    <button type="button" onClick={() => removeAttachment(attachment.id)} aria-label={`Remove ${attachment.name}`}>
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}

            <div className="composer-bottom-row">
              <div className="left-icons" ref={settingsRef}>
                <button
                  className="composer-icon-btn"
                  type="button"
                  aria-label="Add attachment"
                  title="Attach up to 5 text files (max 256KB each)"
                  onClick={triggerAttachmentPicker}
                >
                  <img src={paperclipIcon} alt="" aria-hidden="true" className="ui-icon" />
                </button>
                <button
                  className={`composer-icon-btn${webSearchEnabled ? " active-web" : ""}`}
                  type="button"
                  aria-label="Toggle web search"
                  aria-pressed={webSearchEnabled}
                  title={webSearchEnabled ? "Web search enabled" : "Enable OpenRouter web search + web fetch tools"}
                  onClick={() => setWebSearchEnabled((prev) => !prev)}
                >
                  <img src={globeIcon} alt="" aria-hidden="true" className="ui-icon" />
                </button>
                <button
                  className={`composer-icon-btn${isSettingsOpen ? " active-settings" : ""}`}
                  type="button"
                  aria-label="Open parameters"
                  aria-expanded={isSettingsOpen}
                  title="Adjust temperature, max tokens, and reasoning effort"
                  onClick={() => setIsSettingsOpen((prev) => !prev)}
                >
                  <img src={slidersHorizontalIcon} alt="" aria-hidden="true" className="ui-icon" />
                </button>

                {isSettingsOpen && (
                  <div className="composer-settings-popover" role="dialog" aria-label="Model parameters">
                    <label>
                      <span>Temperature: {temperature.toFixed(2)}</span>
                      <input
                        type="range"
                        min={0}
                        max={2}
                        step={0.05}
                        value={temperature}
                        onChange={(event) => setTemperature(Number(event.target.value))}
                      />
                    </label>
                    <label>
                      <span>Reasoning effort</span>
                      <select
                        value={reasoningEffort}
                        onChange={(event) => setReasoningEffort(event.target.value as ReasoningEffort)}
                      >
                        {REASONING_EFFORT_OPTIONS.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      <span>Debate mode</span>
                      <select
                        value={debateMode}
                        onChange={(event) => setDebateMode(event.target.value as DebateMode)}
                      >
                        {DEBATE_MODE_OPTIONS.map((option) => (
                          <option key={option} value={option}>
                            {option === "off"
                              ? "Off (skip debate and go straight to fusion)"
                              : option === "partial"
                                ? "Partial (one reviewer per response)"
                                : "Full (all other reviewers per response)"}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="settings-toggle-row">
                      <span>Personas per source model</span>
                      <input
                        type="checkbox"
                        checked={personaEnabled}
                        onChange={(event) => setPersonaEnabled(event.target.checked)}
                      />
                    </label>
                  </div>
                )}
              </div>
              <div className="composer-right-actions">
                <button
                  type="button"
                  className={`auto-optimize-compact${autoOptimizeEnabled ? " active" : ""}`}
                  aria-pressed={autoOptimizeEnabled}
                  disabled={isOptimizingPrompt || isOptimizationReviewPending}
                  onClick={() => setAutoOptimizeEnabled((prev) => !prev)}
                >
                  <img src={sparklesIcon} alt="" aria-hidden="true" className="ui-icon" />
                  Auto-Optimize
                </button>
                {isOptimizationReviewPending && (
                  <div className="optimize-review-actions" role="group" aria-label="Auto-optimize confirmation">
                    <button
                      type="button"
                      className="optimize-review-btn approve"
                      onClick={handleApproveOptimizedPrompt}
                      disabled={isRunning || isOptimizingPrompt}
                    >
                      ✓
                    </button>
                    <button
                      type="button"
                      className="optimize-review-btn reject"
                      onClick={handleRejectOptimizedPrompt}
                      disabled={isRunning || isOptimizingPrompt}
                    >
                      ✕
                    </button>
                  </div>
                )}
                <button className="send-btn" type="button" onClick={handleSend} disabled={!canRun}>
                  <img src={sendIcon} alt="" aria-hidden="true" className="ui-icon" />
                  {isOptimizingPrompt ? "Optimizing..." : isRunning ? "Running..." : "Send"}
                </button>
              </div>
            </div>
          </section>
        )}

        {error && <p className="error main-error">{error}</p>}

        {showResultsPanel && (
          <section className="panel results-panel">
            <div className="result-tabs" role="tablist" aria-label="Result tabs">
              <div className="result-tab-list">
                <button
                  type="button"
                  className={`result-tab-btn${activeResultTab === "fusion" ? " active" : ""}`}
                  role="tab"
                  aria-selected={activeResultTab === "fusion"}
                  onClick={() => setActiveResultTab("fusion")}
                >
                  Fusion Answer
                </button>
                <button
                  type="button"
                  className={`result-tab-btn${activeResultTab === "summaries" ? " active" : ""}`}
                  role="tab"
                  aria-selected={activeResultTab === "summaries"}
                  onClick={() => setActiveResultTab("summaries")}
                >
                  Agent Summaries
                </button>
                {hasSourceResults && (
                  <button
                    type="button"
                    className={`result-tab-btn${activeResultTab === "sources" ? " active" : ""}`}
                    role="tab"
                    aria-selected={activeResultTab === "sources"}
                    onClick={() => setActiveResultTab("sources")}
                  >
                    Sources ({sourceResults.length})
                  </button>
                )}
              </div>

              <div className="result-tab-actions" role="group" aria-label="Result actions">
                <button type="button" className="result-action-btn" onClick={handleCopyResult}>
                  <img src={copyIcon} alt="" aria-hidden="true" className="ui-icon" />
                  Copy
                </button>
                <button type="button" className="result-action-btn" onClick={handleExportResult}>
                  <img src={downloadIcon} alt="" aria-hidden="true" className="ui-icon" />
                  Export
                </button>
                <button
                  type="button"
                  className="result-action-btn regenerate"
                  onClick={handleRegenerateFusion}
                  disabled={!canRegenerateFusion}
                >
                  <img src={rotateCcwIcon} alt="" aria-hidden="true" className="ui-icon" />
                  Regenerate
                </button>
              </div>
            </div>

            {activeResultTab === "fusion" && (
              <div className="fusion-answer-panel">
                <h3>Fusion Response</h3>
                {hasFusionOutput ? (
                  <div className="markdown-body">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{fusionOutput}</ReactMarkdown>
                  </div>
                ) : (
                  <p className="muted">
                    {stepState.fusion === "active"
                      ? "Fusion model is synthesizing the final answer."
                      : stepState.critique === "active" || stepState.critique === "done"
                        ? "Critique is complete. Waiting for fusion output."
                        : "Run in progress. Fusion answer will appear when synthesis begins."}
                  </p>
                )}
              </div>
            )}

            {activeResultTab === "summaries" && (
              <div className="agent-summaries-grid">
                {summaryAgents.map((agent) => {
                  const result = sourceResultsByAgent.get(agent.id);
                  const status = getAgentStatus(agent.id);
                  const display = agentDisplayModel(agent.model);
                  const reviewerLabel = (item: DebateResult) => {
                    const reviewerId = item.reviewer_agent_id?.trim() || item.reviewer_model;
                    return reviewerId === item.reviewer_model
                      ? item.reviewer_model
                      : agentDisplayName({ id: reviewerId, model: item.reviewer_model });
                  };

                  return (
                    <article key={agent.id} className="summary-card">
                      <div className="summary-card-head">
                        <h4>{agentDisplayName(agent)}</h4>
                        <span className={`summary-status status-${status}`}>{formatAgentStatus(status)}</span>
                      </div>
                      <p className="summary-provider">{display.provider}</p>

                      <div className="summary-section">
                        <h5>Initial Response</h5>
                        {result ? (
                          result.status === "ok" ? (
                            <pre>{result.content}</pre>
                          ) : (
                            <pre>{result.error || "Model returned an error."}</pre>
                          )
                        ) : (
                          <p className="muted">
                            {isRunning ? "Waiting for this agent response..." : "No response yet."}
                          </p>
                        )}
                      </div>

                      <div className="summary-section">
                        <h5>Critique / Debate</h5>
                        {(() => {
                          const agentDebates = debateResults.filter(
                            (item) => (item.target_agent_id?.trim() || item.target_model) === agent.id
                          );
                          if (agentDebates.length > 0) {
                            return (
                              <details className="debate-dropdown" open={agentDebates.length <= 1}>
                                <summary>
                                  Reviewer critiques ({agentDebates.length})
                                </summary>
                                <div className="debate-dropdown-body">
                                  {agentDebates.map((item, idx) => (
                                    <article key={`${item.target_agent_id ?? item.target_model}-${item.reviewer_agent_id ?? item.reviewer_model}-${idx}`} className="debate-entry">
                                      <h6>{reviewerLabel(item)}</h6>
                                      <pre>{item.content}</pre>
                                    </article>
                                  ))}
                                </div>
                              </details>
                            );
                          }

                          return (
                            <p className="muted">
                              {stepState.critique === "active"
                                ? "Critique is in progress..."
                                : hasSourceResults
                                  ? "No reviewer critique captured for this response yet."
                                  : "Awaiting source responses before critique."}
                            </p>
                          );
                        })()}
                      </div>
                    </article>
                  );
                })}
                {summaryAgents.length === 0 && (
                  <article className="summary-card">
                    <div className="summary-section">
                      <h5>Agent Summaries</h5>
                      <p className="muted">No successful agent responses were returned for this run.</p>
                    </div>
                  </article>
                )}
              </div>
            )}

            {activeResultTab === "sources" && hasSourceResults && (
              <div className="output-block">
                <h3>Source Responses</h3>
                {sourceResults.map((result) => (
                  <article key={agentKey(result)} className="output-item">
                    <div className="output-head">
                      <strong>{agentDisplayName({ id: agentKey(result), model: result.model })}</strong>
                      <span className={result.status === "ok" ? "ok" : "error"}>{result.status}</span>
                    </div>
                    {result.persona && (
                      <div className="persona-output-block">
                        <p className="persona-chip">{result.persona.title}</p>
                        <p className="persona-chip">Temp {result.persona.temperature.toFixed(2)}</p>
                        <p className="persona-description">{result.persona.description}</p>
                      </div>
                    )}
                    <pre>{result.status === "ok" ? result.content : result.error}</pre>
                  </article>
                ))}
              </div>
            )}
          </section>
        )}

          </>
        )}

        {activePage === "direct" && (
          <>
            <section className="panel composer-panel direct-composer-panel">
              <DirectChatTranscript messages={directMessages} isRunning={isDirectRunning} />

              <textarea
                value={directPrompt}
                onChange={(event) => setDirectPrompt(event.target.value)}
                onPaste={onDirectComposerPaste}
                placeholder="Message your selected model..."
                rows={4}
              />

              <input
                ref={directFileInputRef}
                type="file"
                className="composer-file-input"
                onChange={onDirectAttachmentChange}
                multiple
                accept=".txt,.md,.json,.csv,.py,.js,.ts,.tsx,.jsx,.html,.css,.yaml,.yml,.xml,.log,text/*,image/*"
              />

              {directAttachments.length > 0 && (
                <div className="attachment-chip-row" role="list" aria-label="Attached files">
                  {directAttachments.map((attachment) => (
                    <div key={attachment.id} className="attachment-chip" role="listitem">
                      {isImageAttachment(attachment) && attachment.thumb_url && (
                        <img
                          className="attachment-thumb"
                          src={attachment.thumb_url}
                          alt={`Attached image ${attachment.name}`}
                        />
                      )}
                      <span>{attachment.name}</span>
                      <button type="button" onClick={() => removeDirectAttachment(attachment.id)} aria-label={`Remove ${attachment.name}`}>
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              )}

              <div className="composer-bottom-row">
                <div className="left-icons" ref={directSettingsRef}>
                  <button
                    className="composer-icon-btn"
                    type="button"
                    aria-label="Add attachment"
                    title="Attach text files or images, or paste an image with Ctrl+V"
                    onClick={triggerDirectAttachmentPicker}
                  >
                    <img src={paperclipIcon} alt="" aria-hidden="true" className="ui-icon" />
                  </button>
                  <button
                    className={`composer-icon-btn${directWebSearchEnabled ? " active-web" : ""}`}
                    type="button"
                    aria-label="Toggle web search"
                    aria-pressed={directWebSearchEnabled}
                    title={directWebSearchEnabled ? "Web search enabled" : "Enable OpenRouter web search + web fetch tools"}
                    onClick={() => setDirectWebSearchEnabled((prev) => !prev)}
                  >
                    <img src={globeIcon} alt="" aria-hidden="true" className="ui-icon" />
                  </button>
                  <button
                    className={`composer-icon-btn${isDirectSettingsOpen ? " active-settings" : ""}`}
                    type="button"
                    aria-label="Open direct chat parameters"
                    aria-expanded={isDirectSettingsOpen}
                    title="Adjust direct chat parameters"
                    onClick={() => setIsDirectSettingsOpen((prev) => !prev)}
                  >
                    <img src={slidersHorizontalIcon} alt="" aria-hidden="true" className="ui-icon" />
                  </button>

                  {isDirectSettingsOpen && (
                    <div className="composer-settings-popover" role="dialog" aria-label="Direct chat parameters">
                      <label>
                        <span>Temperature: {directTemperature.toFixed(2)}</span>
                        <input
                          type="range"
                          min={0}
                          max={2}
                          step={0.05}
                          value={directTemperature}
                          onChange={(event) => setDirectTemperature(Number(event.target.value))}
                        />
                      </label>
                      <label>
                        <span>Reasoning effort</span>
                        <select
                          value={directReasoningEffort}
                          onChange={(event) => setDirectReasoningEffort(event.target.value as ReasoningEffort)}
                        >
                          {REASONING_EFFORT_OPTIONS.map((option) => (
                            <option key={option} value={option}>
                              {option}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                  )}
                </div>

                <div className="composer-right-actions">
                  <button
                    className="send-btn"
                    type="button"
                    onClick={() => {
                      void sendDirectMessage();
                    }}
                    disabled={!canSendDirect}
                  >
                    <img src={sendIcon} alt="" aria-hidden="true" className="ui-icon" />
                    {isDirectRunning ? "Running..." : "Send"}
                  </button>
                </div>
              </div>
            </section>

            {directError && <p className="error main-error">{directError}</p>}
          </>
        )}

      </main>

      {activePage === "fusion" && <aside className="right-column">
        <section className="panel orchestration-panel">
          <div className="side-head">
            <h3>
              <img src={brainCogIcon} alt="" aria-hidden="true" className="ui-icon flow-head-icon" />
              ORCHESTRATION
            </h3>
            <span className="live-badge">{orchestrationStatusLabel}</span>
          </div>

          <div className="flow-list">
            {ORCHESTRATION_FLOW.map((item) => {
              const status = stepState[item.id];

              return (
                <div key={item.id} className={`flow-item ${status}`}>
                  <div className="flow-dot-wrap">
                    <span className="flow-dot">
                      <img src={item.icon} alt="" aria-hidden="true" className="ui-icon" />
                    </span>
                  </div>
                  <div>
                    <p className="flow-title">{item.label}</p>
                    <p className="flow-detail">{stepDetail[item.id]}</p>
                    <p className="flow-status">
                      {status === "done" ? "Complete" : status === "active" ? "Live" : "Pending"}
                    </p>
                  </div>
                  <div className="flow-check">{status === "done" ? "✓" : "○"}</div>
                </div>
              );
            })}
          </div>
        </section>

        <section className="panel live-panel">
          <h3>LIVE AGENTS</h3>

          {liveAgentRows.length === 0 && <p className="muted">No agents selected.</p>}

          {liveAgentRows.map((row, index) => {
            const tone = AGENT_COLORS[index % AGENT_COLORS.length];
            return (
              <div key={row.agent.id} className="live-row">
                <span className={`live-dot tone-${tone}`} />
                <span className="live-name">{row.displayName}</span>
                <span className="live-bars" aria-hidden="true">
                  <i />
                  <i />
                  <i />
                  <i />
                  <i />
                </span>
                <span className="live-latency">
                  {row.latency ? `${(row.latency / 1000).toFixed(1)}s` : row.status === "running" ? "..." : "--"}
                </span>
                {row.persona && <span className="live-persona-label">{row.persona.title}</span>}
                {row.status === "running" && <span className="loader" />}
                {row.status === "completed" && <span className="done-tick">✓</span>}
                {row.status === "error" && <span className="error-dot">!</span>}
              </div>
            );
          })}
        </section>
      </aside>}

      {appVersion && <div className="app-version-badge">v{appVersion}</div>}
    </div>
  );
}

export default App;
