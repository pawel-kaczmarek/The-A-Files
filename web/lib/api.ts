import type {
  AttackInfo,
  AttackPresets,
  CatalogDataset,
  Corpus,
  Dataset,
  DatasetDetail,
  DesignInfo,
  Experiment,
  ExperimentConfig,
  ExperimentInput,
  ExperimentPlan,
  ExperimentType,
  Inspection,
  MethodInfo,
  MetricInfo,
  PlatformStats,
  RowsPage,
  RunInfo,
  SubsetRule,
  Summary,
  SystemStatus,
} from "@/lib/types";

export const API_BASE = process.env.NEXT_PUBLIC_TAF_API_URL ?? "http://127.0.0.1:8000";
import type { LiteratureEntry, ResearchComparison } from "@/lib/research";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number
  ) {
    super(message);
  }
}

function formatError(status: number, body: string): string {
  try {
    const parsed = JSON.parse(body) as { detail?: unknown };
    if (typeof parsed.detail === "string") return parsed.detail;
    if (Array.isArray(parsed.detail)) {
      return parsed.detail
        .map((item) => {
          const entry = item as { msg?: string; loc?: unknown[] };
          const where = Array.isArray(entry.loc) ? entry.loc.filter((part) => part !== "body").join(".") : "";
          return where ? `${where}: ${entry.msg}` : (entry.msg ?? JSON.stringify(item));
        })
        .join("; ");
    }
  } catch {
    // not JSON
  }
  return `API ${status}: ${body.slice(0, 300)}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    throw new ApiError(formatError(response.status, await response.text()), response.status);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

const json = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

function query(params: Record<string, string | number | boolean | null | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) search.set(key, String(value));
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}

export const api = {
  // System
  health: () => request<SystemStatus>("/api/health"),
  stats: () => request<PlatformStats>("/api/stats"),

  // Catalogue
  methods: () => request<MethodInfo[]>("/api/catalog/methods"),
  literature: () => request<LiteratureEntry[]>("/api/catalog/literature"),
  research: (id: string, params: Record<string, string | number | undefined>) =>
    request<ResearchComparison>(`/api/runs/${id}/research${query(params)}`),
  metrics: () => request<MetricInfo[]>("/api/catalog/metrics"),
  attacks: () => request<AttackInfo[]>("/api/catalog/attacks"),
  designs: () => request<DesignInfo[]>("/api/catalog/designs"),
  corpora: () => request<Corpus[]>("/api/catalog/corpora"),
  presets: (sampleRate = 16000) => request<AttackPresets>(`/api/catalog/presets${query({ sample_rate: sampleRate })}`),
  catalogDatasets: () => request<CatalogDataset[]>("/api/catalog/datasets"),

  // Experiments
  experiments: (includeArchived = false) =>
    request<Experiment[]>(`/api/experiments${query({ include_archived: includeArchived })}`),
  experiment: (id: string) => request<Experiment>(`/api/experiments/${id}`),
  createExperiment: (body: ExperimentInput) => request<Experiment>("/api/experiments", json(body)),
  updateExperiment: (id: string, body: ExperimentInput) =>
    request<Experiment>(`/api/experiments/${id}`, { method: "PUT", body: JSON.stringify(body) }),
  duplicateExperiment: (id: string) => request<Experiment>(`/api/experiments/${id}/duplicate`, { method: "POST" }),
  archiveExperiment: (id: string, archived: boolean) =>
    request<Experiment>(`/api/experiments/${id}/archive${query({ archived })}`, { method: "POST" }),
  deleteExperiment: (id: string) => request<void>(`/api/experiments/${id}`, { method: "DELETE" }),
  preview: (name: string, type: ExperimentType, config: ExperimentConfig) =>
    request<ExperimentPlan>("/api/experiments/preview", json({ ...config, name: name || "preview", experiment_type: type })),
  experimentRuns: (id: string) => request<RunInfo[]>(`/api/experiments/${id}/runs`),
  startRun: (id: string) => request<RunInfo>(`/api/experiments/${id}/runs`, { method: "POST" }),

  // Runs
  runs: (limit = 100) => request<RunInfo[]>(`/api/runs${query({ limit })}`),
  run: (id: string) => request<RunInfo>(`/api/runs/${id}`),
  summary: (id: string) => request<{ run_id: string; status: string; summary: Summary }>(`/api/runs/${id}/summary`),
  rows: (
    id: string,
    params: { method?: string; attack?: string; status?: string; file_name?: string; offset?: number; limit?: number }
  ) => request<RowsPage>(`/api/runs/${id}/rows${query(params)}`),
  facets: (id: string) =>
    request<{ method: string[]; attack: (string | null)[]; file_name: string[]; status: string[] }>(`/api/runs/${id}/facets`),
  manifest: (id: string) => request<Record<string, unknown>>(`/api/runs/${id}/manifest.json?download=false`),
  report: async (id: string, extension: "md" | "tex") => {
    const response = await fetch(`${API_BASE}/api/runs/${id}/report.${extension}?download=false`);
    if (!response.ok) throw new ApiError(formatError(response.status, await response.text()), response.status);
    return response.text();
  },
  cancelRun: (id: string) => request<RunInfo>(`/api/runs/${id}/cancel`, { method: "POST" }),
  deleteRun: (id: string) => request<void>(`/api/runs/${id}`, { method: "DELETE" }),
  inspect: (runId: string, rowId: number) => request<Inspection>(`/api/runs/${runId}/rows/${rowId}/inspect`),

  // Datasets
  datasets: () => request<Dataset[]>("/api/datasets"),
  dataset: (id: string) => request<DatasetDetail>(`/api/datasets/${id}`),
  prepareCorpus: (body: { corpus_id: string; name?: string; rule: SubsetRule; source_path?: string | null }) =>
    request<Dataset>("/api/datasets/prepare", json(body)),
  registerLocal: (body: { name: string; path: string; corpus_id?: string | null }) =>
    request<Dataset>("/api/datasets/local", json(body)),
  synthetic: (body: { name?: string; sample_rate: number; duration_seconds: number; seed: number }) =>
    request<Dataset>("/api/datasets/synthetic", json(body)),
  uploadDataset: async (name: string, files: File[]) => {
    const form = new FormData();
    form.set("name", name);
    for (const file of files) form.append("files", file);
    const response = await fetch(`${API_BASE}/api/datasets/upload`, { method: "POST", body: form });
    if (!response.ok) throw new ApiError(formatError(response.status, await response.text()), response.status);
    return (await response.json()) as Dataset;
  },
  deleteDataset: (id: string) => request<void>(`/api/datasets/${id}`, { method: "DELETE" }),
};

export const urls = {
  runEvents: (id: string) => `${API_BASE}/api/runs/${id}/events`,
  allRunEvents: () => `${API_BASE}/api/runs/events`,
  exportCsv: (id: string) => `${API_BASE}/api/runs/${id}/export.csv`,
  exportSummaryCsv: (id: string) => `${API_BASE}/api/runs/${id}/export_summary.csv`,
  config: (id: string) => `${API_BASE}/api/runs/${id}/config.json`,
  manifest: (id: string) => `${API_BASE}/api/runs/${id}/manifest.json`,
  report: (id: string, extension: "md" | "tex") => `${API_BASE}/api/runs/${id}/report.${extension}`,
  audio: (runId: string, rowId: number, signal: string) => `${API_BASE}/api/runs/${runId}/rows/${rowId}/audio/${signal}.wav`,
  docs: () => `${API_BASE}/docs`,
};
