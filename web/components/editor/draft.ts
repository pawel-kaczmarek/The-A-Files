import type { DesignInfo, ExperimentConfig, ExperimentInput, ExperimentType } from "@/lib/types";

/** Render a value the way Python's ``ast.literal_eval`` reads it back. */
export function literal(value: unknown): string {
  if (typeof value === "boolean") return value ? "True" : "False";
  if (value === null || value === undefined) return "None";
  if (typeof value === "number") return String(value);
  const text = String(value);
  if (/^-?\d+(\.\d+)?(e-?\d+)?$/i.test(text) || /^(True|False|None)$/.test(text)) return text;
  return `'${text.replace(/\\/g, "\\\\").replace(/'/g, "\\'")}'`;
}

export function formatSpec(name: string, parameters: Record<string, unknown>): string {
  const entries = Object.entries(parameters).filter(([, value]) => value !== "" && value !== undefined);
  if (!entries.length) return name;
  return `${name}:${entries.map(([key, value]) => `${key}=${literal(value)}`).join(",")}`;
}

function parseLiteral(text: string): unknown {
  const value = text.trim();
  if (value === "True") return true;
  if (value === "False") return false;
  if (value === "None") return null;
  if (/^-?\d+(\.\d+)?(e-?\d+)?$/i.test(value)) return Number(value);
  const quoted = value.match(/^'(.*)'$|^"(.*)"$/);
  return quoted ? (quoted[1] ?? quoted[2]) : value;
}

export function parseSpec(spec: string): { name: string; parameters: Record<string, unknown>; severity: string | null } {
  const [head, parameterText = ""] = spec.split(/:(.*)/s, 2);
  const [name, severity] = head.split("@");
  const parameters: Record<string, unknown> = {};
  for (const chunk of parameterText.split(",")) {
    const [key, ...rest] = chunk.split("=");
    if (key?.trim() && rest.length) parameters[key.trim()] = parseLiteral(rest.join("="));
  }
  return { name: name.trim(), parameters, severity: severity ?? null };
}

export function parseValues(text: string): (number | string)[] {
  return text
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => (/^-?\d+(\.\d+)?(e-?\d+)?$/i.test(part) ? Number(part) : part));
}

/** A geometric ladder around a default strength, for trade-off sweeps. */
export function strengthLadder(value: unknown): number[] {
  const base = typeof value === "number" && value > 0 ? value : 0.1;
  return [0.25, 0.5, 1, 2, 4].map((factor) => Number((base * factor).toPrecision(3)));
}

const SEED_LIMIT = 1_000_000;

export function newSeed(): number {
  return Math.floor(Math.random() * SEED_LIMIT);
}

export function defaultDraft(type: ExperimentType, design?: DesignInfo): ExperimentInput {
  const config: ExperimentConfig = {
    dataset_id: "vctk",
    file_limit: null,
    methods: [],
    metrics: [],
    attacks: [],
    attack_preset: null,
    attack_sweep: null,
    method_sweep: null,
    payload_lengths: design?.default_payload_lengths ?? [16],
    repetitions: 1,
    random_seed: newSeed(),
    max_workers: 2,
    advanced_options: {},
  };
  switch (type) {
    case "perceptual_quality":
      config.methods = ["LSB_METHOD", "QIM_METHOD", "IMPROVED_SPREAD_SPECTRUM_METHOD"];
      config.metrics = ["SNR_METRIC", "SNR_SEG_METRIC", "STOI_METRIC", "PESQ_METRIC"];
      break;
    case "attack_robustness":
      config.methods = ["QIM_METHOD", "DSSS_METHOD", "IMPROVED_SPREAD_SPECTRUM_METHOD"];
      config.attacks = ["awgn@moderate", "mp3@moderate", "low_pass@moderate", "resample@moderate", "time_shift@moderate"];
      break;
    case "robustness_curve":
      config.methods = ["QIM_METHOD", "DSSS_METHOD", "IMPROVED_SPREAD_SPECTRUM_METHOD"];
      config.attack_sweep = { target: "awgn", parameter: "snr_db", values: [40, 30, 20, 15, 10, 5, 0] };
      break;
    case "embedding_capacity":
      config.methods = ["LSB_METHOD", "QIM_METHOD", "DCT_B1_METHOD"];
      break;
    case "detectability":
      config.methods = ["LSB_METHOD", "QIM_METHOD"];
      config.payload_lengths = [16, 32];
      config.advanced_options = { window_length: 16000, test_fraction: 0.3 };
      break;
    case "tradeoff_curve":
      config.methods = [];
      config.method_sweep = { target: "QIM_METHOD", parameter: "step_scale", values: [0.025, 0.05, 0.1, 0.2, 0.4] };
      config.metrics = ["SNR_METRIC", "STOI_METRIC"];
      config.attacks = ["mp3:bitrate_kbps=64"];
      break;
    case "method_comparison":
      config.methods = ["QIM_METHOD", "DSSS_METHOD", "IMPROVED_SPREAD_SPECTRUM_METHOD", "ECHO_METHOD"];
      config.metrics = ["SNR_METRIC", "STOI_METRIC"];
      config.attacks = ["awgn@moderate", "mp3@moderate"];
      config.max_workers = 1;
      break;
    default:
      config.methods = ["LSB_METHOD", "QIM_METHOD"];
      config.metrics = ["SNR_METRIC"];
  }
  return { name: "", experiment_type: type, research_question: "", hypothesis: "", description: "", tags: [], config };
}

export type StepId = "protocol" | "data" | "methods" | "conditions" | "measures" | "design" | "review";

/** The steps a design needs, in order. */
export function stepsFor(type: ExperimentType): StepId[] {
  const steps: StepId[] = ["protocol", "data", "methods"];
  if (!["perceptual_quality", "embedding_capacity", "detectability"].includes(type)) steps.push("conditions");
  if (type !== "detectability") steps.push("measures");
  steps.push("design", "review");
  return steps;
}

/** Client-side checks for quick feedback; the backend re-validates everything. */
export function stepProblems(step: StepId, draft: ExperimentInput, design?: DesignInfo): string[] {
  const config = draft.config;
  const problems: string[] = [];
  if (step === "protocol" && !draft.name.trim()) problems.push("name");
  if (step === "data" && !config.dataset_id) problems.push("dataset");
  if (step === "methods") {
    const count = config.methods.length + (config.method_sweep ? config.method_sweep.values.length : 0);
    if (design?.requires_sweep === "method" && (!config.method_sweep || config.method_sweep.values.length < 2)) problems.push("sweep");
    else if (count < (design?.min_methods ?? 1)) problems.push("methods");
  }
  if (step === "conditions") {
    if (design?.requires_sweep === "attack" && (!config.attack_sweep || config.attack_sweep.values.length < 2)) problems.push("sweep");
    if (design?.requires_attacks && !config.attacks.length && !config.attack_preset) problems.push("attacks");
  }
  if (step === "measures" && design?.requires_metrics && !config.metrics.length) problems.push("metrics");
  if (step === "design" && (config.payload?.kind ?? "random") === "random"
    && !config.payload_rates_bps?.length && !config.payload_lengths.length) problems.push("payloads");
  return problems;
}
