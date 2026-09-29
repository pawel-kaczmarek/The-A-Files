// API types. They mirror the Pydantic models in src/taf: the backend owns the
// schema, the frontend transports and renders it.

export type ExperimentType =
  | "perceptual_quality"
  | "attack_robustness"
  | "robustness_curve"
  | "embedding_capacity"
  | "detectability"
  | "tradeoff_curve"
  | "method_comparison"
  | "dataset_benchmark"
  | "research_experiment";

export type Property = "imperceptibility" | "robustness" | "capacity" | "security" | "multi_criteria";

export type RunStatus = "queued" | "running" | "completed" | "failed" | "cancelled" | "interrupted";

// ---------------------------------------------------------------- catalogue

/** A backend text in every interface language (English is the fallback). */
export interface LocalizedText {
  en: string;
  pl: string;
}

export interface ReferenceInfo {
  citation: string;
  year: number | null;
  doi: string | null;
  url: string | null;
  link: string | null;
}

/** Fields every method, metric and attack takes from its card (taf.models.card). */
export interface CatalogueCard {
  title: LocalizedText;
  summary: LocalizedText;
  details: LocalizedText;
  abbreviation: string;
  references: ReferenceInfo[];
  requires: string[];
  extra: string | null;
  /** Whether the requirements are installed on the server. */
  available: boolean;
  reference: string | null;
  year: number | null;
  doi: string | null;
}

export interface MethodParameter {
  name: string;
  default: unknown;
  type: string | null;
  required: boolean;
  is_key: boolean;
}

export interface MethodInfo extends CatalogueCard {
  name: string;
  class_name: string;
  /** The method's own label, as recorded in result rows. */
  description: string;
  packaged: boolean;
  family: string | null;
  family_label: LocalizedText;
  purpose: "steganography" | "watermarking" | null;
  purpose_label: LocalizedText | null;
  strength_parameter: string | null;
  parameters: MethodParameter[];
  requires_tensorflow: boolean;
  needs_long_input: boolean;
}

export interface MetricInfo extends CatalogueCard {
  domain?: string | null;
  name: string;
  /** The metric's own label, as recorded in result rows. */
  label: string;
  class_name: string;
  category: string;
  category_label: LocalizedText;
  packaged: boolean;
  requires_tensorflow: boolean;
  higher_is_better: boolean | null;
  components: string[];
  scale: string | null;
  intrusive: boolean;
}

export interface SweepPreset {
  parameter: string;
  values: (number | string)[];
  unit: string;
}

export interface AttackInfo extends CatalogueCard {
  name: string;
  class_name: string;
  description: string;
  family: string;
  family_label: LocalizedText;
  parameters: { name: string; default: unknown }[];
  changes_length_or_rate: boolean;
  stochastic: boolean;
  has_severity: boolean;
  sweep: SweepPreset | null;
}

export interface DesignInfo {
  type: ExperimentType;
  title: string;
  description: string;
  property: Property;
  factors: string[];
  measures: string[];
  analyses: string[];
  requires_metrics: boolean;
  requires_attacks: boolean;
  requires_sweep: "attack" | "method" | null;
  min_methods: number;
  default_payload_lengths: number[];
}

export interface CatalogDataset {
  id: string;
  label: string;
  kind: string;
  file_count: number;
  domain: string | null;
  sample_rate: number | null;
  total_duration_seconds: number | null;
}

export interface Corpus {
  id: string;
  name: string;
  domain: string;
  language: string | null;
  description: string;
  license: string;
  citation: string;
  year: number;
  native_sample_rate: number | null;
  doi: string | null;
  url: string | null;
  download: { url: string; archive: string; size_mb: number | null } | null;
  downloadable: boolean;
  tags: string[];
}

export interface AttackPresets {
  sample_rate: number;
  suites: Record<string, string[]>;
  sweeps: Record<string, SweepPreset>;
  pipelines: Record<string, string[]>;
}

// --------------------------------------------------------------- experiments

export interface ParameterSweep {
  target: string;
  parameter: string;
  values: (number | string)[];
}

export interface ExperimentConfig {
  subset_seed?: number | null;
  audio_category?: string | null;
  audio_source?: string | null;
  channel_policy?: "mono" | "reject";
  payload?: { kind: "random" | "text" | "binary" | "bits"; value?: string | null };
  payload_rates_bps?: number[];
  dataset_id?: string | null;
  dataset_path?: string | null;
  file_limit?: number | null;
  selected_files?: string[];
  selected_file_sha256?: Record<string, string>;
  methods: string[];
  metrics: string[];
  attacks: string[];
  attack_preset?: string | null;
  attack_sweep?: ParameterSweep | null;
  method_sweep?: ParameterSweep | null;
  payload_lengths: number[];
  repetitions: number;
  random_seed?: number | null;
  max_workers?: number;
  advanced_options?: Record<string, unknown>;
  notes?: string | null;
  description?: string | null;
}

export interface ExperimentInput {
  name: string;
  experiment_type: ExperimentType;
  research_question?: string | null;
  hypothesis?: string | null;
  description?: string | null;
  tags: string[];
  config: ExperimentConfig;
}

export interface RunBrief {
  id: string;
  number: number;
  status: RunStatus;
  experiment_version: number;
  total_rows: number;
  completed_rows: number;
  ok_rows: number;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface Experiment extends Omit<ExperimentInput, "config"> {
  id: string;
  config: ExperimentConfig;
  version: number;
  archived: boolean;
  created_at: string;
  updated_at: string;
  run_count: number;
  latest_run: RunBrief | null;
  problems: string[];
}

export interface RunInfo extends RunBrief {
  experiment_id: string;
  experiment_name: string;
  experiment_type: ExperimentType;
  config: ExperimentConfig & { random_seed?: number | null };
  plan: Partial<ExperimentPlan>;
}

export interface PlanWarning {
  code: string;
  message: string;
}

export interface ExperimentPlan {
  experiment_type: ExperimentType;
  file_count: number;
  method_count: number;
  payload_length_count: number;
  repetitions: number;
  attack_variant_count: number;
  metric_count: number;
  encode_operations: number;
  estimated_result_rows: number;
  estimated_metric_calculations: number;
  warnings: PlanWarning[];
}

// ------------------------------------------------------------------ results

export interface ResultRow {
  channels?: number;
  source_channels?: number | null;
  bit_depth?: number | null;
  audio_category?: string | null;
  audio_source?: string | null;
  audio_sha256?: string | null;
  file_id?: string | null;
  payload_kind?: string | null;
  payload_seed?: number | null;
  payload_sha256?: string | null;
  payload_bytes?: number | null;
  requested_payload_rate_bps?: number | null;
  payload_bits_per_sample?: number | null;
  exact_goodput_bps?: number | null;
  encode_rtf?: number | null;
  decode_rtf?: number | null;
  experiment_id: string;
  file_name: string;
  sample_rate: number | null;
  duration_seconds: number | null;
  method: string;
  method_type: string | null;
  method_parameters: Record<string, unknown>;
  payload_length: number;
  payload_rate_bps: number | null;
  repetition: number;
  message_bits: string | null;
  decoded_bits: string | null;
  attack: string | null;
  attack_parameters: Record<string, unknown>;
  metrics: Record<string, number | null>;
  metric_errors: Record<string, string>;
  attack_metrics: Record<string, number | null>;
  bit_accuracy: number | null;
  ber: number | null;
  decode_success: boolean;
  encode_time_seconds: number | null;
  decode_time_seconds: number | null;
  attack_time_seconds: number | null;
  status: "ok" | "error";
  failure_kind: string | null;
  error: string | null;
}

export interface RowsPage {
  total: number;
  offset: number;
  limit: number;
  rows: { id: number; row: ResultRow }[];
}

export interface Estimate {
  estimate: number | null;
  ci95_low: number | null;
  ci95_high: number | null;
  n: number;
  clusters: number;
}

export interface GroupStats {
  rows: number;
  error_rows: number;
  completion_rate: number | null;
  failures: Record<string, number>;
  decode_success_rate: number | null;
  avg_bit_accuracy: number | null;
  avg_ber: number | null;
  ber_imputed: Estimate;
  avg_ber_imputed: number | null;
  perfect_extraction_rate: number | null;
  usable_extraction_rate: number | null;
  avg_metrics: Record<string, number>;
  avg_encode_time_seconds: number | null;
  avg_decode_time_seconds: number | null;
}

export interface PairwiseTest {
  a: string;
  b: string;
  better: string | null;
  median_difference: number;
  rank_biserial: number;
  p_value: number;
  p_holm: number;
  significant: boolean;
}

export interface Comparison {
  available: boolean;
  reason?: string;
  higher_is_better?: boolean;
  blocks?: number;
  treatments?: string[];
  mean_ranks?: Record<string, number>;
  omnibus?: { test: string; statistic: number | null; p_value: number; significant: boolean };
  pairwise?: PairwiseTest[];
  critical_difference?: number;
  alpha?: number;
}

export interface ParetoResult {
  objectives: string[];
  excluded_objectives: string[];
  front: string[];
  dominated_by: Record<string, string[]>;
}

export type Summary = Record<string, unknown> & {
  overall?: GroupStats;
  statistics?: Record<string, Comparison | Record<string, Comparison>>;
  pareto?: ParetoResult;
  evaluation?: Evaluation;
};

// -------------------------------------------- evaluation block (backend: scenarios/evaluation.py)

export interface Distribution {
  count: number;
  clusters: number;
  mean: number | null;
  median: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  q1: number | null;
  q3: number | null;
  iqr: number | null;
  ci95_low: number | null;
  ci95_high: number | null;
}

export interface BoxSummary {
  n: number;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  mean: number;
  whisker_low: number;
  whisker_high: number;
  outliers: number;
}

export type RecoveryKey = "exact" | "le_1pct" | "le_5pct";

export interface BerBlock {
  trials: number;
  files: number;
  completion_rate: number;
  failures: Record<string, number>;
  ber: Distribution;
  box: BoxSummary | null;
  ber_imputed: Estimate;
  recovery: Record<RecoveryKey, Estimate>;
}

export interface EvaluationAttack {
  attack: string;
  family: string | null;
  level: string;
  rank: number | null;
  parameters: Record<string, unknown>;
  per_method: (BerBlock & { method: string; delta_ber: Estimate & { median: number | null } })[];
  most_resistant: string[];
}

export interface Correlation {
  x: string;
  y: string;
  method: string;
  group: string | null;
  available: boolean;
  reason?: string;
  rho?: number;
  rho_ci95_low?: number | null;
  rho_ci95_high?: number | null;
  p_value?: number;
  p_holm?: number;
  significant?: boolean;
  n_observations?: number;
  n_points?: number;
  n_files?: number;
  levels?: number;
}

export interface Fact {
  kind: string;
  text: string;
  [key: string]: unknown;
}

export interface Evaluation {
  version: number;
  settings: Record<string, unknown> & { confidence: number; bootstrap_resamples: number; bootstrap_seed: number; alpha: number; recovery_thresholds: Record<RecoveryKey, number>; stable_ber: number };
  resolution: { min_payload_bits: number | null; ber_step: number | null; le_1pct_equals_exact: boolean };
  methods: { method: string; clean: BerBlock | null; attacked: BerBlock | null; between_file_sd: number | null }[];
  attacks: EvaluationAttack[];
  most_destructive: Record<string, { attack: string; delta_ber: Estimate }[]>;
  severity: {
    family: string;
    parameter: string | null;
    levels: string[];
    curves: { method: string; points: { level: string; rank: number; attack: string | null; ber: Estimate; delta_ber: Estimate | null }[]; trend: Correlation }[];
  }[];
  quality: {
    metrics: { name: string; higher_is_better: boolean | null; per_method: (Distribution & { method: string })[] }[];
    attack_damage: { name: string; higher_is_better: boolean | null; per_attack: { attack: string; per_method: (Distribution & { method: string })[] }[] }[];
    psnr_available: boolean;
  };
  payload: {
    levels: number[];
    available: boolean;
    metrics: string[];
    per_method: {
      method: string;
      points: { payload_bits: number; payload_bps: Distribution; ber: Estimate; ber_imputed: Estimate; completion_rate: number; quality: Record<string, Estimate> }[];
    }[];
  };
  runtime: {
    max_workers: number | null;
    comparable: boolean;
    per_method: { method: string; encode_seconds: Distribution; decode_seconds: Distribution; total_seconds: Distribution; real_time_factor: Distribution }[];
    comparison?: Comparison;
  };
  correlations: Correlation[];
  facts: Fact[];
}

// ----------------------------------------------------------------- datasets

export type DatasetStatus = "pending" | "downloading" | "preparing" | "ready" | "failed";

export interface Dataset {
  id: string;
  name: string;
  kind: "corpus" | "upload" | "local" | "synthetic";
  corpus_id: string | null;
  status: DatasetStatus;
  progress: number;
  stage: string | null;
  path: string | null;
  domain: string | null;
  language: string | null;
  license: string | null;
  citation: string | null;
  description: string | null;
  file_count: number;
  total_duration_seconds: number | null;
  sample_rate: number | null;
  rule: Record<string, unknown>;
  error: string | null;
  created_at: string;
}

export interface DatasetDetail extends Dataset {
  manifest: {
    files?: { file: string; source?: string; speaker?: string | null; sample_rate: number; duration_seconds: number; sha256?: string; channels?: number; bit_depth?: number | null; subtype?: string; category?: string }[];
    speakers?: number;
    candidates?: number;
    source_sha256?: string | null;
    [key: string]: unknown;
  };
}

export interface SubsetRule {
  max_files: number;
  target_sample_rate: number | null;
  min_duration_seconds: number;
  max_duration_seconds: number | null;
  excerpt_seconds: number | null;
  excerpt_offset_seconds: number;
  seed: number;
}

// ------------------------------------------------------------------ system

export interface SystemStatus {
  status: "ok" | "degraded";
  version: string;
  database: { ok: boolean; version?: string; error?: string };
  data_dir: string;
}

export interface PlatformStats {
  experiments: number;
  runs: Record<string, number>;
  datasets: number;
  methods: number;
  metrics: number;
  attacks: number;
  designs: number;
}

export interface SpectrogramData {
  times: number[];
  frequencies: number[];
  db: number[][];
  floor_db: number;
}

export interface Inspection {
  row: ResultRow;
  sample_rate: number;
  duration_seconds: number;
  reproduced: boolean;
  decoded_bits: string;
  attack_metadata: Record<string, unknown>;
  signals: Record<string, { spectrogram: SpectrogramData; envelope: { min: number[]; max: number[] } }>;
}
