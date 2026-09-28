export interface LiteratureEntry {
  id: string; title: string; authors: string[]; year: number; venue: string;
  family: string; purpose: string; url: string; doi: string | null; arxiv: string | null;
  datasets: string[]; metrics: string[]; attacks: string[]; payload: string;
  results: { metric: string; value: number; unit: string; condition: string; locator: string }[];
  limitations: string; source_url: string; implemented_methods: string[]; verified_on: string;
}

export interface ResearchStats {
  mean: number | null; median: number | null; count: number; clusters: number;
  ci95_low: number | null; ci95_high: number | null;
}

export interface ResearchComparison {
  reference: string; attack: string | null; group: string; rows: number;
  facets: Record<string, string[]>; timing_reliable: boolean; interpretation: string;
  groups: { label: string; rows: number; completed: number; exact: number; measures: Record<string, ResearchStats> }[];
}

export const researchLabels: Record<string, string> = {
  method: "Method", audio_category: "Audio category", payload_kind: "Payload kind",
  audio_source: "Audio source", requested_payload_rate_bps: "Requested payload (bits/s)",
  payload_length: "Exact payload (bits)", sample_rate: "Sample rate (Hz)",
  source_channels: "Source channels", bit_depth: "PCM bit depth", ber: "BER (completed trials)",
  payload_rate_bps: "Offered payload (bits/s)", payload_bits_per_sample: "Payload bits/sample",
  exact_goodput_bps: "Exact-message goodput (bits/s)", encode_rtf: "Encode real-time factor",
  decode_rtf: "Decode real-time factor",
};
