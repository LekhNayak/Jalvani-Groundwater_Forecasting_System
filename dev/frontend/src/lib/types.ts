export type { Lang } from "./langs";
import type { Lang } from "./langs";
export type Role = "farmer" | "government";
export type Horizon = "t+7" | "t+14" | "t+30";
export const HORIZONS: Horizon[] = ["t+7", "t+14", "t+30"];

export interface Meta {
  built: string;
  model: { source: string; run: string; pipeline: string; operational_forecast: boolean; last_issue_date: string | null;
           backtest_period: [string, string] };
  freshness: { groundwater_last_reading: string | null; rain_as_of: string; rain_provisional_from: string | null;
               soil_as_of: string; extraction_year: number };
  counts: { districts: number; with_forecast: number; insufficient_data: number; no_data: number; wells_used: number };
  languages: Lang[];
  features: { chat: "groq" | "guide"; speech: boolean; speech_languages?: string[] };
}

export type Status = "forecast" | "insufficient_data" | "no_data";

export interface DistrictLite {
  district: string; slug: string; status: Status; tier: string | null; wells: number | null;
  layers: { vs_usual_m: number | null; forecast_change_t30_m: number | null; stage_pct: number | null;
            category: string | null; wells: number | null; reliability_t30: Reliability | null };
}

export type Reliability = "good" | "fair" | "limited";
export interface HorizonForecast {
  target_date: string; q10: number; q50: number; q90: number; change_m: number; change_q10_m: number;
  change_q90_m: number; interval: string;
  likely_margin_m?: number | null; likely_lo?: number | null; likely_hi?: number | null;
}
export interface RainWindow { from: string; to: string; mm: number | null; usual_mm: number | null; pct_of_usual: number | null;
                              provisional_days: number; usual_years: number[] }
export interface Tip { key: string; level: "warn" | "good" | "info"; params: Record<string, number | string> }
export interface Flag { key: string; weight: number; action: string; params: Record<string, unknown> }

export interface District {
  district: string; slug: string; status: Status; status_reason?: string;
  monitoring: { wells: number; completeness_pct: number | null; tier: string | null; single_well: boolean };
  groundwater: { last_reading: string; level_m: number; depth_m: number; change_30d_m: number | null; change_1y_m: number | null;
                 usual_level_m: number | null; vs_usual_m: number | null; vs_usual: "about" | "deeper" | "shallower" | null;
                 usual_years: number[] } | null;
  reliability: Partial<Record<Horizon, { mase: number; vs_sn: number; coverage_pct: number; label: Reliability }>> | null;
  forecast: { issue_date: string; level_now_m: number; horizons: Partial<Record<Horizon, HorizonForecast>> } | null;
  forecast_unavailable?: { code: "not_produced" | "too_few_readings" | "no_window"; share_pct?: number } | null;
  rain: { as_of: string; provisional_from: string | null; last_7d: RainWindow; last_30d: RainWindow; season: RainWindow } | null;
  soil: { as_of: string; sm_l1: { pct: number; usual_pct: number; vs_usual: string | null }; sm_l2: { pct: number; usual_pct: number; vs_usual: string | null } } | null;
  extraction: { latest_year: number; stage_pct: number; category: string; recharge_bcm: number; extraction_bcm: number;
                extractable_bcm: number; history: { year: number; stage_pct: number; category: string }[];
                use_split_pct?: { irrigation: number; industrial: number; domestic: number } | null } | null;
  advice: Tip[]; flags: Flag[]; direction: Partial<Record<Horizon, string | null>>;
}

export interface Table { cols: string[]; rows: (string | number | null)[][] }
export interface History {
  district: string; observed_weekly: [string, number][]; rain_monthly: [string, number][];
  backtest: Partial<Record<Horizon, Table>>; latest_path: Table | null;
}

export interface PriorityRow {
  district: string; slug: string; status: Status; score: number; actions: string[]; flags: Flag[];
  stage_pct: number | null; category: string | null; depth_m: number | null; vs_usual_m: number | null;
  forecast_change_t30_m: number | null; wells: number | null; reliability_t30: Reliability | null; season_rain_pct: number | null;
}

export interface Overview {
  counts: Meta["counts"]; freshness: Meta["freshness"]; model: Meta["model"];
  extraction_categories: Record<string, number>; flag_counts: Record<string, number>;
  weights: Record<string, { weight: number; action: string }>;
}

export type ChatAction = { type: "navigate"; route: string } | { type: "highlight"; element: string } |
                         { type: "start_tour"; tour: "farmer" | "government" };
