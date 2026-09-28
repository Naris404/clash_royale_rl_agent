import { useEffect, useMemo, useState } from "react";
import type { Strings } from "../i18n";

interface MetricSeries {
  tag: string;
  steps: number[];
  values: number[];
  latest: number;
  minimum: number;
  maximum: number;
  points: number;
}

interface TrainingMetrics {
  training: boolean;
  latest_run: string | null;
  latest_step: number;
  updated_at?: number;
  series: MetricSeries[];
  runs: Array<{ name: string; updated_at: number }>;
  models: Array<{ name: string; path: string; size_bytes: number; updated_at: number }>;
}

interface Props {
  strings: Strings;
  onMenu: () => void;
  onPlay: () => void;
}

const FEATURED = [
  "rollout/win_rate_vs_logic",
  "eval/mean_reward",
  "train/explained_variance",
  "train/policy_gradient_loss",
  "train/value_loss",
  "train/entropy_loss",
  "train/approx_kl",
  "time/fps",
];

function formatValue(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (Math.abs(value) >= 10) return value.toFixed(1);
  if (Math.abs(value) >= 1) return value.toFixed(3);
  return value.toPrecision(3);
}

function metricName(tag: string): string {
  return tag
    .replace("rollout/", "")
    .replace("train/", "")
    .replace("eval/", "")
    .replace("time/", "")
    .replaceAll("_", " ");
}

function MetricChart({ metric, compact = false }: { metric: MetricSeries; compact?: boolean }) {
  const width = 600;
  const height = compact ? 90 : 180;
  const padding = compact ? 5 : 18;
  const min = Math.min(...metric.values);
  const max = Math.max(...metric.values);
  const span = Math.max(max - min, 1e-9);
  const points = metric.values
    .map((value, index) => {
      const x = padding + (index / Math.max(1, metric.values.length - 1)) * (width - padding * 2);
      const y = padding + ((max - value) / span) * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div className={`metric-chart ${compact ? "metric-chart-compact" : ""}`}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={metric.tag}>
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} />
        <polyline points={points} />
      </svg>
      {!compact && (
        <div className="chart-range">
          <span>{formatValue(min)}</span>
          <span>{metric.steps.at(-1)?.toLocaleString() ?? "0"} steps</span>
          <span>{formatValue(max)}</span>
        </div>
      )}
    </div>
  );
}

export function Dashboard({ strings, onMenu, onPlay }: Props) {
  const [data, setData] = useState<TrainingMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const response = await fetch("/api/training/metrics", { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const next = (await response.json()) as TrainingMetrics;
        if (active) {
          setData(next);
          setError(null);
        }
      } catch (reason) {
        if (active) setError(reason instanceof Error ? reason.message : String(reason));
      }
    };
    void load();
    const timer = window.setInterval(load, 5000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const ordered = useMemo(() => {
    if (!data) return [];
    return [...data.series].sort((a, b) => {
      const ai = FEATURED.indexOf(a.tag);
      const bi = FEATURED.indexOf(b.tag);
      return (ai < 0 ? 999 : ai) - (bi < 0 ? 999 : bi) || a.tag.localeCompare(b.tag);
    });
  }, [data]);
  const featured = ordered.filter((metric) => FEATURED.includes(metric.tag));

  return (
    <main className="dashboard-page">
      <header className="dashboard-header">
        <div>
          <span className="hero-eyebrow">MaskablePPO</span>
          <h1>{strings.dashboard}</h1>
          <p>{strings.dashboard_desc}</p>
        </div>
        <nav className="dashboard-nav">
          <button className="btn" onClick={onPlay}>▶ {strings.play}</button>
          <button className="btn btn-primary" onClick={onMenu}>⌂ {strings.menu}</button>
        </nav>
      </header>

      {error && <div className="dashboard-error">{error}</div>}
      {!data && !error && <div className="dashboard-loading">{strings.coach_waiting}</div>}
      {data && (
        <>
          <section className="dashboard-summary">
            <article>
              <span className={`live-dot ${data.training ? "live-dot-active" : ""}`} />
              <small>{data.training ? strings.training_active : strings.training_idle}</small>
              <strong>{data.latest_step.toLocaleString()}</strong>
              <label>{strings.latest_step}</label>
            </article>
            <article><small>{strings.latest_run}</small><strong>{data.latest_run ?? "—"}</strong></article>
            <article><small>{strings.metrics_count}</small><strong>{data.series.length}</strong></article>
            <article><small>{strings.model_files}</small><strong>{data.models.length}</strong></article>
          </section>

          {ordered.length === 0 ? (
            <div className="dashboard-empty">{strings.no_metrics}</div>
          ) : (
            <>
              <section className="dashboard-section">
                <div className="dashboard-section-title">
                  <h2>{strings.performance}</h2>
                  {data.updated_at && <span>{strings.updated}: {new Date(data.updated_at * 1000).toLocaleTimeString()}</span>}
                </div>
                <div className="featured-metrics">
                  {featured.map((metric) => (
                    <article className="featured-metric" key={metric.tag}>
                      <div><span>{metricName(metric.tag)}</span><strong>{formatValue(metric.latest)}</strong></div>
                      <MetricChart metric={metric} />
                    </article>
                  ))}
                </div>
              </section>

              <section className="dashboard-section">
                <div className="dashboard-section-title"><h2>{strings.all_metrics}</h2></div>
                <div className="all-metrics-grid">
                  {ordered.map((metric) => (
                    <article className="metric-tile" key={metric.tag}>
                      <div>
                        <span>{metricName(metric.tag)}</span>
                        <strong>{formatValue(metric.latest)}</strong>
                      </div>
                      <MetricChart metric={metric} compact />
                      <small>{metric.points.toLocaleString()} points · {metric.tag}</small>
                    </article>
                  ))}
                </div>
              </section>
            </>
          )}
        </>
      )}
    </main>
  );
}
