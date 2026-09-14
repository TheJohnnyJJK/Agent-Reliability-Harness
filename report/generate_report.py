"""Builds report/benchmark_report.html from results/summary.json.

Same shape as lead-router's dashboard/generate_dashboard.py: one static
HTML file, no templating engine, every number computed from stored data
rather than hand-typed. Run harness/metrics.py first to produce
results/summary.json - this script only reads it.

Colors follow the dataviz skill's validated default palette, not picked by
eye: the two bars in the pass@1/pass^5 chart use categorical slots 1 and 2
(blue, orange) since they're two *metrics*, not framework identities; the
taxonomy chart and every pass/fail dot use the fixed status palette
(good/warning/serious/critical), since "did this trial succeed, and how
did it fail" is state, not series identity - reusing status colors there
instead of inventing new ones keeps that meaning consistent everywhere it
appears in the report.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import datetime  # noqa: E402

from harness.metrics import RESULTS_PATH, SUMMARY_PATH, write_summary  # noqa: E402
from tasks.schema import load_golden_tasks  # noqa: E402

OUT_PATH = ROOT / "report" / "benchmark_report.html"

FRAMEWORK_ORDER = ["langgraph", "crewai", "ag2"]
FRAMEWORK_LABELS = {"langgraph": "LangGraph", "crewai": "CrewAI", "ag2": "AG2"}
TRIALS_PER_TASK = 5

TAXONOMY_ORDER = [
    "success", "silent_wrong_answer", "hallucinated_parameter", "tool_call_error", "infinite_loop",
]
TAXONOMY_LABELS = {
    "success": "Success",
    "silent_wrong_answer": "Silent wrong answer",
    "hallucinated_parameter": "Hallucinated parameter",
    "tool_call_error": "Tool call error",
    "infinite_loop": "Infinite loop",
}
# Status palette from the dataviz skill's reference instance - fixed roles,
# never reused for framework/series identity.
TAXONOMY_COLORS = {
    "success": "#0ca30c",              # status: good
    "silent_wrong_answer": "#4a3aa7",  # categorical slot 7 (violet) - the one outcome that isn't a status role
    "hallucinated_parameter": "#fab219",  # status: warning
    "tool_call_error": "#ec835a",      # status: serious
    "infinite_loop": "#d03b3b",        # status: critical
}
PASS_COLOR = "#0ca30c"   # status: good
FAIL_COLOR = "#d03b3b"   # status: critical
METRIC_COLORS = {"pass_at_1": "#2a78d6", "pass_at_5": "#eb6834"}  # categorical slots 1, 2


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _bar_chart(summary: dict) -> str:
    """Grouped bar chart: framework on the x-axis, pass@1/pass^5 as the two
    series (one axis, 0-100%) - see module docstring for the color choice."""
    frameworks = [f for f in FRAMEWORK_ORDER if f in summary["frameworks"]]
    if not frameworks:
        return ""

    width, height, left_pad = 720, 160, 40
    group_w = width / len(frameworks)
    bar_w, gap = 28, 6
    plot_h = height - 24

    gridlines = "".join(
        f'<line x1="0" y1="{plot_h - plot_h * p:.1f}" x2="{width}" y2="{plot_h - plot_h * p:.1f}" '
        f'stroke="var(--grid)" stroke-width="1"/>'
        f'<text x="-8" y="{plot_h - plot_h * p + 3:.1f}" text-anchor="end" class="axis-label">{int(p * 100)}%</text>'
        for p in (0.0, 0.5, 1.0)
    )

    bars = []
    labels = []
    for i, fw in enumerate(frameworks):
        stats = summary["frameworks"][fw]
        gx = i * group_w + group_w / 2
        for j, metric in enumerate(("pass_at_1", "pass_at_5")):
            value = stats[metric]
            bx = gx - bar_w - gap / 2 + j * (bar_w + gap)
            bh = plot_h * value
            by = plot_h - bh
            color = METRIC_COLORS[metric]
            metric_label = "pass@1" if metric == "pass_at_1" else "pass^5"
            bars.append(
                f'<rect class="bar" x="{bx:.2f}" y="{by:.2f}" width="{bar_w}" height="{bh:.2f}" '
                f'rx="3" ry="3" fill="{color}" '
                f'data-fw="{_esc(FRAMEWORK_LABELS[fw])}" data-metric="{metric_label}" '
                f'data-value="{value:.0%}"/>'
            )
        labels.append(
            f'<text x="{gx:.1f}" y="{plot_h + 18}" text-anchor="middle" class="axis-label">'
            f"{_esc(FRAMEWORK_LABELS[fw])}</text>"
        )

    return f'''<svg viewBox="0 0 {width + left_pad} {height}" class="chart" role="img"
  aria-label="pass at 1 and pass hat 5 by framework">
  <g transform="translate({left_pad},0)">
    {gridlines}
    {"".join(bars)}
    {"".join(labels)}
  </g>
</svg>
<div class="legend">
  <span><span class="sw" style="background:{METRIC_COLORS["pass_at_1"]}"></span>pass@1 (trial 1 succeeded)</span>
  <span><span class="sw" style="background:{METRIC_COLORS["pass_at_5"]}"></span>pass^5 (all 5 trials succeeded)</span>
</div>'''


def _taxonomy_chart(summary: dict) -> str:
    """One stacked column per framework, normalized to 100% of that
    framework's recorded trials, segmented by outcome category."""
    frameworks = [f for f in FRAMEWORK_ORDER if f in summary["frameworks"]]
    if not frameworks:
        return ""

    col_w, gap_between, height = 90, 40, 220
    width = len(frameworks) * col_w + (len(frameworks) - 1) * gap_between

    cols = []
    labels = []
    for i, fw in enumerate(frameworks):
        stats = summary["frameworks"][fw]
        total = stats["total_trials"] or 1
        cx = i * (col_w + gap_between)
        cy = 0.0
        segs = []
        for cat in TAXONOMY_ORDER:
            count = stats["taxonomy_counts"].get(cat, 0)
            if count == 0:
                continue
            seg_h = height * (count / total)
            segs.append(
                f'<rect class="tax-seg" x="{cx:.1f}" y="{cy:.1f}" width="{col_w}" '
                f'height="{max(seg_h - 2, 0):.1f}" rx="2" ry="2" fill="{TAXONOMY_COLORS[cat]}" '
                f'data-fw="{_esc(FRAMEWORK_LABELS[fw])}" data-cat="{_esc(TAXONOMY_LABELS[cat])}" '
                f'data-count="{count}" data-total="{total}"/>'
            )
            cy += seg_h
        cols.append("".join(segs))
        labels.append(
            f'<text x="{cx + col_w / 2:.1f}" y="{height + 18}" text-anchor="middle" class="axis-label">'
            f"{_esc(FRAMEWORK_LABELS[fw])}</text>"
        )

    legend_items = "".join(
        f'<span><span class="sw" style="background:{TAXONOMY_COLORS[cat]}"></span>{_esc(TAXONOMY_LABELS[cat])}</span>'
        for cat in TAXONOMY_ORDER
    )

    return f'''<svg viewBox="0 0 {width} {height + 30}" class="chart" role="img"
  aria-label="failure taxonomy distribution by framework">
  {"".join(cols)}
  {"".join(labels)}
</svg>
<div class="legend">{legend_items}</div>'''


def _dot_pattern(pattern: list[bool]) -> str:
    dots = "".join(
        f'<span class="dot" style="background:{PASS_COLOR if p else FAIL_COLOR}" '
        f'title="trial {i + 1}: {"pass" if p else "fail"}"></span>'
        for i, p in enumerate(pattern)
    )
    return f'<span class="dots">{dots}</span>'


def _task_table(summary: dict, notes_by_id: dict[str, str]) -> str:
    rows = sorted(summary["tasks"], key=lambda t: (t["task_id"], t["framework"]))
    body = []
    for row in rows:
        note = notes_by_id.get(row["task_id"], "")
        body.append(
            "<tr>"
            f'<td>{_esc(row["task_id"])}</td>'
            f'<td>{_esc(row["category"])}</td>'
            f'<td>{_esc(FRAMEWORK_LABELS.get(row["framework"], row["framework"]))}</td>'
            f'<td>{_dot_pattern(row["trial_pattern"])}</td>'
            f'<td class="num">{row["pass_at_1"]:.0%}</td>'
            f'<td class="num">{row["pass_at_5"]:.0%}</td>'
            f'<td class="notes">{_esc(note)}</td>'
            "</tr>"
        )
    return "".join(body)


def build(summary: dict) -> str:
    all_tasks = load_golden_tasks()
    notes_by_id = {t.task_id: t.notes for t in all_tasks}
    generated_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    theoretical_max = len(all_tasks) * len(FRAMEWORK_ORDER) * TRIALS_PER_TASK
    recorded = summary["total_trials_recorded"]

    if recorded == 0:
        body = '''
  <div class="card">
    <h2>No trials recorded yet</h2>
    <p class="muted">results/trials.jsonl is empty. Run the benchmark first:</p>
    <pre>python -m harness.runner
python -m harness.metrics
python -m report.generate_report</pre>
    <p class="muted">A smoke test first is cheap: <code>python -m harness.runner --frameworks langgraph --tasks 2 --trials 1</code></p>
  </div>'''
    else:
        hero_tiles = "".join(
            f'''<div class="tile">
      <div class="label">{_esc(FRAMEWORK_LABELS[fw])}</div>
      <div class="value">{summary["frameworks"][fw]["pass_at_1"]:.0%}</div>
      <div class="sub">pass@1 &middot; pass^5 {summary["frameworks"][fw]["pass_at_5"]:.0%}
        &middot; {summary["frameworks"][fw]["total_trials"]} trials</div>
    </div>'''
            for fw in FRAMEWORK_ORDER if fw in summary["frameworks"]
        )
        body = f'''
  <div class="hero-row">{hero_tiles}</div>

  <div class="card">
    <h2>Completion rate by framework</h2>
    <p class="muted">pass@1 is what a single run per task would show you. pass^5 - every one of 5
      repeated trials passing - is the number that actually means "reliable."</p>
    {_bar_chart(summary)}
  </div>

  <div class="card">
    <h2>How each framework fails, when it fails</h2>
    <p class="muted">Every recorded trial, classified once by harness/taxonomy.py - never by hand.</p>
    {_taxonomy_chart(summary)}
  </div>

  <div class="card">
    <h2>Every task, every trial</h2>
    <p class="muted">One row per task per framework. Each dot is one of 5 trials, in order.</p>
    <table class="eval-table">
      <thead><tr><th>Task</th><th>Category</th><th>Framework</th><th>Trials</th>
        <th>pass@1</th><th>pass^5</th><th>Why this task exists</th></tr></thead>
      <tbody>{_task_table(summary, notes_by_id)}</tbody>
    </table>
  </div>'''

    honesty_note = (
        f"{recorded} of a theoretical {theoretical_max} trials recorded "
        f"({len(all_tasks)} tasks &times; {len(FRAMEWORK_ORDER)} frameworks &times; "
        f"{TRIALS_PER_TASK} trials)."
        if recorded < theoretical_max
        else f"All {theoretical_max} trials recorded - the full benchmark ran to completion."
    )

    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Reliability Harness - Benchmark Report</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  :root{{
    --paper:#eef1ee; --ink:#1b211f; --ink-soft:#4b544f; --ink-faint:#7c847e;
    --line:#c9d1cb; --line-soft:#dde3de; --accent:#1c6e63; --accent-ink:#0f3f38;
    --tile:#e2e8e3; --grid:#d6ddd8;
  }}
  @media (prefers-color-scheme: dark){{
    :root{{
      --paper:#12181a; --ink:#e9ede9; --ink-soft:#a3ada6; --ink-faint:#6f7973;
      --line:#2b3533; --line-soft:#232d2b; --accent:#57bfae; --accent-ink:#8fd9cc;
      --tile:#1a2220; --grid:#26312e;
    }}
  }}
  *{{box-sizing:border-box}}
  body{{background:var(--paper);color:var(--ink);font-family:"Public Sans",-apple-system,sans-serif;
       line-height:1.55;margin:0;padding:48px 24px 80px}}
  .wrap{{max-width:860px;margin:0 auto}}
  h1{{font-family:"Source Serif 4",Georgia,serif;font-size:28px;margin:0 0 4px}}
  h2{{font-family:"Source Serif 4",Georgia,serif;font-size:18px;margin:0 0 10px}}
  .eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:12px;letter-spacing:.08em;
           text-transform:uppercase;color:var(--ink-faint)}}
  .muted{{color:var(--ink-soft);font-size:14px;margin:0 0 16px}}
  .card{{border:1px solid var(--line);background:var(--tile);padding:20px 24px;margin:20px 0;border-radius:2px}}

  .hero-row{{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);
            border:1px solid var(--line);margin:24px 0}}
  .hero-row .tile{{background:var(--tile);padding:18px 16px}}
  .tile .label{{font-family:"IBM Plex Mono",monospace;font-size:11px;text-transform:uppercase;
               letter-spacing:.06em;color:var(--ink-faint);margin-bottom:8px}}
  .tile .value{{font-family:"Public Sans",sans-serif;font-weight:700;color:var(--accent-ink);font-size:34px}}
  .tile .sub{{font-size:12px;color:var(--ink-soft);margin-top:6px}}

  .chart{{width:100%;height:auto;overflow:visible}}
  .axis-label{{font-family:"IBM Plex Mono",monospace;font-size:10px;fill:var(--ink-faint)}}
  .bar, .tax-seg{{cursor:default}}

  .legend{{display:flex;gap:18px;flex-wrap:wrap;margin-top:12px;font-size:13px;color:var(--ink-soft)}}
  .legend span.sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}}

  .dots{{display:inline-flex;gap:3px}}
  .dot{{width:9px;height:9px;border-radius:50%;display:inline-block}}

  .eval-table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}}
  .eval-table th, .eval-table td{{padding:8px 10px;border-bottom:1px solid var(--line-soft);text-align:left;vertical-align:middle}}
  .eval-table th{{font-family:"IBM Plex Mono",monospace;font-size:11px;text-transform:uppercase;color:var(--ink-faint)}}
  .eval-table td.num{{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}}
  .eval-table td.notes{{color:var(--ink-soft);max-width:280px}}

  pre{{background:var(--paper);border:1px solid var(--line);padding:12px 14px;border-radius:2px;
       font-family:"IBM Plex Mono",monospace;font-size:13px;overflow-x:auto}}
  code{{font-family:"IBM Plex Mono",monospace;font-size:13px}}

  .tooltip{{position:fixed;pointer-events:none;background:var(--ink);color:var(--paper);
           font-family:"IBM Plex Mono",monospace;font-size:12px;padding:6px 10px;border-radius:3px;
           opacity:0;transform:translate(-50%,-120%);transition:opacity .1s;z-index:10;white-space:nowrap}}
  .tooltip.show{{opacity:1}}
  .tooltip b{{color:#fff}}

  footer{{margin-top:36px;color:var(--ink-faint);font-size:12px;border-top:1px solid var(--line);padding-top:16px}}
</style>
</head>
<body>
<div class="wrap">

  <span class="eyebrow">Agent reliability harness - benchmark report</span>
  <h1>LangGraph vs. CrewAI vs. AG2</h1>
  <p class="muted">Same task, same tools, same model, same golden tasks - only the orchestration
    framework changes. Generated {generated_at}.</p>
{body}

  <footer>
    Generated by report/generate_report.py from results/summary.json (built by harness/metrics.py
    from results/trials.jsonl). {honesty_note} Every figure above is computed from stored trial
    records - none are hand-typed.
  </footer>
</div>

<div class="tooltip" id="tooltip"></div>
<script>
  const tip = document.getElementById('tooltip');
  const tipBold = document.createElement('b');
  const tipRest = document.createTextNode('');
  tip.append(tipBold, tipRest);

  // Every value used below is generated server-side from fixed labels and
  // numbers, never raw model output - textContent is used on principle
  // anyway, matching lead-router/dashboard/generate_dashboard.py's tooltip.
  function showTip(x, y, bold, rest) {{
    tipBold.textContent = bold;
    tipRest.textContent = rest;
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
    tip.classList.add('show');
  }}
  function hideTip() {{ tip.classList.remove('show'); }}
  function followPointer(e) {{ tip.style.left = e.clientX + 'px'; tip.style.top = e.clientY + 'px'; }}

  document.querySelectorAll('.bar').forEach(bar => {{
    bar.addEventListener('pointerenter', e => {{
      showTip(e.clientX, e.clientY, bar.dataset.fw, `: ${{bar.dataset.metric}} = ${{bar.dataset.value}}`);
    }});
    bar.addEventListener('pointermove', followPointer);
    bar.addEventListener('pointerleave', hideTip);
  }});
  document.querySelectorAll('.tax-seg').forEach(seg => {{
    seg.addEventListener('pointerenter', e => {{
      showTip(e.clientX, e.clientY, seg.dataset.fw, `: ${{seg.dataset.cat}} - ${{seg.dataset.count}}/${{seg.dataset.total}}`);
    }});
    seg.addEventListener('pointermove', followPointer);
    seg.addEventListener('pointerleave', hideTip);
  }});
</script>
</body>
</html>
'''


def main() -> None:
    if not SUMMARY_PATH.exists() and not RESULTS_PATH.exists():
        summary = {"total_trials_recorded": 0, "frameworks": {}, "tasks": []}
    else:
        summary = write_summary()
    OUT_PATH.write_text(build(summary), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
