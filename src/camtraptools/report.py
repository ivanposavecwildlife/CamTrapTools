"""Self-contained HTML health report for a Camtrap DP package."""

from datetime import datetime
from html import escape
from pathlib import Path
from string import Template

import pandas as pd

from camtraptools import __version__, check

CHART_WIDTH = 760
LABEL_WIDTH = 120
ROW_HEIGHT = 32
AXIS_HEIGHT = 30


def x_position(time: pd.Timestamp, first: pd.Timestamp, last: pd.Timestamp) -> float:
    """Map a time to a horizontal position in the chart, in pixels."""
    share = (time - first) / (last - first)
    return LABEL_WIDTH + share * (CHART_WIDTH - LABEL_WIDTH - 10)


def chart_range(
    media: pd.DataFrame, deployments: pd.DataFrame
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the first and last plausible time to show in the chart."""
    times = pd.concat(
        [
            media["timestamp"],
            deployments["deploymentStart"],
            deployments["deploymentEnd"],
        ]
    )
    times = times[(times >= check.EARLIEST_PLAUSIBLE) & (times <= pd.Timestamp.now())]
    first, last = times.min(), times.max()
    if first == last:
        last = first + pd.Timedelta(days=1)
    return first, last


def time_axis(first: pd.Timestamp, last: pd.Timestamp, height: float) -> str:
    """Return SVG grid lines and date labels along the bottom of the chart."""
    span_days = (last - first).days
    if span_days > 120:
        frequency, label_format = "MS", "%b %Y"
    elif span_days > 21:
        frequency, label_format = "W-MON", "%d %b"
    else:
        frequency, label_format = "D", "%d %b"
    parts = []
    for tick in pd.date_range(first.normalize(), last, freq=frequency):
        if tick < first:
            continue
        x = x_position(tick, first, last)
        parts.append(
            f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="0" y2="{height}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{height + 20}" text-anchor="middle">{tick:{label_format}}</text>'
        )
    return "\n".join(parts)


def deployment_row(
    deployment_id: str,
    media: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    first: pd.Timestamp,
    last: pd.Timestamp,
    y: float,
    max_gap_days: float,
) -> str:
    """Return the SVG for one deployment: its period, gaps and one tick per image.

    Elements with a ``data-tip`` attribute show that text as a tooltip on hover.
    """
    parts = [
        f'<text class="label" x="{LABEL_WIDTH - 12}" y="{y + 4}" text-anchor="end">{escape(deployment_id)}</text>'
    ]
    if not (pd.isna(start) or pd.isna(end)):
        start, end = max(start, first), min(end, last)
        x1, x2 = x_position(start, first, last), x_position(end, first, last)
        tip = f"{deployment_id}: {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}"
        parts.append(
            f'<rect class="period" x="{x1:.1f}" y="{y - 9}" width="{max(x2 - x1, 1):.1f}" height="18" rx="4" '
            f'data-tip="{escape(tip)}"/>'
        )
        for gap_start, gap_end, _ in check.find_gaps(media, start, end, max_gap_days):
            gx1, gx2 = (
                x_position(gap_start, first, last),
                x_position(gap_end, first, last),
            )
            days = (gap_end - gap_start) / pd.Timedelta(days=1)
            tip = f"No images for {days:.1f} days ({gap_start:%Y-%m-%d} to {gap_end:%Y-%m-%d})"
            parts.append(
                f'<rect class="gap" x="{gx1:.1f}" y="{y - 9}" width="{gx2 - gx1:.1f}" height="18" rx="4" '
                f'data-tip="{escape(tip)}"/>'
            )
            parts.append(
                f'<text class="gap-label" x="{(gx1 + gx2) / 2:.1f}" y="{y + 4}" text-anchor="middle">⚠ {days:.0f} d</text>'
            )
    for row in media.dropna(subset=["timestamp"]).itertuples():
        if first <= row.timestamp <= last:
            x = x_position(row.timestamp, first, last)
            tip = f"{row.timestamp:%Y-%m-%d %H:%M}: {row.fileName}"
            parts.append(
                f'<line class="image" x1="{x:.1f}" x2="{x:.1f}" y1="{y - 6}" y2="{y + 6}"/>'
            )
            parts.append(
                f'<line class="hit" x1="{x:.1f}" x2="{x:.1f}" y1="{y - 9}" y2="{y + 9}" data-tip="{escape(tip)}"/>'
            )
    return "\n".join(parts)


def timeline_svg(
    media: pd.DataFrame, deployments: pd.DataFrame, max_gap_days: float
) -> str:
    """Return an SVG timeline with one row per deployment."""
    if media["timestamp"].isna().all():
        return "<p>No dated images to chart.</p>"
    first, last = chart_range(media, deployments)
    rows = []
    groups = list(media.groupby("deploymentID"))
    for index, (deployment_id, deployment_media) in enumerate(groups):
        y = index * ROW_HEIGHT + ROW_HEIGHT / 2
        start = deployments.loc[deployment_id, "deploymentStart"]
        end = deployments.loc[deployment_id, "deploymentEnd"]
        rows.append(
            deployment_row(
                str(deployment_id),
                deployment_media,
                start,
                end,
                first,
                last,
                y,
                max_gap_days,
            )
        )
    height = len(groups) * ROW_HEIGHT
    return (
        f'<svg class="timeline" viewBox="0 0 {CHART_WIDTH} {height + AXIS_HEIGHT}" role="img" '
        f'aria-label="Timeline of images per deployment">\n'
        f"{time_axis(first, last, height)}\n" + "\n".join(rows) + "\n</svg>"
    )


def summary_table_html(summary: pd.DataFrame, results: dict[str, list[str]]) -> str:
    """Return an HTML table with one row of key numbers and the status per deployment."""
    rows = []
    for deployment_id, issues in results.items():
        numbers = summary.loc[deployment_id]
        if issues:
            status = f'<span class="status warning"><span class="icon">⚠</span> {len(issues)} issue(s)</span>'
        else:
            status = '<span class="status good"><span class="icon">✓</span> OK</span>'
        per_day = (
            "–"
            if pd.isna(numbers["imagesPerDay"])
            else f"{numbers['imagesPerDay']:.1f}"
        )
        night = (
            "–" if pd.isna(numbers["nightShare"]) else f"{numbers['nightShare']:.0%}"
        )
        rows.append(
            f'<tr><th scope="row">{escape(str(deployment_id))}</th>'
            f'<td class="num">{numbers["days"]:.1f}</td><td class="num">{int(numbers["images"])}</td>'
            f'<td class="num">{per_day}</td><td class="num">{night}</td>'
            f'<td class="num">{numbers["longestGapDays"]:.1f}</td><td>{status}</td></tr>'
        )
    header = (
        '<tr><th scope="col">Deployment</th><th scope="col" class="num">Days</th>'
        '<th scope="col" class="num">Images</th><th scope="col" class="num">Images per day</th>'
        '<th scope="col" class="num">At night</th><th scope="col" class="num">Longest gap (days)</th>'
        '<th scope="col">Status</th></tr>'
    )
    return (
        f"<table>\n<thead>{header}</thead>\n<tbody>\n"
        + "\n".join(rows)
        + "\n</tbody>\n</table>"
    )


def issues_html(results: dict[str, list[str]]) -> str:
    """Return the list of issues, grouped by deployment."""
    sections = []
    for deployment_id, issues in results.items():
        if issues:
            items = "".join(f"<li>{escape(issue)}</li>" for issue in issues)
            sections.append(f"<h3>{escape(str(deployment_id))}</h3>\n<ul>{items}</ul>")
    if not sections:
        return '<p><span class="status good"><span class="icon">✓</span> No problems found.</span></p>'
    return "\n".join(sections)


PAGE = Template("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Camera health report: $name</title>
<style>
  :root {
    color-scheme: light;
    --surface: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
    --line: #e4e2dd; --period: #e4e2dd; --image: #2a78d6;
    --good: #0ca30c; --warning: #fab219;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --surface: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
      --line: #383835; --period: #383835; --image: #3987e5;
    }
  }
  body { margin: 0; background: var(--surface); color: var(--text-primary);
         font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 820px; margin: 0 auto; padding: 32px 16px 48px; }
  h1 { font-size: 24px; margin: 0 0 4px; }
  h2 { font-size: 18px; margin: 36px 0 12px; }
  h3 { font-size: 15px; margin: 20px 0 4px; }
  .meta, .note { color: var(--text-secondary); margin: 0; }
  .facts { display: flex; gap: 32px; margin: 24px 0 0; flex-wrap: wrap; }
  .facts div { display: flex; flex-direction: column; }
  .facts strong { font-size: 28px; font-variant-numeric: tabular-nums; }
  .facts span { color: var(--text-secondary); }
  .chart { overflow-x: auto; }
  svg.timeline { width: 100%; min-width: 600px; height: auto; display: block; }
  svg .grid { stroke: var(--line); stroke-width: 1; }
  svg .tick, svg .label { fill: var(--text-secondary); font-size: 12px; }
  svg .label { fill: var(--text-primary); font-size: 13px; }
  svg .period { fill: var(--period); }
  svg .gap { fill: var(--warning); fill-opacity: 0.35; stroke: var(--warning); stroke-width: 1; }
  svg .gap-label { fill: var(--text-primary); font-size: 11px; pointer-events: none; }
  svg .image { stroke: var(--image); stroke-width: 1.5; }
  svg .hit { stroke: transparent; stroke-width: 7; }
  svg .hit:hover { stroke: var(--image); stroke-opacity: 0.3; }
  svg .gap:hover, svg .period:hover { stroke: var(--text-secondary); }
  .tip { position: fixed; z-index: 1; pointer-events: none; padding: 4px 8px; border-radius: 4px;
         background: var(--text-primary); color: var(--surface); font-size: 12px; white-space: nowrap; }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); }
  thead th { color: var(--text-secondary); font-weight: 600; font-size: 13px; }
  .num { text-align: right; }
  .table-scroll { overflow-x: auto; }
  .status { white-space: nowrap; font-weight: 600; }
  .status.good .icon { color: var(--good); }
  .status.warning .icon { color: var(--warning); }
  ul { margin: 0; padding-left: 20px; }
  footer { margin-top: 48px; color: var(--text-secondary); font-size: 13px; }
</style>
</head>
<body>
<main>
<h1>Camera health report</h1>
<p class="meta">Package <strong>$name</strong> · generated $generated</p>
<div class="facts">
  <div><strong>$deployments</strong><span>deployments</span></div>
  <div><strong>$images</strong><span>images</span></div>
  <div><strong>$with_issues</strong><span>with issues</span></div>
</div>
<h2>Timeline</h2>
<p class="note">Each tick is an image; grey bars are deployment periods; ⚠ marks gaps over $max_gap_days days. Hover for details.</p>
<div class="chart">
$timeline
</div>
<h2>Deployments</h2>
<div class="table-scroll">
$table
</div>
<h2>Issues</h2>
$issues
<footer>Generated by CamTrapTools $version</footer>
</main>
<div class="tip" id="tip" hidden></div>
<script>
  const tip = document.getElementById("tip");
  for (const element of document.querySelectorAll("[data-tip]")) {
    element.addEventListener("pointerenter", () => { tip.textContent = element.dataset.tip; tip.hidden = false; });
    element.addEventListener("pointermove", (event) => {
      tip.style.left = `$${Math.min(event.clientX + 12, innerWidth - tip.offsetWidth - 8)}px`;
      tip.style.top = `$${event.clientY + 16}px`;
    });
    element.addEventListener("pointerleave", () => { tip.hidden = true; });
  }
</script>
</body>
</html>
""")


def build_report(package_dir: Path, max_gap_days: float = check.MAX_GAP_DAYS) -> str:
    """Return the complete HTML report for the package in ``package_dir``."""
    media, deployments = check.load_package(package_dir)
    results = check.run(package_dir, max_gap_days)
    summary = check.summarize(package_dir)
    return PAGE.substitute(
        name=escape(package_dir.resolve().name),
        generated=f"{datetime.now():%Y-%m-%d %H:%M}",
        deployments=len(results),
        images=len(media),
        with_issues=sum(1 for issues in results.values() if issues),
        max_gap_days=f"{max_gap_days:g}",
        timeline=timeline_svg(media, deployments, max_gap_days),
        table=summary_table_html(summary, results),
        issues=issues_html(results),
        version=__version__,
    )


def run(package_dir: Path, out_file: Path, max_gap_days: float = check.MAX_GAP_DAYS) -> None:
    """Write the HTML report for ``package_dir`` to ``out_file``."""
    out_file.write_text(build_report(package_dir, max_gap_days), encoding="utf-8")
