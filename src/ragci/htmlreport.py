"""A sweep result you can send to someone.

The terminal renders a sweep badly now: there is a ranking, a per-finalist significance
test, sometimes a projection, and a note about which scores are not comparable with which.
That is a page, not a table.

Self-contained by construction — no CDN, no fonts, no scripts. A report that needs the
network is a report that renders as a blank box in six months.
"""

import html
from datetime import datetime
from pathlib import Path

from ragci.sweep import Comparison, SweepEvaluation, SweepOutcome

STYLE = """
:root { color-scheme: light dark; --fg: #1a1a1a; --muted: #6b6b6b; --line: #e2e2e2;
        --bg: #fff; --bar: #4a7c9e; --warn: #8a6d1f; --warn-bg: #fdf6e3; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e8e8e8; --muted: #9a9a9a; --line: #333; --bg: #161616;
          --bar: #6fa8cc; --warn: #d4b656; --warn-bg: #2a2416; }
}
* { box-sizing: border-box; }
body { margin: 0 auto; padding: 2.5rem 1.5rem; max-width: 52rem; background: var(--bg);
       color: var(--fg); font: 15px/1.6 ui-sans-serif, system-ui, -apple-system, sans-serif; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.05rem; margin: 2.5rem 0 .75rem; }
.sub { color: var(--muted); margin: 0 0 2rem; font-size: .9rem; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: .5rem .6rem; border-bottom: 1px solid var(--line); }
th { font-weight: 600; font-size: .8rem; text-transform: uppercase;
     letter-spacing: .04em; color: var(--muted); }
td.n, th.n { text-align: right; }
.win { font-weight: 600; }
.bar { height: .5rem; background: var(--bar); border-radius: 2px; display: inline-block;
       vertical-align: middle; }
.note { background: var(--warn-bg); border-left: 3px solid var(--warn); color: var(--warn);
        padding: .75rem 1rem; margin: 1rem 0; border-radius: 0 3px 3px 0; }
.ok { border-left-color: var(--bar); background: transparent; color: var(--fg);
      border-left: 3px solid var(--bar); }
code { font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; }
footer { margin-top: 3rem; color: var(--muted); font-size: .82rem; }
"""


def _params(config: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(config.items())) or "default"


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _bar(score: float, widest: float) -> str:
    width = 0.0 if widest <= 0 else 100 * score / widest
    return f'<span class="bar" style="width:{width:.1f}px"></span>'


def _rows(evaluations: list[SweepEvaluation], winner: dict, widest: float) -> str:
    out = []
    for evaluation in evaluations:
        mark = ' class="win"' if evaluation.config == winner else ""
        out.append(
            f"<tr{mark}><td>{_escape(_params(evaluation.config))}</td>"
            f'<td class="n">{evaluation.score:.3f}</td>'
            f"<td>{_bar(evaluation.score, widest)}</td>"
            f'<td class="n">{evaluation.n_cases}</td></tr>'
        )
    return "\n".join(out)


def _verdict(outcome: SweepOutcome) -> str:
    if not outcome.comparisons:
        return ""
    undecided: list[Comparison] = outcome.contenders
    if not undecided:
        worst = max(c.p_value for c in outcome.comparisons)
        return (
            f'<p class="note ok"><strong>Winner confirmed.</strong> Ahead of all '
            f"{len(outcome.comparisons)} finalist(s), p&nbsp;&le;&nbsp;{worst:.4f} after "
            "Holm-Bonferroni correction.</p>"
        )

    rows = "\n".join(
        f"<tr><td>{_escape(_params(c.against))}</td>"
        f'<td class="n">{c.advantage:+.3f}</td>'
        f'<td class="n">[{c.ci_low:.3f}, {c.ci_high:.3f}]</td>'
        f'<td class="n">{c.p_value:.4f}</td></tr>'
        for c in undecided
    )
    return (
        '<p class="note"><strong>No clear winner.</strong> The top configuration is not '
        "statistically separable from the rest of the final rung. Choosing it over these "
        "is a preference, not a measured improvement.</p>"
        "<table><thead><tr><th>not separable from</th><th class='n'>advantage</th>"
        "<th class='n'>95% CI</th><th class='n'>p</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _projection(outcome: SweepOutcome) -> str:
    curve = outcome.pool_curve
    if curve is None:
        return ""
    if curve.flat:
        return (
            '<h2>Projection</h2><p class="note">The score did not move across the pool '
            "sizes measured, so there is no trend to extend. A plateau over the range "
            "measured is not evidence of one beyond it.</p>"
        )
    if not curve.reliable:
        return (
            '<h2>Projection</h2><p class="note">The measured points do not follow the '
            f"log-linear shape retrieval degrades along (R²&nbsp;=&nbsp;{curve.r_squared:.2f}). "
            "Extrapolating from them would invent a number.</p>"
        )
    return (
        f"<h2>Projection</h2><p>At <strong>{curve.target_pool_size:,}</strong> documents: "
        f"<strong>{curve.extrapolated:.3f}</strong> "
        f"(95% CI [{curve.ci_low:.3f}, {curve.ci_high:.3f}], R²&nbsp;=&nbsp;"
        f"{curve.r_squared:.2f}). A projection, not a measurement.</p>"
    )


def render_html(outcome: SweepOutcome, *, generated_at: str | None = None) -> str:
    """One self-contained page. Nothing is fetched at render time."""
    latest = {repr(e.config): e for e in outcome.evaluations}
    deepest = max(e.rung for e in outcome.evaluations)
    finalists = [e for e in latest.values() if e.rung == deepest]
    eliminated = [e for e in latest.values() if e.rung != deepest]

    finalists.sort(key=lambda e: -e.score)
    eliminated.sort(key=lambda e: (-e.rung, -e.score))
    widest = max((e.score for e in finalists), default=1.0) or 1.0

    spent = sum(e.n_cases for e in outcome.evaluations)
    saved = 100 * (1 - spent / outcome.full_grid_cost) if outcome.full_grid_cost else 0

    tie = (
        '<p class="note"><strong>The cut was a tie-break.</strong> Configurations were '
        "eliminated while tied with the survivors, so which of them advanced was decided "
        "by ordering rather than by evidence. Add cases and sweep again.</p>"
        if outcome.arbitrary_elimination
        else ""
    )

    # Eliminated configurations were scored on fewer cases. Charting them beside the
    # finalists would invite a comparison the numbers do not support, so they get their
    # own table, below, with the reason attached.
    others = (
        "<h2>Eliminated earlier</h2>"
        "<p class='sub'>Scored on fewer cases than the finalists, so these numbers are "
        "not comparable with the ones above — that is what elimination means.</p>"
        "<table><thead><tr><th>configuration</th><th class='n'>score</th><th></th>"
        "<th class='n'>cases</th></tr></thead>"
        f"<tbody>{_rows(eliminated, outcome.winner, widest)}</tbody></table>"
        if eliminated
        else ""
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rag-ci sweep — {_escape(_params(outcome.winner))}</title>
<style>{STYLE}</style></head><body>
<h1>rag-ci sweep</h1>
<p class="sub">{outcome.n_configs} configurations, {len(outcome.rungs)} rung(s),
{spent} case-evaluations instead of {outcome.full_grid_cost} ({saved:.0f}% saved).</p>

{tie}
{_verdict(outcome)}

<h2>Final rung</h2>
<table><thead><tr><th>configuration</th><th class="n">score</th><th></th>
<th class="n">cases</th></tr></thead>
<tbody>{_rows(finalists, outcome.winner, widest)}</tbody></table>

{others}
{_projection(outcome)}

<footer>Generated by rag-ci{f" on {_escape(generated_at)}" if generated_at else ""}.
Scores are means over the cases shown; the verdict above is a paired bootstrap over
per-case differences, corrected for multiple comparisons.</footer>
</body></html>
"""


def save_html(outcome: SweepOutcome, path: Path) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    Path(path).write_text(render_html(outcome, generated_at=stamp), encoding="utf-8")
