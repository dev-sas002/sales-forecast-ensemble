"""
Plain-language forecast explanation, with and without an API key.

A quantile forecast is three numbers a week for twelve weeks. A demand planner
reviewing forty of those before Friday does not read them; they read the ones
someone flagged. Turning the numbers into two sentences — what the model
expects, how that compares to recent trading, what drove it, and how much to
trust it — is the difference between a dashboard and a tool.

**The architecture is deliberate, and it is the interesting part.** The
language model is given no data access, no model access, and nothing to
compute. It receives a :class:`~src.explain.facts.ForecastFacts` — every
number already calculated from the forecast, the feature table, LightGBM's own
SHAP contributions and the measured backtest — and is asked only to phrase
them. It cannot invent a driver, because the drivers are the model's actual
attributions; it cannot invent an accuracy claim, because the accuracy is the
measured one. The worst failure available to it is clumsy prose.

**It degrades to nothing worse than a plainer sentence.** With no API key —
which is the default, and what the test suite runs under — `explain()` renders
the same facts from a template. The endpoint returns 200, the dashboard shows
an explanation, and the only difference is that the writing is mechanical. No
feature of this system requires a key, and no test makes a network call: the
tests that cover the Claude path inject a fake client.

`anthropic` is an optional dependency, imported inside the call. Installing it
is `pip install anthropic`; not installing it is a supported configuration.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from src.explain.facts import ForecastFacts

logger = logging.getLogger(__name__)

#: Opus 5 for a short, careful piece of writing over supplied numbers. The
#: request is tiny and infrequent — one per explained series, on demand.
MODEL = "claude-opus-5"

#: Two or three sentences. Capped so a runaway response cannot become a cost
#: or a latency problem in a dashboard callback.
MAX_TOKENS = 600

SYSTEM_PROMPT = """\
You write short demand-planning notes for a supply chain team.

You are given a JSON object of facts already computed from a forecasting
model: its predicted units, how that compares with recent trading, the
model's own feature attributions, any data-quality flags, and the model's
measured out-of-sample accuracy.

Rules:
- Use only the numbers in the JSON. Never introduce a figure that is not there,
  and never round one into a different claim.
- Two to four sentences. No headings, no bullet points, no preamble.
- Say what is expected, how it compares to recent weeks, what drove it, and
  what the uncertainty means in practice.
- If anomaly_count is above zero, mention that the inputs have flagged weeks
  and that the forecast inherits them.
- Write for a planner deciding an order quantity, not for a data scientist.
- Plain English. No jargon, no hedging padding, no "it is important to note".
"""


@dataclass(frozen=True, slots=True)
class Explanation:
    """The note, and how it was produced — the caller always knows which."""

    text: str
    source: str  # "claude" | "template"
    model: str | None = None


def narrator_status() -> dict[str, object]:
    """
    Whether AI narration is available, and why not if it is not.

    Surfaced in the dashboard and at `/health` so the degraded path is visible
    rather than silent — an AI feature that has quietly stopped working and is
    serving templates is worse than one that says so.
    """
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    try:
        import anthropic  # noqa: F401

        installed = True
    except ImportError:
        installed = False

    if installed and has_key:
        reason = "ready"
    elif not installed:
        reason = "the optional `anthropic` package is not installed"
    else:
        reason = "ANTHROPIC_API_KEY is not set"

    return {
        "available": installed and has_key,
        "model": MODEL if installed and has_key else None,
        "reason": reason,
    }


def _pct(value: float) -> str:
    return f"{value:+.0%}"


def render_template(facts: ForecastFacts) -> str:
    """
    The no-API-key explanation: the same facts, phrased mechanically.

    Every sentence here is a direct restatement of a field. That is the point —
    it is the floor the AI path is measured against, and it is what runs in
    tests, in CI, and on any machine with no credentials.
    """
    parts = [
        f"{facts.sku} in {facts.region}: the model expects "
        f"{facts.total_forecast:,.0f} units over the next {facts.horizon_weeks} "
        f"weeks, around {facts.mean_weekly:,.0f} a week."
    ]

    if facts.trend_direction != "unknown":
        parts.append(
            f"That is {facts.trend_direction} the "
            f"{facts.recent_mean_weekly:,.0f} units a week it has averaged "
            f"recently ({_pct(facts.change_vs_recent)}), and the window falls in "
            f"{facts.seasonal_position}."
        )

    if facts.drivers:
        named = ", ".join(d.description for d in facts.drivers[:3])
        parts.append(f"The largest influences on the median were {named}.")

    if facts.interval_width_pct:
        parts.append(
            f"The 80% interval spans about {facts.interval_width_pct:.0%} of the "
            "median, so plan the range rather than the point."
        )

    if facts.backtest_wape is not None:
        lift = (
            f", {facts.backtest_lift:.0%} better than a seasonal-naive baseline"
            if facts.backtest_lift is not None
            else ""
        )
        parts.append(f"Out of sample this model scores {facts.backtest_wape:.3f} WAPE{lift}.")

    if facts.anomaly_count:
        parts.append(
            f"Note that {facts.anomaly_count} week(s) of input for this series are "
            "flagged as suspect, so the forecast inherits whatever caused them."
        )

    return " ".join(parts)


def _call_claude(facts: ForecastFacts, client=None) -> str:
    """One Messages request. `client` is injected by the tests."""
    import json

    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        # A short piece of writing over numbers that are already computed —
        # low effort is the right spend, and the facts leave nothing to reason
        # about beyond how to say it.
        output_config={"effort": "low"},
        messages=[
            {
                "role": "user",
                "content": json.dumps(facts.to_dict(), default=str, sort_keys=True),
            }
        ],
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()


def explain(facts: ForecastFacts, client=None) -> Explanation:
    """
    Narrate a forecast, falling back to the template on any failure.

    "Any failure" is meant literally: no key, no package, a rate limit, a
    network partition, a malformed response. A demand planner opening the
    dashboard during an Anthropic incident should see a slightly duller
    sentence, not a 500 — the explanation is a convenience layered on top of
    numbers that are already correct, and it must never be able to take the
    product down.
    """
    status = narrator_status()
    if client is None and not status["available"]:
        logger.debug("AI narration unavailable (%s); rendering template", status["reason"])
        return Explanation(text=render_template(facts), source="template")

    try:
        text = _call_claude(facts, client=client)
        if not text:
            raise ValueError("empty response")
        return Explanation(text=text, source="claude", model=MODEL)
    except Exception as exc:
        logger.warning("AI narration failed (%s); rendering template", exc)
        return Explanation(text=render_template(facts), source="template")
