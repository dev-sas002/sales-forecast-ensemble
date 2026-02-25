"""
Tests for forecast explanation.

Two properties matter and both are about trust rather than about output.

**The facts are computed, not generated.** Every number the narrator is given
comes from arithmetic on the forecast, the history and the model's own SHAP
contributions. If that stops being true, the explanation becomes a plausible
story with no connection to the model, which is worse than no explanation.

**Nothing requires a key.** These tests make no network call and set no
credentials: the Claude path is exercised with an injected fake client, and
the default path is the template. If a test here ever needs `ANTHROPIC_API_KEY`
the degradation has stopped working.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.explain.facts import ForecastFacts, build_facts
from src.explain.narrative import (
    MODEL,
    explain,
    narrator_status,
    render_template,
)


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    """Every test runs as a developer with no credentials — the default."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def forecast_rows(medians, start="2026-01-05"):
    dates = pd.date_range(start, periods=len(medians), freq="W-MON")
    return pd.DataFrame(
        {
            "sku": "SKU-1",
            "region": "NE",
            "date": dates.strftime("%Y-%m-%d"),
            "q0.1": [m * 0.7 for m in medians],
            "q0.5": list(medians),
            "q0.9": [m * 1.3 for m in medians],
        }
    )


def history_rows(units):
    weeks = pd.date_range("2024-01-01", periods=len(units), freq="W-MON")
    return pd.DataFrame(
        {
            "sku": "SKU-1",
            "region": "NE",
            "week": weeks,
            "units_sold": np.asarray(units, dtype=float),
            "avg_price": 2.5,
        }
    )


class TestFactsAreArithmetic:
    def test_the_total_is_the_sum_of_the_medians(self):
        facts = build_facts("SKU-1", "NE", forecast_rows([10.0, 20.0, 30.0]))
        assert facts.total_forecast == pytest.approx(60.0)
        assert facts.mean_weekly == pytest.approx(20.0)
        assert facts.horizon_weeks == 3

    def test_the_change_is_measured_against_recent_trading(self):
        facts = build_facts(
            "SKU-1",
            "NE",
            forecast_rows([120.0] * 4),
            history=history_rows([100.0] * 60),
        )
        assert facts.recent_mean_weekly == pytest.approx(100.0)
        assert facts.change_vs_recent == pytest.approx(0.2)
        assert facts.trend_direction == "above"

    def test_a_flat_forecast_is_reported_as_in_line(self):
        facts = build_facts(
            "SKU-1",
            "NE",
            forecast_rows([100.0] * 4),
            history=history_rows([100.0] * 60),
        )
        assert facts.trend_direction == "in line with"

    def test_without_history_the_comparison_is_omitted_not_guessed(self):
        facts = build_facts("SKU-1", "NE", forecast_rows([10.0, 20.0]))
        assert facts.trend_direction == "unknown"
        assert np.isnan(facts.change_vs_recent)

    def test_interval_width_is_relative_to_the_median(self):
        facts = build_facts("SKU-1", "NE", forecast_rows([100.0]))
        # q0.9 - q0.1 = 130 - 70 = 60, over a median of 100.
        assert facts.interval_width_pct == pytest.approx(0.6)

    def test_anomalies_in_the_history_are_counted_and_quoted(self):
        units = [150.0] * 20 + [0.0] * 5 + [150.0] * 20
        facts = build_facts("SKU-1", "NE", forecast_rows([150.0] * 4), history=history_rows(units))
        assert facts.anomaly_count > 0
        assert facts.anomaly_notes

    def test_backtest_lift_is_derived_from_the_measured_summary(self):
        facts = build_facts(
            "SKU-1",
            "NE",
            forecast_rows([10.0]),
            backtest={"wape_model": 0.2, "wape_naive": 0.4, "coverage": 0.72},
        )
        assert facts.backtest_wape == 0.2
        assert facts.backtest_lift == pytest.approx(0.5)

    def test_the_calibrated_coverage_is_preferred_when_present(self):
        facts = build_facts(
            "SKU-1",
            "NE",
            forecast_rows([10.0]),
            backtest={
                "wape_model": 0.2,
                "wape_naive": 0.4,
                "coverage": 0.72,
                "calibrated_coverage": 0.81,
            },
        )
        assert facts.backtest_coverage == 0.81

    def test_facts_serialise_to_json_safe_primitives(self):
        import json

        facts = build_facts(
            "SKU-1",
            "NE",
            forecast_rows([10.0, 12.0]),
            history=history_rows([11.0] * 60),
            drivers=[("roll_4w_mean", 3.2)],
        )
        json.dumps(facts.to_dict(), default=str)  # must not raise
        assert facts.to_dict()["drivers"][0]["description"] == ("the trailing four-week average")


class TestDegradationWithoutAKey:
    def test_the_narrator_reports_itself_unavailable(self):
        status = narrator_status()
        assert status["available"] is False
        assert status["model"] is None
        assert isinstance(status["reason"], str) and status["reason"]

    def test_explain_returns_a_template_note_not_an_error(self):
        facts = build_facts(
            "SKU-1", "NE", forecast_rows([50.0] * 4), history=history_rows([40.0] * 60)
        )
        note = explain(facts)
        assert note.source == "template"
        assert note.model is None
        assert "SKU-1" in note.text and "NE" in note.text

    def test_the_template_quotes_only_facts_it_was_given(self):
        facts = build_facts(
            "SKU-1",
            "NE",
            forecast_rows([25.0] * 4),
            backtest={"wape_model": 0.238, "wape_naive": 0.327},
        )
        text = render_template(facts)
        assert "100" in text  # the total
        assert "0.238" in text  # the measured WAPE
        assert "27%" in text or "28%" in text  # the derived lift

    def test_the_template_mentions_flagged_inputs(self):
        units = [150.0] * 20 + [0.0] * 5 + [150.0] * 20
        facts = build_facts("SKU-1", "NE", forecast_rows([150.0] * 4), history=history_rows(units))
        assert "flagged as suspect" in render_template(facts)

    def test_a_minimal_facts_object_still_renders(self):
        facts = ForecastFacts(
            sku="S",
            region="R",
            horizon_weeks=0,
            total_forecast=0.0,
            mean_weekly=0.0,
            recent_mean_weekly=float("nan"),
            change_vs_recent=float("nan"),
            interval_width_pct=0.0,
            trend_direction="unknown",
            seasonal_position="unknown",
        )
        assert render_template(facts)


class TestTheClaudePath:
    """Exercised with an injected client — never a real request."""

    @staticmethod
    def fake_client(text="Demand looks steady; order to the middle of the band."):
        block = MagicMock()
        block.type = "text"
        block.text = text
        response = MagicMock()
        response.content = [block]
        client = MagicMock()
        client.messages.create.return_value = response
        return client

    def test_a_successful_call_is_reported_as_coming_from_claude(self):
        facts = build_facts("SKU-1", "NE", forecast_rows([10.0]))
        note = explain(facts, client=self.fake_client())
        assert note.source == "claude"
        assert note.model == MODEL
        assert note.text.startswith("Demand looks steady")

    def test_the_model_is_sent_the_facts_and_nothing_else(self):
        import json

        client = self.fake_client()
        facts = build_facts("SKU-1", "NE", forecast_rows([10.0, 11.0]))
        explain(facts, client=client)

        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["model"] == MODEL
        payload = json.loads(kwargs["messages"][0]["content"])
        # The payload is exactly the computed facts — no dataframe, no model,
        # nothing for the narrator to recompute or misread.
        assert payload["sku"] == "SKU-1"
        assert payload["total_forecast"] == pytest.approx(21.0)
        assert set(payload) == set(facts.to_dict())

    def test_the_system_prompt_forbids_inventing_numbers(self):
        client = self.fake_client()
        explain(build_facts("S", "R", forecast_rows([1.0])), client=client)
        system = client.messages.create.call_args.kwargs["system"]
        assert "Use only the numbers in the JSON" in system

    def test_an_api_failure_falls_back_to_the_template(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("rate limited")

        facts = build_facts("SKU-1", "NE", forecast_rows([10.0]))
        note = explain(facts, client=client)

        assert note.source == "template"
        assert "SKU-1" in note.text

    def test_an_empty_response_falls_back_rather_than_serving_nothing(self):
        note = explain(
            build_facts("SKU-1", "NE", forecast_rows([10.0])),
            client=self.fake_client(text="   "),
        )
        assert note.source == "template"
