"""Tests for the GDPval weighted-rubric grader.

The grading core (rubric parsing + weighted aggregation) is pure and tested
without a judge. ``grade_rubric`` takes an injected ``judge_fn`` so the
per-criterion LLM call is stubbed. ``evaluate`` is tested end-to-end by
monkeypatching the litellm judge factory.
"""

from __future__ import annotations

import json

from rllm.eval.reward_fns import gdpval as g
from rllm.eval.reward_fns.gdpval import Criterion, aggregate, parse_rubric
from rllm.types import Episode

# --------------------------------------------------------------------------- #
# Rubric parsing
# --------------------------------------------------------------------------- #


class TestParseRubric:
    def test_parses_json_string(self):
        raw = json.dumps(
            [
                {"score": 2, "criterion": "Has a summary section", "rubric_item_id": "a"},
                {"score": 5, "criterion": "Recommended value is 220", "rubric_item_id": "b"},
            ]
        )
        crits = parse_rubric(raw)
        assert [c.score for c in crits] == [2.0, 5.0]
        assert crits[0].text == "Has a summary section"
        assert crits[1].rubric_item_id == "b"

    def test_accepts_decoded_list(self):
        crits = parse_rubric([{"score": 1, "criterion": "x"}])
        assert len(crits) == 1 and crits[0].score == 1.0

    def test_skips_malformed_items(self):
        crits = parse_rubric(
            [
                {"score": 1, "criterion": "ok"},
                {"score": "notnum", "criterion": "bad"},
                {"criterion": "no score"},
                {"score": 2, "criterion": "   "},  # blank text
            ]
        )
        assert len(crits) == 1 and crits[0].text == "ok"


# --------------------------------------------------------------------------- #
# Weighted aggregation
# --------------------------------------------------------------------------- #


class TestAggregate:
    def _crits(self):
        return [Criterion(2, "a"), Criterion(3, "b"), Criterion(5, "c")]

    def test_all_met_is_full_score(self):
        r = aggregate(self._crits(), [True, True, True])
        assert r.reward == 1.0
        assert r.earned == 10 and r.total_possible == 10

    def test_none_met_is_zero(self):
        r = aggregate(self._crits(), [False, False, False])
        assert r.reward == 0.0 and r.earned == 0

    def test_partial_is_weight_normalized(self):
        # met a(2) and c(5) of possible 10 -> 0.7
        r = aggregate(self._crits(), [True, False, True])
        assert r.reward == 0.7
        assert r.earned == 7 and r.total_possible == 10

    def test_negative_penalty_subtracts_and_clips_at_zero(self):
        crits = [Criterion(4, "good"), Criterion(-10, "fabricates data")]
        # good met (+4), penalty condition also present (-10) -> earned -6, clip 0
        r = aggregate(crits, [True, True])
        assert r.earned == -6 and r.total_possible == 4
        assert r.reward == 0.0

    def test_negative_penalty_not_triggered(self):
        crits = [Criterion(4, "good"), Criterion(-10, "fabricates data")]
        r = aggregate(crits, [True, False])
        assert r.earned == 4 and r.reward == 1.0

    def test_no_positive_weight_is_ungraded(self):
        r = aggregate([Criterion(-2, "penalty only")], [False])
        assert r.ungraded is True and r.reward == 0.0


# --------------------------------------------------------------------------- #
# grade_rubric with injected judge
# --------------------------------------------------------------------------- #


class TestGradeRubric:
    def test_injected_judge_drives_score(self):
        crits = [Criterion(1, "mentions revenue"), Criterion(1, "mentions risk"), Criterion(2, "concludes GO")]

        def judge(text, score, deliverable, prompt):
            return text.split()[-1].lower() in deliverable.lower()

        deliverable = "The memo covers revenue and recommends GO."
        r = g.grade_rubric(crits, deliverable, "prompt", judge)
        # "revenue" yes(1), "risk" no(0), "GO" yes(2) -> 3/4
        assert r.reward == 0.75
        assert [c["met"] for c in r.per_criterion] == [True, False, True]

    def test_judge_exception_counts_as_not_met(self):
        crits = [Criterion(1, "a"), Criterion(1, "b")]

        def judge(text, score, deliverable, prompt):
            if text == "a":
                raise RuntimeError("judge down")
            return True

        r = g.grade_rubric(crits, "x", "p", judge)
        assert r.reward == 0.5  # only "b" counted


# --------------------------------------------------------------------------- #
# Deliverable text extraction
# --------------------------------------------------------------------------- #


class TestExtraction:
    def test_missing_file_returns_empty(self):
        assert g.extract_deliverable_text("/no/such/file.xlsx") == ""

    def test_text_file_roundtrip(self, tmp_path):
        p = tmp_path / "out.md"
        p.write_text("# Report\nrevenue is up", encoding="utf-8")
        text = g.extract_deliverable_text(p)
        assert "out.md" in text and "revenue is up" in text

    def test_csv_extracted(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,2\n", encoding="utf-8")
        assert "1,2" in g.extract_deliverable_text(p)


# --------------------------------------------------------------------------- #
# evaluate() end-to-end (judge factory monkeypatched)
# --------------------------------------------------------------------------- #


class TestEvaluate:
    def _task(self, rubric):
        from pathlib import Path

        from rllm.types import Task

        return Task(
            id="t1",
            instruction="do the thing",
            metadata={"prompt": "do the thing", "rubric_json": json.dumps(rubric), "judge_model": "stub/model"},
            dataset_dir=Path("/"),
        )

    def test_no_rubric_is_ungraded(self):
        from pathlib import Path

        from rllm.types import Task

        task = Task(id="t", instruction="x", metadata={}, dataset_dir=Path("/"))
        out = g.evaluate(task, Episode(artifacts={"deliverable_text": "anything"}))
        assert out.reward == 0.0 and out.metadata.get("ungraded") is True

    def test_full_pass_with_stub_judge(self, monkeypatch):
        rubric = [{"score": 2, "criterion": "c1"}, {"score": 3, "criterion": "c2"}]
        monkeypatch.setattr(g, "_make_litellm_judge", lambda *a, **k: (lambda *args: True))
        task = self._task(rubric)
        ep = Episode(artifacts={"deliverable_text": "a great deliverable"})
        out = g.evaluate(task, ep)
        assert out.reward == 1.0 and out.is_correct is True
        assert out.metadata["criteria_met"] == 2 and out.metadata["criteria_total"] == 2

    def test_partial_with_stub_judge(self, monkeypatch):
        rubric = [{"score": 1, "criterion": "yes"}, {"score": 1, "criterion": "no"}]
        monkeypatch.setattr(g, "_make_litellm_judge", lambda *a, **k: (lambda text, *args: text == "yes"))
        out = g.evaluate(self._task(rubric), Episode(artifacts={"deliverable_text": "d"}))
        assert out.reward == 0.5 and out.is_correct is True

    def test_no_judge_configured_is_ungraded(self, monkeypatch):
        # Force resolution to find no judge model.
        monkeypatch.setattr(g, "_resolve_judge", lambda task: ("", None, None))
        rubric = [{"score": 1, "criterion": "c"}]
        out = g.evaluate(self._task(rubric), Episode(artifacts={"deliverable_text": "d"}))
        assert out.reward == 0.0 and out.metadata.get("reason") == "no_judge_configured"


# --------------------------------------------------------------------------- #
# _parse_met tolerance
# --------------------------------------------------------------------------- #


class TestParseMet:
    def test_json_forms(self):
        assert g._parse_met('{"met": 1, "reasoning": "ok"}') is True
        assert g._parse_met('{"met": 0}') is False

    def test_prose_fallback(self):
        assert g._parse_met("Yes, this is met.") is True
        assert g._parse_met("No.") is False
