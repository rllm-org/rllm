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

    def test_no_deliverable_is_ungraded_not_zero(self):
        # Fail-closed: nothing surfaced -> ungraded, and the judge is never called.
        rubric = [{"score": 1, "criterion": "c"}]
        out = g.evaluate(self._task(rubric), Episode(artifacts={}))
        assert out.reward == 0.0 and out.metadata.get("reason") == "no_deliverable_surfaced"
        assert out.metadata.get("ungraded") is True

    def test_never_reads_transcript(self, monkeypatch):
        # Even with a rich transcript on the episode, no deliverable => ungraded.
        rubric = [{"score": 1, "criterion": "c"}]
        called = {"n": 0}
        monkeypatch.setattr(g, "_make_litellm_judge", lambda *a, **k: (lambda *args: called.__setitem__("n", called["n"] + 1) or True))
        ep = Episode(artifacts={"answer": "I produced a perfect deliverable", "conversation": [{"role": "assistant", "content": "trust me"}]})
        out = g.evaluate(self._task(rubric), ep)
        assert out.metadata.get("reason") == "no_deliverable_surfaced"
        assert called["n"] == 0  # judge never invoked


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


# --------------------------------------------------------------------------- #
# Pairwise grader
# --------------------------------------------------------------------------- #


class TestParseWinner:
    def test_json(self):
        assert g._parse_winner('{"winner": "A", "reasoning": "x"}') == "a"
        assert g._parse_winner('{"winner": "B"}') == "b"
        assert g._parse_winner('{"winner": "tie"}') == "tie"

    def test_unparseable(self):
        assert g._parse_winner("no verdict here 12345") is None


class TestScorePairwise:
    def test_consistent_win_both_positions(self):
        # judge always prefers the candidate regardless of slot
        def judge(prompt, rubric, a, b):
            return "a" if a == "CAND" else "b"

        # call1: a=CAND -> 'a' -> win(1); call2: a=REF,b=CAND -> 'b' -> win(1)
        assert g.score_pairwise("CAND", "REF", "p", "r", judge) == 1.0

    def test_consistent_loss(self):
        def judge(prompt, rubric, a, b):
            return "a" if a == "REF" else "b"  # always prefers REF

        assert g.score_pairwise("CAND", "REF", "p", "r", judge) == 0.0

    def test_tie(self):
        assert g.score_pairwise("C", "R", "p", "r", lambda *a: "tie") == 0.5

    def test_position_bias_averages_to_tie(self):
        # judge always picks slot A regardless of content -> pure position bias
        def judge(prompt, rubric, a, b):
            return "a"

        # call1 (cand=A) -> win(1); call2 (cand=B) -> 'a'=REF -> loss(0); mean 0.5
        assert g.score_pairwise("C", "R", "p", "r", judge) == 0.5

    def test_both_calls_fail_returns_none(self):
        assert g.score_pairwise("C", "R", "p", "r", lambda *a: None) is None


class TestEvaluatePairwise:
    def _task(self, *, rubric=None, ref_text="EXPERT DELIVERABLE"):
        from pathlib import Path

        from rllm.types import Task

        meta = {"prompt": "do the thing", "judge_model": "stub/model", "reference_deliverable_text": ref_text}
        if rubric is not None:
            meta["rubric_json"] = json.dumps(rubric)
        return Task(id="t1", instruction="do the thing", metadata=meta, dataset_dir=Path("/"))

    def test_no_reference_is_ungraded(self):
        from pathlib import Path

        from rllm.types import Task

        task = Task(id="t", instruction="x", metadata={"prompt": "p", "judge_model": "m"}, dataset_dir=Path("/"))
        out = g.evaluate_pairwise(task, Episode(artifacts={"deliverable_text": "cand"}))
        assert out.reward == 0.0 and out.metadata.get("reason") == "no_reference_deliverable"

    def test_win_with_stub_judge(self, monkeypatch):
        # stub the pairwise judge factory to always prefer the candidate
        monkeypatch.setattr(g, "_make_litellm_pairwise_judge", lambda *a, **k: (lambda prompt, rubric, x, y: "a" if x == "CAND" else "b"))
        task = self._task(rubric=[{"score": 2, "criterion": "c1"}])
        out = g.evaluate_pairwise(task, Episode(artifacts={"deliverable_text": "CAND"}))
        assert out.reward == 1.0 and out.is_correct is True
        assert out.metadata["reference_source"] == "metadata.reference_deliverable_text"

    def test_tie_is_not_a_win(self, monkeypatch):
        monkeypatch.setattr(g, "_make_litellm_pairwise_judge", lambda *a, **k: (lambda *args: "tie"))
        out = g.evaluate_pairwise(self._task(), Episode(artifacts={"deliverable_text": "CAND"}))
        assert out.reward == 0.5 and out.is_correct is False

    def test_no_judge_is_ungraded(self, monkeypatch):
        monkeypatch.setattr(g, "_resolve_judge", lambda task: ("", None, None))
        out = g.evaluate_pairwise(self._task(), Episode(artifacts={"deliverable_text": "CAND"}))
        assert out.reward == 0.0 and out.metadata.get("reason") == "no_judge_configured"

    def test_no_candidate_deliverable_is_ungraded(self, monkeypatch):
        # Fail-closed: no candidate surfaced -> ungraded; judge never called.
        called = {"n": 0}
        monkeypatch.setattr(g, "_make_litellm_pairwise_judge", lambda *a, **k: (lambda *args: called.__setitem__("n", called["n"] + 1) or "a"))
        out = g.evaluate_pairwise(self._task(), Episode(artifacts={"answer": "I made a great file"}))
        assert out.reward == 0.0 and out.metadata.get("reason") == "no_deliverable_surfaced"
        assert called["n"] == 0
