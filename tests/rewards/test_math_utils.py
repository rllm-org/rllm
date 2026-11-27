from rllm.rewards.math_utils.utils import grade_answer_mathd, grade_answer_sympy


class TestGradeAnswerEmptyInputs:
    """Empty ground truths and empty given answers must never grade as a match."""

    def test_sympy_empty_vs_empty(self):
        assert grade_answer_sympy("", "") is False

    def test_sympy_whitespace_vs_whitespace(self):
        assert grade_answer_sympy("   ", "  ") is False

    def test_sympy_empty_given_vs_real_ground_truth(self):
        assert grade_answer_sympy("", "42") is False

    def test_sympy_real_given_vs_empty_ground_truth(self):
        assert grade_answer_sympy("42", "") is False

    def test_sympy_real_answer_still_matches(self):
        assert grade_answer_sympy("42", "42") is True

    def test_mathd_empty_vs_empty(self):
        assert grade_answer_mathd("", "") is False

    def test_mathd_whitespace_vs_whitespace(self):
        assert grade_answer_mathd("   ", "  ") is False

    def test_mathd_empty_given_vs_real_ground_truth(self):
        assert grade_answer_mathd("", "42") is False

    def test_mathd_real_given_vs_empty_ground_truth(self):
        assert grade_answer_mathd("42", "") is False

    def test_mathd_real_answer_still_matches(self):
        assert grade_answer_mathd("42", "42") is True
