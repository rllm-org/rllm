import pytest

from rllm.rewards import RewardConfig
from rllm.rewards.reward_fn import f1_reward_fn
from rllm.rewards.reward_types import RewardInput
from rllm.rewards.search_reward import RewardSearchFn


class TestF1RewardFn:
    """Tests for the standalone token-level F1 reward function."""

    def test_exact_match(self):
        """Identical prediction and ground truth score a perfect F1."""
        output = f1_reward_fn({"ground_truth": "Paris"}, "Paris")
        assert output.reward == 1.0

    def test_partial_overlap(self):
        """Token overlap yields the harmonic mean of precision and recall."""
        # gt -> {hello, world}; pred -> {hello, there, world}
        # precision=2/3, recall=2/2 -> F1 = 0.8
        output = f1_reward_fn({"ground_truth": "Hello, world!"}, "hello there world")
        assert output.reward == pytest.approx(0.8)

    def test_normalization_drops_articles_and_punctuation(self):
        """Articles, casing, and punctuation are normalized away before scoring."""
        # "the quick brown fox" -> "quick brown fox" == prediction
        output = f1_reward_fn({"ground_truth": "the quick brown fox"}, "quick brown fox")
        assert output.reward == 1.0

    def test_no_overlap(self):
        """Disjoint tokens score zero."""
        output = f1_reward_fn({"ground_truth": "Paris"}, "London")
        assert output.reward == 0.0

    def test_empty_ground_truth(self):
        """An empty ground truth produces a zero reward, not a crash."""
        output = f1_reward_fn({"ground_truth": ""}, "anything")
        assert output.reward == 0.0

    def test_none_ground_truth(self):
        """A ``None`` ground truth is treated as empty (zero reward)."""
        output = f1_reward_fn({"ground_truth": None}, "anything")
        assert output.reward == 0.0

    def test_missing_ground_truth_key(self):
        """A task_info without a ground_truth key defaults to empty text."""
        output = f1_reward_fn({}, "anything")
        assert output.reward == 0.0

    def test_all_tokens_normalized_away(self):
        """When both texts reduce to no tokens after normalization, reward is zero."""
        output = f1_reward_fn({"ground_truth": "a an the"}, "a an the")
        assert output.reward == 0.0


class TestRewardSearchFnHelpers:
    """Tests for the pure helper methods on ``RewardSearchFn``."""

    @pytest.fixture
    def reward(self):
        return RewardSearchFn(RewardConfig())

    def test_normalize_answer(self, reward):
        """Normalization lowercases, strips punctuation/articles, and fixes whitespace."""
        assert reward.normalize_answer("The, Quick!!  Brown fox") == "quick brown fox"

    def test_f1_score_exact(self, reward):
        assert reward.f1_score("paris", "paris") == (1.0, 1.0, 1.0)

    def test_f1_score_partial(self, reward):
        # common {quick, brown} over 3 pred / 3 gt tokens -> 2/3 each
        f1, precision, recall = reward.f1_score("quick brown fox", "the quick brown dog")
        assert f1 == pytest.approx(2 / 3)
        assert precision == pytest.approx(2 / 3)
        assert recall == pytest.approx(2 / 3)

    def test_f1_score_no_overlap(self, reward):
        assert reward.f1_score("cat", "dog") == (0, 0, 0)

    def test_f1_score_yes_no_mismatch(self, reward):
        """yes/no/noanswer answers must match exactly or score zero."""
        assert reward.f1_score("yes", "no") == (0, 0, 0)

    def test_exact_match_score_normalizes(self, reward):
        """Exact match is computed on the normalized forms."""
        assert reward.exact_match_score("Paris.", "paris") is True

    def test_extract_boxed_answer(self, reward):
        assert reward.extract_answer_from_response(r"The capital is \boxed{Paris}") == "Paris"

    def test_extract_strips_think_tags_and_reads_bold(self, reward):
        response = "<think>let me recall</think> The answer is **Napoleon Bonaparte**"
        assert reward.extract_answer_from_response(response) == "Napoleon Bonaparte"

    def test_extract_year(self, reward):
        assert reward.extract_answer_from_response("It happened in 1969 somewhere") == "1969"


class TestRewardSearchFnCall:
    """End-to-end tests for ``RewardSearchFn.__call__`` via ``RewardInput``."""

    @pytest.fixture
    def reward(self):
        return RewardSearchFn(RewardConfig())

    def test_exact_match_full_reward(self, reward):
        output = reward(RewardInput(task_info={"ground_truth": "Paris"}, action=r"The capital is \boxed{Paris}"))
        assert output.is_correct is True
        assert output.reward == 1.0
        assert output.metadata["evaluation_method"] == "exact_match"

    def test_partial_match_scales_reward_by_f1(self, reward):
        output = reward(RewardInput(task_info={"ground_truth": "the quick brown dog"}, action=r"\boxed{quick brown fox}"))
        assert output.is_correct is True
        assert output.metadata["evaluation_method"] == "f1_score"
        # reward = correct_reward * f1; default correct_reward == 1.0
        assert output.reward == pytest.approx(output.metadata["f1_score"])
        assert output.reward == pytest.approx(2 / 3)

    def test_below_f1_threshold_is_incorrect(self, reward):
        """F1 below the 0.3 threshold (and no exact match) scores the incorrect reward."""
        output = reward(RewardInput(task_info={"ground_truth": "alpha beta gamma delta"}, action=r"\boxed{alpha zzz yyy www}"))
        assert output.is_correct is False
        assert output.reward == 0.0
        assert output.metadata["f1_score"] < 0.3

    def test_multiple_ground_truths_matches_any(self, reward):
        output = reward(RewardInput(task_info={"ground_truth": ["London", "Paris"]}, action=r"\boxed{Paris}"))
        assert output.is_correct is True
        assert output.reward == 1.0
        assert output.metadata["evaluation_method"] == "exact_match"

    def test_answer_key_used_when_ground_truth_absent(self, reward):
        """The ``answer`` key is honored as a fallback for ``ground_truth``."""
        output = reward(RewardInput(task_info={"answer": "Paris"}, action=r"\boxed{Paris}"))
        assert output.is_correct is True
        assert output.reward == 1.0

    def test_missing_ground_truth_returns_unk_error(self, reward):
        output = reward(RewardInput(task_info={}, action="anything"))
        assert output.is_correct is False
        assert output.reward == RewardConfig().unk_error_reward
        assert "error" in output.metadata
