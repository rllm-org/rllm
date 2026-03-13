"""Tests for dataset transform functions."""

from __future__ import annotations

from rllm.data.transforms import (
    ceval_transform,
    countdown_transform,
    gpqa_diamond_transform,
    gsm8k_transform,
    hmmt_transform,
    hotpotqa_transform,
    humaneval_transform,
    livecodebench_transform,
    math500_transform,
    mbpp_transform,
    mmlu_pro_transform,
    mmlu_redux_transform,
    mmmlu_transform,
    supergpqa_transform,
)


class TestGSM8KTransform:
    def test_basic_transform(self):
        row = {
            "question": "Janet sells 16 ducks. How many does she have left?",
            "answer": "She started with 20.\n#### 4",
        }
        result = gsm8k_transform(row)
        assert result["question"] == "Janet sells 16 ducks. How many does she have left?"
        assert result["ground_truth"] == "4"
        assert result["data_source"] == "gsm8k"

    def test_no_hash_marks(self):
        row = {"question": "Q", "answer": "42"}
        result = gsm8k_transform(row)
        assert result["ground_truth"] == "42"


class TestMath500Transform:
    def test_basic_transform(self):
        row = {
            "problem": "Solve x^2 = 9",
            "solution": "x = 3 or x = -3",
            "answer": "3",
        }
        result = math500_transform(row)
        assert result["question"] == "Solve x^2 = 9"
        assert result["ground_truth"] == "3"
        assert result["data_source"] == "math500"


class TestCountdownTransform:
    def test_basic_transform(self):
        row = {"target": 10, "nums": [2, 3, 5]}
        result = countdown_transform(row)
        assert result["target"] == 10
        assert result["nums"] == [2, 3, 5]
        assert result["data_source"] == "countdown"


class TestHotpotQATransform:
    def test_basic_transform(self):
        row = {"question": "Who wrote Hamlet?", "answer": "Shakespeare"}
        result = hotpotqa_transform(row)
        assert result["question"] == "Who wrote Hamlet?"
        assert result["ground_truth"] == "Shakespeare"
        assert result["data_source"] == "hotpotqa"


class TestGPQADiamondTransform:
    def test_basic_transform(self):
        row = {
            "Question": "What is the Higgs boson?",
            "Correct Answer": "A fundamental particle",
            "Incorrect Answer 1": "A type of star",
            "Incorrect Answer 2": "A chemical element",
            "Incorrect Answer 3": "A mathematical constant",
        }
        result = gpqa_diamond_transform(row)
        assert result["question"] == "What is the Higgs boson?"
        assert result["data_source"] == "gpqa_diamond"
        assert len(result["choices"]) == 4
        assert "A fundamental particle" in result["choices"]
        # ground_truth should be a letter pointing to the correct answer
        gt_idx = ord(result["ground_truth"]) - ord("A")
        assert result["choices"][gt_idx] == "A fundamental particle"

    def test_deterministic_shuffle(self):
        row = {
            "Question": "Test question",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong1",
            "Incorrect Answer 2": "Wrong2",
            "Incorrect Answer 3": "Wrong3",
        }
        result1 = gpqa_diamond_transform(row)
        result2 = gpqa_diamond_transform(row)
        assert result1["choices"] == result2["choices"]
        assert result1["ground_truth"] == result2["ground_truth"]


class TestSuperGPQATransform:
    def test_basic_transform(self):
        row = {
            "question": "What is 2+2?",
            "options": ["3", "4", "5", "6"],
            "answer": "4",
        }
        result = supergpqa_transform(row)
        assert result["question"] == "What is 2+2?"
        assert result["choices"] == ["3", "4", "5", "6"]
        assert result["ground_truth"] == "B"
        assert result["data_source"] == "supergpqa"

    def test_answer_not_in_options(self):
        row = {
            "question": "Test",
            "options": ["A", "B", "C"],
            "answer": "Z",
        }
        result = supergpqa_transform(row)
        assert result["ground_truth"] == ""


class TestCEvalTransform:
    def test_basic_transform(self):
        row = {
            "question": "中国的首都是？",
            "A": "上海",
            "B": "北京",
            "C": "广州",
            "D": "深圳",
            "answer": "B",
        }
        result = ceval_transform(row)
        assert result["question"] == "中国的首都是？"
        assert result["choices"] == ["上海", "北京", "广州", "深圳"]
        assert result["ground_truth"] == "B"
        assert result["data_source"] == "ceval"


class TestMMLUProTransform:
    def test_basic_transform(self):
        row = {
            "question": "What is the speed of light?",
            "options": ["3×10^8 m/s", "3×10^6 m/s", "3×10^10 m/s"],
            "answer": "A",
            "category": "physics",
        }
        result = mmlu_pro_transform(row)
        assert result["question"] == "What is the speed of light?"
        assert result["choices"] == ["3×10^8 m/s", "3×10^6 m/s", "3×10^10 m/s"]
        assert result["ground_truth"] == "A"
        assert result["data_source"] == "mmlu_pro"
        assert result["category"] == "physics"


class TestMMLUReduxTransform:
    def test_integer_answer(self):
        row = {
            "question": "Test question",
            "A": "Option A",
            "B": "Option B",
            "C": "Option C",
            "D": "Option D",
            "answer": 1,
        }
        result = mmlu_redux_transform(row)
        assert result["ground_truth"] == "B"  # index 1 -> B

    def test_string_answer(self):
        row = {
            "question": "Test",
            "A": "A",
            "B": "B",
            "C": "C",
            "D": "D",
            "answer": "C",
        }
        result = mmlu_redux_transform(row)
        # Non-integer answer treated as string
        assert result["ground_truth"] == "C"


class TestMMMLUTransform:
    def test_basic_transform(self):
        row = {
            "question": "¿Cuál es la capital de Francia?",
            "A": "Londres",
            "B": "París",
            "C": "Berlín",
            "D": "Madrid",
            "answer": "B",
        }
        result = mmmlu_transform(row)
        assert result["question"] == "¿Cuál es la capital de Francia?"
        assert result["choices"] == ["Londres", "París", "Berlín", "Madrid"]
        assert result["ground_truth"] == "B"
        assert result["data_source"] == "mmmlu"


class TestHMMTTransform:
    def test_basic_transform(self):
        row = {
            "problem": "Find the value of x such that x^2 = 4",
            "answer": "2",
            "problem_idx": 1,
            "problem_type": ["Algebra"],
        }
        result = hmmt_transform(row)
        assert result["question"] == "Find the value of x such that x^2 = 4"
        assert result["ground_truth"] == "2"
        assert result["data_source"] == "hmmt"


class TestHumanEvalTransform:
    def test_basic_transform(self):
        row = {
            "task_id": "HumanEval/0",
            "prompt": "def has_close_elements(numbers, threshold):",
            "canonical_solution": "    for i in range(len(numbers)):\n        pass",
            "test": "def check(candidate):\n    assert candidate([1.0], 0.5) == False",
            "entry_point": "has_close_elements",
        }
        result = humaneval_transform(row)
        assert result["question"] == "def has_close_elements(numbers, threshold):"
        assert "check(candidate)" in result["ground_truth"]
        assert result["data_source"] == "humanevalplus"
        assert result["entry_point"] == "has_close_elements"


class TestMBPPTransform:
    def test_basic_transform(self):
        row = {
            "task_id": 1,
            "text": "Write a function to find min cost path",
            "code": "def min_cost(cost, m, n):\n    pass",
            "test_list": [
                "assert min_cost([[1, 2, 3]], 0, 2) == 6",
                "assert min_cost([[1]], 0, 0) == 1",
            ],
            "test_setup_code": "",
        }
        result = mbpp_transform(row)
        assert result["question"] == "Write a function to find min cost path"
        assert "min_cost" in result["ground_truth"]
        assert "check(candidate)" in result["ground_truth"]
        assert result["data_source"] == "humanevalplus"

    def test_empty_tests(self):
        row = {"text": "Test", "test_list": [], "code": ""}
        result = mbpp_transform(row)
        assert result["question"] == "Test"


class TestLiveCodeBenchTransform:
    def test_basic_transform(self):
        row = {
            "question_content": "Given an array, find the sum",
            "public_test_cases": '[{"input": "1 2 3", "output": "6"}]',
            "question_id": "Q1",
        }
        result = livecodebench_transform(row)
        assert result["question"] == "Given an array, find the sum"
        assert result["data_source"] == "livecodebench"
        assert isinstance(result["ground_truth"], list)
        assert result["ground_truth"][0]["input"] == "1 2 3"

    def test_already_parsed_tests(self):
        row = {
            "question_content": "Test",
            "public_test_cases": [{"input": "1", "output": "1"}],
        }
        result = livecodebench_transform(row)
        assert isinstance(result["ground_truth"], list)
