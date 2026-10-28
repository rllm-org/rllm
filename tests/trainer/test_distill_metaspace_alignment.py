"""Offline regression tests using real Hugging Face tokenizer decoders."""

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from rllm.trainer.distill.advantage import compute_distill_reverse_kl
from rllm.trainer.distill.alignment import align_teacher_logprobs, build_byte_offsets


def _metaspace_tokenizer(text, prepend_scheme="always", replacement="▁"):
    pre_tokenizer = pre_tokenizers.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme)
    pieces = list(dict.fromkeys(piece for piece, _ in pre_tokenizer.pre_tokenize_str(text)))
    backend = Tokenizer(models.Unigram([("<unk>", 0.0)] + [(piece, -1.0) for piece in pieces], unk_id=0))
    backend.pre_tokenizer = pre_tokenizer
    backend.decoder = decoders.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme)
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")


def _bytelevel_tokenizer(text):
    pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    pieces = list(dict.fromkeys(piece for piece, _ in pre_tokenizer.pre_tokenize_str(text)))
    backend = Tokenizer(models.WordLevel({"<unk>": 0, **{piece: i + 1 for i, piece in enumerate(pieces)}}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizer
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")


@pytest.mark.parametrize(
    "text,prepend_scheme,replacement,expected_offsets",
    [
        ("hello world", "always", "▁", [0, 5, 11]),
        ("café 世界", "always", "▁", [0, 5, 12]),
        ("hello world", "never", "▁", [0, 5, 11]),
        (" hello world", "never", "▁", [0, 6, 12]),
        ("hello world", "first", "▁", [0, 5, 11]),
        ("hello world", "always", "▂", [0, 5, 11]),
    ],
)
def test_metaspace_offsets_match_decoded_text(text, prepend_scheme, replacement, expected_offsets):
    tokenizer = _metaspace_tokenizer(text, prepend_scheme, replacement)
    ids = tokenizer.encode(text, add_special_tokens=False)

    offsets, reconstructed = build_byte_offsets(tokenizer, ids)

    assert reconstructed == text
    assert reconstructed == tokenizer.decode(ids)
    assert offsets == expected_offsets
    assert offsets[-1] == len(reconstructed.encode("utf-8"))


@pytest.mark.parametrize("metaspace_side", ["student", "teacher"])
@pytest.mark.parametrize("text", ["hello world", "café 世界"])
def test_cross_tokenizer_alignment_preserves_teacher_signal(metaspace_side, text):
    metaspace = _metaspace_tokenizer(text)
    bytelevel = _bytelevel_tokenizer(text)
    student, teacher = (metaspace, bytelevel) if metaspace_side == "student" else (bytelevel, metaspace)
    student_ids = student.encode(text, add_special_tokens=False)
    teacher_ids = teacher.encode(text, add_special_tokens=False)
    assert len(student_ids) == len(teacher_ids) == 2
    student_logprobs = [-1.0, -2.0]
    teacher_logprobs = [-0.2, -0.3]

    aligned = align_teacher_logprobs(
        student_ids=student_ids,
        student_tokenizer=student,
        teacher_ids=teacher_ids,
        teacher_tokenizer=teacher,
        teacher_logprobs=teacher_logprobs,
        student_logprobs=student_logprobs,
        reasoning_str="",
        content_str=text,
    )

    assert aligned == pytest.approx(teacher_logprobs)
    advantages = compute_distill_reverse_kl(aligned, student_logprobs, clip_min=-5.0, clip_max=5.0)
    assert advantages == pytest.approx([0.8, 1.7])


def test_bytelevel_offsets_preserve_partial_utf8_bytes():
    # ByteLevel can split one UTF-8 character across tokens; decoding each token
    # independently would introduce replacement characters and wrong offsets.
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "Ã": 1, "©": 2}, unk_token="<unk>"))
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")

    offsets, reconstructed = build_byte_offsets(tokenizer, [1, 2])

    assert offsets == [0, 1, 2]
    assert reconstructed == tokenizer.decode([1, 2]) == "é"


def test_empty_metaspace_sequence_has_no_bytes():
    tokenizer = _metaspace_tokenizer("hello world")
    assert build_byte_offsets(tokenizer, []) == ([0], "")


@pytest.mark.parametrize("metaspace_side", ["student", "teacher"])
def test_cross_tokenizer_alignment_with_different_token_counts(metaspace_side):
    text = "hello world"
    metaspace = _metaspace_tokenizer(text)
    bytelevel_pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    characters = list(dict.fromkeys("".join(piece for piece, _ in bytelevel_pre_tokenizer.pre_tokenize_str(text))))
    backend = Tokenizer(models.BPE({"<unk>": 0, **{char: i + 1 for i, char in enumerate(characters)}}, [], unk_token="<unk>"))
    backend.pre_tokenizer = bytelevel_pre_tokenizer
    backend.decoder = decoders.ByteLevel()
    character_tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    student, teacher = (metaspace, character_tokenizer) if metaspace_side == "student" else (character_tokenizer, metaspace)
    student_ids = student.encode(text, add_special_tokens=False)
    teacher_ids = teacher.encode(text, add_special_tokens=False)
    assert sorted([len(student_ids), len(teacher_ids)]) == [2, 11]
    teacher_logprobs = [-0.1] * 11 if metaspace_side == "student" else [-0.2, -0.3]

    aligned = align_teacher_logprobs(
        student_ids=student_ids,
        student_tokenizer=student,
        teacher_ids=teacher_ids,
        teacher_tokenizer=teacher,
        teacher_logprobs=teacher_logprobs,
        student_logprobs=[-1.0] * len(student_ids),
        reasoning_str="",
        content_str=text,
    )

    expected = [-0.5, -0.6] if metaspace_side == "student" else [-0.04] * 5 + [-0.05] * 6
    assert aligned == pytest.approx(expected)
    assert sum(aligned) == pytest.approx(sum(teacher_logprobs))


def test_metaspace_standalone_prefix_marker_has_zero_byte_width():
    backend = Tokenizer(models.Unigram([("<unk>", 0.0), ("▁", -1.0), ("hello", -1.0), ("▁world", -1.0)], unk_id=0))
    backend.decoder = decoders.Metaspace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")

    offsets, reconstructed = build_byte_offsets(tokenizer, [1, 2, 3])

    assert offsets == [0, 0, 5, 11]
    assert reconstructed == tokenizer.decode([1, 2, 3]) == "hello world"
