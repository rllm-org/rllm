"""Utilities for aligning logprobs for cross-tokenizer distillation."""

import logging
from bisect import bisect_left, bisect_right

from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-to-unicode mapping for byte-level BPE tokenizers."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]

    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs, strict=False))


BYTE_ENCODER: dict[int, str] = _bytes_to_unicode()
BYTE_DECODER: dict[str, int] = {v: k for k, v in BYTE_ENCODER.items()}


def _token_str_to_bytes(token_str: str | None) -> bytes:
    """Convert token string to raw bytes."""
    if token_str is None:
        return b""
    try:
        return bytes([BYTE_DECODER[c] for c in token_str])
    except KeyError:
        return token_str.encode("utf-8", errors="replace")


def build_byte_offsets(tokenizer: PreTrainedTokenizer, token_ids: list[int]) -> list[int]:
    """
    Build cumulative byte offsets for token sequence.

    Returns list of length n+1 where offsets[i] is byte position where token i starts,
    and offsets[n] is total byte length.
    """
    if not token_ids:
        return [0]

    token_strs = tokenizer.convert_ids_to_tokens(token_ids)
    offsets = [0]
    cumulative = 0
    for token_str in token_strs:
        token_bytes = _token_str_to_bytes(token_str)
        cumulative += len(token_bytes)
        offsets.append(cumulative)

    return offsets


def find_content_byte_ranges(
    text: str,
    reasoning_str: str,
    content_str: str,
) -> dict[str, tuple[int, int]]:
    """
    Find byte ranges for reasoning and content regions.

    Returns dict mapping region name to (start_byte, end_byte).
    Only includes regions that were found.
    """
    text_bytes = text.encode("utf-8")
    regions = {}

    if reasoning_str:
        reasoning_bytes = reasoning_str.encode("utf-8")
        idx = text_bytes.find(reasoning_bytes)
        if idx != -1:
            regions["reasoning"] = (idx, idx + len(reasoning_bytes))

    if content_str:
        content_bytes = content_str.encode("utf-8")
        search_start = regions.get("reasoning", (0, 0))[1]
        idx = text_bytes.find(content_bytes, search_start)
        if idx != -1:
            regions["content"] = (idx, idx + len(content_bytes))

    return regions


def _find_tokens_in_range(
    offsets: list[int],
    range_start: int,
    range_end: int,
) -> tuple[int, int]:
    """Find token indices that overlap with [range_start, range_end)."""
    n_tokens = len(offsets) - 1
    if n_tokens == 0:
        return 0, 0

    first = bisect_right(offsets, range_start) - 1
    first = max(0, first)
    last = bisect_left(offsets, range_end)
    last = min(n_tokens, last)

    return first, last


def _compute_overlap_mapping(
    student_offsets: list[int],
    teacher_offsets: list[int],
    s_region: tuple[int, int],
    t_region: tuple[int, int],
) -> tuple[list[list[int]], list[int]]:
    """
    Map student tokens to overlapping teacher tokens using two-pointer sweep.

    Returns (student_to_teachers, teacher_usage_count) where student_to_teachers[i]
    is list of teacher token indices overlapping student token i.
    """
    n_student = len(student_offsets) - 1
    n_teacher = len(teacher_offsets) - 1

    student_to_teachers: list[list[int]] = [[] for _ in range(n_student)]
    teacher_usage_count = [0] * n_teacher

    s_start, s_end = s_region
    t_start, t_end = t_region

    s_first, s_last = _find_tokens_in_range(student_offsets, s_start, s_end)
    t_first, t_last = _find_tokens_in_range(teacher_offsets, t_start, t_end)

    if s_first >= s_last or t_first >= t_last:
        return student_to_teachers, teacher_usage_count

    t_ptr = t_first

    for s_idx in range(s_first, s_last):
        s_tok_start = student_offsets[s_idx]
        s_tok_end = student_offsets[s_idx + 1]

        s_rel_start = max(0, s_tok_start - s_start)
        s_rel_end = min(s_end - s_start, s_tok_end - s_start)

        if s_rel_start >= s_rel_end:
            continue

        while t_ptr < t_last:
            t_tok_end = teacher_offsets[t_ptr + 1]
            t_rel_end = min(t_end - t_start, t_tok_end - t_start)
            if t_rel_end > s_rel_start:
                break
            t_ptr += 1

        for t_idx in range(t_ptr, t_last):
            t_tok_start = teacher_offsets[t_idx]
            t_tok_end = teacher_offsets[t_idx + 1]

            t_rel_start = max(0, t_tok_start - t_start)
            t_rel_end = min(t_end - t_start, t_tok_end - t_start)

            if t_rel_start >= t_rel_end:
                continue

            if t_rel_start >= s_rel_end:
                break

            student_to_teachers[s_idx].append(t_idx)
            teacher_usage_count[t_idx] += 1

    return student_to_teachers, teacher_usage_count


def _merge_overlap_mappings(
    mappings: list[tuple[list[list[int]], list[int]]],
    n_student: int,
    n_teacher: int,
) -> tuple[list[list[int]], list[int]]:
    """Merge overlap mappings from multiple regions."""
    student_to_teachers: list[list[int]] = [[] for _ in range(n_student)]
    teacher_usage_count = [0] * n_teacher

    for s2t, t_count in mappings:
        for s_idx, t_indices in enumerate(s2t):
            for t_idx in t_indices:
                if t_idx not in student_to_teachers[s_idx]:
                    student_to_teachers[s_idx].append(t_idx)
                    teacher_usage_count[t_idx] += 1

    return student_to_teachers, teacher_usage_count


def align_teacher_logprobs(
    student_ids: list[int],
    student_tokenizer: PreTrainedTokenizer,
    teacher_ids: list[int],
    teacher_tokenizer: PreTrainedTokenizer,
    teacher_logprobs: list[float],
    student_logprobs: list[float],
    reasoning_str: str,
    content_str: str,
) -> list[float]:
    """
    Align teacher logprobs to student tokens using byte-level alignment.

    Maps teacher logprobs to student tokens based on byte coverage in shared content
    regions. Returns aligned logprobs where content tokens get aggregated teacher
    logprobs and format tokens get 0.0. On failure, returns student_logprobs.

    Args:
        student_ids: Student token IDs
        student_tokenizer: Student tokenizer
        teacher_ids: Teacher token IDs
        teacher_tokenizer: Teacher tokenizer
        teacher_logprobs: Teacher logprobs per token
        student_logprobs: Student logprobs (fallback)
        reasoning_str: Reasoning content (may be empty)
        content_str: Answer content (may be empty)

    Returns:
        Aligned teacher logprobs for each student token.

    Raises:
        ValueError: If both reasoning_str and content_str are empty.
    """
    if not reasoning_str and not content_str:
        raise ValueError("At least one of reasoning_str or content_str must be provided for alignment.")

    n_student = len(student_ids)
    n_teacher = len(teacher_ids)

    student_offsets = build_byte_offsets(student_tokenizer, student_ids)
    teacher_offsets = build_byte_offsets(teacher_tokenizer, teacher_ids)

    student_text = student_tokenizer.decode(student_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    teacher_text = teacher_tokenizer.decode(teacher_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    student_regions = find_content_byte_ranges(student_text, reasoning_str, content_str)
    teacher_regions = find_content_byte_ranges(teacher_text, reasoning_str, content_str)

    if reasoning_str:
        if "reasoning" not in student_regions or "reasoning" not in teacher_regions:
            logger.warning(f"reasoning_str not found in decoded text. Zeroing out sample. reasoning_str[:50]={reasoning_str[:50]!r}")
            return list(student_logprobs)

    if content_str:
        if "content" not in student_regions or "content" not in teacher_regions:
            logger.warning(f"content_str not found in decoded text. Zeroing out sample. content_str[:50]={content_str[:50]!r}")
            return list(student_logprobs)

    mappings = []
    for region_name in student_regions:
        if region_name not in teacher_regions:
            continue

        s2t, t_count = _compute_overlap_mapping(
            student_offsets,
            teacher_offsets,
            student_regions[region_name],
            teacher_regions[region_name],
        )
        mappings.append((s2t, t_count))

    student_to_teachers, teacher_usage_count = _merge_overlap_mappings(mappings, n_student, n_teacher)

    aligned_logprobs = []
    for i in range(n_student):
        teacher_indices = student_to_teachers[i]

        if not teacher_indices:
            aligned_logprobs.append(0.0)
        else:
            total = 0.0
            for j in teacher_indices:
                usage = max(1, teacher_usage_count[j])
                total += teacher_logprobs[j] / usage
            aligned_logprobs.append(total)

    return aligned_logprobs


def visualize_alignment(
    student_ids: list[int],
    student_tokenizer: PreTrainedTokenizer,
    teacher_ids: list[int],
    teacher_tokenizer: PreTrainedTokenizer,
    teacher_logprobs: list[float],
    student_logprobs: list[float],
    reasoning_str: str,
    content_str: str,
    max_tokens: int = 50,
) -> None:
    """
    Visualize alignment between student and teacher tokens.

    Prints student and teacher tokens with their logprobs, plus summary statistics.
    """
    aligned = align_teacher_logprobs(
        student_ids,
        student_tokenizer,
        teacher_ids,
        teacher_tokenizer,
        teacher_logprobs,
        student_logprobs,
        reasoning_str,
        content_str,
    )

    student_decoded = [student_tokenizer.decode([tid], clean_up_tokenization_spaces=False) for tid in student_ids[:max_tokens]]
    teacher_decoded = [teacher_tokenizer.decode([tid], clean_up_tokenization_spaces=False) for tid in teacher_ids[:max_tokens]]

    def fmt(tok: str, lp: float) -> str:
        tok_display = repr(tok)[1:-1]
        return f"'{tok_display}' ({lp:.5f})"

    student_parts = []
    for tok, lp in zip(student_decoded, aligned[:max_tokens], strict=False):
        student_parts.append(fmt(tok, lp))

    teacher_parts = []
    for tok, lp in zip(teacher_decoded, teacher_logprobs[:max_tokens], strict=False):
        teacher_parts.append(fmt(tok, lp))

    print()
    print("student -> " + " ".join(student_parts))
    print()
    print("teacher -> " + " ".join(teacher_parts))
    print()

    student_offsets = build_byte_offsets(student_tokenizer, student_ids)
    teacher_offsets = build_byte_offsets(teacher_tokenizer, teacher_ids)
    student_text = student_tokenizer.decode(student_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    teacher_text = teacher_tokenizer.decode(teacher_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    student_regions = find_content_byte_ranges(student_text, reasoning_str, content_str)
    teacher_regions = find_content_byte_ranges(teacher_text, reasoning_str, content_str)

    mappings = []
    for region_name in student_regions:
        if region_name not in teacher_regions:
            continue
        s2t, t_count = _compute_overlap_mapping(
            student_offsets,
            teacher_offsets,
            student_regions[region_name],
            teacher_regions[region_name],
        )
        mappings.append((s2t, t_count))

    student_to_teachers, teacher_usage_count = _merge_overlap_mappings(mappings, len(student_ids), len(teacher_ids))

    n_format = sum(1 for t_indices in student_to_teachers if len(t_indices) == 0)
    n_content = len(student_to_teachers) - n_format

    print(f"Summary: {len(student_ids)} student tokens, {len(teacher_ids)} teacher tokens")
    print(f"         {n_content} content tokens, {n_format} format tokens")

    aligned_sum = sum(lp for lp in aligned if lp != 0.0)
    teacher_content_sum = sum(lp for lp, usage in zip(teacher_logprobs, teacher_usage_count, strict=False) if usage > 0)

    n_perfect = 0
    for t_indices in student_to_teachers:
        if len(t_indices) > 0 and len(t_indices) == 1 and teacher_usage_count[t_indices[0]] == 1:
            n_perfect += 1

    perfect_pct = 100.0 * n_perfect / n_content if n_content > 0 else 0.0

    print(f"\nLogprob mass: teacher content={teacher_content_sum:.4f}, aligned sum={aligned_sum:.4f}, diff={abs(teacher_content_sum - aligned_sum):.6f}")
    print(f"Perfect 1:1 transfers: {n_perfect}/{n_content} ({perfect_pct:.1f}% of content tokens)")
