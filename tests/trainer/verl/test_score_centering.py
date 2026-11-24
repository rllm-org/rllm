import math

import pytest
import torch

from rllm.trainer.verl.score_centering import candidate_log_probs, centered_token_loss


@pytest.mark.parametrize("chunk_size", [1, 3, 128])
def test_chunked_gather_matches_full_vocabulary_gradient(chunk_size):
    torch.manual_seed(4)
    logits = torch.randn(7, 11, requires_grad=True)
    original = logits.detach().clone()
    labels = torch.randint(0, 11, (7,))
    ids = torch.stack([torch.randperm(11)[:4] for _ in range(7)])
    weights = torch.randn(7, 6)
    label_lp, head_lp, tail_lp = candidate_log_probs(logits, labels, ids, chunk_size)
    actual = torch.cat((label_lp[:, None], head_lp, tail_lp[:, None]), -1)
    logp = logits.log_softmax(-1)
    in_head = torch.zeros(7, 11, dtype=torch.bool).scatter(-1, ids, True)
    expected_tail = logp.masked_fill(in_head, -torch.inf).logsumexp(-1)
    expected = torch.cat((logp.gather(-1, labels[:, None]), logp.gather(-1, ids), expected_tail[:, None]), -1)
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad((actual * weights).sum(), logits)[0]
    ge = torch.autograd.grad((expected * weights).sum(), logits)[0]
    torch.testing.assert_close(ga, ge)
    torch.testing.assert_close(logits, original)


def sc_loss(logits, samples, sample_q, head_ids, head_q, advantage, cap):
    """Per-sample SC loss through the real chunked gather."""
    n = len(samples)
    expanded = logits.expand(n, -1)
    sample_lp, head_lp, tail_lp = candidate_log_probs(expanded, samples, head_ids.expand(n, -1), 2)
    losses, _, _ = centered_token_loss(sample_lp, sample_q.log(), head_lp, head_q.log().expand(n, -1), tail_lp, advantage, cap)
    return losses


def expected_sc_gradient(logits, q, advantage, cap, head_ids=None):
    """E_{y~q}[grad loss(y)] by enumerating every token of a small vocabulary."""
    samples = torch.arange(len(q))
    head_ids = samples if head_ids is None else head_ids
    losses = sc_loss(logits, samples, q, head_ids, q[head_ids], advantage, cap)
    return torch.autograd.grad((q * losses).sum(), logits)[0]


@pytest.mark.parametrize("cap", [None, 0.7, 2.0])
def test_exact_constant_signal_has_zero_expected_gradient(cap):
    logits = torch.tensor([0.3, -0.7, 1.1, 0.0], requires_grad=True)
    q = torch.tensor([0.4, 0.1, 0.2, 0.3])
    gradient = expected_sc_gradient(logits, q, torch.ones(4), cap)
    torch.testing.assert_close(gradient, torch.zeros(4), atol=2e-7, rtol=0)


@pytest.mark.parametrize("cap", [None, 0.7, 2.0])
def test_exact_gradient_matches_independent_enumeration_and_constant_shift(cap):
    logits = torch.tensor([0.3, -0.7, 1.1, 0.0], requires_grad=True)
    q = torch.tensor([0.4, 0.1, 0.2, 0.3])
    advantages = torch.tensor([0.5, -0.8, 1.4, 0.1])
    actual = expected_sc_gradient(logits, q, advantages, cap)
    shifted = expected_sc_gradient(logits, q, advantages + 3.7, cap)
    p = logits.detach().softmax(-1)
    weights = torch.ones(4) if cap is None else (p / q).clamp(max=cap)
    scores = torch.eye(4) - p.unsqueeze(0)
    center = (q[:, None] * weights[:, None] * scores).sum(0)
    reference = -(q[:, None] * advantages[:, None] * (weights[:, None] * scores - center)).sum(0)
    torch.testing.assert_close(actual, reference, atol=3e-7, rtol=1e-5)
    torch.testing.assert_close(actual, shifted, atol=5e-7, rtol=1e-5)


@pytest.mark.parametrize("cap", [None, 0.7, 2.0])
def test_head_tail_matches_declared_proportional_tail_model(cap):
    logits = torch.tensor([0.3, -0.7, 1.1, 0.0], requires_grad=True)
    p = logits.detach().softmax(-1)
    q = torch.cat((torch.tensor([0.35, 0.25]), 0.4 * p[2:] / p[2:].sum()))
    advantages = torch.tensor([0.5, -0.8, 1.4, 0.1])
    exact = expected_sc_gradient(logits, q, advantages, cap)
    approximate = expected_sc_gradient(logits, q, advantages, cap, torch.tensor([0, 1]))
    torch.testing.assert_close(approximate, exact, atol=2e-7, rtol=1e-5)


@pytest.mark.parametrize("cap", [None, 2.0])
@pytest.mark.parametrize("sharpness", [14.0, 20.0, 30.0])
def test_confident_trainer_tail_is_stable_in_fp32(cap, sharpness):
    """Trainer tail mass far below fp32 resolution of 1 - sum_H p; fp64 reference."""
    torch.manual_seed(0)
    vocab, k = 2000, 32
    z = torch.randn(vocab, dtype=torch.float64)
    z[:k] += sharpness
    sampler = z.clone()
    sampler[k:] += 6.0  # sampler keeps materially more tail mass than the trainer
    logq = sampler.log_softmax(-1)
    head = logq.argsort(descending=True)[:k]
    y = head[3]

    ref_logits = z.clone().requires_grad_()
    logp = ref_logits.log_softmax(-1)
    p = logp.detach().exp()
    tail = torch.ones(vocab, dtype=torch.bool)
    tail[head] = False
    q_model = p * (1 - logq[head].exp().sum()) / p[tail].sum()
    q_model[head] = logq[head].exp()
    w = (lambda r: torch.ones_like(r)) if cap is None else (lambda r: r.clamp(max=cap))
    reference = -(w(p[y] / logq[y].exp()) * logp[y] - (q_model * w(p / q_model) * logp).sum())
    expected = torch.autograd.grad(reference, ref_logits)[0]

    logits = z.float().requires_grad_()
    loss = sc_loss(logits[None], y[None], logq[y].exp().float()[None], head, logq[head].exp().float(), torch.ones(1), cap)
    actual = torch.autograd.grad(loss.sum(), logits)[0]
    assert torch.isfinite(loss).all()
    assert ((actual.double() - expected).norm() / expected.norm()) < 1e-5


def test_q_and_advantage_are_detached_and_padding_slots_are_safe():
    logits = torch.tensor([[0.3, -0.7, 1.1, 0.0]], requires_grad=True)
    head_logq = torch.tensor([[0.6, 0.4, 0.0]]).log().requires_grad_()
    advantage = torch.ones(1, requires_grad=True)
    sample_lp, head_lp, tail_lp = candidate_log_probs(logits, torch.tensor([0]), torch.tensor([[0, 2, 3]]))
    loss, _, _ = centered_token_loss(sample_lp, torch.tensor([0.6]).log(), head_lp, head_logq, tail_lp, advantage, 2)
    loss.sum().backward()
    assert head_logq.grad is None and advantage.grad is None
    assert torch.isfinite(logits.grad).all()


def test_head_covering_vocabulary_has_finite_loss_and_gradient():
    logits = torch.tensor([[0.3, -0.7, 1.1, 0.0]], requires_grad=True)
    q = torch.tensor([[0.4, 0.1, 0.2, 0.3]])
    sample_lp, head_lp, tail_lp = candidate_log_probs(logits, torch.tensor([2]), torch.arange(4)[None])
    loss, _, _ = centered_token_loss(sample_lp, q[:, 2].log(), head_lp, q.log(), tail_lp, torch.ones(1))
    loss.sum().backward()
    assert torch.isfinite(loss).all() and abs(loss.item()) < 10
    assert torch.isfinite(logits.grad).all()


def test_overfull_sampler_head_is_rejected():
    with pytest.raises(ValueError, match="head probability mass"):
        centered_token_loss(torch.tensor([-0.1]), torch.tensor([-0.1]), torch.tensor([[-0.1, -0.1]]), torch.tensor([[math.log(0.6), math.log(0.6)]]), torch.tensor([-3.0]), torch.ones(1))
