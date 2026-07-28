"""Render the static model-metric figure used by the GLM-5.2 RL report."""

from __future__ import annotations

from html import escape
from pathlib import Path

STEPS = list(range(1, 13))
TRAIN_SUCCESS = [
    65 / 128 * 100,
    69 / 128 * 100,
    70 / 128 * 100,
    91 / 128 * 100,
    71 / 128 * 100,
    95 / 128 * 100,
    85 / 127 * 100,
    95 / 128 * 100,
    86 / 128 * 100,
    82 / 128 * 100,
    69 / 128 * 100,
    64 / 128 * 100,
]
ENTROPY = [0.3438, 0.2454, 0.1726, 0.1332, 0.1243, 0.0975, 0.0837, 0.0778, 0.0664, 0.0546, 0.0488, 0.0387]
KLD = [0.00374, 0.00547, 0.00641, 0.00708, 0.00724, 0.00631, 0.00625, 0.00576, 0.00545, 0.00504, 0.00419, 0.00406]
PPO_CLIPPED = [4.36, 5.01, 5.18, 4.86, 4.64, 4.17, 3.85, 3.65, 3.58, 3.41, 3.03, 3.00]
EVAL_STEPS = [0, 3, 6, 9, 12]
EVAL_PASSED = [44, 48, 51, 50, 37]
EVAL_RATE = [passed / 89 * 100 for passed in EVAL_PASSED]

WIDTH = 1200
HEIGHT = 760
OUTPUT = Path(__file__).with_name("glm52_opencode_medium_metrics.svg")


def _xy(panel: tuple[int, int, int, int], step: float, value: float, y_min: float, y_max: float) -> tuple[float, float]:
    x, y, width, height = panel
    plot_x = x + 62
    plot_y = y + 38
    plot_width = width - 82
    plot_height = height - 86
    return (
        plot_x + step / 12 * plot_width,
        plot_y + (y_max - value) / (y_max - y_min) * plot_height,
    )


def _panel(
    parts: list[str],
    panel: tuple[int, int, int, int],
    title: str,
    y_min: float,
    y_max: float,
    y_ticks: list[float],
    tick_format: str,
) -> None:
    x, y, width, height = panel
    plot_x = x + 62
    plot_y = y + 38
    plot_width = width - 82
    plot_height = height - 86
    parts.append(f'<text class="panel-title" x="{x}" y="{y + 17}">{escape(title)}</text>')
    for tick in y_ticks:
        _, tick_y = _xy(panel, 0, tick, y_min, y_max)
        parts.append(f'<line class="grid" x1="{plot_x}" y1="{tick_y:.1f}" x2="{plot_x + plot_width}" y2="{tick_y:.1f}"/>')
        parts.append(f'<text class="tick" x="{plot_x - 10}" y="{tick_y + 4:.1f}" text-anchor="end">{format(tick, tick_format)}</text>')
    for step in (0, 3, 6, 9, 12):
        tick_x, _ = _xy(panel, step, y_min, y_min, y_max)
        parts.append(f'<line class="grid epoch" x1="{tick_x:.1f}" y1="{plot_y}" x2="{tick_x:.1f}" y2="{plot_y + plot_height}"/>')
        parts.append(f'<text class="tick" x="{tick_x:.1f}" y="{plot_y + plot_height + 22}" text-anchor="middle">{step}</text>')
    parts.append(f'<line class="axis" x1="{plot_x}" y1="{plot_y + plot_height}" x2="{plot_x + plot_width}" y2="{plot_y + plot_height}"/>')
    parts.append(f'<line class="axis" x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_height}"/>')
    parts.append(f'<text class="axis-label" x="{plot_x + plot_width / 2:.1f}" y="{y + height - 3}" text-anchor="middle">Optimizer step</text>')


def _series(
    parts: list[str],
    panel: tuple[int, int, int, int],
    steps: list[int],
    values: list[float],
    y_min: float,
    y_max: float,
    css_class: str,
) -> None:
    points = [_xy(panel, step, value, y_min, y_max) for step, value in zip(steps, values, strict=True)]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    parts.append(f'<polyline class="series {css_class}" points="{path}"/>')
    for x, y in points:
        parts.append(f'<circle class="point {css_class}" cx="{x:.1f}" cy="{y:.1f}" r="3.8"/>')


def render() -> str:
    panels = {
        "reward": (64, 82, 510, 280),
        "entropy": (644, 82, 510, 280),
        "kld": (64, 420, 510, 280),
        "ppo": (644, 420, 510, 280),
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        '<title id="title">GLM-5.2 OpenCode reinforcement-learning metrics over twelve optimizer steps</title>',
        '<desc id="desc">Four plots show training and Terminal-Bench success, policy entropy, trainer-to-rollout KLD, and PPO clipping fraction.</desc>',
        """<style>
            text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #24292f; }
            .chart-title { font-size: 20px; font-weight: 600; }
            .panel-title { font-size: 15px; font-weight: 600; }
            .tick, .note { font-size: 11px; fill: #57606a; }
            .axis-label, .legend { font-size: 12px; fill: #57606a; }
            .axis { stroke: #8c959f; stroke-width: 1; }
            .grid { stroke: #d8dee4; stroke-width: 1; }
            .epoch { stroke-dasharray: 3 4; }
            .series { fill: none; stroke-width: 2.5; stroke-linejoin: round; stroke-linecap: round; }
            .point { stroke: none; }
            .series.train { stroke: #0969da; }
            .series.eval { stroke: #bf8700; }
            .series.entropy { stroke: #8250df; }
            .series.kld { stroke: #1a7f37; }
            .series.ppo { stroke: #cf222e; }
            .point.train { fill: #0969da; }
            .point.eval { fill: #bf8700; }
            .point.entropy { fill: #8250df; }
            .point.kld { fill: #1a7f37; }
            .point.ppo { fill: #cf222e; }
            @media (prefers-color-scheme: dark) {
                text { fill: #f0f6fc; }
                .tick, .note, .axis-label, .legend { fill: #8c959f; }
                .axis { stroke: #6e7681; }
                .grid { stroke: #30363d; }
                .series.train { stroke: #58a6ff; }
                .series.eval { stroke: #d29922; }
                .series.entropy { stroke: #d2a8ff; }
                .series.kld { stroke: #3fb950; }
                .series.ppo { stroke: #ff7b72; }
                .point.train { fill: #58a6ff; }
                .point.eval { fill: #d29922; }
                .point.entropy { fill: #d2a8ff; }
                .point.kld { fill: #3fb950; }
                .point.ppo { fill: #ff7b72; }
            }
        </style>""",
        '<text class="chart-title" x="64" y="40">GLM-5.2 OpenCode: model metrics across 12 optimizer steps</text>',
    ]

    reward_panel = panels["reward"]
    _panel(parts, reward_panel, "Success rate (%)", 35, 80, [40, 50, 60, 70, 80], ".0f")
    _series(parts, reward_panel, STEPS, TRAIN_SUCCESS, 35, 80, "train")
    _series(parts, reward_panel, EVAL_STEPS, EVAL_RATE, 35, 80, "eval")
    parts.extend(
        [
            '<line class="series train" x1="368" y1="102" x2="390" y2="102"/><text class="legend" x="398" y="106">train success</text>',
            '<line class="series eval" x1="368" y1="121" x2="390" y2="121"/><text class="legend" x="398" y="125">TB2.1 pass@1</text>',
        ]
    )
    for step, passed, rate in zip(EVAL_STEPS, EVAL_PASSED, EVAL_RATE, strict=True):
        x, y = _xy(reward_panel, step, rate, 35, 80)
        anchor = "start" if step == 0 else "end" if step == 12 else "middle"
        offset = 5 if step == 0 else -5 if step == 12 else 0
        parts.append(f'<text class="note" x="{x + offset:.1f}" y="{y - 9:.1f}" text-anchor="{anchor}">{passed}/89</text>')

    entropy_panel = panels["entropy"]
    _panel(parts, entropy_panel, "Policy entropy", 0, 0.36, [0, 0.1, 0.2, 0.3], ".1f")
    _series(parts, entropy_panel, STEPS, ENTROPY, 0, 0.36, "entropy")
    for step in (1, 12):
        x, y = _xy(entropy_panel, step, ENTROPY[step - 1], 0, 0.36)
        parts.append(f'<text class="note" x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle">{ENTROPY[step - 1]:.3f}</text>')

    kld_panel = panels["kld"]
    _panel(parts, kld_panel, "Trainer ↔ rollout KLD", 0, 0.008, [0, 0.002, 0.004, 0.006, 0.008], ".3f")
    _series(parts, kld_panel, STEPS, KLD, 0, 0.008, "kld")
    peak = max(range(len(KLD)), key=KLD.__getitem__)
    x, y = _xy(kld_panel, STEPS[peak], KLD[peak], 0, 0.008)
    parts.append(f'<text class="note" x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle">max {KLD[peak]:.5f}</text>')

    ppo_panel = panels["ppo"]
    _panel(parts, ppo_panel, "PPO clipped tokens (%)", 0, 6, [0, 2, 4, 6], ".0f")
    _series(parts, ppo_panel, STEPS, PPO_CLIPPED, 0, 6, "ppo")
    for step in (1, 12):
        x, y = _xy(ppo_panel, step, PPO_CLIPPED[step - 1], 0, 6)
        parts.append(f'<text class="note" x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle">{PPO_CLIPPED[step - 1]:.2f}%</text>')

    parts.append('<text class="note" x="64" y="744">Evaluation uses one OpenCode rollout on each of the same 89 Terminal-Bench 2.1 tasks.</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    OUTPUT.write_text(render())
    print(OUTPUT)
