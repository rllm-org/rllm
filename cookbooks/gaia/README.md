# GAIA — `rllm eval gaia`

A minimal **search + browse ReAct agent** for the GAIA benchmark, matching how
frontier models are evaluated on GAIA (a multi-tool ReAct loop — cf. HuggingFace
[Open Deep Research](https://huggingface.co/blog/open-deep-research) and Princeton
HAL, the two reference GAIA scaffolds). Tools: rLLM's Tavily search + extract.

Scope: the **text-only** GAIA subset (`gaia_transform` skips file-attachment
tasks). Code-execution / file / multimodal tools are the follow-up for harder levels.

## Setup

```bash
export TAVILY_API_KEY=...      # web search + page extract (single key)
export HF_TOKEN=...            # GAIA is a gated HF dataset — accept terms at
                              #   https://huggingface.co/datasets/gaia-benchmark/GAIA
rllm model setup               # configure your model provider (e.g. an OpenAI model)
```

## Run

```bash
# from the repo root (so `cookbooks` is importable)
PYTHONPATH=. rllm eval gaia --agent cookbooks.gaia.agent:agent --max-examples 5
```

`gaia` resolves to the `validation` split (public answers) and is scored by
`gaia_reward_fn` (GAIA's official quasi-exact-match). Start with a small
`--max-examples` to sanity-check the pipeline before a full run.

## Notes

- `default_agent` for `gaia` (and `browsecomp`/`hle_search`) is `"search"`, which
  is **not** a built-in agent — pass `--agent cookbooks.gaia.agent:agent`
  explicitly, or register a search agent. Promoting this to a built-in `search`
  agent would fix all three benchmarks at once (open question for maintainers).
- Accuracy depends on the model + tools; the contribution here is the native
  integration (dataset + official scorer + a runnable search/browse harness),
  not a leaderboard number.
