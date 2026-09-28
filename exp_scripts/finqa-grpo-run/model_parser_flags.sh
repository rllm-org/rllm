#!/usr/bin/env bash
# Resolve the vLLM tool-call / reasoning parser flags for a model by INSPECTING
# ITS CHAT TEMPLATE, not by matching its name.
#
# Why this exists: FinQA passes tools=TOOL_SPECS to chat.completions.create, so
# the server must parse tool calls out of the raw text. A mismatched parser does
# not error -- it silently yields zero tool calls, the agent never queries SQL,
# and the run looks like a capability failure instead of a config bug.
#
# Two incompatible formats are in play here:
#   Hermes JSON  <tool_call>{"name": ..., "arguments": {...}}</tool_call>   Qwen3-4B-Instruct-2507
#   Qwen3 XML    <tool_call><function=N><parameter=P>v</parameter></...>    Qwen3.5-4B
#
# Unrecognised templates are a hard failure: guessing is exactly the mode that
# produces a silent zero.
resolve_parser_flags() {
  local model="${1:?resolve_parser_flags MODEL_PATH}"
  local tpl
  tpl=$(python3 - "$model" <<'PY'
import json, sys, pathlib
m = pathlib.Path(sys.argv[1])
p = m / "chat_template.jinja"
if p.exists():
    print(p.read_text()); raise SystemExit
tc = m / "tokenizer_config.json"
if tc.exists():
    print(json.loads(tc.read_text()).get("chat_template", "") or "")
PY
)
  if [[ -z "$tpl" ]]; then
    echo "[parser-flags] FATAL: no chat template found under $model" >&2
    echo "[parser-flags] expected: chat_template.jinja, or chat_template inside tokenizer_config.json" >&2
    return 2
  fi

  PARSER_FLAGS=(--enable-auto-tool-choice)
  local tool_fmt=""
  if [[ "$tpl" == *"<function="* ]]; then
    tool_fmt=qwen3_coder
  elif [[ "$tpl" == *'{"name": '* || "$tpl" == *'{\"name\": '* ]]; then
    tool_fmt=hermes
  fi
  if [[ -z "$tool_fmt" ]]; then
    echo "[parser-flags] FATAL: cannot identify the tool-call format of $model" >&2
    echo "[parser-flags] the template contains neither '<function=' (qwen3_coder) nor '{\"name\": ' (hermes)" >&2
    echo "[parser-flags] fix: inspect the template and extend resolve_parser_flags rather than guessing" >&2
    return 2
  fi
  PARSER_FLAGS+=(--tool-call-parser "$tool_fmt")

  # Thinking models emit <think>...</think>; without a reasoning parser that text
  # stays in message.content and pollutes whatever the flow reads as the answer.
  local reasoning="none"
  if [[ "$tpl" == *"<think>"* ]]; then
    # The reasoning parser assumes the output STARTS inside a think block and
    # ends it with </think>. When thinking is forced off the prompt already
    # closed that block, the output is pure content, </think> never arrives, and
    # the parser drops everything -- content=None on every turn. Measured: 0/522
    # on FinQA val before this was fixed. So the two are mutually exclusive.
    if [[ "${RLLM_DISABLE_THINKING:-0}" != 1 ]]; then
      reasoning=qwen3
      PARSER_FLAGS+=(--reasoning-parser qwen3)
    fi
    # RLLM_DISABLE_THINKING=1 measures a thinking model in the same non-reasoning
    # regime as a plain instruct model. Done by deriving a template that always
    # emits a closed <think></think> pair, so the assistant turn starts after it.
    if [[ "${RLLM_DISABLE_THINKING:-0}" == 1 ]]; then
      local cache="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.derived_templates"
      local derived="$cache/$(basename "$model")-nothink.jinja"
      mkdir -p "$cache"
      python3 - "$model" "$derived" <<'PY' || return 2
import json, pathlib, re, sys
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
p = src / "chat_template.jinja"
tpl = p.read_text() if p.exists() else json.loads((src / "tokenizer_config.json").read_text()).get("chat_template", "")
pat = re.compile(r"\{%-?\s*if\s+enable_thinking\b.*?\{%-?\s*endif\s*-?%\}", re.S)
hits = pat.findall(tpl)
if len(hits) != 1:
    sys.exit(f"[parser-flags] FATAL: expected exactly 1 enable_thinking block, found {len(hits)}")
out = pat.sub("{{- '<think>\\n\\n</think>\\n\\n' }}", tpl)
if "<think>" not in out:
    sys.exit("[parser-flags] FATAL: derived template lost its think markers")
dst.write_text(out)
PY
      PARSER_FLAGS+=(--chat-template "$derived")
      reasoning="forced-off"
    fi
  fi

  echo "[parser-flags] $(basename "$model"): tool=$tool_fmt reasoning=$reasoning" >&2
  return 0
}
