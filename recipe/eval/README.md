# rLLM evaluation (ref. harbor)

`rllm eval` -> SWE-bench Verified와 SWE-bench Pro 평가 방법

실행 방식은 두 가지:

| Way | `--agent` | Runtime |
| --- | --- | --- |
| **harbor** | `harbor:mini-swe-agent` 처럼 `harbor:` 접두어 | Harbor `Trial.run()` — 환경 빌드, 에이전트, 검증기를 Harbor가 모두 담당 |
| **native** | `mini-swe-agent`, `oracle` 처럼 접두어 없음 | rLLM `SandboxTaskHooks` + rLLM Harness + `ShellScriptEvaluator` |

두 방식 모두 `--sandbox-backend docker`(local Docker daemon) 사용

---

## 0. 환경 세팅 및 rLLM 배경 지식

### 0.1 Container

- base image: `pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel`. 
- 평가 시 agent sandbox/verifier sandbox는 이 container 안에서 host Docker daemon을 빌려 dood (docker out of docker)로 launch 
- 이에 따라 rllm을 사용할 Container를 launch할 때 세 가지 옵션 설정 필요:

  * `-v /var/run/docker.sock:/var/run/docker.sock` — 호스트 Docker daemon 공유
  * `--network host` — host 주소로 rLLM 모델 gateway에 접속 
      - native는 `host.docker.internal`, harbor는 docker0 gateway IP `172.17.0.1`
      - gateway는 rLLM 프로세스(=Container) 안에서 `0.0.0.0`에 바인드되므로, 이 Container가 host 네트워크에 있어야 그 주소를 gateway로 활용 가능. bridge 네트워크로 띄우면 agent container/verifier container에서 접근 불가.
  * `-v /path/to/shared:/path/to/shared` — Host shared directory 볼륨 마운트. rLLM integrated harbor를 통한 평가 수행 시, Harbor가 `$RLLM_HOME/harbor_trials/{trial}/verifier`를 task container에 bind-mount하고, 이는 host docker daemon이 resolve하므로 해당 경로가 host에도 **반드시 같은 위치**에 존재해야 접근 가능.

```bash
# Host 
docker run -d --name {CONTAINER_NAME} 
  --gpus all \
  --network host \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /raid/rllm-work:/raid/rllm-work \ # i.e. /path/to/shared
  -v /gpfs/home/exaone/.cache/huggingface:/root/.cache/huggingface # i.e. change to shared hf directory
  -v /path/to/rllm:/workspace/rllm \
  --shm-size 32g \
  pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel sleep infinity
docker exec -it {CONTAINER_NAME} bash
```

### 0.2 환경 설치

Container 안에는 Docker CLI와 compose 플러그인 설치 필수 (Harbor는 `docker compose`를 호출).
```
apt-get update && apt-get install curl
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
cd /workspace/rllm
uv venv --python 3.12
source .venv/bin/activate

uv pip install -e ".[harbor]"

apt-get update && apt-get install -y docker.io docker-compose-v2
uv add docker
```

### 0.3 저장 위치

rLLM은 데이터셋, 평가 결과, Harbor trial log를 `$RLLM_HOME`(기본 `~/.rllm`) 아래에 둔다. Host가 아닌 Container 내부에서 dood 형태로 실행하는 본 환경에서는 두 이유로 **host와 같은 경로로 마운트된 큰 볼륨**(위 예시의 `/raid/rllm-work`) 아래를 가리키게 한다.

1. 용량. 태스크 디렉토리 수백 개와 에피소드 JSON이 쌓이고, Container Root에 두면 Container와 함께 사라진다.
2. Harbor 방식은 `$RLLM_HOME/harbor_trials/...`를 task container에 bind-mount하는데, 이 mount는 host docker daemon이 해석한다. Container 안에만 있는 경로(`~/.rllm`, `/workspace/rllm`)를 주면 검증기의 reward 파일이 host에만 남아 모든 태스크가 `RewardFileNotFoundError`로 끝난다. 이 위치를 따로 바꾸는 설정은 없고, `RLLM_HOME`을 host와 공유하는 디렉토리 경로로 바꾸는 방법이 있다.

아래는 이 문서 전체에서 가정하는 값이다. 바꾸면 이전에 pull한 데이터셋과 결과가 다른 위치에 있어 없는 것처럼 보인다. 
프로젝트 내부에서 해당 경로를 고정해서 사용하면, rLLM 설정 (config.json), 프로젝트 내부 인원이 특정 벤치마크를 평가한 결과, SWE-Bench Verified Subset, rLLM을 통해 build한 학습 데이터 등 모든 것을 공유해서 사용할 수 있다.

```bash
export RLLM_HOME=/raid/rllm-work/rllm-home
```

`recipe/qwen3_5_swe_grpo/env.sh`를 `source`해도 같은 값이 잡힌다.

### 0.3.1 환경변수

| Var | Default | Why |
| --- | --- | --- |
| `RLLM_HOME` | `~/.rllm` | 데이터셋, 결과, `config.json`(모델 setup), Harbor Trial 등
| `RLLM_HARBOR_SESSION_TIMEOUT_S` | `900` | Harbor 방식 전용. Trial 1개(에이전트 + Verifier)의 상한. SWE-bench Pro의 Go 저장소는 verifier 단계만 15분 넘게 걸리므로, 기본값이면 oracle test조차 timeout으로 0점. `3000` 이상으로 설정 권장 |
| `RLLM_AGENT_IMAGE` | `auto` | native 방식 전용. `--agent-image` 플래그와 같은 값(`auto` / `skip` / `repo:tag`) |
| `HF_HOME` | `~/.cache/huggingface` | `swebench_pro` 빌더가 HF 데이터셋을 받는 곳 |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` 등 | 없으면 `empty` | proxy를 사용할 경우 실제 값 필요 X. export되어 있으면 task container 안으로 전달되므로, 신뢰하지 않는 task(에이전트가 임의 명령을 실행함)를 평가할 때는 shell에 실제 키 설정 X |
| `RLLM_API_KEY` | 없음 | `rllm login`을 했거나 이 변수가 있으면 평가 결과와 에피소드가 rLLM UI 서버로 업로드. 외부로 나가면 안 되는 결과라면 `--no-ui` 명시 |

※ Harbor Registry에서 받은 task directory는 `RLLM_HOME`이 아니라 **`~/.cache/harbor/tasks`**에 들어가고, `$RLLM_HOME/datasets/<name>/default.parquet`의 `task_path`가 그곳을 가리킨다. Container를 새로 만들면 이 cache가 사라져 parquet은 있는데 태스크 디렉토리가 없는 상태가 된다. 그때는 `rllm dataset pull harbor:<name>`을 다시 실행해야한다.

### 0.3.2 데이터셋 용어 정리: catalog, parquet, task directory

세 개념이 서로 연관되어 있다.

| 개념 | 실체 | 역할 |
| --- | --- | --- |
| **catalog** | `rllm/registry/datasets.json` (코드에 포함) | 데이터셋 이름 → 어디서 어떻게 받는지. `rllm dataset list`가 보여주는 것. `harbor:<이름>`은 이 파일에 없고 호출 시 Harbor 레지스트리에 물어 항목을 즉석에서 만든다 |
| **parquet (DatasetRegistry)** | `$RLLM_HOME/datasets/<이름>/<split>.parquet` + `registry.json` | rLLM의 표 형식 Dataset. **태스크당 한 행.** `rllm eval`/`rllm train`은 항상 이걸 통해 태스크 목록을 얻는다 |
| **task directory (Harbor 형식)** | `task.toml`, `instruction.md`, `environment/Dockerfile`, `tests/test.sh`, `solution/solve.sh` | **실제로 실행되는 것.** Harbor가 정한 형식이고 rLLM도 그대로 쓴다. 디렉토리 묶음 위에 `dataset.toml`(이름, split, 기본 에이전트)이 붙는다 |

parquet과 task directory의 관계가 핵심: 수학 문제(e.g. gsm8k)처럼 텍스트만 있는 데이터셋은 parquet 행 자체가 모델 입력으로 사용되는 instruction. SWE 태스크처럼 컨테이너가 필요한 데이터셋은 parquet 행이 포인터. 행의 `task_path`가 가리키는 디렉토리가 진짜 내용.

`rllm dataset pull`이 종류별로 하는 일:
- `rllm dataset pull`: CLI 명령으로, catalog 항목을 읽어 실제로 데이터를 받아오는 동작. 항목 종류에 따라 하는 일이 다르다.

| type | ex | pull | task directory | parquet row |
| --- | --- | --- | --- | --- |
| HF 데이터셋 | `gsm8k` | HF에서 행을 받아 transform 후 등록 | 없음 (eval 시 즉석 materialise) | Instruction 자체 |
| **builder** | `swebench_pro`, `rllm-swesmith`, `deepswe` | `builder` 함수를 `out_dir=$RLLM_HOME/datasets/<이름>`으로 호출. 원본에서 디렉토리 **생성** + `dataset.toml` + 등록 | `$RLLM_HOME/datasets/<이름>/<task_id>/` | 포인터 |
| **`harbor:<이름>`** | `harbor:swebench-verified`, `harbor:swebenchpro` | Harbor Registry Client로 완성된 디렉토리를 다운로드 + 등록 | `~/.cache/harbor/tasks/<해시>/<task_id>/` (`RLLM_HOME` 무관) | 포인터 |

`rllm dataset pull swebench_pro`와 `python -m rllm.data.swebench_pro_builder ...`는 **같은 함수**(`build_benchmark`)를 호출하고 결과물 형식도 같다. pull은 출력 위치가 `$RLLM_HOME/datasets/swebench_pro`로 고정되고 731개 전체를 만들며 parquet도 등록한다. 모듈 직접 호출은 `--out-dir`, `--task-ids`, `--limit`을 받고 parquet은 등록하지 않는다. 그래서 subset은 직접 호출로 만든다.

`rllm eval <X>`가 X를 해석하는 순서:

1. 경로(`./`, `/`로 시작) → 그 디렉토리의 `dataset.toml`/`task.toml`을 `BenchmarkLoader`가 읽는다. parquet 사용 X.
2. `harbor:<이름>` → 레지스트리에 물어 catalog 항목 합성 → parquet 없으면 pull → 행의 `task_path`로 Task 생성.
3. Pure 이름 → `$RLLM_HOME/datasets/<X>/dataset.toml`이 있고 에이전트가 `harbor:*`가 아니면 1번처럼 디렉토리를 직접 읽는다. 아니면 catalog → parquet(없으면 pull).

```
catalog (datasets.json)           "이름 → 받는 방법"
        │  rllm dataset pull
        ▼
task directory                     실제 내용 (task.toml, Dockerfile, test.sh ...)
   ├─ harbor:*  → ~/.cache/harbor/tasks/...
   └─ builder      → $RLLM_HOME/datasets/<이름>/...
        │  등록
        ▼
parquet ($RLLM_HOME/datasets/<이름>/default.parquet)   태스크당 한 행, task_path 포인터
        │  rllm eval / rllm train
        ▼
실행 결과
   ├─ $RLLM_HOME/eval_results/<실행>/       
   └─ $RLLM_HOME/harbor_trials/<트라이얼>/  
```

`default_verl.parquet`는 같은 행을 verl 학습기 형식으로 한 번 더 저장한 사본이다. 평가에서는 무관하다.

### 0.4 모델 연결 — 두 가지

`rllm eval`은 모델 호출을 항상 **rLLM model gateway**를 거쳐 보낸다. gateway 뒤에 무엇을 두느냐에 따라 두 가지로 나뉜다.

**(A) Proprietary 모델 (OpenAI, Anthropic, Gemini 등)**

provider & API 키를 등록한다. 이후 `rllm eval`이 LiteLLM proxy를 자동으로 띄워 routing한다.

```bash
rllm model setup            # provider(openai / anthropic / gemini / openrouter / ...), API key, model 선택
rllm model show             # 현재 설정 확인
```

이후 명령에서는 `--model`만 바꾸면 된다. 생략하면 setup에서 고른 모델을 사용한다.

```bash
rllm eval <benchmark> --agent <agent> --sandbox-backend docker --model gpt-5.5
```

**How the API KEY is used during eval** setup이 받은 키는 `$RLLM_HOME/config.json`에 저장된다. `rllm eval`은 그 키로 local LiteLLM
proxy를 띄우고, gateway가 proxy로 요청을 넘긴다. task container 안의 에이전트에는 실제 키가 아니라 placeholder(`sk-rllm-gateway` 또는 `empty`)가 들어간다. 즉 **평가 중 실제 키는 rLLM 프로세스 밖으로 나가지 않는다.**

```
container(에이전트, placeholder 키) → gateway → LiteLLM proxy(실제 키) → api.openai.com
```

그래서 키는 setup에만 주고, shell에는 export하지 않는 것을 권장한다. `OPENAI_API_KEY`가 shell에 export되어 있으면 setup이 그 값을 기본값으로 읽어 편하지만, 그 상태로 `rllm eval`을 실행하면 placeholder 대신 **실제 키를 task container에 넣는다**(동작에는 차이가 없지만 에이전트가 임의 명령을 실행하는 container에 키가 노출된다).

setup을 마친 뒤 `unset OPENAI_API_KEY` 하면 된다. `config.json`은 평문이므로 공유 볼륨에 둘 때는 `chmod 600 $RLLM_HOME/config.json`.

예외가 하나 있다. provider를 `custom`(OpenAI compatible endpoint)으로 잡고 키를 입력하면 proxy를 띄우지 않고 `rllm eval`이 그 키를 자기 환경변수 `OPENAI_API_KEY`에 넣는다(`rllm/cli/eval.py`). 그러면 rLLM이 그 값을 읽어 **실제 키가 task container에 들어간다.** 키를 설정한 vLLM을 `custom`으로 평가할 때 해당한다.

proxy 시작에 실패하면 `uv pip install "litellm[proxy]"`.

**(B) vLLM / SGLang으로 서빙한 로컬 모델**

OpenAI compatible endpoint를 `--base-url`로 직접 준다. 이때 `--model`은 필수이며, 서버의 `served-model-name`과 정확히 같아야 한다.

```bash
# vLLM
vllm serve Qwen/Qwen3-8B --port 8000 --served-model-name Qwen/Qwen3-8B --max-model-len 65536

# SGLang
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000 --served-model-name Qwen/Qwen3-8B

rllm eval <benchmark> --agent <agent> --sandbox-backend docker \
    --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-8B
```

`--network host`로 띄웠으므로 서빙 서버가 호스트나 다른 컨테이너에 있어도 `127.0.0.1:<port>`로 닿는다.

모델명 규칙: mini-swe-agent 계열 에이전트는 `provider/model` 형식을 요구해서 rLLM이 접두어를 자동으로 붙인다. `qwen`, `deepseek`, `gpt-*`, 알 수 없는 이름은 `openai/`가 붙어 OpenAI 호환 경로로 나가고, `claude`가 들어간 이름은 `anthropic/`이 붙는다. `Qwen/Qwen3-8B`처럼 HF 조직명이 앞에 오는 이름도 `openai/` 접두어가 붙는다(`openai/Qwen/Qwen3-8B`). litellm이 `openai/`를 떼고 보내므로 서버에는 `Qwen/Qwen3-8B` 그대로 도착한다.

### 0.5 결과 확인

결과는 `$RLLM_HOME/eval_results/<benchmark>_<model>_<timestamp>/`에 `meta.json`, `results.json`, `episodes/`로
남는다. 콘솔에는 Accuracy, Errors, `--attempts`를 썼다면 pass@k가 찍힌다.

```bash
rllm view <run-id>          # 에피소드를 브라우저에서 훑어보기
```

`Errors`는 모델이 못 푼 것이 아니라 **인프라 실패**(이미지 빌드 실패, timeout, gateway 접속 불가 등)다. 0이 아니면 먼저 원인을 잡는다. Console에 처음 다섯 개가 찍히고 나머지는 `results.json`에 있다.

---

## 1. Harbor 방식

| 벤치마크 | 이름 | 태스크 수 | 베이스 이미지 |
| --- | --- | --- | --- |
| SWE-bench Verified | `harbor:swebench-verified` | 500 | `swebench/sweb.eval.x86_64.<repo>_1776_<instance>` |
| SWE-bench Pro | `harbor:swebenchpro` | 731 | 태스크별 사전 빌드 이미지 |

첫 실행 때 task directory(only text, not image)를 자동으로 내려받아 `$RLLM_HOME/datasets/<name>/`에 등록한다. 미리 받아두려면:

```bash
rllm dataset pull harbor:swebench-verified
rllm dataset pull harbor:swebenchpro
```

Docker 이미지는 태스크가 실행될 때 Harbor가 `environment/Dockerfile`을 빌드하면서 `FROM` 이미지를 pull한다.
Verified 500개 전체의 베이스 이미지는 약 2TB다. 전체 평가 전에 디스크를 확인한다.

### 1.1 전체 평가

```bash
export RLLM_HARBOR_SESSION_TIMEOUT_S=3600   

# (A) proprietary
rllm eval harbor:swebench-verified --agent harbor:mini-swe-agent --sandbox-backend docker \
    --model gpt-5.5 --sandbox-concurrency 8
rllm eval harbor:swebenchpro       --agent harbor:mini-swe-agent --sandbox-backend docker \
    --model gpt-5.5 --sandbox-concurrency 8

# (B) vLLM / SGLang
rllm eval harbor:swebench-verified --agent harbor:mini-swe-agent --sandbox-backend docker \
    --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-8B --sandbox-concurrency 8
rllm eval harbor:swebenchpro       --agent harbor:mini-swe-agent --sandbox-backend docker \
    --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-8B --sandbox-concurrency 8
```

`--agent`를 생략하면 Harbor 데이터셋의 기본값인 `harbor:mini-swe-agent`가 쓰인다. 다른 Harbor 스캐폴드도 같은 형식으로 고른다: `harbor:terminus-2`, `harbor:swe-agent`, `harbor:openhands`, `harbor:codex`, `harbor:claude-code`, `harbor:oracle`(정답 패치만 적용, LLM 미호출) 등.

### 1.2 Subset 평가

Harbor 경로에서 태스크를 고르는 수단은 **Index**다. Index는 `$RLLM_HOME/datasets/<name>/default.parquet`의 행 순서다.

```bash
# 앞에서 10개
rllm eval harbor:swebench-verified --agent harbor:mini-swe-agent --sandbox-backend docker \
    --max-examples 10 --model gpt-5.5

# 특정 Index. '0', '3,7,12', '0-9' 형식
rllm eval harbor:swebench-verified --agent harbor:mini-swe-agent --sandbox-backend docker \
    --task-indices 0-4,17,42 --model gpt-5.5
```

instance_id로 고르고 싶으면 id → Index를 먼저 뽑는다.

```bash
python - <<'PY'
import os, pandas as pd
name = "swebench-verified"                     #  or "swebenchpro"
want = {"django__django-11265", "sympy__sympy-16792"}
df = pd.read_parquet(f"{os.environ['RLLM_HOME']}/datasets/{name}/default.parquet")
print(",".join(str(i) for i, t in enumerate(df["task_id"]) if t in want))
PY
```

출력값을 `--task-indices`에 그대로 넣는다. Index는 데이터셋을 다시 pull해도 바뀌지 않는다 (registry manifest 순서).

같은 태스크를 여러 번 평가해 pass@k를 보려면 `--attempts 4`처럼 준다. 이때는 온도를 0보다 크게 주는 것을 권장한다(`--temperature 0.7`).

#### 1.2.1 subset을 Harbor harness로 평가

Harbor harness(`harbor:*`)가 필요로 하는 것은 데이터셋 이름이 `harbor:`로 시작하는 것이 아니라, **parquet 행에 `task_path`가 있는 것**이다(`HarborRuntime`은 `task.metadata["task_path"]`로 Trial을 만든다). 
그래서 native와 같은 방법으로 task directory를 `$RLLM_HOME` 아래에 복사해 등록한 subset도 Harbor harness로 돌릴 수 있다. 
단, 1.2 방법으로 충분히 같은 일을 할 수 있기에 권장하지 않는다. 별도의 이유로 subset을 분리하고 싶은 경우에만 참고로 사용한다.

조건: 
- **parquet 행에 task_path 추가** 
- **`--evaluator harbor_reward_fn` 명시**: `harbor:` 이름이 아니면 CLI가 catalog에서 Evaluator를 찾지 못해 멈추는데, 이 플래그가 그 분기를 지나게 한다.
- **사본의 `task.toml`에서 `docker_image` 줄 삭제**: 이 값이 있으면 Harbor는 Dockerfile을 빌드하지 않고 베이스 이미지를 그대로 `sh -c "sleep infinity"`로 띄운다.

예시:
```bash
python - <<'PY'
import os, re, shutil, pandas as pd
from pathlib import Path
from rllm.data import DatasetRegistry
home = Path(os.environ["RLLM_HOME"]); name = "swebench_pro_mine"
want = {"instance_qutebrowser__qutebrowser-e64622cd2df5b521342cf4a62e0d4cb8f8c9ae5a-v363c8a7e5ccdf6968fc7ab84a2053ac78036691d"}
src = pd.read_parquet(home / "datasets/swebench_pro/test.parquet")        # harbor 소스면 datasets/swebenchpro/default.parquet, 열 이름은 task_id
out = home / "datasets" / name; out.mkdir(parents=True, exist_ok=True)
rows = []
for r in src.itertuples():
    tid = r.id
    if tid not in want: continue
    dst = out / tid
    if not dst.exists(): shutil.copytree(r.task_path, dst)
    toml = dst / "task.toml"
    toml.write_text(re.sub(r"^docker_image\s*=.*\n", "", toml.read_text(), flags=re.M))   # Harbor가 Dockerfile을 빌드할 수 있도록
    rows.append({"id": tid, "task_id": tid, "task_path": str(dst), "instruction": r.instruction, "question": r.instruction})
DatasetRegistry.register_dataset(name=name, data=rows, split="test", source="swebench_pro (subset)", category="agentic")
print(len(rows), "tasks ->", out)
PY
rllm eval swebench_pro_mine --split test \
    --agent harbor:mini-swe-agent --evaluator harbor_reward_fn --sandbox-backend docker ...
```

### 1.3 Harbor 경로에서 알아둘 것

* **timeout**: Harbor trial 하나에 rLLM이 거는 상한은 `RLLM_HARBOR_SESSION_TIMEOUT_S`(기본 900초)다. SWE-bench 태스크의 `task.toml`은 Agent와 Verifier에 각 3000초를 주므로 기본값이면 긴 rollout이 먼저 잘린다. 3600 이상으로 올리는 것을 권장한다. 단, harbor와 동일한 평가 결과를 재현하고자 할 경우, task.toml 설정을 따른다.
  - `[agent].timeout_sec`와 `[verifier].timeout_sec`(두 벤치마크 모두 3000초)를 Harbor가 그대로 적용하고, 그 위에 rLLM의 `RLLM_HARBOR_SESSION_TIMEOUT_S`가 Trial 전체 상한으로 한 번 더 걸린다. 둘 중 짧은 쪽이 이긴다. 단, Harbor는 Verifier 타임아웃을 **한 번 재시도**한다(`stop_after_attempt(2)`).
* **concurrency**: `--sandbox-concurrency`가 동시에 뜨는 sandbox 수(default 64)이며, 로컬 Docker에서는 CPU, 메모리, 디스크 I/O를 보고 8 안팎에서 시작한다. `--concurrency`는 LLM 호출 동시성으로 별도다.
* **자원 제한**: `task.toml` 그대로. Harbor는 `[environment]`의 cpus/memory를 compose `deploy.resources.limits`로 적용하고, rLLM은 이를 바꾸지 않는다. 레지스트리의 `swebenchpro`는 CPU 1개, 4GB라서 Go 저장소의 Verifier가 시간 안에 끝나지 않는다(task instance: flipt). 이 태스크들을 Harbor 방식으로 채점하려면 자원 상한을 올릴 방법이 필요하다.
* **태스크마다 Harness 설치**: Harbor의 mini-swe-agent는 컨테이너 안에서 `apt-get install build-essential` 후 `uv tool install mini-swe-agent`를 매번 실행한다. native 방식의 `--agent-image`에 해당하는 캐시가 없어 태스크당 1~2분이 추가된다.
* **gateway 주소**: Harbor의 compose 파일은 태스크 컨테이너에 `host.docker.internal`을 넣어주지 않고, Linux Docker Engine은 그 이름을 기본으로 모른다(cf. Docker Desktop). 그래서 rLLM은 Linux에서 `docker network inspect bridge`로 얻은 docker0 게이트웨이 IP(보통 `172.17.0.1`)로 URL을 바꿔 넘긴다. rLLM 컨테이너가 `--network host`라는 전제 위에서만 성립한다.
* **이미지 다운로드 시점**: 실행 중에 태스크별로 받는다. 사전 다운로드는 필요 없다. Harbor는 `environment/Dockerfile`을 빌드해 태스크별 파생 이미지 `<trial>-main:latest`(Verified 기준 약 7.5 GB)를 만들고, 정상 종료 시 `compose down --rmi all`로 지운다. 비정상 종료 시에는 남는다. 베이스 이미지(`swebench/sweb.eval...`, `jefzda/sweap-images:...`, 개당 약 4 GB)는 어느 방식도 지우지 않는다. 

---

## 2. Native 방식

native 방식은 Harbor 런타임 없이 rLLM이 직접 컨테이너를 만들고, 그 안에서 하네스(예: mini-swe-agent CLI)를 실행하여 추론을 진행한 뒤, task directory의 `tests/test.sh`를 실행해 `reward.txt`를 읽는다.

```
SandboxTaskHooks      docker build -t rllm-task-<task_id> --rm . && docker run (+ Harness CLI image mount)
MiniSweAgentHarness   mini-swe-agent CLI를 Container 안에서 실행
  └─ litellm ───────► rLLM 모델 gateway (host.docker.internal) ──► LiteLLM proxy
ShellScriptEvaluator  /tests/test.sh → /logs/verifier/reward.txt → reward
```

Harness는 `rllm agent list`에 나오는 rLLM harness를 사용한다.

### 2.1 SWE-bench Verified

데이터 소스는 Harbor 방식과 **같은** `harbor:swebench-verified`다. `--agent`에 `harbor:` 접두어가 없으면 rLLM이 Harbor 채점기(`harbor_reward_fn`)를 건너뛰고 태스크별 `tests/test.sh`로 채점한다.

```bash
rllm dataset pull harbor:swebench-verified

# 환경 점검: 정답 패치를 적용하고 검증기만 돌린다. LLM 미호출 (LLM을 호출하지 않지만 CLI가 Provider 설정을 요구하므로 죽은 포트를 준다)
# oracle 점검에서 1.0이 안 나오는 태스크는 모델도 절대 풀 수 없으니 제외한다 (알려진 예: `astropy__astropy-7606`).
rllm eval harbor:swebench-verified --agent oracle --sandbox-backend docker \
    --max-examples 5 --concurrency 3 --no-ui \
    --base-url http://127.0.0.1:1/v1 --model oracle-dummy

# (A) proprietary
rllm eval harbor:swebench-verified --agent mini-swe-agent --sandbox-backend docker \
    --agent-image auto --sandbox-concurrency 8 --model gpt-5.5

# (B) vLLM / SGLang
rllm eval harbor:swebench-verified --agent mini-swe-agent --sandbox-backend docker \
    --agent-image auto --sandbox-concurrency 8 \
    --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-8B
```

`--agent-image auto`는 mini-swe-agent CLI를 한 번 Docker 이미지로 구워 두고 모든 task container에 read only로 마운트한다. 없으면 태스크마다 `uv tool install`을 반복한다. 

### 2.2 SWE-bench Pro

rLLM catalog의 `swebench_pro` 항목이 전용 builder를 갖고 있다. HuggingFace `ScaleAI/SWE-bench_Pro`와 `scaleapi/SWE-bench_Pro-os`의 인스턴스별 `run_script.sh`/`parser.py`를 합쳐 task directory를 만든다.
Verifier는 upstream `swe_bench_pro_eval.py` 흐름(agent diff → base_commit으로 reset → diff apply → gold test checkout → F2P/P2P eval)을 `tests/test.sh`로 재현한다.

```bash
rllm dataset pull swebench_pro          # $RLLM_HOME/datasets/swebench_pro/ 에 731개 task directory 생성

rllm eval swebench_pro --agent oracle --sandbox-backend docker \
    --max-examples 5 --concurrency 3 --no-ui \
    --base-url http://127.0.0.1:1/v1 --model oracle-dummy

# (A) proprietary
rllm eval swebench_pro --agent mini-swe-agent --sandbox-backend docker \
    --agent-image auto --sandbox-concurrency 8 --model gpt-5.5

# (B) vLLM / SGLang
rllm eval swebench_pro --agent mini-swe-agent --sandbox-backend docker \
    --agent-image auto --sandbox-concurrency 8 \
    --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-8B
```

이미지는 `jefzda/sweap-images:<tag>`이며 태스크마다 다르다. 없으면 `docker run` 시점에 pull한다. builder는 `task.toml`에 CPU 4개, 메모리 16 GB를 적어 두는데, Docker 백엔드는 이 값으로 컨테이너 자원을 제한한다. JS/Go 테스트 스위트가 커서 `--sandbox-concurrency`를 너무 올리면 호스트 메모리가 먼저 마른다.

### 2.2.1 harbor:swebenchpro vs swebench_pro

`harbor:swebenchpro`와 `swebench_pro`는 같은 벤치마크, 같은 Docker 이미지에 대해 **독립적으로 만들어진 두 벌의 task directory**다. rLLM builder는 Harbor 태스크를 변환하는 것이 아니라 원본 데이터에서 처음부터 만든다.

| | `harbor:swebenchpro` | `swebench_pro` |
| --- | --- | --- |
| 만드는 쪽 | Harbor adapter가 만들어 레지스트리에 올린 것 | `rllm/data/swebench_pro_builder.py` (`rllm dataset pull swebench_pro`) |
| 입력 | 완성된 태스크 디렉토리 731개를 다운로드 | HF `ScaleAI/SWE-bench_Pro` 행 + `scaleapi/SWE-bench_Pro-os`의 `run_script.sh`, `parser.py` |
| `task.toml` 자원 | `cpus = 1`, `memory_mb = 4096` | `cpus = 4`, `memory_mb = 16384` (builder `_DEFAULT_RESOURCES`) |
| `tests/test.sh` | adapter의 래퍼 | builder가 합성한 래퍼 |
| Docker 이미지 | `jefzda/sweap-images:<tag>` | 같은 이미지 |
| 저장 위치 | `~/.cache/harbor/tasks/...` | `$RLLM_HOME/datasets/swebench_pro/<instance_id>/` |

HF 데이터셋에는 CPU·메모리 정보가 없다. 4 / 16384는 업스트림 평가 스크립트가 1–4 CPU, 5–30 GiB를 쓴다는 점을 근거로 builder 작성자가 설정한 값이다(ref. builder 주석).

어느 명령이 어느 소스를 쓰는지:

| 명령 | 소스 | 컨테이너 자원 |
| --- | --- | --- |
| `rllm eval harbor:swebenchpro --agent harbor:mini-swe-agent` | Harbor | CPU 1, 4 GB |
| `rllm eval harbor:swebenchpro --agent mini-swe-agent` | **Harbor** (native harness라도 task directory는 Harbor) | CPU 1, 4 GB. native 경로도 `task.toml` 값을 그대로 적용한다 |
| `rllm eval swebench_pro --agent mini-swe-agent` | rLLM builder | CPU 4, 16 GB |

### 2.3 Subset만 평가

- **Index로 고르기**: Harbor 방식과 같다 (`--max-examples`, `--task-indices`). 
- **instance_id로 고정한 subset 만들기**: task directory를 `$RLLM_HOME/datasets/<subset명>/` 아래에 **복사**해 별도 dataset으로 만든다. 사본이므로 `task.toml`(timeout, 자원)을 고쳐도 원본에 영향이 없고, 어떤 이미지가 pull됐든 평가 대상이 바뀌지 않는다.

### 2.4 Native 경로에서 알아둘 것

* **이미지 자원 제한**: `task.toml`의 `[environment]` cpus/memory를 Docker 컨테이너에 적용한다. Verified는 CPU 1, 메모리 4 GB로 작다.
* **네트워크**: task container는 기본 bridge 네트워크에 붙고 `--add-host=host.docker.internal:host-gateway`가 자동으로 들어간다.

---

## Appendix

### 1. 파생 이미지 정리(공유 호스트라면 다른 사람의 실행분이 섞여 있으니 목록을 먼저 본다):

```bash
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}} {{.CreatedSince}}' | grep -- '-main:latest'   # Harbor 잔여
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}} {{.CreatedSince}}' | grep '^rllm-task-'       # native rllm 잔여
docker images --format '{{.Repository}}:{{.Tag}}' | grep '^rllm-task-' | xargs -r docker rmi             # 삭제 예
docker images --format '{{.Repository}}:{{.Tag}}' | grep -- '-main:latest' | xargs -r docker rmi
docker network ls --format '{{.Name}}' | grep '_default$' | grep -E 'instance_|__' | xargs -r docker network rm  # Harbor 잔여 네트워크
docker image prune                                                                                       # dangling 레이어
```

`docker rmi`는 실행 중인 컨테이너가 쓰는 이미지는 거부하므로 진행 중인 평가를 깨뜨리지 않는다. 베이스 이미지를 지우면 다음 실행 때 다시 받는다.

---

### 2. Native 방식과 Harbor 방식의 차이

| | Harbor | native |
| --- | --- | --- |
| container | Harbor가 `docker compose`로 태스크별 `environment/Dockerfile`을 **빌드**해서 띄움 (`<trial>-main:latest`) | rLLM도 docker 백엔드에서는 `docker build`로 태스크별 이미지(`rllm-task-<task_id>`)를 만들어 띄움. RUN 단계를 컨테이너 안에서 재생하는 것은 modal/daytona 백엔드만 |
| derived image per task | 정상 종료 시 `compose down --rmi all`로 삭제. 비정상 종료 시 남음 | **삭제하지 않음** (재실행 캐시로 남김) |
| harness | Harbor 스캐폴드 (`harbor/agents/`). 20종, `harbor:<이름>` | rLLM 하네스 (`rllm/harnesses/`). `rllm agent list` |
| harness install | 태스크마다 컨테이너 안에서 설치 | `--agent-image auto`로 한 번 구워 마운트 |
| verifier | Harbor verifier → `TrialResult.verifier_result` → `harbor_reward_fn` | 태스크의 `tests/test.sh`를 rLLM이 직접 실행 → `reward.txt` |
| trajectory | Harbor ATIF 트라젝토리를 rLLM Episode로 변환 | gateway가 모든 LLM 호출을 캡처해 Step/Episode 구성 (학습과 동일 경로) |
| timeout | `task.toml` 값 + `RLLM_HARBOR_SESSION_TIMEOUT_S` 상한 | `task.toml` 값만 |
| log | `$RLLM_HOME/harbor_trials/<trial>/` (Harbor 형식) | `$RLLM_HOME/eval_results/<run>/episodes/` |
| train | `examples/harbor_swe` (RemoteAgentFlowEngine + tinker) | `recipe/qwen3_5_swe_grpo` (AgentFlowEngine + verl) |


