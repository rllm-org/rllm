# rLLM으로 SWE-bench Verified / SWE-bench Pro 평가하기

`rllm eval` 한 명령으로 SWE-bench Verified와 SWE-bench Pro(731 태스크)를 평가하는 방법을
정리한다. 실행 방식은 두 가지다.

| 방식 | `--agent` | 태스크를 실제로 굴리는 것 |
| --- | --- | --- |
| **Harbor** | `harbor:mini-swe-agent` 처럼 `harbor:` 접두어 | Harbor `Trial.run()` — 환경 빌드, 에이전트, 검증기를 Harbor가 모두 담당 |
| **native** | `mini-swe-agent`, `oracle` 처럼 접두어 없음 | rLLM `SandboxTaskHooks` + rLLM 하네스 + `ShellScriptEvaluator` |

두 방식 모두 샌드박스는 `--sandbox-backend docker`(로컬 Docker 데몬)를 쓴다. 차이는 문서 끝의
[native vs Harbor](#native-방식과-harbor-방식의-차이) 절에 정리했다.

---

## 0. 공통 준비

### 0.1 컨테이너

기준 이미지는 `pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel`. 평가용 샌드박스는 이 컨테이너 **안에서**
호스트의 Docker 데몬을 빌려 형제 컨테이너로 뜬다. 그래서 두 가지가 필요하다.

* `-v /var/run/docker.sock:/var/run/docker.sock` — 호스트 Docker 데몬 공유
* `--network host` — 태스크 컨테이너는 **호스트 주소**로 rLLM 모델 게이트웨이에 접속한다(native는
  `host.docker.internal`, Harbor는 docker0 게이트웨이 IP `172.17.0.1`). 게이트웨이는 rLLM 프로세스(= 이
  컨테이너) 안에서 `0.0.0.0`에 바인드되므로, 이 컨테이너가 호스트 네트워크에 있어야 그 주소가 곧 게이트웨이가
  된다. 브리지 네트워크로 띄우면 어느 쪽도 도달하지 못하며, 이를 우회하는 설정은 없다.

```bash
# 호스트에서
docker run -d --name rllm-eval --gpus all --network host \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /raid/rllm-work:/raid/rllm-work \
  -v /path/to/rllm:/workspace/rllm \
  pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel sleep infinity
docker exec -it rllm-eval bash
```

컨테이너 안에는 Docker CLI와 compose 플러그인이 있어야 한다 (Harbor는 `docker compose`를 호출한다).

```bash
apt-get update && apt-get install -y docker.io git curl
docker compose version      # 플러그인 확인
docker info                 # 소켓이 연결됐는지 확인
```

### 0.2 rLLM 설치

```bash
cd /workspace/rllm
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[harbor]"          # harbor==0.3.0 포함
```

`harbor` extra는 native 방식에도 필요하다. SWE-bench Verified 태스크 디렉토리를 Harbor 레지스트리에서
받아오기 때문이다.

### 0.3 저장 위치

rLLM은 데이터셋, 평가 결과, Harbor 트라이얼 로그를 `$RLLM_HOME`(기본 `~/.rllm`) 아래에 둔다. 설정하지 않아도
동작하지만, 이 환경에서는 두 이유로 **호스트와 같은 경로로 마운트된 큰 볼륨**(위 예시의 `/raid/rllm-work`)
아래를 가리키게 한다.

1. 용량. 태스크 디렉토리 수백 개와 에피소드 JSON이 쌓이고, 컨테이너 루트에 두면 컨테이너와 함께 사라진다.
2. Harbor 방식은 `$RLLM_HOME/harbor_trials/...`를 태스크 컨테이너에 bind-mount하는데, 이 마운트는 **호스트의**
   Docker 데몬이 해석한다. 컨테이너 안에만 있는 경로(`~/.rllm`, `/workspace/rllm`)를 주면 검증기의 reward
   파일이 호스트에만 남아 모든 태스크가 `RewardFileNotFoundError`로 끝난다(1.3절, 트러블슈팅 참고).
   이 위치를 따로 바꾸는 설정은 없다. `RLLM_HOME`을 옮기는 것이 방법이다.

아래는 이 문서 전체에서 가정하는 값이다. 한 번 정하면 팀 안에서 고정한다. 바꾸면 이전에 pull한 데이터셋과
결과가 다른 위치에 있어 없는 것처럼 보인다.

```bash
export RLLM_HOME=/raid/rllm-work/rllm-home
export HF_HOME=/raid/rllm-work/hf
```

`recipe/qwen3_5_swe_grpo/env.sh`를 `source`해도 같은 값이 잡힌다.

### 0.3.1 함께 신경 쓸 환경변수

| 변수 | 기본값 | 왜 신경 써야 하나 |
| --- | --- | --- |
| `RLLM_HOME` | `~/.rllm` | 데이터셋, 결과, `config.json`(모델 setup), Harbor 트라이얼. 위 0.3절. 값을 바꾸면 **모델 setup도 다시** 해야 한다. |
| `RLLM_HARBOR_SESSION_TIMEOUT_S` | `900` | Harbor 방식 전용. 트라이얼 하나(에이전트 + 검증기)의 상한. SWE-bench Pro의 Go 저장소는 **검증기만 15분 넘게** 걸리므로 기본값이면 정답 패치조차 타임아웃으로 0점이 된다. `3000` 이상으로. |
| `RLLM_AGENT_IMAGE` | `auto` | native 방식 전용. `--agent-image` 플래그와 같은 값(`auto` / `skip` / `repo:tag`). |
| `HF_HOME` | `~/.cache/huggingface` | `swebench_pro` 빌더가 HF 데이터셋을 받는 곳. 큰 볼륨으로. |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` 등 | 없으면 rLLM이 `empty`를 채움 | 프록시를 쓰면 실제 값이 필요 없다. **export되어 있으면 태스크 컨테이너 안으로 전달**되므로, 신뢰하지 않는 태스크(에이전트가 임의 명령을 실행함)를 돌릴 때는 셸에 실제 키를 두지 않는다. |
| `RLLM_API_KEY` | 없음 | `rllm login`을 했거나 이 변수가 있으면 **평가 결과와 에피소드가 rLLM UI 서버로 업로드**된다. 외부로 나가면 안 되는 결과라면 `--no-ui`를 명시한다. |

Harbor가 레지스트리에서 받은 태스크 디렉토리는 `RLLM_HOME`이 아니라 **`~/.cache/harbor/tasks`**(하드코딩, 환경변수
없음)에 들어가고, `$RLLM_HOME/datasets/<name>/default.parquet`의 `task_path`가 거기를 가리킨다. 컨테이너를 새로
만들면 이 캐시가 사라져 parquet은 있는데 태스크 디렉토리가 없는 상태가 된다. 그때는 `rllm dataset pull
harbor:<name>`을 다시 실행한다(텍스트만이라 빠르다).

### 0.4 모델 연결 — 두 가지

`rllm eval`은 모델 호출을 항상 **rLLM 모델 게이트웨이**를 거쳐 보낸다. 게이트웨이 뒤에 무엇을 두느냐에 따라
두 가지로 나뉜다.

**(A) Proprietary 모델 (OpenAI, Anthropic, Gemini 등)**

한 번만 프로바이더와 API 키를 등록한다. 이후 `rllm eval`이 LiteLLM 프록시를 자동으로 띄워 라우팅한다.

```bash
rllm model setup            # provider(openai / anthropic / gemini / openrouter / ...), API key, model 선택
rllm model show             # 현재 설정 확인
```

이후 명령에서는 `--model`만 바꾸면 된다. 생략하면 setup에서 고른 모델을 쓴다.

```bash
rllm eval <benchmark> --agent <agent> --sandbox-backend docker --model gpt-5.5
```

**키가 흐르는 길.** setup이 받은 키는 `$RLLM_HOME/config.json`에 저장된다. `rllm eval`은 그 키로 로컬 LiteLLM
프록시를 띄우고, 게이트웨이가 프록시로 요청을 넘긴다. 태스크 컨테이너 안의 에이전트에는 실제 키가 아니라
자리표시자(`sk-rllm-gateway` 또는 `empty`)가 들어간다. 즉 **평가 중 실제 키는 rLLM 프로세스 밖으로 나가지 않는다.**

```
컨테이너(에이전트, 자리표시자 키) → 게이트웨이 → LiteLLM 프록시(실제 키) → api.openai.com
```

그래서 키는 setup에만 주고, 셸에는 export하지 않는 것을 권장한다. `OPENAI_API_KEY`가 셸에 export되어 있으면
setup이 그 값을 기본값으로 읽어 편하지만, 그 상태로 `rllm eval`을 돌리면 하네스가 자리표시자 대신 **실제 키를
태스크 컨테이너에 넣는다**(동작에는 차이가 없지만 에이전트가 임의 명령을 실행하는 컨테이너에 키가 노출된다).
setup을 마친 뒤 `unset OPENAI_API_KEY` 하면 된다. `config.json`은 평문이므로 공유 볼륨에 둘 때는
`chmod 600 $RLLM_HOME/config.json`.

프록시 시작에 실패하면 `uv pip install "litellm[proxy]"`.

**(B) vLLM / SGLang으로 서빙한 로컬 모델**

OpenAI 호환 엔드포인트를 `--base-url`로 직접 준다. 이때 `--model`은 필수이며, 서버의 `served-model-name`과
정확히 같아야 한다.

```bash
# vLLM (같은 컨테이너 또는 같은 호스트에서)
vllm serve Qwen/Qwen3-8B --port 8000 --served-model-name Qwen/Qwen3-8B --max-model-len 65536

# 또는 SGLang
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000 --served-model-name Qwen/Qwen3-8B

rllm eval <benchmark> --agent <agent> --sandbox-backend docker \
    --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-8B
```

`--network host`로 띄웠으므로 서빙 서버가 호스트나 다른 컨테이너에 있어도 `127.0.0.1:<port>`로 닿는다.

모델명 규칙 하나. mini-swe-agent 계열 에이전트는 `provider/model` 형식을 요구해서 rLLM이 접두어를 자동으로
붙인다. `qwen`, `deepseek`, `gpt-*`, 알 수 없는 이름은 `openai/`가 붙어 OpenAI 호환 경로로 나가고, `claude`가
들어간 이름은 `anthropic/`이 붙는다. `Qwen/Qwen3-8B`처럼 HF 조직명이 앞에 오는 이름도 `openai/` 접두어가
붙는다(`openai/Qwen/Qwen3-8B`). litellm이 `openai/`를 떼고 보내므로 서버에는 `Qwen/Qwen3-8B` 그대로
도착한다. 게이트웨이는 `/v1/chat/completions`만 제공하므로 Claude 계열은 에이전트 내부 litellm이 Anthropic
메시지 API로 호출하려 할 수 있다. 이 조합은 아직 검증하지 않았다. 검증 전까지는 OpenAI 호환으로 나가는
모델명을 권장한다.

### 0.5 결과 확인

결과는 `$RLLM_HOME/eval_results/<benchmark>_<model>_<timestamp>/`에 `meta.json`, `results.json`, `episodes/`로
남는다. 콘솔에는 Accuracy, Errors, `--attempts`를 썼다면 pass@k가 찍힌다.

```bash
rllm view <run-id>          # 에피소드를 브라우저에서 훑어보기
```

`Errors`는 모델이 못 푼 것이 아니라 **인프라 실패**(이미지 빌드 실패, 타임아웃, 게이트웨이 접속 불가 등)다.
0이 아니면 먼저 원인을 잡는다. 콘솔에 처음 다섯 개가 찍히고 나머지는 `results.json`에 있다.

---

## 1. Harbor 방식

Harbor 레지스트리에 두 벤치마크가 모두 등록되어 있다. 이름은 다음과 같고, `harbor:` 접두어로 부른다.

| 벤치마크 | 이름 | 태스크 수 | 베이스 이미지 |
| --- | --- | --- | --- |
| SWE-bench Verified | `harbor:swebench-verified` | 500 | `swebench/sweb.eval.x86_64.<repo>_1776_<instance>` (태스크마다 다름) |
| SWE-bench Pro | `harbor:swebenchpro` | 731 | 태스크별 사전 빌드 이미지 |

첫 실행 때 태스크 디렉토리(텍스트만, 이미지는 아님)를 자동으로 내려받아 `$RLLM_HOME/datasets/<name>/`에
등록한다. 미리 받아두려면:

```bash
rllm dataset pull harbor:swebench-verified
rllm dataset pull harbor:swebenchpro
```

Docker 이미지는 태스크가 실행될 때 Harbor가 `environment/Dockerfile`을 빌드하면서 `FROM` 이미지를 pull한다.
Verified 500개 전체의 베이스 이미지는 약 2 TB다. 전체 평가 전에 디스크를 확인한다.

### 1.1 전체 평가

```bash
export RLLM_HARBOR_SESSION_TIMEOUT_S=3600     # 아래 "타임아웃" 참고

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

`--agent`를 생략하면 Harbor 데이터셋의 기본값인 `harbor:mini-swe-agent`가 쓰인다. 다른 Harbor 스캐폴드도
같은 형식으로 고른다: `harbor:terminus-2`, `harbor:swe-agent`, `harbor:openhands`, `harbor:codex`,
`harbor:claude-code`, `harbor:oracle`(정답 패치만 적용, LLM 미호출) 등.

### 1.2 Subset만 평가

Harbor 경로에서 태스크를 고르는 수단은 **인덱스**다. 인덱스는 `$RLLM_HOME/datasets/<name>/default.parquet`의
행 순서다.

```bash
# 앞에서 N개
rllm eval harbor:swebench-verified --agent harbor:mini-swe-agent --sandbox-backend docker \
    --max-examples 10 --model gpt-5.5

# 특정 인덱스. '0', '3,7,12', '0-9' 형식
rllm eval harbor:swebench-verified --agent harbor:mini-swe-agent --sandbox-backend docker \
    --task-indices 0-4,17,42 --model gpt-5.5
```

instance_id로 고르고 싶으면 id → 인덱스를 먼저 뽑는다.

```bash
python - <<'PY'
import os, pandas as pd
name = "swebench-verified"                     # 또는 "swebenchpro"
want = {"django__django-11265", "sympy__sympy-16792"}
df = pd.read_parquet(f"{os.environ['RLLM_HOME']}/datasets/{name}/default.parquet")
print(",".join(str(i) for i, t in enumerate(df["task_id"]) if t in want))
PY
```

출력값을 `--task-indices`에 그대로 넣는다. 인덱스는 데이터셋을 다시 pull해도 바뀌지 않는다 (레지스트리
manifest 순서).

같은 태스크를 여러 번 굴려 pass@k를 보려면 `--attempts 4`처럼 준다. 이때는 온도를 0보다 크게 준다
(`--temperature 0.7`).

### 1.3 Harbor 경로에서 알아둘 것

* **타임아웃**: Harbor 트라이얼 하나에 rLLM이 거는 상한은 `RLLM_HARBOR_SESSION_TIMEOUT_S`(기본 900초)다.
  SWE-bench 태스크의 `task.toml`은 에이전트와 검증기에 각 3000초를 주므로 기본값이면 긴 롤아웃이 먼저 잘린다.
  3600 이상으로 올린다.
* **동시성**: `--sandbox-concurrency`가 동시에 뜨는 샌드박스 수다(기본 64). 로컬 Docker에서는 CPU, 메모리,
  디스크 I/O를 보고 8 안팎에서 시작한다. `--concurrency`는 LLM 호출 동시성으로 별도다.
* **자원 제한은 `task.toml` 그대로.** Harbor는 `[environment]`의 cpus/memory를 compose `deploy.resources.limits`로
  적용하고, rLLM은 이를 바꾸지 않는다. 레지스트리의 `swebenchpro`는 CPU 1개, 4 GB라서 Go 저장소의 검증기가 시간
  안에 끝나지 않는다(flipt: 링커 십여 개가 각 3% CPU로 30분 이상). 이 태스크들을 Harbor 방식으로 채점하려면 자원
  상한을 올릴 방법이 필요하다(미해결, 아래 검증 상태 참고).
* **타임아웃은 `task.toml`을 따른다.** `[agent].timeout_sec`와 `[verifier].timeout_sec`(두 벤치마크 모두 3000초)를
  Harbor가 그대로 적용하고, 그 위에 rLLM의 `RLLM_HARBOR_SESSION_TIMEOUT_S`가 트라이얼 전체 상한으로 한 번 더 걸린다.
  둘 중 짧은 쪽이 이긴다. 단, Harbor는 검증기 타임아웃을 **한 번 재시도**한다(`stop_after_attempt(2)`). flipt에서
  실측: 3000초에 타임아웃 → 즉시 두 번째 시도 → 다시 3000초. 그래서 검증기가 멈춘 태스크는 사실상 100분을
  잡아먹고, 이를 끊는 것은 `RLLM_HARBOR_SESSION_TIMEOUT_S`뿐이다. 이 상한으로 잘린 트라이얼은 컨테이너 정리가
  끝나기 전에 프로세스가 죽을 수 있으니, 실행 후 `docker ps | grep instance_`로 고아 컨테이너를 확인한다.
* **태스크마다 에이전트를 새로 설치한다.** Harbor의 mini-swe-agent는 컨테이너 안에서 `apt-get install
  build-essential` 후 `uv tool install mini-swe-agent`를 매번 돈다. native 방식의 `--agent-image`에 해당하는
  캐시가 없어 태스크당 1~2분이 추가된다.
* **게이트웨이 주소**: Harbor의 compose 파일은 태스크 컨테이너에 `host.docker.internal`을 넣어주지 않고,
  Linux Docker Engine은 그 이름을 기본으로 모른다(Docker Desktop만 안다). 그래서 rLLM은 Linux에서
  `docker network inspect bridge`로 얻은 docker0 게이트웨이 IP(보통 `172.17.0.1`)로 URL을 바꿔 넘긴다.
  rLLM 컨테이너가 `--network host`라는 전제 위에서만 성립한다.
* **트라이얼 디렉토리는 호스트와 같은 경로여야 한다.** Harbor는 `trials/<트라이얼>/verifier` 등을 태스크 컨테이너에
  bind-mount하고 검증기가 거기에 `reward.txt`를 쓴다. 이 마운트는 **호스트의 Docker 데몬**이 해석하므로, rLLM이
  컨테이너 안에서 돌 때 그 경로가 호스트에 같은 위치로 존재하지 않으면 reward 파일이 호스트에만 남고 Harbor는
  `RewardFileNotFoundError`를 낸다(검증기는 정상 채점했는데도). rLLM은 이 디렉토리를 `$RLLM_HOME/harbor_trials`로
  잡는다. `RLLM_HOME`을 `-v /raid/rllm-work:/raid/rllm-work`처럼 **같은 경로로 마운트한 볼륨** 아래에 두는 이유가
  이것이다. 트라이얼 로그(`trial.log`, `agent/`, `verifier/test-stdout.txt`)도 여기에 남는다.
* **`Docker Compose is configured to build using Bake, but buildx isn't installed`** 경고는 무해하다.

---

## 2. Native 방식

native 방식은 Harbor 런타임 없이 rLLM이 직접 컨테이너를 만들고(`docker run <태스크 이미지>`), 그 안에서
하네스(예: mini-swe-agent CLI)를 돌린 뒤, 태스크 디렉토리의 `tests/test.sh`를 실행해 `reward.txt`를 읽는다.

```
SandboxTaskHooks      docker run <task image>  (+ 에이전트 CLI 이미지 마운트)
MiniSweAgentHarness   mini-swe-agent CLI를 컨테이너 안에서 실행
  └─ litellm ───────► rLLM 모델 게이트웨이 (host.docker.internal) ──► LiteLLM 프록시 또는 vLLM
ShellScriptEvaluator  /tests/test.sh → /logs/verifier/reward.txt → reward
```

에이전트는 `rllm agent list`에 나오는 rLLM 하네스를 쓴다. 이 문서는 `mini-swe-agent`와 `oracle`을 쓴다.

### 2.1 SWE-bench Verified

데이터 소스는 Harbor 방식과 **같은** `harbor:swebench-verified`다. `--agent`에 `harbor:` 접두어가 없으면
rLLM이 Harbor 채점기(`harbor_reward_fn`)를 건너뛰고 태스크별 `tests/test.sh`로 채점한다.

```bash
rllm dataset pull harbor:swebench-verified

# 환경 점검: 정답 패치를 적용하고 검증기만 돌린다. LLM 미호출.
# (LLM을 부르지 않지만 CLI가 프로바이더 설정을 요구하므로 죽은 포트를 준다)
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

`--agent-image auto`는 mini-swe-agent CLI를 한 번 Docker 이미지로 구워 두고 모든 태스크 컨테이너에 읽기 전용
마운트한다. 없으면 태스크마다 `uv tool install`을 반복한다. oracle 점검에서 1.0이 안 나오는 태스크는 정책
모델도 절대 풀 수 없으니 제외한다 (알려진 예: `astropy__astropy-7606`).

### 2.2 SWE-bench Pro

rLLM 카탈로그의 `swebench_pro` 항목이 전용 빌더를 갖고 있다. HuggingFace `ScaleAI/SWE-bench_Pro`와
`scaleapi/SWE-bench_Pro-os`의 인스턴스별 `run_script.sh`/`parser.py`를 합쳐 태스크 디렉토리를 만든다.
검증기는 업스트림 `swe_bench_pro_eval.py` 흐름(agent diff 캡처 → base_commit으로 리셋 → diff 재적용 → 골드
테스트 체크아웃 → F2P/P2P 판정)을 `tests/test.sh`로 재현한다.

```bash
rllm dataset pull swebench_pro          # $RLLM_HOME/datasets/swebench_pro/ 에 731개 태스크 디렉토리 생성

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

이미지는 `jefzda/sweap-images:<tag>`이며 태스크마다 다르다. 없으면 `docker run` 시점에 pull한다. 빌더는
`task.toml`에 CPU 4개, 메모리 16 GB를 적어 두는데, Docker 백엔드는 이 값으로 컨테이너 자원을 제한한다.
JS/Go 테스트 스위트가 커서 `--sandbox-concurrency`를 너무 올리면 호스트 메모리가 먼저 마른다.

### 2.3 Subset만 평가

**인덱스로 고르기**는 Harbor 방식과 같다 (`--max-examples`, `--task-indices`). Verified의 인덱스는
`datasets/swebench-verified/default.parquet`, Pro는 `datasets/swebench_pro/test.parquet`의 행 순서다.

**instance_id로 고정된 subset을 만들기**는 벤치마크마다 다르다.

*SWE-bench Verified* — 레시피 스크립트가 Harbor 캐시에서 태스크 디렉토리를 복사해 별도 이름으로 등록한다.
로컬 Docker 데몬에 베이스 이미지가 **있는** 태스크만 고르고, 없는 태스크가 지정되면 `docker pull` 명령을
찍어 주고 멈춘다.

```bash
# 한 줄에 instance_id 하나. '#' 주석 가능
cat > my_val_ids.txt <<'IDS'
django__django-11265
sympy__sympy-16792
IDS
python recipe/qwen3_5_swe_grpo/scripts/prepare_datasets.py --val-only \
    --val-ids my_val_ids.txt --val-name swebench_verified_mine

rllm eval swebench_verified_mine --agent mini-swe-agent --sandbox-backend docker --split test \
    --agent-image auto --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-8B
```

이 스크립트는 복사한 `task.toml`의 타임아웃을 에이전트 3600초, 검증기 1800초로 바꿔 쓴다.
`recipe/qwen3_5_swe_grpo/val_tasks.txt`가 난이도 균형을 맞춘 10개 예시다.

*SWE-bench Pro* — 빌더를 직접 호출해 다른 디렉토리에 일부만 만든다.

```bash
python -m rllm.data.swebench_pro_builder \
    --out-dir $RLLM_HOME/datasets/swebench_pro_mine --name swebench_pro_mine \
    --task-ids <instance_id_1> <instance_id_2> ...
# 또는 앞에서 N개만: --limit 20

rllm eval swebench_pro_mine --agent mini-swe-agent --sandbox-backend docker \
    --agent-image auto --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-8B
```

`$RLLM_HOME/datasets/<이름>/dataset.toml`이 있으면 `rllm eval <이름>`이 그 디렉토리를 그대로 읽는다.
경로를 직접 줘도 된다: `rllm eval /path/to/dir --agent mini-swe-agent ...`.

### 2.4 Native 경로에서 알아둘 것

* **타임아웃은 `task.toml`에서 읽는다.** Verified는 업스트림 값이 에이전트/검증기 각 3000초다. 바꾸려면
  디렉토리를 복사한 뒤(위 subset 절) 다음처럼 고친다. 실행 중인 job 아래에서는 하지 않는다.
  ```bash
  find $RLLM_HOME/datasets/swebench_verified_mine -name task.toml -exec \
      sed -i '/^\[agent\]/,/^\[/ s/^timeout_sec = .*/timeout_sec = 3600.0/' {} +
  ```
* **이미지 자원 제한**: `task.toml`의 `[environment]` cpus/memory를 Docker 컨테이너에 적용한다. Verified는
  CPU 1, 메모리 4 GB로 작다.
* **네트워크**: 태스크 컨테이너는 기본 브리지 네트워크에 붙고 `--add-host=host.docker.internal:host-gateway`가
  자동으로 들어간다. 검증기도 네트워크가 필요하다(Verified의 `test.sh`는 `uv run`으로 `swebench` 등을 받는다).

---

## 검증 상태 (2026-09-08, DGX03)

| 경로 | 확인한 범위 |
| --- | --- |
| Harbor + docker, `harbor:mini-swe-agent`, HF 모델명 | 태스크 이미지 빌드 → 컨테이너 기동 → mini-swe-agent 설치 → 게이트웨이(`172.17.0.1:<port>`)에 첫 LLM 요청 도달까지. 업스트림은 일부러 죽은 포트를 줘서 게이트웨이가 `No healthy workers available`을 반환하는 것으로 도달을 확인했다. **실제 모델을 붙인 end-to-end 점수는 아직 안 냈다.** |
| Harbor + docker, `harbor:oracle`, SWE-bench Verified | 로컬에 이미지가 있는 20개 태스크 전부 실행. **19/20 = 1.0, 오류 0.** 유일한 0은 `astropy__astropy-7606`으로, native 경로에서도 채점 불가로 알려진 태스크(PASS_TO_PASS의 빈 pytest param id) |
| native + docker, `oracle` | `recipe/qwen3_5_swe_grpo`에서 SWE-bench Verified 12/12, SWE-smith 1/1 (레시피 README 참고). 이번에 `sympy__sympy-16792` 1/1 재확인 |
| native + docker, `mini-swe-agent` + vLLM(verl rollout) | 같은 레시피의 학습 검증 경로로 상시 사용 중 |
| Harbor + docker, `harbor:oracle`, SWE-bench Pro | 로컬에 이미지가 있는 110개 중 언어·저장소가 겹치지 않게 11개(Go, Python, JS, TS). **`task.toml` 자원(CPU 1, 4 GB)으로 10/11 = 1.0**, teleport는 검증기 10분. flipt는 검증기 3000초 초과(`go test ./...` 링크 단계가 CPU 1개에서 30분 이상 정지) 후 Harbor가 재시도에 들어가 결과 없음. 이 태스크는 `task.toml` 자원(CPU 1, 4 GB)으로는 Harbor 방식에서 채점 불가. rLLM native `swebench_pro` 빌더는 같은 태스크에 CPU 4, 16 GB를 쓴다 |
| native, `swebench_pro` 빌더 | 빌더 코드 확인만. 실행은 아직 |

---

## 3. 트러블슈팅

| 증상 | 원인 / 조치 |
| --- | --- |
| `Harbor tasks require Docker — ...` | 컨테이너 안에 Docker CLI가 없거나 `/var/run/docker.sock`이 마운트되지 않았다. |
| Harbor: `invalid volume specification: '.../trials/<task>:0/verifier:rw'` | rLLM 세션 id(`<task_id>:<rollout>`)를 그대로 Harbor 트라이얼 이름으로 쓰던 버그. `rllm/integrations/harbor/runtime.py`의 `safe_trial_name`이 `:`를 `-`로 바꾼다. 이 수정이 없는 트리에서는 Harbor 방식이 Docker에서 전부 실패한다. |
| Harbor: `Unable to determine API key for model Qwen/Qwen3-8B: Unknown model` | HF 조직명이 붙은 모델명에 프로바이더 접두어가 안 붙던 버그. `rllm/integrations/harbor/trial_helper.py`의 `qualify_model_name`이 `openai/`를 붙인다. |
| Harbor: 에이전트가 `Name or service not known` (host.docker.internal) | Linux Docker Engine은 이 이름을 모르고 Harbor compose는 `extra_hosts`를 넣지 않는다. rLLM이 docker0 게이트웨이 IP로 바꿔 넘기도록 고쳤다(`trial_helper.container_host_for_gateway`). 이 수정이 없는 트리에서는 Harbor 방식의 모든 LLM 호출이 실패한다. |
| 에이전트가 게이트웨이에 `Connection refused` | rLLM 컨테이너가 `--network host`가 아니다. 점검: 이 컨테이너에서 `python3 -m http.server 18099 --bind 0.0.0.0`을 띄우고 `docker run --rm python:3.12-slim python -c "import urllib.request;print(urllib.request.urlopen('http://172.17.0.1:18099/').status)"`가 200이어야 한다. |
| `No configuration found. Run rllm setup first` (oracle 실행 시) | oracle도 프로바이더 설정을 요구한다. `--base-url http://127.0.0.1:1/v1 --model oracle-dummy`를 준다. |
| `rllm eval swebench_verified` 가 `Agent 'swe' not found` | 접두어 없는 `swebench_verified`는 레거시 카탈로그 항목으로, 가리키는 에이전트와 채점기가 현재 코드에 없다. `harbor:swebench-verified`를 쓴다. |
| Harbor: 모든 태스크가 `RewardFileNotFoundError`, `verifier/test-stdout.txt`가 0바이트 | 트라이얼 디렉토리가 호스트와 다른 경로다(예: 컨테이너 전용 `/tmp`, 호스트에 없는 `/workspace/rllm`). `docker run --rm -v <trials 경로>:/x python:3.12-slim ls /x`로 호스트 쪽 내용을 보면 거기에 `reward.txt`가 있다. `RLLM_HOME`을 같은 경로로 마운트된 볼륨 아래로 옮긴다. |
| 롤아웃이 900초 근처에서 일괄 종료 (Harbor) | `RLLM_HARBOR_SESSION_TIMEOUT_S`를 올린다. |
| 디스크 부족 | Verified 500개 베이스 이미지 약 2 TB. `docker system df`로 확인하고 subset을 쓴다. |

---

## Native 방식과 Harbor 방식의 차이

| | Harbor | native |
| --- | --- | --- |
| 컨테이너 생성 | Harbor가 `docker compose`로 태스크별 `environment/Dockerfile`을 **빌드**해서 띄움 | rLLM이 `docker run <베이스 이미지>` 후 Dockerfile의 RUN 단계를 재생 |
| 에이전트 | Harbor 스캐폴드 (`harbor/agents/`). 20종, `harbor:<이름>` | rLLM 하네스 (`rllm/harnesses/`). `rllm agent list` |
| 에이전트 설치 | 태스크마다 컨테이너 안에서 설치 | `--agent-image auto`로 한 번 구워 마운트 |
| 채점 | Harbor verifier → `TrialResult.verifier_result` → `harbor_reward_fn` | 태스크의 `tests/test.sh`를 rLLM이 직접 실행 → `reward.txt` |
| 트라젝토리 | Harbor ATIF 트라젝토리를 rLLM Episode로 변환 | 게이트웨이가 모든 LLM 호출을 캡처해 Step/Episode 구성 (학습과 동일 경로) |
| 타임아웃 | `task.toml` 값 + `RLLM_HARBOR_SESSION_TIMEOUT_S` 상한 | `task.toml` 값만 |
| 로그 | `$RLLM_HOME/harbor_trials/<trial>/` (Harbor 형식) | `$RLLM_HOME/eval_results/<run>/episodes/` |
| 학습 연계 | `examples/harbor_swe` (RemoteAgentFlowEngine + tinker) | `recipe/qwen3_5_swe_grpo` (AgentFlowEngine + verl) |

요약하면 **Harbor 방식은 Harbor 생태계의 기준 구현을 그대로 재현**한다. 리더보드 수치와 비교하거나 다른 Harbor
스캐폴드(terminus-2, openhands 등)를 바꿔 끼울 때 맞다. **native 방식은 rLLM이 학습 때 쓰는 경로와 같다**.
게이트웨이가 토큰 단위 트레이스를 남기고, 하네스와 검증기를 rLLM 코드 안에서 직접 손볼 수 있다. RL 학습 전후
비교 평가는 native로 하는 것이 일관적이다. 데이터 소스와 검증 스크립트는 두 방식이 같으므로, 같은 에이전트
(mini-swe-agent)라면 점수 차이는 스캐폴드 버전과 타임아웃 차이에서만 나와야 한다.
