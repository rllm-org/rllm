# Initial Setup

apt-get update && apt-get install git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
cd /workspace/rllm
uv venv --python 3.12
source .venv/bin/activate

uv pip install -e ".[verl,harbor]"

## Docker CLI (+ compose plugin)
apt-get update && apt-get install -y docker.io docker-compose-v2
uv add docker

: rllm.cli.eval.py::240-248
if (_is_harbor_agent or _is_harbor_source) and _runs_on_local_docker:
    from rllm.integrations.harbor.utils import diagnose_docker

    ok, reason, hint = diagnose_docker()
    if not ok:
        if hint:
            console.print(f"  [dim]{hint}[/]")
        console.print("  [dim]Or run on a remote backend, e.g. [bold]--sandbox-backend modal[/].[/]")
        fail(f"Harbor tasks require Docker — {reason}.")


## ipykernel 설치
uv add --dev ipykernel
uv run python -m ipykernel install --user --name=rllm --display-name="Python (rllm uv)" # Jupyter 커널로 등록

## gsm8k sample data eval with rllm 
rllm eval gsm8k \
  --base-url http://27.122.129.133:8010/v1 \
  --model LGAI-EXAONE/EXAONE-4.5-33B \
  --max-examples 2 \
  --concurrency 1 \s
  --no-ui

## view results
rllm view gsm8k_LGAI-EXAONE_EXAONE-4.5-33B_20260828_085827


## swebench pro evaluation

```
Gateway session URL: task 별 rLLM이 만드는 LLM proxy endpoint로, agent sandbox dotenv로 입력되는 값
ex. OPENAI_API_BASE=http://127.0.0.1:47699/sessions/abc123/v1
  - /sessions/abc123: task rollout 전용 session
  - /v1: OpenAI-compatible API prefix

Gateway 동작 방식:
→ sandbox 안에서 이 URL로 chat completion을 보내면
→ gateway가 받아서 vLLM으로 forward
→ gateway가 trace 저장 → rLLM이 나중에 steps로 채움

문제:
- sandbox -> gateway unreachable -> Episode trajectories = []
  
원인:
- DockerSandbox로 띄운 container 내부: 
    - host.docker.internal -> FAIL (Name or service not known)
    - 27.122.129.133 (host ip) -> OK
    - Sandbox dotenv: OPENAI_API_BASE=http://host.docker.internal:<gateway_port>/sessions/<uid>/v1
- rllm.gateway.manager.container_reachable_url
    - ```python
    def container_reachable_url(url: str, backend: str | None) -> str:
        """Rewrite host loopback addresses so a process inside a container can reach the gateway.

        The gateway binds to 127.0.0.1 on the host; inside a Docker container
        that loopback addresses the container itself. Docker Desktop resolves
        ``host.docker.internal`` to the host, and on Linux Docker 20.10+ the
        same hostname works when the container is started with
        ``--add-host=host.docker.internal:host-gateway``. Keyed on the backend
        that actually provisioned the task's sandbox; a no-op for everything
        but ``docker`` (remote backends get a public tunnel URL instead).
        """
        if backend != "docker":
            return url
        return re.sub(
            r"(https?://)(?:127\.0\.0\.1|localhost)(:\d+|/|$)",
            r"\1host.docker.internal\2",
            url,
        )
    ``` 
- 주석에는 ``--add-host=host.docker.internal:host-gateway``가 필요하다고 적혀있지만, 실제 container 생성 시 적용되지 않음
    - rllm.sandbox.backends.docker.DockerSandbox
    - ```python
        self._container = self._client.containers.run(
            image,
            command="sleep infinity",
            name=f"rllm-sandbox-{name}",
            detach=True,
            remove=False,
        )
        ```

해결 방안:
- DockerSandbox에 extra_hosts 추가 (코드 주석과 일치)
  - `extra_hosts={"host.docker.internal": "host-gateway"}`
  - extra_hosts 설정 = container 생성 시 /etc/hosts에 hostname mapping을 추가하는 옵션


```


- 1) materialize
python3 << 'PY'
import subprocess, json
from datasets import load_dataset
from rllm.data.swebench_pro_builder import build_benchmark

local = set(subprocess.check_output(
    ["docker","images","jefzda/sweap-images","--format","{{.Tag}}"], text=True # text=False -> Bytes
).split())
ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
ids = [r["instance_id"] for r in ds if r["dockerhub_tag"] in local][:3]
print("Selected:", *ids, sep="\n ")
build_benchmark(name="swebench_pro", split="test",
    out_dir="/root/.rllm/datasets/swebench_pro", task_ids=ids, clean=True)
PY

- 2) eval
rllm eval swebench_pro \
  --agent mini-swe-agent \
  --sandbox-backend docker \
  --base-url http://27.122.129.133:8010/v1 \
  --model LGAI-EXAONE/EXAONE-4.5-33B \
  --max-examples 3 \
  --concurrency 1 \
  --no-ui \
  --no-snapshot # backend=docker -> meaningless (whereas in daytona or modal, it's useful.)

**oracle
rllm eval harbor:swebench-verified --agent oracle --sandbox-backend docker --max-examples 1

## snapshot

```bash
rllm snapshot create swebench_pro --sandbox-backend daytona 
```

1) rllm snapshot create swebench_pro --sandbox-backend daytona
  → task env를 Daytona cloud에 bake (base image + Dockerfile RUN + mini-swe-agent install 등)
  → ref(SNAPSHOT 이름)를 ~/.rllm/snapshots.json에 기록
    - env_key = (backend, base_image, Dockerfile RUN, install_script) -> content hash
      - ex. rllm-env-a1b2c3d4e5f6
    - group_id = 한 번의 snapshot create 실행 기록
    - ref = daytona/modal 측 실제 snapshot ID
      - modal: im-abc123...
      - daytona: rllm-env-a1b2c3d4e5f6 (보통 env_key와 동일한 이름)
      - eval 시: env_key → registry lookup → ref → remote에서 boot.
      

2) rllm eval ... --sandbox-backend daytona --snapshot
  → task마다 env_key로 registry 조회
  → hit → Daytona snapshot에서 boot (빠름)
  → miss → cold start (처음부터 build)

3) rllm eval ... --no-snapshot
  → registry 안 봄 → 항상 cold start

```json
{
  "version": 2,
  "envs": {
    "rllm-env-a1b2c3d4e5f6": {
      "backend": "daytona",
      "ref": "rllm-env-a1b2c3d4e5f6",
      "base_image": "jefzda/sweap-images:some-tag",
      "created_at": "2026-08-29T10:00:00+00:00",
      "expires_at": "2026-09-05T10:00:00+00:00"
    }
  },
  "groups": {
    "swebench-pro-first3-deadbeef": {
      "dataset": "swebench_pro",
      "backend": "daytona",
      "slice": {"kind": "max_examples", "value": 3},
      "tasks": [
        {"id": "instance_qutebrowser__...", "env_key": "rllm-env-a1b2c3d4e5f6"},
        {"id": "instance_gravitational__...", "env_key": "rllm-env-b2c3d4e5f6a"}
      ],
      "created_at": "2026-08-29T10:00:00+00:00",
      "ttl_hours": 168
    }
  }
}
```


## GatewayManager & rllm-model-gateway

```bash
Agent (OpenAI SDK)
    │  POST /sessions/{sid}/v1/chat/completions
    ▼
┌───────────────────────────────────────────
│  rllm-model-gateway (FastAPI)            
│  1. Session routing (URL에서 sid 추출) 
│  2. Request mutation (logprobs 등 주입)
│  3. Trace capture (token id, logprob 저장)
└───────────────────────────────────────────
    │  POST /v1/chat/completions
    ▼
Upstream worker (vLLM, LiteLLM, OpenAI 등)
```

- How to capture vLLM token_id / log_prob?
  - Middleware가 vLLM 전용 파라미터 주입 (rllm_model_gateway.middleware)
    - payload["logprob"] = True; payload["return_token_ids"] = True
    - ```json
      {
        "prompt_token_ids": [1, 2, 3, ...],
        "choices": [{
          "message": {"role": "assistant", "content": "hello"},
          "token_ids": [4, 5, 6],
          "logprobs": {
            "content": [
              {"token": "hello", "logprob": -0.5},
              ...
            ]
          }
        }]
      }
      ```
  - Proxy 응답 파싱 -> TraceRecord 저장 (rllm_model_gateway.proxy; data_process)

- Training rollout 1회 전체 타임라인

```bash
Trainer                          Agent (sandbox)              Gateway              vLLM
  │                                   │                         │                   │
  │ gw.start(rollout_engine)          │                         │                   │
  │ ─────────────────────────────────►│                         │ uvicorn 기동       │
  │                                   │                         │ worker 등록        │
  │                                   │                         │                   │
  │ gw.create_session("task-0")       │                         │                   │
  │ ─────────────────────────────────►│                         │ session 메타 저장  │
  │                                   │                         │                   │
  │ session_url 전달                   │                         │                   │
  │ ─────────────────────────────────►│                         │                   │
  │                                   │ POST .../sessions/task-0/v1/chat/completions
  │                                   │ ───────────────────────►│                   │
  │                                   │                         │ inject logprobs   │
  │                                   │                         │ ─────────────────►│
  │                                   │                         │◄──────────────────│ (token_ids, logprobs)
  │                                   │                         │ TraceRecord 저장  │
  │                                   │◄───────────────────────│ (sanitized resp)  │
  │                                   │                         │                   │
  │ gw.get_traces("task-0")           │                         │                   │
  │ ─────────────────────────────────►│                         │                   │
  │◄─ [TraceRecord(prompt_ids, completion_ids, logprobs)]        │                   │
  │                                   │                         │                   │
  │ trace → Step → PPO loss           │                         │                   │
```

- gateway launch 방식 (subprocess vs thread)
  - thread
    - trainer와 같은 프로세스 → Tinker in-process handler 주입 가능
    - startup/teardown 가벼움
    - eval (EvalGatewayManager)도 항상 thread
  - process
    - verl은 Ray/multi-process 환경 → gateway를 격리된 프로세스로 두는 편이 안전
    - subprocess는 stdout/stderr를 parent에 inherit (pipe 버퍼 hang 방지)
    - health poll로 /health 확인 후 worker 등록

```bash
[thread mode — Tinker/eval]
┌─────────────────────────────┐
│  trainer / eval runner      │
│  ├─ GatewayManager          │
│  └─ uvicorn thread ──► :9090│
└─────────────────────────────┘

[process mode — verl]
┌──────────────┐     ┌─────────────────────┐
│ trainer/verl │────►│ rllm-model-gateway  │
│ GatewayMgr   │     │ (subprocess) :9090  │
└──────────────┘     └─────────────────────┘
```

- 1 Worker = 1 upstream inference server 
  
```bash
Session "task-0"                    Worker pool
  ├─ trace 1 (turn 1)  ──route──►  http://vllm-0:8000
  ├─ trace 2 (turn 2)  ──route──►  http://vllm-0:8000  (sticky!)
  └─ trace 3 (turn 3)  ──route──►  http://vllm-0:8000

Session "task-1"                    Worker pool
  └─ trace 1           ──route──►  http://vllm-1:8000  (least-loaded)
```

- tunnel (rllm.gateway.tunnel)
  - `cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8010`

## SandboxedAgentFlow, BaseCliHarness

- rllm.engine.agentflow_engine:_run_flow_only
  - SandboxTaskHooks.setup()
  - run_agent_flow()
  - BaseCliHarness.run

```bash
AgentFlow (Protocol)
├── ReActHarness              # gsm8k, etc., sandbox X
└── SandboxedAgentFlow (ABC)  # swebench, etc.
    ├── BashHarness           # LLM=host, exec=sandbox
    └── BaseCliHarness
        └── MiniSweAgentHarness  # LLM=sandbox (gateway URL)
```

## rllm native swebench-pro vs harbor swebench-pro
rLLM의 SandboxTaskHooks / MiniSweAgentHarness / Gateway trace pipeline을 거치지 않고, Harbor trial이 전체 lifecycle을 관리

즉, rLLM Native는 agent 실행과 verifier가 rLLM engine에서 분리되고, Harbor trial은 하나의 Trial 객체가 env → agent → verifier를 순서대로 orchestrate

| Component | Native (mini-swe-agent) | Harbor trial (`harbor:mini-swe-agent`) |
|-----------|-------------------------|------------------------------------------|
| **Sandbox** | rLLM SandboxTaskHooks | Harbor EnvironmentFactory |
| **Agent** | rLLM MiniSweAgentHarness | Harbor AgentFactory scaffold |
| **LLM trace** | rLLM Gateway (token id, logprob) | ATIF trajectory.json |
| **Verifier** | rLLM ShellScriptEvaluator (agent 후 별도 실행) | Harbor Verifier (trial pipeline 내) |
| **결과** | Episode + gateway traces | Episode + harbor_reward artifact |

AgentFlowEngine
  → SandboxTaskHooks (sandbox 생성/해제)
  → MiniSweAgentHarness (sandbox 안에서 mini-swe-agent CLI)
  → GatewayManager (LLM proxy + trace 저장)
  → ShellScriptEvaluator (tests/test.sh 실행)

HarborRuntime
  → harbor.trial.Trial (환경 + agent + verifier 일괄 실행)
  → Harbor agent scaffold (Harbor 쪽 mini-swe-agent)
  → episode.artifacts["harbor_reward"] 에 결과 저장
  → HarborEvaluator

- What is Harbor Trial?
  - Task
  - TrialConfig (rllm.integrations.harbor.trial_helper::build_harbor_trial_config)
  - Trial.run()
    - Environment setup
      - task.toml + environment/Dockerfile -> start container
      - /logs/agent, /logs/verifier 등 directory mount
    - Agent setup
      - Harbor AgentFactory -> scaffold install
    - Agent execution
    - Verification
      - tests/ directory upload to sandbox's /tests
      - tests/test.sh 실행 -> stdout save to /logs/verifier/test-stdout.txt
      - test.sh -> write reward.txt or reward.json
    - Cleanup
    - TrialResult
      ```python
      class TrialResult(BaseModel):
        id: UUID = Field(default_factory=uuid4)
        task_name: str
        trial_name: str
        trial_uri: str
        task_id: LocalTaskId | GitTaskId | PackageTaskId
        source: str | None = None
        task_checksum: str
        config: TrialConfig
        agent_info: AgentInfo
        agent_result: AgentContext | None = None
        verifier_result: VerifierResult | None = None
        exception_info: ExceptionInfo | None = None
        started_at: datetime | None = None
        finished_at: datetime | None = None
        environment_setup: TimingInfo | None = None
        agent_setup: TimingInfo | None = None
        agent_execution: TimingInfo | None = None
        verifier: TimingInfo | None = None
      ```
      - rllm trial_helper::trial_result_to_reward -> extract reward

rLLM -> Harbor 단일 진입점: rllm.integrations.harbor.trial_helper::run_harbor_task
  - rllm.integrations.harbor.runtime::HarborRuntime

Container mount structure (TrialPaths docstring)
```bash
/                          (container)
├── logs/
│   ├── agent/             ← trial_dir/agent/ 과 mount
│   └── verifier/          ← trial_dir/verifier/ 과 mount
├── tests/                 ← verifier가 upload
└── solution/              ← oracle만 copy
```

After trial, trials/<trial_name>
```bash
trials/<trial_name>/
├── config.json         # TrialConfig (재현용)
├── result.json         # TrialResult (reward, timing, exception)
├── trial.log           # Harbor 내부 로그
├── agent/
│   └── trajectory.json # ATIF 포맷 agent trajectory
├── verifier/
│   ├── test-stdout.txt
│   ├── reward.txt      # ← 점수 (0.0 or 1.0 등)
│   └── reward.json
└── artifacts/          # task에서 수집한 부가 파일
```

- Why HarborRuntime uses different Protocol for eval and training?

Engine 이해 필요: AgentFlowEngine vs RemoteAgentFlowEngine
- 둘 다 Gateway로 LLM trace를 수집한다는 점은 같음
- 차이는 "누가 env / agent / verifier를 orchestrate하느냐"

1) AgentFlowEngine - rLLM이 지휘

rLLM Engine (AgentFlowEngine)
  ├─ hooks.setup()        ← SandboxTaskHooks: 샌드박스 생성
  ├─ agent_flow.run()     ← MiniSweAgentHarness 등
  ├─ gateway traces       ← token id / logprob
  ├─ evaluator            ← ShellScriptEvaluator (test.sh 별도 실행)
  └─ hooks.teardown()

- rLLM 역할: sandbox, gateway, evaluator까지 단계별로 분리 제어
- 쓰는 곳: rllm eval (항상), native 학습 (agent_flow + hooks/evaluator)
- Harbor eval도 여기 — HarborRuntime을 AgentFlow로 끼워 넣음
  - Trial은 Harbor가 돌리지만, Engine 파이프라인은 AgentFlowEngine

2) RemoteAgentFlowEngine — 원격 runtime이 지휘

rLLM Engine (RemoteAgentFlowEngine)
  ├─ gateway session 생성 → inference URL 전달
  ├─ runtime.execute_tasks()  ← env + agent + verifier 전부 원격에서
  ├─ gateway traces 수집
  └─ trace + remote reward → Episode 조립

- rLLM 역할: gateway + trace merge + Episode 만들기
- 원격 역할: 환경, 에이전트, 채점까지 한 덩어리로 실행
- 쓰는 곳: remote_runtime.enabled=true 학습 (Harbor, AgentCore 등)

UnifiedTrainer는 셋 중 하나를 선택:
  1. agent_flow + evaluator → AgentFlowEngine (gateway-based, local)
  2. remote_runtime → RemoteAgentFlowEngine (gateway-based, remote)
  3. workflow_class → UnifiedWorkflowEngine (direct)

Engine: AgentFlowEngine
- rllm 역할: sandbox hooks, gateway, evaluator
- Remote 역할: AgentFlow
```bash
rllm eval --agent harbor:mini-swe-agent
  → load_agent() → HarborRuntime
  → AgentFlowEngine._run_flow_only()
      config.base_url = gateway session URL
      HarborRuntime.arun(task, config)
        → run_harbor_task(inference_url=config.base_url)
        → Trial.run()  # env + agent + verifier
        → outcome_to_episode()  # ATIF steps + harbor_reward artifact
  → gateway.aget_traces() → enrich_episode_with_traces()
  → HarborEvaluator (read reward from artifact)
```

Engine: RemoteAgentFlowEngine
- rllm 역할: gateway, trace merge
- Remote 역할: env + agent + reward 전부
```bash
rllm train ... remote_runtime.enabled=true backend=harbor
  → UnifiedTrainer → RemoteAgentFlowEngine
  → process_task_with_retry():
      1. gateway.acreate_session(session_id)
      2. inference_url = gateway.get_session_url(session_id)
      3. HarborRuntime.execute_tasks([TaskSubmission(...)])
           → run_harbor_task(inference_url=..., model=MODEL_PLACEHOLDER)
           → Trial.run()
           → RemoteTaskResult(reward, trial_uri, ...)
      4. gateway.aget_traces(session_id)
      5. _build_episode(traces, result)  ← Episode 조립
```

```bash
                    HarborRuntime
                         │
          ┌──────────────┴──────────────┐
          │                             │
   AgentFlow protocol              RemoteAgentRuntime protocol
   (eval)                          (train)
          │                             │
   arun(task, config)            execute_tasks(submissions)
   → Episode                      → RemoteTaskResult (reward)
          │                             │
          └────── run_harbor_task() ────┘   ← 실제 Trial 실행은 공통
```
- HarborRuntime은 eval에서 AgentFlow로 사용됨
  - rllm eval --agent harbor:mini-swe-agent
  - HarborRuntime.arun()이 AgentFlow 규격을 충족
    - Task, AgentConfig를 받고 Episode 반환
- 반면 train에서는 RemoteAgentRuntime으로 사용됨
  - remote_runtime.enabled=true, backend=harbor
  - HarborRuntime.execute_tasks()이 RemoteAgentRuntime 규격을 충족

- 학습 시 Native (Harbor ❌) — AgentFlowEngine
| 구성요소 | 담당 |
|---|---|
| **Sandbox** | rLLM `SandboxTaskHooks` |
| **Agent** | rLLM `MiniSweAgentHarness` (sandbox 안 CLI) |
| **LLM trace** | rLLM Gateway (token id, logprob) |
| **Verifier** | rLLM `ShellScriptEvaluator` (agent **후** `tests/test.sh` 별도 실행) |
| **Reward** | evaluator 결과 |
| **Episode** | gateway traces + enrich + evaluator reward |

- 학습 시 Harbor (Harbor ✅) — RemoteAgentFlowEngine + HarborRuntime
| 구성요소 | 담당 |
|---|---|
| **Sandbox + Agent + Verifier** | Harbor `Trial.run()` 한 번에 |
| **LLM trace** | rLLM Gateway (`inference_url`을 Trial에 전달) |
| **Reward** | Harbor verifier → `reward.txt` / `reward.json` |
| **Episode** | gateway traces + remote reward (`_build_episode`) |


## Registry

## Gateway Store


## 컨테이너 디스크: overlayfs와 호스트 디스크의 관계

**증상**: 학습 중 `/`가 100%(0바이트)가 되어 Ray가 `is over 95% full`을 뱉고 쓰기가 ENOSPC로 실패.
컨테이너 안에서 `du`로는 2.6G밖에 안 보이는데 `df`는 1.7T 사용 중.

### 1. 이 호스트에는 컨테이너 저장소가 두 개 있다

| 런타임 | 저장 위치 | 디스크 | 내용 |
|---|---|---|---|
| **containerd** | `/var/lib/containerd/.../snapshots/` | **md0** (1.8T) | **개발 컨테이너 자신**의 레이어 |
| **Docker** | `/raid/docker_home` | md1 (28T) | 마운트된 socket으로 만드는 sandbox/이미지 |

```bash
# 내 컨테이너의 upperdir이 어디인지 = 내 쓰기가 어느 디스크를 먹는지
cat /proc/self/mountinfo | awk '$5=="/"'
# → upperdir=/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots/64546/fs
```

`docker info`의 `Docker Root Dir`은 **socket으로 말을 거는 Docker 데몬**의 것이다.
"docker 이미지를 md1로 보냈으니 안전"해도, **컨테이너 안에서 파일을 쓰는 행위 자체는 md0을 먹는다.**
이 비대칭이 혼란의 근원.

### 2. overlayfs 구조 — 쓰기는 upperdir로

```
merged (내가 보는 /)
  ├── upperdir  = snapshots/64546/fs    ← 내가 쓴 것만. 컨테이너별 1개. 쓰기가 여기 쌓임
  └── lowerdir  = snapshots/190~64545   ← 이미지 레이어. 읽기 전용, 여러 컨테이너 공유
```

컨테이너를 지우면 upperdir이 사라져 공간이 돌아온다 → 그래서 평소에 잘 안 보인다.

### 3. `df /`가 호스트 숫자를 보여주는 이유

overlayfs는 **자기 크기가 없다.** `statfs()`가 upperdir을 담은 파일시스템으로 그대로 전달된다.

```
컨테이너:  overlay    1.8T  1.7T  0  100%  /
호스트:    /dev/md0   1.8T  1.7T  0  100%  /      ← 같은 디스크
```

숫자가 같은 건 우연이 아니다. 그리고 `Avail`은 **그 디스크를 쓰는 모든 컨테이너/호스트가 공유**한다.
→ **내가 아무것도 안 써도 남이 채우면 내 `df`도 0이 된다.**

### 4. bind mount만 컨테이너 레이어를 벗어난다

```bash
findmnt -no SOURCE,FSTYPE -T <경로>   # 경로별 실제 디스크 확인
```

```
/                 overlay              ← md0  ⚠
/tmp              overlay              ← md0  ⚠
/root             overlay              ← md0  ⚠
/home/devuser     overlay              ← md0  ⚠
/var              overlay              ← md0  ⚠
/workspace/rllm   gpfs[/home/...]      ← gpfs ✅  bind mount
/raid             /dev/md1 ext4        ← md1  ✅  bind mount
/dev/shm          tmpfs                ← RAM (디스크 아님)
```

### 5. 실무적으로 터지는 지점 — 도구 기본 경로

| 도구 | 기본 경로 | 컨테이너에서 |
|---|---|---|
| Ray (session, object spill) | `/tmp/ray` | **md0** ⚠ |
| HuggingFace 캐시 | `~/.cache/huggingface` | **md0** ⚠ |
| pip / uv 캐시 | `~/.cache` | **md0** ⚠ |
| torch.compile (inductor) | `/tmp/torchinductor_*` | **md0** ⚠ |
| Hydra output | `./outputs` (cwd 상대) | cwd에 따라 |
| rLLM `episode_log_dir` | `logs/...` (cwd 상대) | cwd에 따라 |

규모 감각: Qwen3.5-4B 스냅샷 하나 **8.8GB**, 에피소드 JSON 개당 **2.5MB**(32개/step).
기본값을 그대로 두면 모델 캐시 하나로 컨테이너 레이어가 10GB 부푼다.

> 이번엔 `cwd`가 `/workspace/rllm`(gpfs)여서 Hydra output/transcript는 처음부터 md0을 피했다.
> 의도가 아니라 마운트 구조 덕. 운에 맡기지 말 것.

### 6. 컨테이너 안에서 범인을 못 찾는 이유

- 마운트 네임스페이스 분리 → 호스트 `/var/lib/containerd`가 **보이지 않는다**
- 내가 보는 `/usr` 12G, `/opt` 4.9G는 **lowerdir(공유 이미지 레이어)**. 내 upperdir 실사용량은 훨씬 작다
- 즉 **`du`(내 merged view) ≠ `df`(호스트 디스크 전체)**

"내 건 2.6G인데 왜 1.7T가 차 있지?"의 답. 나머지는 다른 컨테이너들의 upperdir + containerd 이미지
스냅샷이고 **호스트에서만 보인다.**

### 7. 호스트에서 범인 찾기

```bash
sudo du -x -h -d1 / 2>/dev/null | sort -h | tail -20   # md0만 (다른 fs 제외)
sudo du -sh /var/lib/containerd /var/lib/docker         # 컨테이너 저장소 두 곳
sudo du -sh /var/log; sudo journalctl --disk-usage
sudo lsof -nP +L1 2>/dev/null | awk '/deleted/'         # 삭제됐지만 열려있는 파일
                                                        # ↑ du는 안 보이는데 df만 찰 때의 주범
```

### 8. 규칙

**컨테이너에서는 캐시·temp·출력을 전부 명시적으로 bind mount 볼륨으로 보낸다. 기본값을 믿지 않는다.**

새 환경 진입 시 먼저:

```bash
cat /proc/self/mountinfo | awk '$5=="/"'   # upperdir이 어느 디스크인가
findmnt -no SOURCE,FSTYPE -T /tmp          # overlay면 /tmp 쓰기 금지
df -h /                                    # Avail은 남과 공유되는 값
```

그리고 큰 볼륨 하나에 몰아준다 (레시피의 `RLLM_SCRATCH`가 이 역할):

```bash
export RLLM_SCRATCH=/raid/rllm-work
export HF_HOME=$RLLM_SCRATCH/hf
export RLLM_HOME=$RLLM_SCRATCH/rllm-home
export RAY_TMPDIR=$RLLM_SCRATCH/ray
export TMPDIR=$RLLM_SCRATCH/tmp        # torch inductor 등 /tmp 기본값까지 커버
export UV_CACHE_DIR=$RLLM_SCRATCH/uv
```

### 9. 이번 사고 결론 — 그리고 오진

- **직접 원인**: 호스트 md0(`/`) 100%, 0바이트. 컨테이너 `/`가 그 위에 있어 같이 0
- **내 작업물 위치**: SWE 이미지 71개·모델·데이터셋 전부 md1/gpfs. 여기까진 맞음

**오진했던 것**: 진행 중에 `du`로 "내 기여분은 2.6G뿐, 나머지는 컨테이너 밖 소비자"라고 결론냈다.
그런데 **학습 프로세스를 종료한 직후 `/`가 0 → 14G로 회복**했다. 종료 시점과 정확히 일치하므로
상당 부분이 내 프로세스가 붙잡고 있던 것이었을 가능성이 높다.

원인은 이 문서 6절에서 스스로 적어둔 함정이다:

> **`du`는 "삭제됐지만 프로세스가 열고 있는 파일"을 못 본다.**

Ray object store, vLLM/torch 컴파일 temp 같은 것들이 unlink된 뒤에도 fd로 살아 있으면
`df`는 차 있는데 `du`로는 안 보인다. 프로세스가 죽는 순간 공간이 돌아온다.
게다가 이번 런은 `TMPDIR`을 설정하기 *전에* 시작했으므로 vLLM 8개의 컴파일 캐시가 md0으로 갔다.

**교훈 두 개:**

1. `du`만 보고 "내 것이 아니다"라고 결론내지 말 것. `df`와 `du`가 안 맞으면 **거의 항상**
   삭제-열린 파일이다. 확인:
   ```bash
   sudo lsof -nP +L1 | awk '/deleted/'          # NLINK=0 인 열린 파일
   # 또는 결정적 검증: 의심 프로세스를 멈추고 df가 회복되는지 본다
   ```
2. 컨테이너 안에서 대형 워크로드를 돌릴 때 `TMPDIR`은 **시작 전에** 옮겨야 한다.
   런 도중에 env를 바꿔도 이미 뜬 프로세스에는 적용되지 않는다.

## compact_filtering: 인프라 실패가 모델 실패로 학습되는 문제

**증상**: 없다. 그게 문제다. 학습은 정상적으로 돌고 리워드도 나오는데, verifier가 죽거나
프롬프트가 컨텍스트를 넘겨 죽은 에피소드가 **reward 0으로 그룹에 들어간다.**
"정책이 못 풀었다"와 "환경이 고장났다"가 구분되지 않는다.

### 1. 이름이 오해를 부른다 — 마스킹이 아니라 제거

config 키가 `mask_*`라서 loss masking처럼 읽히는데, 실제 동작은 에피소드를 배치에서 빼는 것이다.

```python
# rllm/trainer/algorithms/transform.py:121
if compact_filtering_config and compact_filtering_config.should_mask(termination_reason):
    continue                       # 에피소드가 trajectory 가 되지 못한다
```

trajectory로 변환조차 되지 않으므로 GRPO 그룹에서 사라진다. **그룹 크기가 줄면 어드밴티지의
평균·표준편차가 남은 롤아웃만으로 계산된다.** 8개 중 3개가 걸러지면 5개짜리 그룹으로 정규화된다.
한 태스크의 롤아웃이 전부 걸러지면 프롬프트 그룹 전체가 버려진다
(`buffer.py:365` → `buffer.py:210` `filter_reason = "compact_filtering"`).

### 2. 판정은 `termination_reason` 하나로만 — 그리고 그걸 붙이는 코드가 없었다

`should_mask`(`rllm/trainer/algorithms/config.py:143`)는 `termination_reason`만 본다.
그런데 네이티브 경로(`AgentFlowEngine` + `MiniSweAgentHarness`)에는 생산자가 사실상 없었다.

| 이유 | 생산자 | 네이티브 경로 |
|---|---|---|
| `ERROR` | `agentflow_engine.py` 재시도 소진 | 작동 |
| `ENV_DONE` | `agentflow_engine.py` 기본값 | **그 외 전부 여기로** |
| `MAX_TURNS_EXCEEDED` | `rllm/workflows/*` (구형 Workflow 경로) | 없음 |
| `MAX_PROMPT_LENGTH_EXCEEDED` | `openai_engine.py`, Harbor `trial_helper.py` | 없음 |
| `TIMEOUT` / `MAX_RESPONSE_LENGTH_EXCEEDED` | Harbor `trial_helper.py` | 없음 |

**분류 로직이 Harbor 통합에만 있고 네이티브 경로에는 대응물이 없다**는 게 핵심이었다.
`trial_helper.map_termination_reason()`은 `finished`/`timed_out`/`exception_type`을 받아
예외 타입을 매핑하는데, CLI 하니스 경로에는 그런 함수가 아예 없었다.

**실측**: fn50 런의 에피소드 414개 **전부** `env_done`. 턴 한도를 소진한 것, verifier가
2시간 58분 폭주한 것, 컨텍스트 초과로 72턴에서 끊긴 것(프롬프트 122,773) 전부 포함.
그래서 지표가 이렇게 나온다:

```
groups/num_trajs_before_filter  64  →  after_filter  64
batch/groups_before_filter       8  →  after_filter   8
```

마스크 다섯 개를 켜두고 **한 건도 걸러지지 않았다.**

### 3. 왜 각 지점에서 정보가 사라지는가

**verifier** — `except Exception`이 삼킨다. 주석은 이렇게 정당화한다:

> Verifier exit != 0 is the *expected* outcome when an agent didn't solve the task

맞는 말이지만, 그 clause는 **"채점해서 0"과 "채점 자체를 못 함"을 구분하지 않는다.**
reward 파일이 없어도 `reward=0.0`으로 내려간다. `_read_reward_from_sandbox`가
`{"error": "no reward file found"}`로 이미 구분해두는데 소비하는 코드가 없었다.

**컨텍스트 초과** — vLLM이 400을 주면 게이트웨이가 에러 바디로도 트레이스를 만든다.
`choices`가 없어 completion 토큰이 비고, 엔진은 그걸 "말단 malformed 트레이스"로 보고
**경고만 남기고 버린다**. 그 자리 주석이 이미 원인을 알고 있었다:

> Common case: vLLM returns an empty body on the final call (e.g. prompt hit max_model_len...)
> Drop the trailing trace rather than burn the whole rollout

버리는 건 맞다(학습할 토큰이 없다). 문제는 **이유가 실려 있지 않아** 버릴 수밖에 없었던 것.

**턴 소진** — CLI 하니스는 `run()`에서 `None`을 반환하고 엔진이 트레이스로 에피소드를 만든다.
그래서 하니스에는 도장 찍을 Episode가 없다. 대신 하니스는 CLI에 넘긴 예산(`step_limit`)을
소유하고 있고, 턴당 트레이스가 1개다.

### 4. 고친 방법 — 공통 seam은 `ENV_DONE` 기본값 직전

`termination_reason`이 `None`일 때만 `ENV_DONE`이 채워지므로, 세 경로 모두 그 전에 값을 넣는다.

```
1. 게이트웨이 업스트림 표식   (enrich 단계에서 episode 에 직접 기록)
2. evaluator 표식             EvalOutput.metadata["termination_reason"]
3. 턴 예산                    len(traces) >= agent_flow.step_limit
4. ENV_DONE                   (기본값)
```

순서에 의미가 있다. 컨텍스트 초과로 죽은 에피소드는 턴을 다 못 썼으니 3번에 안 걸리고,
반대로 턴을 다 쓴 에피소드의 verifier가 타임아웃되면 2번이 이겨 `TIMEOUT`이 된다.
**인프라 원인이 정책 원인보다 앞선다.**

게이트웨이 쪽은 `classify_upstream_error()`가 non-2xx를 구조화된 표식으로 바꿔
트레이스 `metadata`에 넣는다: `{"status", "kind", "message"}`.

### 5. vLLM 에러 분류의 함정

vLLM은 컨텍스트 초과를 **기계가 읽을 코드 없이** 준다. `ErrorInfo`는
`(message, type, param, code)`이고 `code`는 그냥 HTTP 상태다. 결국 **메시지 문자열 매칭**뿐이다.

그리고 요청을 거부하는 경로가 **둘**인데 문구가 다르다 (핀된 vLLM 0.22.1 기준):

| 경로 | 문구 |
|---|---|
| `renderers/params.py:428` | `This model's maximum context length is N tokens. However, you requested ... Please reduce the length of the input prompt ...` |
| `v1/engine/input_processor.py:410` | `The prompt (length N) is longer than the maximum model length of M.` |

`"maximum context length"`만 매칭하면 **두 번째 경로를 통째로 놓친다**. 구버전 얘기가 아니라
핀된 버전에 실재하는 경로다. 바디 형식도 버전마다 다르다 — 0.22.1은 wrapped
(`{"error": {"message": ...}}`), 구버전·타 서버는 flat (`{"object": "error", "message": ...}`).

### 6. 수정 후 실제로 걸러지는 것

| 실패 | `TerminationReason` | 마스크 |
|---|---|---|
| 프롬프트 컨텍스트 초과 | `MAX_PROMPT_LENGTH_EXCEEDED` | `mask_max_prompt_length_exceeded` |
| 업스트림 기타 실패 (500, tool-choice 거부) | `ERROR` | `mask_error` |
| verifier 타임아웃 | `VERIFIER_TIMEOUT` (구 `TIMEOUT`) | `mask_verifier_timeout` |
| 에이전트 타임아웃 (`[agent] timeout_sec`) | `AGENT_TIMEOUT` (cli_harness가 `SandboxExecTimeout`을 잡아 stamp) | `mask_agent_timeout` -- 예산 소진이라 기본 off, 리워드 x0.5 |
| reward 파일 부재 (채점 불가) | `ERROR` | `mask_error` |
| 턴 예산 소진 | `MAX_TURNS_EXCEEDED` | `mask_max_turns_exceeded` |
| 샌드박스 생성 실패 (재시도 소진) | `ERROR` | `mask_error` |

**여전히 안 걸러지는 것**: `MAX_RESPONSE_LENGTH_EXCEEDED`(네이티브 경로에 생산자 없음.
다만 턴당 출력이 `max_tokens`로 제한되고 그 한도에 닿으면 거부가 아니라
`finish_reason: "length"`로 정상 응답하므로 애초에 다른 실패 유형),
그리고 **streaming 경로의 업스트림 실패** — non-2xx면 SSE가 아니라 JSON 바디가 와서
`chunks`가 비고 `if chunks:` 가드 때문에 트레이스가 아예 안 만들어진다. 붙일 대상이 없다.

### 7. 켜는 것 자체가 학습 데이터를 줄인다

이게 함정이다. **필터가 작동하기 시작하면 그만큼 데이터가 사라진다.**
실측으로 오답의 38~55%가 턴 소진이었고, 그 에피소드들이 지금까지는 reward 0으로 그룹에
들어갔지만 이제 제거된다. 그룹이 `min_trajs_per_group` 아래로 떨어지면 태스크가 통째로 빠지고
(`buffer.py:200`), 전 그룹이 비면 스텝이 날아간다.

**턴 소진은 마스킹 대상이 아니라고 본다.** verifier가 죽은 것과 달리
"정책이 주어진 예산 안에 못 풀었다"는 **진짜 학습 신호**다. 그걸 버리면 어려운 태스크가
조용히 커리큘럼에서 빠지고 빨리 끝나는 에피소드만 남는다.

### 8. 규칙

1. **`enable: true`만 보고 필터가 동작한다고 믿지 말 것.** 마스크는 이유를 *소비*할 뿐이고,
   이유를 *생산*하는 코드는 경로마다 다르다. 검증은 config가 아니라 실측으로:
   ```
   episode/termination_reason/*        (buffer.py:344, 이유별 카운트)
   groups/num_trajs_before/after_filter
   ```
   전부 `env_done` 한 종류면 필터는 장식이다.
2. **"인프라 실패인가 정책 실패인가"를 구분하는 지점을 먼저 찾을 것.** 이 스택에서는
   `ENV_DONE` 기본값이 그 지점이었고, 그 앞에 무엇을 끼울 수 있는지가 설계의 전부였다.
3. **삼키는 `except Exception`은 두 가지를 합친다.** 여기서는 "채점해서 0"과 "채점 못 함"이었다.
   리워드가 같은 값으로 내려가는 두 경로가 있으면 거의 항상 구분해야 할 것이 섞여 있다.
4. **에러 분류를 문자열 매칭으로 할 수밖에 없다면, 서버가 그 문구를 내는 경로를 전부 찾을 것.**
   한 경로만 보고 마커를 정하면 나머지가 조용히 generic으로 떨어진다.
5. **필터를 켜기 전에 무엇을 얼마나 잃는지 먼저 재라.** 걸러지는 비율이 과반이면
   그건 필터링 문제가 아니라 데이터/예산 설계 문제다.
