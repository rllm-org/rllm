# SWE-bench Pro subset (`swebenchpro_100`) 평가

- Harbor Registry의 `swebenchpro`에서 100개를 기반으로 subset을 만들고, `rllm eval`로 평가하기 위한 가이드. 
- 환경 설정: Docker Container 빌드, rLLM 패키지 설치 등 [`README.md`](../README.md) 0장 참고.

## 왜 subset인가

- 731개 전체 평가 소요시간 과다
- 731개 중 100개 subset에 대해서만 oracle test 검증 완료
- subset은 저장소, 언어 비율을 전체와 유사한 분포로 샘플링하여 구축 (go 38, python 36, js 23, ts 3; 11개 저장소). id 목록은 `subset-ids.txt` 참고.

## 1. 데이터셋 준비

```bash
# Host와 같은 경로로 마운트된 볼륨 (../README.md 0.3)
export RLLM_HOME=/raid/rllm-work/rllm-home

# Harbor harness 전용. (default=900) 
export RLLM_HARBOR_SESSION_TIMEOUT_S=3300        

# harbor:swebenchpro pull -> subset 추출 ($RLLM_HOME/datasets/swebenchpro_100/) -> registry 등록
python recipe/eval/swebench-pro/prepare_swebenchpro_subset.py
```

## 2. Oracle test (생략 가능, 권장)

정답 패치를 적용해 Verifier가 1.0을 주는지 확인한다. LLM을 호출하지 않지만 CLI가 provider 설정을 요구하므로 죽은 포트를 준다.

```bash
rllm eval swebenchpro_100 \
    --split test \
    --agent harbor:oracle \ 
    --evaluator harbor_reward_fn \
    --sandbox-backend docker \ 
    --concurrency 8 \
    --sandbox-concurrency 8 \
    --no-ui \
    --base-url http://127.0.0.1:1/v1 \
    --model oracle-dummy
```

100/100이 나와야 한다. 0점이나 Errors가 있으면 모델을 돌리기 전에 오류를 해결한다(자원, 이미지, 디스크). 동시 8개 기준 약 30~40분.

## 3. 모델 평가

**Harbor harness 권장.** 재현성을 위해 `../config/qwen3_5.yaml`의 sampling params를 그대로 사용. 다른 값을 쓰려면 해당 파일 수정 대신 새 yaml을 만들어 `--sampling-params @<경로>`로 지정한다.

```bash
# vLLM/SGLang serving model
rllm eval swebenchpro_100 
    --split test \
    --agent harbor:mini-swe-agent \
    --evaluator harbor_reward_fn \
    --sandbox-backend docker \
    --concurrency 8 \
    --sandbox-concurrency 8 \
    --no-ui \
    --base-url http://127.0.0.1:8000/v1 \
    --model Qwen/Qwen3.5-4B \
    --sampling-params @recipe/eval/config/qwen3_5.yaml

# proprietary model: rllm model setup -> API KEY 등록
rllm model setup

rllm eval swebenchpro_100 
    --split test \
    --agent harbor:mini-swe-agent \
    --evaluator harbor_reward_fn \
    --sandbox-backend docker \
    --concurrency 8 \
    --sandbox-concurrency 8 \
    --no-ui \
    --model gpt-5.5 \
    --sampling-params @recipe/eval/config/gpt-5_5.yaml
```

- `--evaluator harbor_reward_fn`은 필수다. 이름이 `harbor:`로 시작하지 않는 데이터셋에서 Harbor harness를 쓸 때 CLI가 Evaluator를 스스로 찾지 못한다.
- pass@k가 필요하면 `--attempts 4`처럼 설정한다. 
- Harbor의 mini-swe-agent는 태스크마다 컨테이너 안에서 설치되므로 native 방식에 비해 태스크당 1~2분이 더 소요된다.

**native rLLM harness.** 같은 사본을 rLLM 하네스로 돌린다. `--evaluator`는 필요 없고, 채점은 태스크의 `tests/test.sh`를 rLLM이 직접 실행한다.
학습(`recipe/qwen3_5_swe_grpo`)과 같은 경로라 학습 전후 비교에는 이쪽이 더 적합하나, 어떤 방식으로 평가하든 이론적으로는 같은 결과가 나와야 정상이다.

```bash
rllm eval swebenchpro_100 
    --split test \
    --agent mini-swe-agent \
    --agent-image auto \
    --sandbox-backend docker \
    --concurrency 8 \
    --sandbox-concurrency 8 \
    --no-ui \
    --base-url http://127.0.0.1:8000/v1 
    --model Qwen/Qwen3.5-4B \
    --sampling-params @recipe/eval/config/qwen3_5.yaml
```

두 harness의 차이는 `../README.md` Appendix 2 참고. 

## 4. 결과

`$RLLM_HOME/eval_results/swebenchpro_100_<model>_<timestamp>/`에 `results.json`과 `episodes/`. 콘솔의 `Errors`는 인프라 실패(타임아웃, 이미지 빌드 실패)이고 0점과 다르다.
Harbor harness는 trial 로그를 `$RLLM_HOME/harbor_trials/<task>-<n>__<실행태그>/`에 추가로 남긴다(실행마다 쌓이므로 주기적으로 지우는 것을 권장한다).

## Appendix. 자원과 시간

- 동시 실행 수 × (CPU 4, 메모리 16 GB)가 호스트 여유 안에 들어야 한다. 동시 8이면 CPU 32, 메모리 128 GB.
- 처음 실행 때 태스크별 베이스 이미지(`jefzda/sweap-images:<tag>`, 약 4 GB × 100)를 받는다. 지우지 않으면 재실행은 빠르다.
- Harbor harness는 태스크별 파생 이미지(약 7 GB)를 만들고 정상 종료 시 지운다. 프로세스를 강제 종료하면 컨테이너·네트워크·이미지가 남으니 `docker ps -a | grep -- -main-1`, `docker images | grep -- -main` 으로 확인한다.

## Appendix. subset vs original Harbor dataset

사본(`$RLLM_HOME/datasets/swebenchpro_100/`)만 바뀌고 Harbor 캐시(`~/.cache/harbor/tasks/`)는 그대로다.

| 변경 | 원본 | 사본 | 이유 |
| --- | --- | --- | --- |
| `task.toml` `[environment].cpus` | 1 | 4 | `pytest -n auto`(ansible), `go test ./...`(Go 4개 저장소), Jest가 호스트 nproc(256)만큼 병렬로 뜬다. CPU 1에서는 Go 검증기가 3000초를 넘기고(정답 패치도 0점) ansible 워커가 죽는다. |
| `task.toml` `[environment].memory_mb` | 4096 | 16384 | 같은 병렬도에서 4 GB는 OOM(`signal: killed`, `Killed`, pytest `BrokenPipeError`). 4 GB로 재검증 시 100개 중 13개 실패, 16 GB에서 전부 통과. |
| element-web 8개의 `tests/run_script.sh` | Jest에 `--maxWorkers=1 --forceExit` | 두 플래그 제거 (업스트림 `SWE-bench_Pro-os`와 동일) | Harbor 어댑터가 추가한 플래그. 단일 워커에서는 테스트 파일 간에 WASM(`matrix-wysiwyg`) 상태가 공유되어 instance `aec454dd`의 정답 패치가 항상 0점. 업스트림에는 이 플래그가 없다. |
| instance id 매칭 | 일부 소문자 (`instance_nodebb__...`) | `subset-ids.txt`의 원본 표기 | 대소문자 무시로 매핑. 등록 행에는 사본 디렉토리 이름과 원본 id 둘 다 남긴다. |
