# 인간 행동 모방 기준선

이 단계의 목표는 Brettspielwelt 인간 로그로 강화학습을 바로 시작하는
것이 아니라, 현재 상태에서 인간이 선택한 합법 행동을 예측할 수 있는지
검증하는 것입니다.

## 모델

`CandidatePolicy`는 고정된 전체 행동 공간을 분류하지 않습니다. 엔진이
상태마다 생성한 합법 행동을 각각 인코딩하고, 공개 상태와 결합해 후보별
점수를 계산합니다. 따라서 추론 결과는 항상 합법 행동 중 하나입니다.

첫 기준선은 다음 입력을 고정 길이 벡터로 사용합니다.

- 자신의 손패
- 상대적 좌석별 남은 카드 수, 획득 트릭 수와 점수
- 현재 소원, 순위, 티츄 및 그랜드 티츄 선언
- 현재 트릭의 최근 행동 12개(순서 보존)
- 각 합법 행동의 종류, 카드, 높이와 부가 정보

손패 Set Attention과 트릭 Transformer를 추가할 때도 동일한 JSONL과
후보 행동 점수 방식을 유지할 수 있습니다.

## 빠른 파이프라인 확인

학습 이미지를 처음 빌드하고, 학습/검증 각각 일부 레코드만 사용합니다.

```powershell
docker compose build train-bc
docker compose run --rm train-bc `
  --epochs 1 `
  --max-train-records 2000 `
  --max-validation-records 500 `
  --output models/experiments/behavior-cloning-smoke.pt
```

`train-bc` 서비스는 NVIDIA GPU 한 개를 예약합니다. CUDA가 정상 연결되면
학습 시작 메시지의 `device`가 `cuda`로 출력되며, 사용할 수 없으면
`--device auto`의 기본 동작에 따라 CPU로 전환됩니다.

출력에는 다음 지표가 JSON으로 표시됩니다.

기본적으로 10,000개 레코드마다 처리량과 진행 상황을 출력합니다.

- `loss`: 선택 행동의 평균 cross-entropy
- `top1`: 가장 높은 점수의 행동이 인간 선택과 일치한 비율
- `top3`: 인간 선택이 상위 세 행동에 포함된 비율
- `uniform_random_top1`: 합법 행동 중 무작위 선택 기준선
- `pass_else_first_top1`: 패스가 가능하면 패스하고 아니면 첫 후보를 고르는 기준선
- `non_forced`: 합법 행동이 둘 이상인 실제 선택 상황의 성능
- `non_pass`: 인간이 패스 외 행동을 선택한 상황의 성능
- `pass_when_available.human_rate`: 패스할 수 있을 때 인간이 패스한 비율
- `by_action`: 패스 및 플레이 조합 종류별 표본 수와 Top-1

## 전체 기준선 학습

```powershell
docker compose run --rm train-bc `
  --device cuda `
  --epochs 5 `
  --batch-size 64 `
  --output models/experiments/behavior-cloning-v1.pt
```

검증 loss가 가장 낮은 epoch만 체크포인트로 저장합니다. `models/`와
생성 데이터셋은 Git에서 제외됩니다. 학습/검증 분할은 원본 게임 ID
단위이므로 같은 게임의 라운드와 인접 결정이 양쪽에 섞이지 않습니다.

## 결과 해석

이 모델은 승리 팀의 행동만 학습한 지도학습 기준선입니다. 높은 Top-1은
인간 행동을 잘 재현한다는 뜻이지, 곧바로 승률이 높다는 뜻은 아닙니다.
학습 후 정책 에이전트로 연결해 Fuegi, MCTS와 좌석 교환 벤치마크를 별도로
수행해야 합니다.

## 실제 게임 벤치마크

학습 체크포인트는 `behavior-cloning` 에이전트로 바로 불러올 수 있습니다.
모델의 플레이 정책만 비교할 수 있도록 거래와 드래곤 전달은 Fuegi와 같은
보조 전략을 사용합니다. 작은 모델의 한 수 추론은 GPU 전송 비용보다 CPU가
빠른 경우가 많으므로 벤치마크 기본 장치는 CPU입니다.

```powershell
docker compose run --rm --entrypoint python train-bc benchmark.py `
  --team-a behavior-cloning `
  --team-b fuegi `
  --games 20 `
  --seed 50000 `
  --target 200 `
  --bc-checkpoint models/experiments/behavior-cloning-v1.pt `
  --output results/benchmarks/smoke/behavior-cloning/behavior-cloning-vs-fuegi-20.csv
```

각 seed는 좌석을 바꿔 두 번 실행됩니다. `--games 20`은 실제 결과 행
40개를 생성합니다. 명령행 벤치마크는 전체 경기 수를 기준으로 약 5%마다
진행 경기 수, 백분율, 경과 시간을 출력합니다.

MCTS가 포함된 벤치마크 CSV와 요약에는 실제 탐색량도 기록됩니다.

- `mcts_search_calls`: MCTS가 행동을 고른 횟수
- `mcts_iterations_requested`: 요청한 rollout iteration 합계
- `mcts_iterations_completed`: 완료한 rollout iteration 합계
- `mcts_average_iterations`: MCTS 결정당 평균 완료 iteration
- `mcts_time_limit_hits`: iteration 상한보다 시간 제한에 먼저 도달한 횟수
- `mcts_iteration_limit_hits`: 요청한 iteration을 모두 완료한 횟수

재현 가능한 고정 탐색량 비교에는 충분히 큰 시간 상한을 사용합니다.

```powershell
--iterations 100 --max-time 2.0
```

실시간 응답 예산 비교에는 높은 iteration 상한과 고정 시간을 사용합니다.

```powershell
--iterations 100000 --max-time 0.2
```

## Baseline B 후보 튜닝

nn_training/tune_baseline_b.py는 v2 train/validation 분할만 사용해 네 MLP 후보를
차례로 학습한다. test 분할은 읽지 않는다.

| 후보 | Hidden size | Learning rate | Dropout |
| --- | ---: | ---: | ---: |
| small-fast | 64 | 0.001 | 0.0 |
| base | 128 | 0.0003 | 0.1 |
| wide | 256 | 0.0003 | 0.1 |
| regularized | 128 | 0.001 | 0.2 |

~~~powershell
docker compose run --rm --entrypoint python train-bc `
  nn_training/tune_baseline_b.py `
  --device cuda `
  --epochs 5 `
  --batch-size 256
~~~

각 후보는 validation Top-1을 우선으로, 동률이면 validation loss가 더 낮은 epoch를
checkpoint로 저장한다. 실행 중에는 매 후보 뒤에 다음 파일이 갱신된다.

- results/tuning/archive/baseline-b-v2/summary.json
- results/tuning/archive/baseline-b-v2/summary.csv

먼저 파이프라인만 확인하려면 전체 feature schema와 전체 epoch 대신 작은 표본을 쓴다.

~~~powershell
docker compose run --rm --entrypoint python train-bc `
  nn_training/tune_baseline_b.py `
  --device cuda `
  --dry-run `
  --output-dir results/tuning/experiments/baseline-b-v2-smoke `
  --model-dir models/experiments/baseline-b-v2-tuning-smoke
~~~
