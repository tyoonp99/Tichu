# 실제 대국 벤치마크 프로토콜

## 목적

동결된 `models/model-c-v2/model-c.pt`가 실제 Tichu 게임에서 기존 정책보다
강한지, 그리고 이후 BC-guided MCTS가 같은 계산 예산에서 더 나은지를 재현 가능하게
측정한다.

## 고정 조건

- 게임 목표 점수: 200점(빠른 반복 측정용)
- 각 seed는 두 번 실행한다. 두 번째 게임은 팀 A/B의 좌석을 교환한다.
- 같은 seed·좌석 교환 쌍은 하나의 관측 단위로 취급한다.
- 보고 지표: 승·패·무승부, 평균 점수 차, 95% 신뢰구간, 게임당 시간.
- 모델과 규칙 엔진은 평가 중 변경하지 않는다.

## 비교 순서

1. Model C v2 vs Baseline B
2. Model C v2 vs Fuegi
3. Model C v2 vs Fuegi-MCTS (고정 iteration 예산)
4. Model C v2 vs Fuegi-MCTS (고정 턴 시간 예산)
5. BC-guided MCTS vs Fuegi-MCTS (동일 iteration/시간 예산)

각 비교는 smoke 30쌍으로 실행 가능성을 확인한 뒤, 본 평가 200쌍 이상으로 확장한다.
최종 승격 판정에는 본 평가와 별도 seed 구간을 사용한다.

## 실행 방법

`model-c` 에이전트는 동결된 Model C v2 체크포인트를 읽고, 매 차례 엔진이 제공한
합법 행동 전체를 후보로 점수화한다. 따라서 선택 결과는 항상 현재 상태의 합법 행동 중
하나이다.

빠른 좌석 교환 smoke(한 seed = 두 실제 게임):

```powershell
docker compose run --rm --entrypoint python train-bc benchmark.py `
  --team-a model-c `
  --team-b fuegi `
  --games 1 `
  --seed 60000 `
  --target 100 `
  --model-c-checkpoint models/model-c-v2/model-c.pt `
  --model-c-device cpu `
  --output results/benchmarks/smoke/model-c/model-c-vs-fuegi-1seed.csv
```

`--games`는 seed 개수이며, 각 seed마다 `swapped=False/True`가 모두 실행되므로 CSV와
최종 요약의 실제 게임 수는 두 배이다. Fuegi-MCTS 비교에서는 `--team-b fuegi-mcts`와
함께 `--iterations` 또는 `--max-time` 중 평가 조건 하나만 고정한다.

## 본 평가 결과: Model C v2 vs Fuegi-MCTS

동결된 `models/model-c-v2/model-c.pt`로 2026-08-22에 실행했다.

- 조건: seed 70,000~70,199, seed당 좌석 교환 2게임, 목표 200점,
  Fuegi-MCTS 고정 10회 탐색
- 표본: 200 seed 쌍, 실제 400게임
- Model C v2: 385승 14패 1무 (승률 96.25%), 평균 점수 차 +196.50
- 좌석 교환 쌍 평균 점수 차의 95% 신뢰구간: +188.55 ~ +204.45
- 상대 MCTS: 16,894회 검색 모두 요청한 10회 탐색을 완료
- 총 대국 시간: 1,410.92초 (게임당 평균 3.53초)

원본 결과: `results/benchmarks/main/model-c-vs-fuegi-mcts-10iter-200seeds.csv`.
이 수치는 고정 iteration 조건의 본 평가이며, 턴 시간 제한 조건의 결과와 혼용하지 않는다.

## 본 평가 결과: Model C v2 vs Baseline B

동결된 Model C v2와 2차 정밀 튜닝에서 선택한 Baseline B `wide.pt`를
2026-08-22에 실행했다.

- 조건: seed 75,000~75,199, seed당 좌석 교환 2게임, 목표 200점
- 표본: 200 seed 쌍, 실제 400게임
- Model C v2: 242승 151패 7무 (승률 60.50%), 평균 점수 차 +39.10
- 좌석 교환 쌍 평균 점수 차의 95% 신뢰구간: +26.69 ~ +51.51
- 좌석별 Model C v2 승수: 교환 전 121승, 교환 후 121승
- 총 대국 시간: 160.49초 (게임당 평균 0.40초)

원본 결과: `results/benchmarks/main/model-c-v2-vs-baseline-b-200seeds.csv`.
신뢰구간 전체가 0보다 크므로 Model C v2를 Baseline B보다 강한 실제 대국 정책으로
판정한다.

## 해석 원칙

- 쌍별 좌석 교환 평균을 기준으로 좌석·마작 선공 효과를 제거한다.
- 단일 승률만이 아니라 평균 점수 차와 실행 시간을 함께 본다.
- MCTS 비교에서는 iteration 제한과 시간 제한 결과를 섞지 않는다.
