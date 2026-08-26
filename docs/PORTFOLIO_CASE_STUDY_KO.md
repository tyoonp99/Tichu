# TICHU AI — 인간 대국 로그로 티츄 엔진을 검증하고 행동 모방 모델을 선택한 프로젝트

> 기간: 2026년 · 역할: 1인 개발 · 기술: Python, Gymnasium, PyTorch, Docker, FastAPI

## 결론 요약

이 프로젝트의 최종 산출물은 **인간 대국 로그를 재생할 수 있는 rules-v2 티츄 엔진**과, 그 로그의 합법 행동 654만 건을 학습한 **Model C v2**다.

| 검증 대상 | 최종 결과 | 근거 |
|---|---:|---|
| 규칙 엔진 | 완료 라운드 95,769개 재생 | 불법 행동 0건, 최종 점수 불일치 0건 |
| 행동 모방 모델 | test Top-1 88.20% | 고정 test 650,605건 |
| 희귀 폭탄 행동 | Top-1 70.14% | 가중 학습 적용 후 독립 test |
| 실제 게임 정책 | Baseline B 대비 +39.10점 | 좌석 교환 포함 400게임, 95% CI +26.69~+51.51 |

초기 목표는 랜덤·규칙 기반·MCTS 에이전트를 비교하는 것이었다. 프로젝트는 인간 로그를 재생하는 과정에서 기존 엔진의 폭탄 처리가 불완전하다는 사실을 발견하면서, **엔진 검증 → 검증 데이터셋 → 모델 선택 → 실제 게임 평가** 순서의 프로젝트로 확장됐다.

---

## 1. 규칙 엔진: 인간 로그가 폭탄 규칙 결함을 발견했다

### 문제

기존 시뮬레이터는 Random, Balanced Random, 전략 팁 기반 Fuegi, MCTS 에이전트를 실행할 수 있었다. 하지만 공개 Brettspielwelt 인간 로그를 재생하자 일부 행동이 불법으로 처리되거나, 라운드가 끝난 뒤 점수가 원본과 달랐다.

가장 큰 원인은 폭탄이었다. 티츄의 폭탄은 일반 차례에만 내는 조합이 아니라, 다른 사람이 카드를 낸 뒤 **차례 밖에서 끼어들 수 있고**, 그 폭탄 위에 다시 더 높은 폭탄을 낼 수 있다. 기존의 “현재 플레이어 한 명이 행동한다”는 전이만으로는 이 흐름을 표현할 수 없었다.

### 해결

`TichuState`에 폭탄 반응 대기열과 원래 차례로 돌아갈 위치를 저장하는 상태를 추가했다. 폭탄을 내지 않는 경우(`PassBombAction`)와 일반 패스를 분리해, 폭탄을 포기해도 정상 트릭에서 패스한 것으로 계산되지 않게 했다.

| 보완한 규칙 | 구현 참조 | 회귀 테스트 참조 |
|---|---|---|
| 비동기 폭탄·재폭탄·폭탄 패스 | [`tichu_state.py`](../gym-tichu/gym_tichu/envs/internals/tichu_state.py#L428-L437), [`tichu_state.py`](../gym-tichu/gym_tichu/envs/internals/tichu_state.py#L740-L803) | [`test_state_transitions.py`](../tests/test_state_transitions.py#L268-L384) |
| 피닉스 단독 높이·스트레이트·풀하우스 비교 | [`cards.py`](../gym-tichu/gym_tichu/envs/internals/cards.py) | [`test_possible_combinations.py`](../tests/test_possible_combinations.py) |
| 마작 소원 강제 이행 | [`tichu_state.py`](../gym-tichu/gym_tichu/envs/internals/tichu_state.py#L486-L509) | [`test_state_transitions.py`](../tests/test_state_transitions.py#L121-L179) |
| 드래곤 트릭 양도·종료 점수 | [`tichu_state.py`](../gym-tichu/gym_tichu/envs/internals/tichu_state.py#L588-L609) | [`test_state_transitions.py`](../tests/test_state_transitions.py#L182-L230) |

피닉스의 동적 높이와 더 높은 풀하우스도 인간 로그 재생 중 발견된 실제 반례를 통해 수정했다. 즉 규칙을 추측해 구현한 뒤 테스트한 것이 아니라, **로그에서 실패한 행동을 재현 테스트로 만든 뒤 전이를 수정**했다.

### 재생 결과

| 재생 단계 | 완료 라운드에서 확인된 문제 | 의미 |
|---|---:|---|
| 초기 재생 | 불법 행동 1,182라운드 | 비동기 폭탄 등 엔진 누락 확인 |
| 폭탄 처리 1차 보완 후 | 불법 행동 543라운드 | 폭탄 외 피닉스·조합 차이 잔존 확인 |
| rules-v2 최종 재생 | 불법 행동 0건, 점수 불일치 0건 | 완료 로그 범위에서 행동·점수 재현 |

최종 검증 범위는 전체 96,257라운드 중 원본이 중간에 끝난 488라운드를 제외한 95,769라운드다. 여기서 “공식 규칙 전체를 증명했다”는 주장은 하지 않는다. 대신 **수집한 완료 인간 로그에서는 엔진의 행동 합법성과 최종 점수가 모두 일치했다**는 재현 결과를 남겼다.

재생기와 원본 로그 파서는 [`brettspielwelt_replay.py`](../scraper/brettspielwelt_replay.py), [`brettspielwelt_parser.py`](../scraper/brettspielwelt_parser.py)에 있다.

---

## 2. 인간 로그는 학습 데이터이기 전에 엔진을 통과한 검증 결과였다

원본 로그를 그대로 모델에 넣지 않았다. 재생 중 불법 행동이나 점수 불일치가 난 게임, 원본이 중간에 끊긴 게임은 제외했다. 따라서 최종 데이터셋은 “인터넷에서 수집한 기록”이 아니라 rules-v2가 합법성을 다시 확인한 게임의 행동 기록이다.

각 행동에는 행동자에게 공개된 정보와 합법 행동 후보만 남겼다. 상대 손패는 포함하지 않았고, 같은 게임의 행동이 train과 test에 동시에 들어가지 않도록 게임 단위로 분할했다.

| 데이터셋 항목 | 결과 |
|---|---:|
| train / validation / test | 5,209,033 / 682,767 / 650,605 결정 |
| 전체 결정 수 | 6,542,405 |
| 게임 단위 분할 | 80 / 10 / 10 |
| 분할 간 게임 ID 중복 | 0건 |
| 선택 행동의 합법 후보 포함 검사 | 전수 통과 |
| 동일 입력 재생성 | gzip 데이터·manifest SHA-256 일치 |

데이터셋 변환 코드는 [`prepare_brettspielwelt_dataset.py`](../scraper/prepare_brettspielwelt_dataset.py), 스키마와 검증 기준은 [`training-dataset-v2.md`](training-dataset-v2.md)에 정리했다.

---

## 3. 모델 선택: 전체 정확도보다 폭탄의 첫 선택을 개선했다

### 판단 기준

모델은 상태만 분류하지 않는다. 엔진이 현재 상태에서 만든 **합법 행동 후보들**을 각각 점수화하고, 그중 가장 높은 후보를 선택한다. 따라서 추론 결과는 항상 합법 행동이다.

- Baseline B: 상태 벡터와 후보 행동을 결합하는 MLP
- Model C: 손패(Set Attention), 현재 트릭(Bi-GRU), 공개 상태, 후보 행동을 분리 인코딩한 뒤 점수화
- Model C v2: Model C에 희귀 조합 가중 손실 적용

구현은 [`behavior_cloning.py`](../nn_training/behavior_cloning.py), [`model_c.py`](../nn_training/model_c.py), 가중치 프로필은 [`train_model_c.py`](../nn_training/train_model_c.py#L17-L24)에 있다.

### Baseline B 튜닝

먼저 단순 MLP 기준선의 크기·학습률·dropout을 비교했다. 모든 후보는 같은 validation 682,767건에서 비교했다.

| 후보 | hidden | learning rate | dropout | best epoch | validation Top-1 | loss |
|---|---:|---:|---:|---:|---:|---:|
| small-fast | 64 | 1e-3 | 0.0 | 5 | 86.91% | 0.3521 |
| base | 128 | 3e-4 | 0.1 | 5 | 87.39% | 0.3378 |
| wide | 256 | 3e-4 | 0.1 | 5 | **87.85%** | **0.3237** |
| regularized | 128 | 1e-3 | 0.2 | 5 | 87.15% | 0.3468 |

선택한 wide 후보는 learning rate만 다시 1e-4 / 3e-4 / 6e-4로 비교했다.

| 후보 | learning rate | best epoch | validation Top-1 | loss |
|---|---:|---:|---:|---:|
| wide-lr-low | 1e-4 | 12 | 88.02% | 0.3197 |
| wide | 3e-4 | 12 | **88.11%** | **0.3157** |
| wide-lr-high | 6e-4 | 11 | 88.03% | 0.3199 |

원본 결과: [`baseline-b-v2/summary.csv`](../results/tuning/archive/baseline-b-v2/summary.csv), [`baseline-b-v2-refine/summary.csv`](../results/tuning/archive/baseline-b-v2-refine/summary.csv).

### Model C v1의 약점: 폭탄 Top-3 99.08%가 좋은 결과는 아니었다

Model C v1은 전체 validation Top-1 88.43%, Top-3 98.65%로 MLP 기준선보다 좋아 보였다. 하지만 세부 분석에서 폭탄 Top-1은 51.31%로 Baseline B의 54.69%보다 낮았다.

폭탄은 validation에서 4,467건뿐이었다. 또 폭탄 반응은 합법 후보가 2~3개인 상황이 자주 있어, Top-3 99.08%는 “추천 3개 안에는 정답이 있다” 이상의 의미를 주기 어려웠다. 실제 게임에서 필요한 것은 첫 번째 추천이다.

| validation 지표 | Baseline B wide | Model C v1 | Model C v2 |
|---|---:|---:|---:|
| 전체 Top-1 | 88.11% | **88.43%** | 88.27% |
| 전체 loss | 0.3157 | **0.3070** | 0.3107 |
| 전체 Top-3 | 98.67% | **98.65%** | 98.62% |
| 폭탄 표본 수 | - | 4,467 | 4,467 |
| 폭탄 Top-1 | 54.69% | 51.31% | **71.21%** |
| 폭탄 Top-3 | - | 99.08% | 99.62% |
| Pair Top-1 | - | 72.40% | **75.54%** |
| FullHouse Top-1 | - | 66.91% | **72.89%** |
| StraightBomb Top-1 | - | 40.08% | **66.50%** |

### 해결: 희귀 조합의 정답에 가중치 부여

Model C v2는 폭탄, Pair, FullHouse가 정답인 레코드에 더 큰 Cross-Entropy 가중치를 적용했다. 전체 Top-1은 88.43%에서 88.27%로 0.16%p 낮아졌지만, 폭탄 Top-1은 51.31%에서 71.21%로 +19.90%p 개선됐다. 모델 선택 기준을 전체 평균이 아니라, 프로젝트에서 중요하게 본 폭탄·복합 조합의 1순위 판단으로 바꾼 결과다.

고정 test에서의 Model C v2 결과는 전체 Top-1 88.20%, Top-3 98.62%, 폭탄 Top-1 70.14%였다. validation과 test에서 폭탄 성능이 크게 벌어지지 않아 최종 체크포인트를 동결했다.

세부 리포트: [`Model C v1 validation`](../results/evaluations/archive/model-c-v1-validation/test-evaluation.json), [`Model C v2 validation`](../results/evaluations/validation/model-c-v2/test-evaluation.json), [`Model C v2 test`](../results/evaluations/final/model-c-v2-test/test-evaluation.json).

---

## 4. 실제 게임 벤치마크: 작은 smoke와 본 평가는 구분했다

행동 모방 정확도는 실제 점수나 승률을 보장하지 않는다. 따라서 같은 seed를 두 번 실행하고 두 번째 게임에서 팀 좌석을 교환했다. 빠른 선별용 30 seed smoke와, 결론용 200 seed 본 평가를 구분했다. 한 seed는 좌석 교환 전후 두 게임이므로 30 seed는 60게임, 200 seed는 400게임이다.

### 모델 간 비교 결과

| 매치업 | 규모 | 전적 (팀 A 기준) | 평균 점수 차 | 해석 |
|---|---:|---:|---:|---|
| Model C v2 vs Model C v1 | 30 seed / 60게임 | 31승 · 27패 · 2무 | +5.17 | v2 우세 신호는 있으나 작은 표본 |
| Model C v2 vs Baseline B | 30 seed / 60게임 | 29승 · 30패 · 1무 | -11.00 | smoke만으로 v2 우세를 결론 내리지 않음 |
| Model C v2 vs Baseline B | 200 seed / 400게임 | 242승 · 151패 · 7무 | **+39.10** | 95% CI +26.69 ~ +51.51, 본 평가에서 우세 확인 |
| Model C v2 vs Fuegi-MCTS (10회) | 200 seed / 400게임 | 385승 · 14패 · 1무 | **+196.50** | 95% CI +188.55 ~ +204.45 |

Baseline B와의 60게임 smoke가 -11점이었는데도 400게임 본 평가를 진행한 이유는, 작은 표본의 변동만으로 모델 선택을 뒤집지 않기 위해서다. 본 평가에서는 신뢰구간까지 확인해 Model C v2의 우세를 판단했다.

원본 CSV: [`v2 vs v1 smoke`](../results/benchmarks/smoke/model-c/model-c-v2-vs-v1-30seeds.csv), [`v2 vs Baseline smoke`](../results/benchmarks/smoke/model-c/model-c-v2-vs-baseline-b-30seeds.csv), [`v2 vs Baseline main`](../results/benchmarks/main/model-c-v2-vs-baseline-b-200seeds.csv), [`v2 vs Fuegi-MCTS main`](../results/benchmarks/main/model-c-vs-fuegi-mcts-10iter-200seeds.csv).

### 채택하지 않은 탐색·강화학습 후보

| 후보 | 비교 조건 | 결과 | 결론 |
|---|---|---:|---|
| BC-guided MCTS | Fuegi-MCTS, 30 seed / 60게임 | 54승 · 5패 · 1무, +178.67 | Fuegi-MCTS는 이겼지만 단독 Model C smoke(+214.17)보다 약함 |
| PPO Dense Reward | Model C v2, 30 seed / 60게임 | 22승 · 37패 · 1무, -35.33 | 95% CI -58.83 ~ -11.84, 조기 종료 |

BC-guided MCTS는 Model C의 점수를 탐색 prior로 사용했지만, 신경망 추론 비용으로 같은 시간 안에 탐색 수가 줄었다. PPO는 트릭 점수를 중간 보상으로 추가했을 때 오히려 장기 전략을 손상시켰다. 두 방법 모두 “구현 여부”가 아니라 실제 대국 결과로 비승격 처리했다.

---

## 최종 산출물과 범위

- **rules-v2:** 폭탄·피닉스·소원·드래곤·점수 전이를 포함한 티츄 엔진
- **Brettspielwelt 데이터 파이프라인:** 로그 다운로드, 재생, 완료 게임 필터링, 행동 데이터셋 변환
- **Model C v2:** 손패 Set Attention + 트릭 Bi-GRU + 합법 후보 점수화 모델
- **벤치마크 도구:** 고정 seed, 좌석 교환, CSV, 평균 점수 차, 신뢰구간
- **웹 시험판:** 사람이 Model C v2와 한 라운드를 플레이하며 엔진과 AI 행동을 확인하는 서버 권위형 UI

웹 시험판은 단일 라운드 MVP다. 티츄 선언·카드 교환은 자동 처리되고, 장기 매치·재접속은 구현하지 않았다. 따라서 사람 상대 승률을 증명하는 결과가 아니라, 후속 플레이테스트와 인간 대국 로그 수집을 위한 기반으로 분리했다.

## 관련 문서와 재현

- [데이터 수집·재생 방법](brettspielwelt-data.md)
- [벤치마크 조건과 원본 결과](benchmark-protocol.md)
- [웹 플레이 안내](web-play.md)

```powershell
docker compose run --rm --entrypoint pytest train-bc -q
```
