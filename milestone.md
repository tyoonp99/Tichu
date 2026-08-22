# 티츄 AI 개발 마일스톤

## 최종 목표

공식 티츄 규칙을 재현하는 검증 가능한 게임 엔진 위에서 인간 대국을 학습하고,
동일한 카드 분배와 좌석 교환 조건으로 전략 강도를 비교할 수 있는 AI를 만든다.
최종 후보는 행동 모방(BC), MCTS, BC-guided MCTS, 자기대전 강화학습이다.

## 문서 관리 원칙

- ✅ 완료, 🔄 진행 중, ⬜ 예정으로 상태를 표시한다.
- 기능 완료·평가 기준 변경·다음 단계 진입 시 이 문서도 함께 갱신한다.
- 완료 단계는 핵심 결과와 재현 근거만 남기고 축약한다.
- 규칙이나 데이터 버전이 바뀌면 이전 결과를 새 버전 성능처럼 사용하지 않는다.

## 평가 원칙

- 같은 seed를 좌석 교환 전후로 실행한다.
- 승률, 평균 점수 차, 좌석별 성능, 오류율을 함께 기록한다.
- 학습·validation·최종 test 게임과 seed를 분리한다.
- MCTS는 요청/완료 iteration, 검색 시간, 종료 조건을 기록한다.
- 고정 iteration 평가와 고정 시간 평가를 구분한다.

---

## 완료 요약

### 마일스톤 0: 실행 가능한 게임 코어 ✅

Docker 기반 Python 3.11 환경, 카드 조합·점수·상태 전이 테스트, Gymnasium API,
자동 게임 실행 및 기본 MCTS 호환성을 확보했다.

### 마일스톤 1: 재현 가능한 벤치마크 ✅

seed·좌석 교환·CSV 결과·진행률·MCTS 탐색량 계측을 구현했다. 동일 seed와 고정
iteration 재실행 결과가 일치한다.

### 마일스톤 2: 전략 기준선 ✅

Random, BalancedRandom, Fuegi, MCTS, Fuegi-MCTS를 구현해 BC와 후속 모델의
시험 상대 및 탐색 구성 요소로 사용할 수 있다.

### 마일스톤 3~4: 인간 로그 및 BC v1 기준선 ✅

초기 인간 로그 파이프라인과 합법 행동 점수 MLP를 구현했다. BC v1은 `rules-v1`
데이터로 학습한 기능 검증용 기준선이며, 기존 성능 수치는 rules-v2 최종 성능으로
사용하지 않는다.

### 마일스톤 5: rules-v2 규칙 엔진 ✅

비동기 폭탄, 재폭탄, 피닉스 조합, 소원, 드래곤 양도 및 종료 상태를 보완했다.

- 최신 완전 게임 로그 10,000개 수집
- 전체 96,257라운드 중 원본 미완성 488개를 제외한 95,769라운드 완전 재생
- 인간 결정 5,367,047개 검증
- 완료 라운드 불법 행동 0개, 점수 불일치 0개, 완전 재생률 100%
- Docker 전체 회귀 테스트 165개 통과

---

## 마일스톤 6: 인간 로그 데이터셋 v2 ✅

### 완료

- 최신 번호 탐색, 중단 재개, 중복 방지, manifest와 진행률을 갖춘 수집기
- `rules-v2` 기반 전체 완료 라운드 행동·점수 검증
- 폭탄·피닉스·소원·종료 반례의 회귀 테스트
- 공개 폭탄 응답 문맥과 추론된 폭탄 보류를 구분하는 스키마 v2
- 게임 단위 80/10/10 고정 분할과 분할 누수 자동 검사
- 플레이어 이름 제외 및 원본 게임 ID 가명화
- 결정적 gzip JSONL, 파일 해시, split manifest와 5게임 생성 검증
- 압축 v2 데이터로 BC 소규모 학습 검증
- 전체 6,542,405결정 생성: train 5,209,033, validation 682,767,
  test 650,605
- 추론 폭탄 보류 1,175,358개와 실제 폭탄 42,505개 구분
- 게임 ID의 분할 중복 0개, 선택 행동의 합법 행동 포함 여부 전수 검사
- 두 번의 전체 생성에서 데이터·manifest SHA-256 완전 일치

완료 조건: 건드리지 않는 test 분할과 스키마 버전이 있는 재현 가능한 데이터셋을
생성하고, 동일 입력에서 같은 분할과 레코드가 만들어진다.

## 마일스톤 7: BC v2 모델 선택 ✅

### 완료

- gzip JSONL을 스트리밍하는 PyTorch IterableDataset·DataLoader
- 가변 길이 합법 행동 목록을 보존하는 배치 collate
- 결정적 버퍼 셔플과 다중 worker 중복 방지 테스트
- 상태·후보 행동 MLP 하이브리드 Baseline B와 후보 축 음의 무한대 마스킹
- 실제 v2 train 배치 256개로 상태·후보·logit 텐서 형태와 마스킹 확인
- Cross-Entropy train·validation 루프, validation Top-1, 최고 checkpoint 저장
- GPU dry run(2 epoch × train/validation 각 10배치)에서 loss·Top-1·checkpoint 확인
- Hidden size·learning rate·dropout 네 후보를 순차 실행하는 튜너와 CSV/JSON 요약
- 1차 validation 튜닝 완료: wide(256, lr 3e-4, dropout 0.1) 선택
- 2차 정밀 튜닝 완료: wide-lr-low·wide·wide-lr-high를 최대 12 epoch 비교
- 강한 MLP 기준선 고정: wide(256, lr 3e-4, dropout 0.1), epoch 12,
  validation Top-1 88.11%, loss 0.3157
- 기준선 validation 세부 분석 완료: 폭탄 Top-1 54.69%, 피닉스 사용 60.52%,
  피닉스 보류 83.95% (전체 Top-3 98.67%)
- Model C(Set Attention 손패 + Bi-GRU 트릭 + 합법 후보 마스킹) 구현·실제 배치 dry run 완료
- Model C validation 비교 완료: 전체 Top-1 88.43%, loss 0.3070으로 MLP보다 개선.
  단, 폭탄 Top-1은 51.31%로 MLP(54.69%)보다 하락하여 최종 선택은 보류
- Model C v2 희귀 조합 Weighted Cross-Entropy 완료: 전체 Top-1 88.27%,
  폭탄 71.21%, Pair 75.54%, FullHouse 72.89%로 핵심 약점 회복
- 고정 test 분할의 Top-1·Top-3·Cross-Entropy 평가 도구와 JSON/CSV 리포트
- 일반 패스·폭탄 대응 패스·조합·폭탄·피닉스 사용·피닉스 보류별 세부 정확도 분석
- 최종 후보 동결: Model C v2 (`models/model-c-v2/model-c.pt`)
- 최종 test 1회 평가 완료: 650,605건, Top-1 88.20%, Top-3 98.62%, loss 0.3116;
  폭탄 70.14%, 피닉스 보류 84.16%, Pair 76.12%, FullHouse 73.38%

완료 조건: 사전에 정한 지표로 BC v2 하나를 선택하고 체크포인트와 test 결과를
고정한다. **완료.**

## 마일스톤 8: 에이전트 통합 및 BC-guided MCTS ✅

- 동결 Model C v2를 엔진의 합법 행동 점수화로 연결한 `ModelCAgent` 구현
- `model-c`·Fuegi·Fuegi-MCTS를 고정 seed·좌석 교환으로 실행하는 벤치마크 등록
- Model C vs Fuegi 1 seed(좌석 교환 2게임) 시범 대국 완료: 오류 0건,
  `results/benchmarks/smoke/model-c/model-c-vs-fuegi-1seed.csv` 저장
- Model C vs Fuegi 30쌍 smoke 완료: 57승 3패, 평균 점수 차 +201.83
- Model C 점수를 PUCT prior로 사용하는 `ModelCGuidedMctsAgent` 구현,
  합법 행동·MCTS 회귀 테스트 19개 통과
- 30쌍 smoke 완료, 동일 seed 표준화 결과:
  - 고정 10회(65,000~65,029): Model C 60승 0패, +214.17;
    BC-guided MCTS 54승 5패 1무, +178.67
  - 턴당 200ms(66,000~66,029): Model C 60승 0패, +210.17;
    BC-guided MCTS 54승 6패, +166.00
- PUCT는 동일 예산의 Fuegi-MCTS를 이겼지만, 단독 Model C보다 약했고
  200ms 조건에서 prior 계산 비용으로 평균 완료 탐색 수도 더 낮았다.
- 본 평가 완료(고정 10회, seed 70,000~70,199, 좌석 교환 400게임):
  Model C v2 385승 14패 1무(96.25%), 평균 점수 차 +196.50,
  쌍별 95% CI +188.55~+204.45; Fuegi-MCTS 16,894회 검색 모두 10회 완료
- PUCT는 prior batching/캐시 또는 탐색 규칙 개선 전에는 후속 본 평가로 승격하지 않는다.

완료 조건: 같은 계산 예산에서 결합 모델의 개선 여부를 통계적으로 판단한다.
**완료.** BC-guided MCTS는 구현·검증했으나, 동일 seed smoke에서 단독 Model C v2보다
낮은 성능과 높은 추론 비용을 보여 최종 후보로 승격하지 않는다.

완료 조건: 같은 계산 예산에서 결합 모델의 개선 여부를 통계적으로 판단한다.

## 마일스톤 9: 자기대전 강화학습 ⬜

- BC v2를 초기 정책으로 사용
- 합법 행동 마스크와 팀 점수 차 보상 사용
- 과거 체크포인트와 기존 에이전트를 섞어 과적합 방지
- 고정 평가군과 보지 않은 seed로 모델 승격 판단

완료 조건: 동일 응답 시간에서 고정 기준선을 통계적으로 유의하게 이기는
체크포인트를 만든다.

---

## 현재 실행 순서

1. Model C v2 대 Fuegi 30쌍 smoke로 단독 정책 기준선 확보
2. BC-guided MCTS 구현: Model C 후보 점수를 PUCT prior로 사용
3. 모든 후보를 30쌍 smoke로 비교 (iteration·시간 조건 분리)
4. 유망한 대결만 200쌍 이상 본 평가로 확대
5. 자기대전 강화학습
6. BC v2가 안정된 뒤 v1 산출물과 저장소 구조 정리
