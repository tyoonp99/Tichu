# 학습 데이터셋 스키마 v1 (보관용)

이 문서는 rules-v1 기반 초기 수집 파이프라인의 기록이다. 현재 학습·평가에는
`../training-dataset-v2.md`를 사용한다.

`collect_dataset.py`는 교사 에이전트가 실제 경기에서 내린 결정을 JSONL 형식으로
저장한다. 한 줄이 한 번의 플레이 또는 패스 결정에 해당한다.

## 정보 범위

각 관측에는 행동하는 플레이어가 결정 시점에 알 수 있는 정보만 포함한다.

- 자신의 손패
- 자신부터 시계 방향으로 정규화한 네 플레이어의 남은 카드 수
- 각 플레이어가 획득한 트릭 수와 공개 점수
- 현재 테이블의 공개 행동
- 소원, 등수, 티츄 선언 정보
- 현재 합법 행동 목록

상대와 팀원의 실제 손패, 전체 `TichuState`, 전체 상태 히스토리는 기록하지 않는다.
플레이어 위치는 관측자 자신을 0으로 두고 오른쪽 상대, 팀원, 왼쪽 상대 순서로
정규화한다.

## 교사 라벨

- `chosen_action`: 교사가 선택한 합법 행동
- `search_statistics`: MCTS가 탐색한 루트 행동별 방문 수, 가용 횟수, 평균 보상
- `outcome`: 경기 종료 후 행동자 팀 기준 점수, 점수 차이, 승리 여부

강제 행동처럼 MCTS 탐색을 실행하지 않은 결정은 `search_statistics`가 빈 배열이다.

## 시드 분리

학습/검증 배정은 시드의 SHA-256 값으로 결정한다. 동일 시드의 기본 경기와 좌석
교환 경기는 항상 같은 파일에 들어가므로 같은 카드 분배가 학습과 검증 양쪽에
노출되지 않는다.

## 당시 실행 예시

```powershell
docker compose run --rm tests python collect_dataset.py `
  --team-a fuegi-mcts `
  --team-b fuegi `
  --record-agent fuegi-mcts `
  --games 100 `
  --seed 40000 `
  --target 100 `
  --iterations 10 `
  --max-time 0.2 `
  --validation-fraction 0.1 `
  --output-dir datasets/tichu-decisions-v1
```

출력 폴더에는 `train.jsonl`, `validation.jsonl`, `metadata.json`이 생성된다.
생성 데이터는 용량이 커질 수 있고 다시 만들 수 있으므로 Git에는 커밋하지 않는다.
