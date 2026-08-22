# 자동 대국 학습 데이터 스키마 v2

`collect_dataset.py`는 교사 에이전트가 실제 경기에서 내린 결정을 JSONL로
저장한다. Brettspielwelt 인간 로그 변환은
[brettspielwelt-data.md](./brettspielwelt-data.md)를 따른다.

한 레코드에는 행동자가 볼 수 있는 다음 정보만 포함한다.

- 자신의 손패와 좌석별 공개 카드 수·획득 트릭·점수
- 현재 트릭, 소원, 순위와 티츄 선언
- 정상 플레이 또는 폭탄 응답 문맥과 공개된 정상 차례 복귀 위치
- 합법 행동 목록, 교사 행동, 선택적인 MCTS 루트 통계
- 경기 종료 후 행동자 팀 기준 점수와 승패

상대와 팀원의 실제 손패, 다른 플레이어의 폭탄 보유 여부, 전체 상태 히스토리는
기록하지 않는다. 좌석은 행동자를 0으로 두고 상대 위치로 정규화한다.

같은 seed의 기본 경기와 좌석 교환 경기는 SHA-256 기반으로 같은 train 또는
validation 분할에 들어간다. 자기대전 학습을 본격화할 때 인간 데이터와 같은
고정 test 분할을 추가한다.

```powershell
docker compose run --rm --entrypoint python tests collect_dataset.py `
  --team-a fuegi-mcts `
  --team-b fuegi `
  --record-agent fuegi-mcts `
  --games 100 `
  --seed 40000 `
  --target 100 `
  --iterations 10 `
  --max-time 0.2 `
  --output-dir datasets/tichu-decisions-v2
```

생성 데이터는 다시 만들 수 있고 용량이 커질 수 있으므로 Git에 커밋하지 않는다.
