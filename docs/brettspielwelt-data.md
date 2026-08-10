# Brettspielwelt 인간 대국 로그 수집

Brettspielwelt의 `.tch` 게임 주소는 숫자 게임 ID를 사용하며, 응답은 HTML이
아닌 일반 텍스트다. 따라서 개별 게임 다운로드에는 BeautifulSoup가 필요하지
않다. 월별 HTML 인덱스를 탐색하는 기능을 추가할 때만 BeautifulSoup를 사용한다.

원본 로그는 재생 검증과 학습 데이터 변환의 입력으로만 사용한다. 플레이어 이름은
학습 데이터 생성 단계에서 익명화하고, 원본과 변환 데이터는 Git에 커밋하지 않는다.

## 최근 1,000개 수집

```powershell
docker compose run --rm tests python scraper/download_brettspielwelt.py `
  --latest-id 2419834 `
  --count 1000
```

기본 저장 위치는 `datasets/raw/brettspielwelt/`다. 각 요청 결과는 같은 디렉터리의
`manifest.jsonl`에 기록된다. 명령을 다시 실행하면 이미 저장된 로그는 다시 받지
않으므로 중단된 수집을 이어갈 수 있다.

수집기는 응답을 다음과 같이 분류한다.

- `complete`: 게임 구조 표식과 최소 한 개의 `Ergebnis:`가 있음
- `incomplete`: 게임 구조는 있으나 완료된 라운드 결과가 없음
- `invalid`: `.tch` 게임 형식이 아님
- `http_*`, `network_error`: 서버 또는 네트워크 오류
- `skipped_complete`, `skipped_incomplete`: 기존 파일을 검증하고 건너뜀

서버 부하를 줄이기 위해 요청 사이에 기본 0.1초 간격을 둔다. 전체 ID 범위를 한 번에
수집하기 전에 1,000개 표본의 정상 비율과 파서 재생 성공률을 먼저 확인한다.

## 엔진 재생 검증

```powershell
docker compose run --rm tests python scraper/brettspielwelt_replay.py
```

완료된 라운드의 모든 행동을 로컬 엔진에 적용하고 최종 점수까지 원본 로그와 같은지
확인한다. 상세 결과는 기본적으로
`datasets/processed/brettspielwelt-replay-report.jsonl`에 저장된다.

## 승리 팀 행동 데이터 생성

```powershell
docker compose run --rm tests python scraper/prepare_brettspielwelt_dataset.py
```

재생과 점수 검증을 모두 통과한 라운드만 사용한다. 논문의 초기 실험과 동일하게 기본값은
승리 팀 행동만 기록한다. 학습/검증 분리는 게임 ID 단위로 고정되므로 같은 게임의 라운드가
양쪽에 섞이지 않는다. 플레이어 이름과 상대의 실제 손패는 출력하지 않는다.

레거시 엔진은 같은 등급의 여러 무늬 중 하나만 합법 행동의 대표로 생성한다. 그래서 인간이
고른 물리적 카드가 대표 카드와 다를 때는 `chosen_action`을 합법 대표 행동으로 정규화하고,
원래 선택은 `original_chosen_action`에 보존한다. `action_canonicalized`로 정규화 여부를
확인할 수 있다.
