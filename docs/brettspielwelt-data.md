# Brettspielwelt 인간 대국 데이터

Brettspielwelt의 `.tch` 로그를 수집하고 rules-v2 엔진으로 완전 재생한 뒤, BC
학습용 데이터셋으로 변환한다. 원본과 생성 데이터는 용량이 크고 재생성 가능하므로
Git에 커밋하지 않는다.

## 1. 최신 완전 게임 10,000개 수집

```powershell
docker compose run --rm --entrypoint python tests `
  scraper/download_brettspielwelt.py `
  --discover-latest `
  --latest-id 2419834 `
  --target-total 10000 `
  --workers 8 `
  --delay 0.1 `
  --progress-every 250
```

기본 저장 위치는 `datasets/raw/brettspielwelt/`다. 수집기는 최신 ID를 자동
탐색하며, 기존 파일 검증·중단 재개·원자적 임시 파일·manifest를 지원한다.

## 2. rules-v2 완전 재생 검증

```powershell
docker compose run --rm --entrypoint python tests `
  scraper/brettspielwelt_replay.py `
  --workers 8 `
  --report datasets/processed/brettspielwelt-replay-report-v2.jsonl
```

현재 원본은 전체 96,257라운드다. 원본 자체가 미완성인 488라운드를 제외한
95,769라운드의 행동과 최종 점수가 엔진과 완전히 일치한다.

## 3. 데이터셋 v2 스모크 생성

전체 변환 전에 최신 5게임으로 출력과 진행률을 확인한다.

```powershell
docker compose run --rm --entrypoint python train-bc `
  scraper/prepare_brettspielwelt_dataset.py `
  --max-games 5 `
  --workers 2 `
  --output-dir datasets/processed/brettspielwelt-decisions-v2-smoke
```

## 4. 전체 데이터셋 v2 생성

```powershell
docker compose run --rm --entrypoint python train-bc `
  scraper/prepare_brettspielwelt_dataset.py `
  --workers 8 `
  --output-dir datasets/processed/brettspielwelt-decisions-v2
```

약 5%마다 처리 파일 수, 백분율과 누적 결정 수가 출력된다. 기본 출력은 결정적
gzip JSONL이며, 같은 입력과 옵션으로 다시 만들면 데이터 파일 SHA-256이 같다.

출력 파일:

- `train.jsonl.gz`, `validation.jsonl.gz`, `test.jsonl.gz`
- `split-manifest.jsonl.gz`: 가명화된 게임별 분할과 레코드 수
- `metadata.json`: 규칙·스키마 버전, 통계, 파일 해시, 누수 검사 결과

현재 전체 생성 결과:

- 전체 6,542,405결정
- train 5,209,033 / validation 682,767 / test 650,605
- 직접 기록 행동 5,367,047 / 추론 폭탄 보류 1,175,358
- 실제 폭탄 42,505(사각 30,672 / 스트레이트 11,833)
- 분할 누수 0, 압축 데이터와 manifest 약 326MiB
- 연속 두 번의 전체 생성에서 모든 데이터 파일 SHA-256 일치

기본 분할은 게임 단위 80/10/10이다. 같은 게임의 모든 라운드와 결정은 반드시
하나의 분할에만 들어가며, test는 최종 BC v2 선택 전까지 사용하지 않는다.

## 스키마 v2의 정보 범위

한 줄은 한 번의 인간 선택이다.

- 자신의 손패와 공개된 손패 수·획득 트릭·점수
- 현재 트릭, 소원, 순위와 티츄 선언
- 현재 합법 행동과 인간이 선택한 행동
- 정상 플레이인지 폭탄 응답인지, 공개된 정상 차례 복귀 위치
- 해당 라운드에서 행동자 팀의 점수와 승패

다른 플레이어의 실제 손패와 폭탄 보유자 전체 목록은 저장하지 않는다. 플레이어
이름은 제외하고, 다운로드 가능한 원본 게임 번호도 안정적인 해시 ID로 가명화한다.

Brettspielwelt 로그는 실제 폭탄만 기록하고 폭탄을 내지 않은 응답은 생략한다.
rules-v2가 폭탄 보유자에게 응답 기회가 있었음을 확인했지만 다음 로그 행동이 폭탄이
아니라면 `decision_source: inferred_bomb_pass`로 기록한다. 직접 기록된 행동과
구분되므로 학습 시 포함 여부나 가중치를 별도로 비교할 수 있다.

## 학습 구성

하나의 전체 행동 데이터셋에서 다음 구성을 파생한다. 같은 레코드를 여러 벌 저장하지
않아 디스크 낭비를 피한다.

- 전체 행동: 필터 없이 사용
- 승리 팀 행동: `outcome.result == "win"`만 사용
- 결과 가중: `outcome.point_diff`를 이용하되 가중식은 validation에서 선택

물리적으로 같은 조합의 무늬 차이 때문에 엔진 대표 행동과 원본 카드가 다를 때는
`chosen_action`을 합법 대표 행동으로 정규화한다. `original_chosen_action`과
`action_canonicalized`로 원래 선택과 정규화 여부를 감사할 수 있다.
