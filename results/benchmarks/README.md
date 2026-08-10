# 벤치마크 결과

이 디렉터리에는 재현하거나 비교할 가치가 있는 기준 실험 결과만 보관합니다.
실행 중 생성되는 임시 결과와 게임 로그는 저장소에 커밋하지 않습니다.

## 파일 구성

- `mcts-baseline-100-fixed.csv`: 오류 수정 후 MCTS 대 BalancedRandom 기준선
- `fuegi-vs-random-20.csv`: Fuegi 대 Random
- `fuegi-vs-balanced-20.csv`: Fuegi 대 BalancedRandom
- `fuegi-vs-mcts-20.csv`: Fuegi 대 MCTS
- `fuegi-mcts-guided-vs-fuegi-20.csv`: Fuegi-guided MCTS 대 Fuegi
- `fuegi-mcts-guided-vs-mcts-10.csv`: Fuegi-guided MCTS 대 MCTS

파일명의 마지막 숫자는 실험에 사용한 기본 시드 수입니다. 각 시드는 좌석 효과를 줄이기 위해
팀 위치를 바꾼 경기까지 포함할 수 있으므로 CSV 행 수는 시드 수보다 많을 수 있습니다.

새 실험은 먼저 기본 출력 파일로 실행한 뒤, 결과가 유효하다고 확인된 경우에만 이 디렉터리로
옮기고 실험 조건을 파일명이나 문서에 기록합니다.
