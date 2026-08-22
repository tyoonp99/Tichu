# 벤치마크 결과

- `main/`: 본 평가. 현재 최종 파일은
  `model-c-vs-fuegi-mcts-10iter-200seeds.csv`다.
- `smoke/model-c/`: Model C와 BC-guided MCTS의 소규모 통합 실험이다.
- `archive/pre-model-c/`: 이전 Random·Fuegi·BC·MCTS 기준선 결과다.

모든 CSV에서 `--games`는 seed 수이며, 각 seed는 좌석 교환 전후 두 게임으로 기록된다.
본 평가의 해석 조건은 `docs/benchmark-protocol.md`를 따른다.
