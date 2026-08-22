# 벤치마크 결과

- `main/`: 본 평가. Model C v2의 Fuegi-MCTS 및 Baseline B 비교 결과를 둔다.
- `smoke/model-c/`: Model C와 BC-guided MCTS의 소규모 통합 실험이다.
- `smoke/selfplay/`: PPO self-play 체크포인트의 비승격 smoke 평가 결과다.
- `archive/pre-model-c/`: 이전 Random·Fuegi·BC·MCTS 기준선 결과다.

모든 CSV에서 `--games`는 seed 수이며, 각 seed는 좌석 교환 전후 두 게임으로 기록된다.
본 평가의 해석 조건은 `docs/benchmark-protocol.md`를 따른다.
