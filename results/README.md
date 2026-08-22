# 실험 결과 안내

- `benchmarks/`: 실제 대국 결과 CSV. 파일 이름의 `smoke`는 30 seed(60게임),
  `main-200seeds`는 본 평가 200 seed(400게임)다.
- `evaluations/`: 고정 validation/test 분할에서의 행동 모방 정확도 분석 JSON·CSV다.
- `tuning/`: Baseline B 및 Model C 선택 전 하이퍼파라미터 탐색 요약이다.

현재 최종 대국 결과는
`benchmarks/model-c-vs-fuegi-mcts-10iter-main-200seeds.csv`이며, 해석 조건은
`docs/benchmark-protocol.md`를 따른다. 모델 가중치와 원본 데이터셋은 용량 때문에
Git에서 제외되며, 결과 CSV와 문서는 재현 근거로 추적한다.
