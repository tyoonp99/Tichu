# 실험 결과 안내

- `benchmarks/main/`: 최종 승격 판단에 사용한 본 평가 CSV다.
- `benchmarks/smoke/`: 30 seed 또는 그보다 작은 통합 점검 결과다.
- `benchmarks/archive/`: Model C 이전 기준선과 미승격 PUCT 비교 결과다.
- `evaluations/final/`: 동결된 후보의 단 한 번의 test 평가 결과다.
- `evaluations/validation/`: 최종 후보 선택에 사용한 validation 분석이다.
- `evaluations/archive/`, `tuning/archive/`: Baseline B·Model C v1 및 중간 튜닝 기록이다.

현재 최종 대국 결과는
`benchmarks/main/model-c-vs-fuegi-mcts-10iter-200seeds.csv`이며, 해석 조건은
`docs/benchmark-protocol.md`를 따른다. 모델 가중치와 원본 데이터셋은 용량 때문에
Git에서 제외되며, 결과 CSV와 문서는 재현 근거로 추적한다.
