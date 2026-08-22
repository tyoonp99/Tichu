# 학습 코드 안내

현재 경로:

- `decision_dataset.py`: gzip JSONL 스트리밍 Dataset/DataLoader
- `model_c.py`, `train_model_c.py`, `train_model_c_v2.py`: 최종 Model C v2 구조와 학습
- `evaluate_baseline_b.py`: validation/test 세부 지표 평가
- `tune_baseline_b*.py`, `train_baseline_b.py`: MLP 기준선과 튜닝 기록

`train_dqn.py`, `visualize_logs.py`, `behavior_cloning.py`,
`train_behavior_cloning.py`는 이전 연구 또는 비교 기준선 코드다. 삭제하지 않으며,
새 강화학습 작업은 Model C v2 체크포인트를 초기 정책으로 별도 경로에서 시작한다.
