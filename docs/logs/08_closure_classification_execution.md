# 개업 후 폐업 분류 실행 로그

- 실행: `src/08_closure_classification.py`
- 시드: 20261128
- 평가: 상권 단위 GroupKFold 5개, 주 임계값=학습 OOF 폐업비율 분위수
- XGBoost: 매 학습 분할 scale_pos_weight=생존 수/폐업 수
- 2년 표본: 5,737개, 폐업 1,915개(33.4%)
