# 머신러닝 생존 모형 실행 로그

- 입력: cafes_analysis 11,501행, cafes_clean 39,602행
- 최종 표본: 7,351행, 사건 3,411건
- 검증: 유형별·거리별 경쟁 수가 기존 200m 수와 점포별 일치
- 평가: 배정 상권 GroupKFold 5겹; 중첩 격자는 메모리 초과로 중단해 보수적 고정값 사용
- 패키지 설치·원본 데이터 수정 없음
- 실행: .venv\Scripts\python.exe src\08_survival_ml.py
