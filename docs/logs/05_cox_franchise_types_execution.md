# 프랜차이즈 유형·거리 Cox 실행 로그

- 입력: cafes_analysis 11,501행, cafes_clean 39,602행
- 최종 표본: 7,351행, 사건 3,411건
- 검증: 유형 및 거리 구간 합계가 기존 200m 경쟁 변수와 점포별 일치
- Wald 검정: lifelines의 군집 샌드위치 공분산(대각·비대각)을 재구성해 사용
- 패키지 설치·원본 데이터 수정 없음
- 실행: .venv\Scripts\python.exe src\05_cox_franchise_types.py
