# Cox 강건성 검증 실행 로그

- 입력: `data/processed/cafes_analysis.parquet` (11,501행)
- 주 분석 공통 표본: 7,351행, 사건 3,411건
- 실행: `src/05_cox.py` 재실행 후 `05_cox_robustness.py`의 ph·radius·samples_core·samples_variants·samples_period·finalize 단계 실행
- 주 분석 계수·적합도는 수정하지 않고, PH 표와 주 분석 보고서의 PH 절만 수정
- 패키지 설치·원본 데이터 수정 없음
- 시간 구간 보조모형 표준오차: 비군집 표준오차
- 완료: PH 재검정, 반경·표본 강건성 표와 그림, 보고서 생성

## 시간구간 재추정

- 실행: `.venv\Scripts\python.exe src\05_cox_time_split.py`
- 입력·공통 표본: `cafes_analysis.parquet`, 7,351개 점포, 사건 3,411건
- 분할: 180일·365일·730일; `CoxPHFitter(entry_col)`와 개업연도 층화, 배정 상권 군집 강건 샌드위치 분산
- 365일 분할: 13,153 person-period, 사건 3,411건
- 분할 무상호작용 계수 검증: 최대 절대차 3.830e-15
- PH 판단 수정: lifelines rank는 누적 사건순서 구현이어서 평균 시간순위 상관과 다르므로 판단에서 제외; KM·log·직접 상관 사용
- 패키지 설치·원본 데이터·06단계 파일 수정 없음
