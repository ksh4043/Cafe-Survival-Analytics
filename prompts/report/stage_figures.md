# Codex 작업: 전처리·공간 결합 단계 그림 보강과 보고서 그림 삽입

AGENTS.md의 "결과 보고서 구성"에 새로 넣은 그림 규칙을 따른다. `.venv`를 쓰고 패키지는 설치하지 말며(없는 패키지가 필요하면 보고), 임시 파일은 `.tmp/`에 둔다. 가설은 경쟁·수요·상쇄라는 이름으로만 쓴다.

## 배경
최종 보고서와 발표 자료를 만들어야 하는데, 지금까지 보고서에는 그림이 파일명으로만 언급되어 있고 전처리·공간 결합 단계는 그림이 하나도 없다. 분석 결과·데이터·정의는 바꾸지 않고 그림만 보강한다.

## 1. 새 그림 (`src/report_figures.py` 한 스크립트로 만든다)
기존 처리된 데이터(`data/processed/`)와 표(`docs/tables/`)만 읽는다. 각 그림은 아래 목적이 드러나게 그리고, 데이터 상태상 그리기 어렵거나 의미가 없으면 건너뛰고 이유를 보고한다.
- `02_sample_flow.png`: 원본 인허가 → 커피숍 → 개업 기간 → 제외(한시적, 0일 이하, 카페가 아닌 업종, 폐업일 없는 폐업 등) → 개인/프랜차이즈 → Cox 주 분석 표본(7,351, 사건 3,411)까지 단계별 건수. 보고서의 표본 설명에 그대로 쓸 수 있는 흐름도 형태. 건수는 `02`·`03`·`05` 보고서와 `05_cox_sample_flow.csv`의 수치와 맞춘다.
- `02_openings_closures_by_quarter.png`: 분기별 개업 수와 폐업 수(개인/프랜차이즈 구분). 개업 기간 경계와 관측 종료일 표시.
- `02_franchise_brands.png`: 표본의 프랜차이즈 상위 브랜드 점포 수(상위 15개 정도).
- `03_cafe_map.png`: 서울 지도 위 개업 기간 표본 점포 위치(영업 중/폐업 색 구분, 개인/프랜차이즈 구분), 상권 영역 경계를 옅게. 좌표계는 EPSG:5181로 통일.
- `03_competition_density_map.png`: 개인 카페 200m 경쟁 수를 점포 위치에 색으로(또는 격자 집계). 경쟁이 어디에 몰리는지 보이게.
- `03_competition_distribution.png`: 100·200·500m 경쟁 수 분포(개인/프랜차이즈 구분).
- `03_trade_area_distance.png`: 가장 가까운 상권까지 거리 분포와 200m 기준선(주 분석 포함·제외 비율 표시).
- `03_competition_vs_demand.png`: 200m 경쟁 수와 로그 상권 총매출·로그 유동인구의 관계(산점도 또는 hexbin + 추세선, 상관계수 표기). 수요가 많은 곳에 경쟁도 몰리는지(상쇄의 전제)를 보여 주는 그림.
- `03_covariate_correlation.png`: Cox 투입 공변량 상관 히트맵(`03_robustness_and_correlation` 결과와 같은 값).

## 2. 기존 보고서에 그림 넣기
- `docs/reports/02_preprocess_report.md`, `03_spatial_join_report.md`, `03_trade_area_variables_report.md`, `03_robustness_and_correlation_report.md`: 위 그림을 관련 결과 문단 바로 아래에 `![설명](../figures/파일명.png)`과 한 줄 설명으로 넣는다.
- `docs/reports/04_kaplan_meier_report.md`: 이미 있는 `04_km_*.png` 13개를 파일명 언급 대신 해당 결과 문단 아래에 삽입한다. 본문 서술은 바꾸지 않는다.
- `docs/reports/05_*`, `06_*` 보고서는 다른 작업이 동시에 수정 중이므로 건드리지 않는다.
- 기존 그림의 글자 크기가 슬라이드에 넣기에 너무 작으면(예: 9pt 미만) 목록으로 보고만 하고 고치지 않는다.

## 산출물
- 스크립트: `src/report_figures.py`
- 그림: `docs/figures/02_*.png`, `docs/figures/03_*.png`
- 그림 목록: `docs/figures/README.md` — 전 단계 그림 파일명, 한 줄 설명, 보고서·발표에서 쓸 만한 용도(본문/부록/발표)를 표로 정리한다. 기존 `04_*`, `05_*` 그림도 포함한다.
- 실행 로그: `docs/logs/report_figures_execution.md`

AGENTS.md 보고 형식에 맞춰 한국어로 보고한다.
