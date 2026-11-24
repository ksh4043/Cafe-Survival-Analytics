# 보고서 그림 실행 로그

- 입력: `data/processed/cafes_analysis.parquet` (11,501행)
- Cox 표본 흐름 입력: `docs/tables/05_cox_sample_flow.csv`
- 지도 좌표계: 인허가 EPSG:5174 → 표시 EPSG:5181; 상권 경계 EPSG:5181
- 생성 그림: 5개 (200dpi PNG)
- 그림 참조 보강: 04_kaplan_meier_report.md: 13개 그림을 결과 문단 아래에 삽입
- 검증: 필수 열·입력 파일·생성 파일 존재를 검사했다. 05·06 단계 보고서는 수정하지 않았다.
- 육안 점검(2026-11-24): 다섯 생성 그림을 열어 글자·범례·기준선·주석의 겹침과 잘림, 표본 흐름 화살표 방향을 확인했다.
