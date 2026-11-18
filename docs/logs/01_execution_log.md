# 작업 01 실행·검증 기록

- AGENTS.md, CLAUDE.md: PowerShell `Get-Content -Encoding UTF8`로 읽음.
- 실제 인터프리터 검색: `Get-Command python,py,uv,conda,git`; `py -0`; 사용자 일반 설치 위치와 C:/Program Files, E:/ 재귀 검색. WindowsApps Python 스텁만 발견.
- uv 설치 시도: GitHub 릴리스 zip을 .tools/에 설치하려 했으나 Invoke-WebRequest는 연결 종료, curl은 SEC_E_NO_CREDENTIALS, Node fetch는 AggregateError로 실패. PyPI 연결도 실패. HTTP PyPI는 HTTPS 필수 403. winget도 WindowsApps 실행 접근 실패.
- Python 가상환경, 의존성 설치·버전 고정, git init: 실행 가능한 도구 및 설치 연결이 없어 미완료. 커밋 없음. requirements.in에 요청 패키지 10개 기록.
- 폴더 생성: data/raw, data/processed, src, outputs. 원본은 data/의 기존 위치 유지.
- 조사 실행: 연결된 Node REPL에서 `await import('file:///E:/projects/DataAnalysis/src/survey.mjs')`. Windows PowerShell .NET CP949 엄격 디코더를 호출하며 원바이트 재인코딩 SHA256 일치를 검사함.
- 독립 검증: PowerShell `ConvertFrom-Csv`로 148,302행, 커피숍 40,665건, 폐업일 없는 폐업 4건을 재확인. 조사 스크립트와 일치.
- 원본 SHA256: 44a5c4cbe64b0dd7d9cbbfa28c0532f003910644ddf22b2cbf4bf24751c16f3b. 실행 전후 동일.
- CP949 임시 디코딩 파일은 data/processed/에 잠시 만들고 읽은 직후 삭제. 데이터 행 제외 0건.
- 좌표 검증은 로컬 역투영 근사·사각형 범위 검사까지. pyproj 설치와 실제 서울 경계 파일이 없어 정확한 WGS84 datum 변환 및 서울 행정경계 밖 건수는 미완료.
- 카페, 폐업, 승계, 관측 기간 정의는 확정하지 않음. 값별 건수와 선택지별 집계는 01_data_survey.md에 수록.
- 추가 확인: Node의 PyPI TCP 443 연결이 EACCES로 차단됨. 추가 연도별 집계의 첫 실행은 시간 제한에 걸렸으나 날짜 파싱 캐시를 적용한 최종 실행은 약 26초에 성공. 행 수·원본 해시·독립 CSV 검증 결과 유지.
