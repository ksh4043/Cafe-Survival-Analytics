# Codex 작업: 작업 회차 번호 제거

AGENTS.md(특히 새로 추가된 "기계적인 번호" 규칙)와 CLAUDE.md를 따른다. 환경 규칙(`.venv` 사용, 설치 금지, 임시 파일 `.tmp/`, 커밋 금지)은 `prompts/02/chain_and_noncafe.md`와 같다.

프롬프트 파일은 `prompts/02/` 아래로 옮겨졌고 이름이 바뀌었다(preprocess_licenses → preprocess_followup → franchise_dictionary_fix → chain_and_noncafe → dictionary_patch). 이제 코드와 산출물에서 `02b`, `02c`, `02d`, `02e` 같은 회차 번호를 모두 없앤다. 분석 결과는 바뀌면 안 된다.

## 1. 검수 파일 이름
- `docs/review/02c_narrowed_brands.csv` → `docs/review/02_narrowed_brands.csv`
- `docs/review/02c_chain_candidates.csv` → `docs/review/02_chain_candidates.csv`
- `docs/review/02d_chain_classification.csv` → `docs/review/02_chain_classification.csv`
- 옛 이름 파일은 남기지 않는다.

## 2. 코드와 사전
- `src/02_preprocess.py`: 경로 상수, 변수명(`pre_02d_*` 등), 주석, 보고서 생성 문자열에서 회차 번호를 없애고 내용을 나타내는 이름으로 바꾼다. 예: `pre_02d_brands` → `brands_before_chain_addition`.
- `src/franchise_brands.csv`의 `비고` 열에 붙은 `02d`, `02e` 표기를 내용 설명으로 바꾼다. 예: "지점 5개 이상 카페 체인 추가", "오타 표기 추가". 코드가 `비고`의 `02d` 문자열로 브랜드를 구분하고 있다면(`str.contains("02d")`), 문자열 검색 대신 명시적인 열(예: `source` 열에 `initial` / `chain_candidates` 같은 값)이나 다른 명확한 방법으로 바꾼다.
- `src/noncafe_patterns.csv`와 그 밖의 `src/` 파일에도 같은 표기가 있으면 고친다.

## 3. 보고서
- `docs/reports/02_preprocess_report.md`에서 "02b 결과", "02c의 294개 후보", "02d 반영 뒤" 같은 표현을 내용으로 바꾼다. 예: "체인 후보 분류 전", "체인 추가 후".
- 이전 결과와 비교하는 표는 유지하되 열 이름을 회차 번호 대신 단계 내용으로 쓴다.

## 4. 검증
- 스크립트를 다시 실행한다. 최종 분석 표본이 개인 8,052, 프랜차이즈 3,463, 사건 4,253으로 그대로인지 확인한다. 두 parquet의 내용(행, 열, 값)이 수정 전과 같은지 비교해 보고한다.
- `.venv`, `data/`, `.git`을 뺀 프로젝트 전체에서 `0[0-9][b-e]` 패턴과 "보완 [0-9]" 패턴을 검색해 남은 것이 없는지 보고한다. 단계 번호(`01_`, `02_`)는 그대로 둔다.
- 그 밖의 정의·사전 패턴·제외 기준은 바꾸지 않는다.
