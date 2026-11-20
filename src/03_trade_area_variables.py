"""Attach pre-opening commercial-area measures to the coffee-shop cohort.

Run from the project root with::

    .venv\\Scripts\\python.exe src\\03_trade_area_variables.py

Raw CSV files are read as CP949 and are never modified.  Every time-varying
measure is keyed to the quarter immediately preceding the permit quarter.
"""

from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
INPUT = ROOT / "data/processed/cafes_spatial.parquet"
AREA = RAW / "area/서울시 상권분석서비스(영역-상권).shp"
OUTPUT = ROOT / "data/processed/cafes_analysis.parquet"
SUMMARY = ROOT / "docs/tables/03_trade_area_summary.csv"
REVIEW = ROOT / "docs/review/03_cafes_analysis_sample.csv"
REPORT = ROOT / "docs/reports/03_trade_area_variables_report.md"
COFFEE = "CS100010"

ENGLISH_COLUMNS = {
    "stdr_yyqu_cd": "quarter", "trdar_se_cd": "trdar_type_code",
    "trdar_se_cd_nm": "trdar_type", "trdar_cd": "trdar_cd",
    "trdar_cd_nm": "trdar_name", "svc_induty_cd": "industry_code",
    "svc_induty_cd_nm": "industry_name", "stor_co": "store_count",
    "opbiz_rt": "open_rate", "clsbiz_rt": "close_rate",
    "frc_stor_co": "franchise_store_count",
}
KOREAN_COLUMNS = {
    "기준_년분기_코드": "quarter", "상권_구분_코드": "trdar_type_code",
    "상권_구분_코드_명": "trdar_type", "상권_코드": "trdar_cd",
    "상권_코드_명": "trdar_name", "서비스_업종_코드": "industry_code",
    "서비스_업종_코드_명": "industry_name", "당월_매출_금액": "sales_amount",
    "점포_수": "store_count", "개업_율": "open_rate", "폐업_률": "close_rate",
    "프랜차이즈_점포_수": "franchise_store_count", "총_유동인구_수": "floating_pop",
    "총_상주인구_수": "resident_pop",
}
MEASURES = [
    "trdar_sales_total", "trdar_sales_coffee", "trdar_floating_pop", "trdar_resident_pop",
    "trdar_coffee_stores", "trdar_coffee_franchise_stores", "trdar_coffee_open_rate",
    "trdar_coffee_close_rate", "trdar_area_sqm",
]


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(해당 없음)"
    values = frame.copy()
    for column in values:
        values[column] = values[column].map(lambda x: "" if pd.isna(x) else str(x).replace("|", "\\|"))
    return "\n".join([
        "| " + " | ".join(values.columns) + " |",
        "| " + " | ".join("---" for _ in values.columns) + " |",
        *("| " + " | ".join(row) + " |" for row in values.itertuples(index=False, name=None)),
    ])


def format_count(number: int) -> str:
    return f"{number:,}"


def normalize_code(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def read_files(folder: str, required: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((RAW / folder).glob("*.csv")):
        frame = pd.read_csv(path, encoding="cp949", low_memory=False)
        frame = frame.rename(columns=KOREAN_COLUMNS | ENGLISH_COLUMNS)
        missing = set(required).difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name}: required columns missing: {sorted(missing)}")
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No CSV file in {RAW / folder}")
    result = pd.concat(frames, ignore_index=True)
    result["quarter"] = pd.to_numeric(result["quarter"], errors="raise").astype("int64")
    result["trdar_cd"] = result["trdar_cd"].map(normalize_code).astype("string")
    if "industry_code" in result:
        result["industry_code"] = result["industry_code"].astype("string")
    return result


def previous_quarter(dates: pd.Series) -> pd.Series:
    quarter = dates.dt.to_period("Q") - 1
    return (quarter.dt.year * 10 + quarter.dt.quarter).astype("int64")


def unique_key(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    duplicated = int(frame.duplicated(keys).sum())
    if duplicated:
        raise ValueError(f"{label}: {duplicated:,} duplicated keys for {keys}")


def make_trade_area_panel() -> tuple[pd.DataFrame, dict[str, object]]:
    sales = read_files("estimated_sales", ["quarter", "trdar_cd", "industry_code", "sales_amount"])
    stores = read_files("store", ["quarter", "trdar_cd", "industry_code", "store_count", "franchise_store_count", "open_rate", "close_rate"])
    floating = read_files("population_by_street", ["quarter", "trdar_cd", "floating_pop"])
    residents = read_files("resident_population", ["quarter", "trdar_cd", "resident_pop"])

    for frame, label in ((sales, "매출"), (stores, "점포")):
        unique_key(frame, ["quarter", "trdar_cd", "industry_code"], label)
    for frame, label in ((floating, "유동인구"), (residents, "상주인구")):
        unique_key(frame, ["quarter", "trdar_cd"], label)

    sales["sales_amount"] = pd.to_numeric(sales["sales_amount"], errors="raise")
    sales_total = sales.groupby(["quarter", "trdar_cd"], as_index=False)["sales_amount"].sum().rename(columns={"sales_amount": "trdar_sales_total"})
    sales_coffee = sales.loc[sales["industry_code"].eq(COFFEE), ["quarter", "trdar_cd", "sales_amount"]].rename(columns={"sales_amount": "trdar_sales_coffee"})
    coffee_stores = stores.loc[stores["industry_code"].eq(COFFEE), ["quarter", "trdar_cd", "store_count", "franchise_store_count", "open_rate", "close_rate"]].copy()
    coffee_stores = coffee_stores.rename(columns={
        "store_count": "trdar_coffee_stores", "franchise_store_count": "trdar_coffee_franchise_stores",
        "open_rate": "trdar_coffee_open_rate", "close_rate": "trdar_coffee_close_rate",
    })
    for column in coffee_stores.columns.difference(["quarter", "trdar_cd"]):
        coffee_stores[column] = pd.to_numeric(coffee_stores[column], errors="raise")
    floating = floating[["quarter", "trdar_cd", "floating_pop"]].rename(columns={"floating_pop": "trdar_floating_pop"})
    residents = residents[["quarter", "trdar_cd", "resident_pop"]].rename(columns={"resident_pop": "trdar_resident_pop"})
    for frame in (floating, residents):
        for column in frame.columns.difference(["quarter", "trdar_cd"]):
            frame[column] = pd.to_numeric(frame[column], errors="raise")

    panel = sales_total.merge(sales_coffee, on=["quarter", "trdar_cd"], how="left")
    panel = panel.merge(coffee_stores, on=["quarter", "trdar_cd"], how="left")
    panel = panel.merge(floating, on=["quarter", "trdar_cd"], how="outer")
    panel = panel.merge(residents, on=["quarter", "trdar_cd"], how="outer")
    unique_key(panel, ["quarter", "trdar_cd"], "결합 상권 패널")
    meta = {
        "sales_codes": set(sales["trdar_cd"].dropna()), "store_codes": set(stores["trdar_cd"].dropna()),
        "floating_codes": set(floating["trdar_cd"].dropna()), "resident_codes": set(residents["trdar_cd"].dropna()),
        "sales_coffee_missing": int(panel["trdar_sales_coffee"].isna().sum()),
        "store_coffee_missing": int(panel["trdar_coffee_stores"].isna().sum()),
    }
    return panel, meta


def variable_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in MEASURES:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        rows.append({
            "변수": column, "유효 표본 수": len(values), "결측 수": int(frame[column].isna().sum()),
            "평균": round(float(values.mean()), 3), "중앙값": round(float(values.median()), 3),
            "1사분위": round(float(values.quantile(.25)), 3), "3사분위": round(float(values.quantile(.75)), 3),
            "최솟값": round(float(values.min()), 3), "최댓값": round(float(values.max()), 3),
        })
    return pd.DataFrame(rows)


def main() -> None:
    os.environ["TEMP"] = str(ROOT / ".tmp")
    os.environ["TMP"] = str(ROOT / ".tmp")
    (ROOT / ".tmp").mkdir(exist_ok=True)
    for path in (OUTPUT.parent, SUMMARY.parent, REVIEW.parent, REPORT.parent):
        path.mkdir(parents=True, exist_ok=True)

    cafes = pd.read_parquet(INPUT).copy()
    if cafes.empty:
        raise ValueError("Spatial cohort is empty")
    cafes["permit_date"] = pd.to_datetime(cafes["permit_date"])
    cafes["trdar_quarter_prior"] = previous_quarter(cafes["permit_date"])
    cafes["trdar_cd_assigned"] = cafes["trdar_cd_nearest"].astype("string")
    for radius in (50, 100, 200, 500):
        cafes[f"trdar_within_{radius}m"] = cafes["trdar_nearest_dist_m"].le(radius).fillna(False).astype("boolean")

    panel, meta = make_trade_area_panel()
    areas = gpd.read_file(AREA)[["TRDAR_CD", "TRDAR_SE_1", "RELM_AR"]].copy()
    areas = areas.rename(columns={"TRDAR_CD": "trdar_cd_assigned", "TRDAR_SE_1": "trdar_type", "RELM_AR": "trdar_area_sqm"})
    areas["trdar_cd_assigned"] = areas["trdar_cd_assigned"].map(normalize_code).astype("string")
    unique_key(areas, ["trdar_cd_assigned"], "상권 SHP")
    cafes = cafes.merge(areas, on="trdar_cd_assigned", how="left", validate="many_to_one")
    panel = panel.rename(columns={"quarter": "trdar_quarter_prior", "trdar_cd": "trdar_cd_assigned"})
    cafes = cafes.merge(panel, on=["trdar_quarter_prior", "trdar_cd_assigned"], how="left", validate="many_to_one")
    cafes.to_parquet(OUTPUT, index=False)
    summary = variable_summary(cafes)
    summary.to_csv(SUMMARY, index=False, encoding="utf-8-sig")
    cafes.head(200).to_csv(REVIEW, index=False, encoding="utf-8-sig")

    area_codes = set(areas["trdar_cd_assigned"].dropna())
    assignment = cafes["trdar_cd_assigned"].notna()
    personal = ~cafes["is_franchise"].fillna(False)
    groups = []
    for label, mask in (("전체 분석 표본", pd.Series(True, index=cafes.index)), ("개인 카페", personal)):
        for limit in (50, 100, 200, 500):
            groups.append({"표본": label, "배정 기준": f"최근접 상권 {limit}m 이내", "점포 수": int((mask & cafes[f"trdar_within_{limit}m"]).sum())})
        groups.append({"표본": label, "배정 기준": "거리 제한 없음(좌표·최근접 상권 있음)", "점포 수": int((mask & assignment).sum())})
    group_table = pd.DataFrame(groups)
    category = cafes.loc[assignment, "trdar_type"].value_counts(dropna=False).rename_axis("상권 구분").reset_index(name="점포 수")
    category["비율(%)"] = (category["점포 수"] / assignment.sum() * 100).round(1)
    correlations = []
    base = cafes.loc[cafes["trdar_within_200m"].fillna(False)].copy()
    for label, mask in (("전체", pd.Series(True, index=base.index)), ("개인 카페", ~base["is_franchise"].fillna(False))):
        for variable in ("trdar_sales_total", "trdar_sales_coffee", "trdar_floating_pop", "trdar_resident_pop"):
            selected = base.loc[mask, [variable, "comp_total_r200"]].dropna()
            correlations.append({"표본": label, "상권 변수": variable, "유효 표본 수": len(selected), "Pearson r": round(float(selected.corr().iloc[0, 1]), 3)})
    correlation = pd.DataFrame(correlations)
    missing_prior = cafes.loc[assignment, MEASURES[:-1]].isna().any(axis=1)
    q1_2026 = cafes["permit_date"].dt.to_period("Q").eq(pd.Period("2026Q1"))
    q1_assigned = int((q1_2026 & assignment).sum())
    q1_complete = int((q1_2026 & cafes["trdar_sales_total"].notna()).sum())
    report = f"""# 상권 변수 결합 보고서

## 주요 결과

분석 대상 {format_count(len(cafes))}개 개업 점포에 개업 **직전 분기**의 최근접 상권 지표를 결합했다. 200m 이내 최근접 상권 배정(주 분석 기본값)을 충족하는 점포는 {format_count(int(cafes['trdar_within_200m'].sum()))}개이며, 이 중 개인 카페는 {format_count(int((personal & cafes['trdar_within_200m']).sum()))}개다. 좌표가 없는 점포도 분석 파일에 남기되 상권 코드와 상권 변수는 결측으로 유지했다.

{markdown_table(group_table)}

상권이 배정된 점포의 상권 구분은 다음과 같다. 상권 폴리곤 밖 점포도 최근접 상권을 붙였으므로 이 분포는 거리 제한 없는 최근접 배정 기준이다.

{markdown_table(category)}

수요 지표와 200m 반경 경쟁 수의 단순 상관은 아래와 같다. 상관은 인과 추정이 아니지만, 수요가 큰 상권에 카페도 함께 모여 경쟁과 수요 효과가 상쇄될 가능성을 보여 주는 기초 진단이다. 특히 커피 매출은 경쟁 카페 수와 함께 커질 수 있어 **수요의 주 변수로 사용하지 않고 참고용**으로만 둔다.

{markdown_table(correlation)}

### 상권 변수 분포

요약표 전체는 `docs/tables/03_trade_area_summary.csv`에 저장했다.

{markdown_table(summary)}

2026년 1분기 개업 점포는 직전 분기인 2025년 4분기 상권 자료를 사용한다. 해당 개업 점포 {format_count(int(q1_2026.sum()))}개 중 최근접 상권이 있는 점포는 {format_count(q1_assigned)}개이고, 총매출이 결합된 점포는 {format_count(q1_complete)}개다. 따라서 최근접 상권은 있으나 2025년 4분기 총매출 행이 없는 {format_count(q1_assigned - q1_complete)}개는 결측으로 남겼다.

## 방법과 검증

### 시점과 결합 규칙

`permit_date`의 분기를 계산한 뒤 정확히 한 분기를 뺀 `trdar_quarter_prior`를 만들었다(예: 2021-05 개업 → 2021Q1). `trdar_cd_nearest`를 모든 좌표 보유 점포의 배정 상권으로 사용했고, 폴리곤 내부 점포는 최근접 거리 0m이므로 원래 포함 상권과 같다. `trdar_within_50m`·`100m`·`200m`·`500m`는 각각 최근접 거리 이하 여부이다. 200m 이내를 주 분석 기본 표본으로 제시하되, 다른 거리와 거리 제한 없음을 강건성 검증 후보로 함께 보존했다. 표본 제외나 변수의 생존모형 활용 여부는 결정하지 않았다.

매출은 상권-분기-업종 행의 `당월_매출_금액`을 모든 업종에 대해 합산해 `trdar_sales_total`, 커피·음료(`CS100010`) 행을 `trdar_sales_coffee`로 두었다. 원자료는 `기준_년분기_코드`로 제공되고 서울 열린데이터광장은 이 데이터셋의 갱신 주기를 분기별, 단위를 원으로 설명한다. 다만 컬럼명은 ‘당월’이므로 이를 3개월 합계라고 재해석하지 않고 **분기 레코드에 제공된 금액 값**으로 사용했다. 원자료에서 이 값은 요일별 매출금액의 합과 일치한다. 향후 월별 원자료나 공식 산식이 추가로 확인되면 분기 합계 여부를 재검토해야 한다.

점포 파일의 커피·음료 행에서 점포 수·프랜차이즈 점포 수·개업률·폐업률을, 길단위인구 및 상주인구 파일에서 총인구를 가져왔다. 상주인구는 제공기관 설명대로 연 4분기에 갱신되고 다음 해 1~3분기에 같은 값이 반복될 수 있다. 상권 면적과 구분은 영역 SHP의 `RELM_AR`, `TRDAR_SE_1`을 사용했다. 면적은 시점에 따라 달라지지 않는 폴리곤 속성이므로 직전 분기 결합과 별개로 최근접 상권 코드에 붙였다.

### 코드 일치와 결측 처리

영역 SHP 1,650개 코드와의 일치 코드는 매출 {len(meta['sales_codes'] & area_codes):,}개, 점포 {len(meta['store_codes'] & area_codes):,}개, 유동인구 {len(meta['floating_codes'] & area_codes):,}개, 상주인구 {len(meta['resident_codes'] & area_codes):,}개다. 커피·음료 행이 없는 상권-분기 조합은 매출 {format_count(meta['sales_coffee_missing'])}개, 점포 {format_count(meta['store_coffee_missing'])}개였다. 이 부재는 0인지 비공개·결측인지 원자료 명세에서 판별할 근거가 없으므로 **0으로 대체하지 않고 결측**으로 보존했다. 상권-분기 전체 행이 없는 경우도 같은 방식이다. 따라서 상권 변수가 없는 {format_count(int(missing_prior.sum()))}개 배정 점포는 자료 부재를 0으로 오인하지 않는다.

2025년 점포 파일은 영문 헤더(`stdr_yyqu_cd`, `trdar_cd`, `svc_induty_cd`, `stor_co`, `frc_stor_co` 등)였으며, 같은 순서·의미의 한글 헤더로 표준화한 후 결합했다. 각 원자료 내부의 상권-분기(업종 포함) 키 중복은 검사하여 0건임을 확인했다.

### 재현성 검증

스크립트는 원자료를 수정하지 않고 CP949로 읽으며, `.tmp/`를 `TEMP`/`TMP`로 지정한다. 결합 뒤 행 수는 입력과 같은 {format_count(len(cafes))}행인지 확인하고, 상권-분기 패널과 SHP 코드의 키 중복을 중단 조건으로 검사한다. 분석 파일 앞 200행은 UTF-8 BOM CSV로 별도 저장해 VS Code에서 바로 검수할 수 있다.

## 산출물과 결정이 필요한 사항

- `data/processed/cafes_analysis.parquet`: {format_count(len(cafes))}행, 기존 공간·경쟁 변수와 상권 변수 및 배정 표지
- `docs/tables/03_trade_area_summary.csv`: 상권 변수 분포
- `docs/review/03_cafes_analysis_sample.csv`: 분석 파일 앞 200행 검수본

상권 밖 점포의 200m 기본 표본 사용, 반경별 강건성 검증의 적용, 상권 변수의 모형 투입 방식은 후속 Cox 설계에서 정해야 한다. 특히 ‘당월 매출’의 공식 산식과 업종 행 부재의 의미는 제공기관 명세를 추가 확보해 확인하는 것이 바람직하다.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {SUMMARY}")
    print(f"Wrote {REVIEW}")
    print(f"Wrote {REPORT}")
    print(f"Rows: {len(cafes):,}; 200m assignment: {int(cafes['trdar_within_200m'].sum()):,}")


if __name__ == "__main__":
    main()
