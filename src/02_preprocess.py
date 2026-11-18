"""Prepare the Seoul coffee-shop licence cohort and franchise review files.

Run from the project root with ``.venv\\Scripts\\python.exe src\\02_preprocess.py``.
The script never changes the raw source file.  It writes all derived data and
reports deterministically; franchise-review sampling uses a fixed random seed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data/raw/refreshment_establishment/식품_휴게음식점_서울특별시.csv"
BRAND_PATH = PROJECT_ROOT / "src/franchise_brands.csv"
PROCESSED_PATH = PROJECT_ROOT / "data/processed/cafes_licenses.parquet"
REPORT_PATH = PROJECT_ROOT / "outputs/02_preprocess_report.md"
REVIEW_PATH = PROJECT_ROOT / "outputs/02_franchise_review.csv"
UNMATCHED_PATH = PROJECT_ROOT / "outputs/02_unmatched_top.csv"

EXPECTED_SHA256 = "44a5c4cbe64b0dd7d9cbbfa28c0532f003910644ddf22b2cbf4bf24751c16f3b"
OBSERVATION_END = pd.Timestamp("2026-09-30")
STUDY_START = pd.Timestamp("2021-04-01")
STUDY_END = pd.Timestamp("2026-03-31")
RANDOM_SEED = 20261118

SEOUL_BOUNDS = {"lon_min": 126.76, "lon_max": 127.19, "lat_min": 37.42, "lat_max": 37.71}


@dataclass(frozen=True)
class CoordinateResult:
    epsg: int
    longitude: pd.Series
    latitude: pd.Series
    in_seoul_box: int
    median_longitude: float
    median_latitude: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_name(value: object) -> str:
    """Normalize spacing, case, punctuation, and Unicode for brand matching."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    return re.sub(r"[\s\-_'`.,·()/\\&]+", "", text)


def study_period_mask(permit_date: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Return the inclusive licence-date study window; dates remain configurable."""
    return permit_date.between(start, end, inclusive="both")


def transform_coordinates(x: pd.Series, y: pd.Series, epsg: int) -> CoordinateResult:
    valid = x.notna() & y.notna()
    longitude = pd.Series(np.nan, index=x.index, dtype="float64")
    latitude = pd.Series(np.nan, index=x.index, dtype="float64")
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon_values, lat_values = transformer.transform(x.loc[valid].to_numpy(), y.loc[valid].to_numpy())
    longitude.loc[valid] = lon_values
    latitude.loc[valid] = lat_values
    in_box = (
        longitude.between(SEOUL_BOUNDS["lon_min"], SEOUL_BOUNDS["lon_max"])
        & latitude.between(SEOUL_BOUNDS["lat_min"], SEOUL_BOUNDS["lat_max"])
    )
    return CoordinateResult(
        epsg=epsg,
        longitude=longitude,
        latitude=latitude,
        in_seoul_box=int(in_box.sum()),
        median_longitude=float(longitude.median()),
        median_latitude=float(latitude.median()),
    )


def choose_coordinate_system(
    results: dict[int, CoordinateResult], reference_summary: pd.DataFrame
) -> int:
    """Choose the candidate supported by matched known-address reference points.

    A Seoul bounding-box check detects gross errors but cannot distinguish these
    close Korean projected CRSs.  The median error for known building addresses
    is used first, falling back to the bounding-box count only if no references
    occur in the source data.
    """
    reference_error = reference_summary.dropna(subset=["기준점 중앙거리(m)"]).groupby("EPSG")["기준점 중앙거리(m)"].mean()
    if not reference_error.empty:
        return int(reference_error.idxmin())
    best_count = max(result.in_seoul_box for result in results.values())
    tied = [epsg for epsg, result in results.items() if result.in_seoul_box == best_count]
    return min(tied)


def load_brand_dictionary(path: Path) -> pd.DataFrame:
    brands = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
    required = {"브랜드명", "매칭 패턴(정규식)", "비고"}
    if set(brands.columns) != required:
        raise ValueError(f"Brand dictionary columns must be {sorted(required)}; found {brands.columns.tolist()}")
    brands["compiled_pattern"] = brands["매칭 패턴(정규식)"].map(re.compile)
    return brands


def classify_franchises(names: pd.Series, brands: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    normalized = names.map(normalize_name)
    brand = pd.Series(pd.NA, index=names.index, dtype="string")
    for record in brands.itertuples(index=False):
        matches = normalized.str.contains(record.compiled_pattern, na=False)
        brand.loc[matches & brand.isna()] = record.브랜드명
    return brand.notna().astype(bool), brand


def format_int(value: int | float) -> str:
    return f"{int(value):,}"


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(해당 없음)"
    columns = [str(column) for column in frame.columns]

    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    rows.extend("| " + " | ".join(cell(value) for value in record) + " |" for record in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def reference_address_diagnostics(frame: pd.DataFrame, coordinate_results: dict[int, CoordinateResult]) -> pd.DataFrame:
    """Compare matched cafe addresses to three known Seoul reference points.

    This is an address-based sanity check, not a substitute for an official CRS
    field.  Multiple business units can share an address, so median error is
    reported rather than assuming every licence sits at a building centroid.
    """
    reference_points = (
        ("서울시청", "세종대로 110", 126.9779692, 37.566535),
        ("서울역", "한강대로 405", 126.97075, 37.55468),
    )
    address = frame["도로명주소"].fillna("").astype(str).str.strip()
    rows: list[dict[str, object]] = []
    for point_name, address_text, point_lon, point_lat in reference_points:
        target = address.str.contains(address_text, regex=False)
        for epsg, result in coordinate_results.items():
            lon = result.longitude.loc[target]
            lat = result.latitude.loc[target]
            valid = lon.notna() & lat.notna()
            # Equirectangular approximation is adequate for a building-level check.
            dx = (lon.loc[valid] - point_lon) * 88_200
            dy = (lat.loc[valid] - point_lat) * 111_000
            distance = np.sqrt(dx**2 + dy**2)
            rows.append(
                {
                    "기준점": point_name,
                    "주소 일치 문자열": address_text,
                    "EPSG": epsg,
                    "해당 카페 행": int(target.sum()),
                    "좌표 보유": int(valid.sum()),
                    "기준점 중앙거리(m)": round(float(np.median(distance)), 1) if len(distance) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write Parquet using the installed pandas engine, failing transparently otherwise."""
    try:
        frame.to_parquet(path, index=False)
    except ImportError as error:
        raise RuntimeError(
            "Parquet writer is unavailable in the prepared environment. "
            "Install neither packages nor write a substitute: add pyarrow or fastparquet to the managed environment."
        ) from error


def main() -> None:
    for directory in (PROCESSED_PATH.parent, REPORT_PATH.parent):
        directory.mkdir(parents=True, exist_ok=True)

    observed_hash = sha256(RAW_PATH)
    if observed_hash != EXPECTED_SHA256:
        raise RuntimeError(f"Raw-file SHA256 mismatch: expected {EXPECTED_SHA256}, got {observed_hash}")

    raw = pd.read_csv(RAW_PATH, encoding="cp949", dtype=str, keep_default_na=True)
    raw_rows = len(raw)
    cafe = raw.loc[raw["업태구분명"].str.strip().eq("커피숍")].copy()
    coffee_rows = len(cafe)

    cafe["permit_date"] = pd.to_datetime(cafe["인허가일자"].str.strip(), format="%Y-%m-%d", errors="coerce")
    cafe["closure_date"] = pd.to_datetime(cafe["폐업일자"].str.strip(), format="%Y-%m-%d", errors="coerce")
    cafe["operation_status"] = cafe["영업상태명"].fillna("").str.strip()
    closure_status = cafe["operation_status"].eq("폐업")
    closure_status_missing_date = closure_status & cafe["closure_date"].isna()
    permit_parse_failure = cafe["permit_date"].isna()
    closure_date_parse_failure = cafe["폐업일자"].notna() & cafe["closure_date"].isna()

    # A licence cannot enter a cohort ending before it opened.
    permit_after_observation = cafe["permit_date"].gt(OBSERVATION_END)
    exclusions = closure_status_missing_date | permit_after_observation | permit_parse_failure
    included = cafe.loc[~exclusions].copy()

    included["event"] = included["closure_date"].le(OBSERVATION_END).astype("int8")
    included["analysis_end_date"] = included["closure_date"].where(included["event"].eq(1), OBSERVATION_END)
    included["survival_days"] = (included["analysis_end_date"] - included["permit_date"]).dt.days.astype("Int64")
    included["in_study_period"] = study_period_mask(included["permit_date"], STUDY_START, STUDY_END)

    included["coordinate_x"] = pd.to_numeric(included["좌표정보(X)"].astype("string").str.strip(), errors="coerce")
    included["coordinate_y"] = pd.to_numeric(included["좌표정보(Y)"].astype("string").str.strip(), errors="coerce")
    coordinate_results = {
        epsg: transform_coordinates(included["coordinate_x"], included["coordinate_y"], epsg)
        for epsg in (5174, 2097, 5181)
    }
    reference_summary = reference_address_diagnostics(included, coordinate_results)
    selected_epsg = choose_coordinate_system(coordinate_results, reference_summary)
    included["coordinate_crs_epsg"] = selected_epsg
    included["longitude"] = coordinate_results[selected_epsg].longitude
    included["latitude"] = coordinate_results[selected_epsg].latitude
    included["has_coordinates"] = included["coordinate_x"].notna() & included["coordinate_y"].notna()

    brands = load_brand_dictionary(BRAND_PATH)
    included["is_franchise"], included["brand"] = classify_franchises(included["사업장명"], brands)

    # Review samples are deterministic but selected independently within each brand.
    franchise_rows = included.loc[
        included["is_franchise"], ["brand", "사업장명", "관리번호", "인허가일자", "도로명주소"]
    ]
    review_parts = [
        group.sample(n=min(len(group), 10), random_state=RANDOM_SEED)
        for _, group in franchise_rows.groupby("brand", sort=True)
    ]
    review = pd.concat(review_parts, ignore_index=True).sort_values(
        ["brand", "사업장명", "관리번호"], kind="stable"
    )
    review.to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")

    unmatched_top = (
        included.loc[~included["is_franchise"], "사업장명"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .rename_axis("사업장명")
        .reset_index(name="건수")
        .head(100)
    )
    unmatched_top.to_csv(UNMATCHED_PATH, index=False, encoding="utf-8-sig")

    parquet_engine_available = any(
        importlib.util.find_spec(package) is not None for package in ("pyarrow", "fastparquet")
    )
    parquet_engine_status = "사용 가능" if parquet_engine_available else "사용 불가"
    parquet_limitation = (
        ""
        if parquet_engine_available
        else "\n5. `pyarrow`와 `fastparquet`가 모두 설치되어 있지 않아 Parquet 파일을 만들 수 없었다. "
        "지시에 따라 패키지를 설치하거나 CSV 대체본을 만들지 않았다. 관리된 환경에 두 패키지 중 하나를 추가한 뒤 같은 명령을 다시 실행해야 한다."
    )

    cohort = included.loc[included["in_study_period"]].copy()
    cohort["permit_year"] = cohort["permit_date"].dt.year
    annual = (
        cohort.groupby(["permit_year", "is_franchise"], dropna=False)
        .agg(개업_수=("관리번호", "size"), 폐업_수=("event", "sum"))
        .reset_index()
    )
    annual["구분"] = np.where(annual["is_franchise"], "프랜차이즈", "개인 카페")
    annual = annual.rename(columns={"permit_year": "개업 연도"})[["개업 연도", "구분", "개업_수", "폐업_수"]]
    annual = annual.rename(columns={"개업_수": "개업 수", "폐업_수": "폐업 수"})

    brand_counts = (
        included.loc[included["is_franchise"], "brand"]
        .value_counts()
        .rename_axis("브랜드")
        .reset_index(name="매칭 수")
        .sort_values(["매칭 수", "브랜드"], ascending=[False, True], kind="stable")
    )
    coordinate_summary = pd.DataFrame(
        [
            {
                "EPSG": result.epsg,
                "서울 근사 경계 내": result.in_seoul_box,
                "중앙 경도": round(result.median_longitude, 6),
                "중앙 위도": round(result.median_latitude, 6),
            }
            for result in coordinate_results.values()
        ]
    )

    zero_or_negative_survival = int(included["survival_days"].le(0).sum())
    coordinate_missing = int((~included["has_coordinates"]).sum())
    report = f"""# 02 인허가 전처리 보고서

실행일: 2026-11-18  
입력: `data/raw/refreshment_establishment/식품_휴게음식점_서울특별시.csv` (CP949)  
입력 SHA256: `{observed_hash}` (지정값과 일치)  
관측 종료일: {OBSERVATION_END.date()}  
개업 기간: {STUDY_START.date()}~{STUDY_END.date()} (양 끝 포함)  
프랜차이즈 검수 표본 시드: {RANDOM_SEED}

## 처리 단계와 행 수

| 단계 | 행 수 | 제외 수 | 제외 사유 |
| --- | ---: | ---: | --- |
| 원본 CP949 읽기 | {format_int(raw_rows)} | 0 | - |
| 업태구분명 `커피숍` 선택 | {format_int(coffee_rows)} | {format_int(raw_rows - coffee_rows)} | 커피숍 외 업태 |
| 폐업 상태·폐업일 및 인허가일 검사 후 유지 | {format_int(len(included))} | {format_int(int(exclusions.sum()))} | 폐업 상태인데 폐업일 없음 {format_int(int(closure_status_missing_date.sum()))}; 인허가일이 관측 종료일 뒤 {format_int(int(permit_after_observation.sum()))}; 인허가일 파싱 실패 {format_int(int(permit_parse_failure.sum()))} |
| 개업 기간 표본 표시 (`in_study_period == True`) | {format_int(len(cohort))} | {format_int(len(included) - len(cohort))} | 산출물에는 전체 유지; 기간 밖은 분석 표본에서만 제외 |

폐업일 비결측인데 파싱에 실패한 행은 {format_int(int(closure_date_parse_failure.sum()))}건이다. 생존기간이 0일 이하인 행은 지시대로 제외하지 않았으며 {format_int(zero_or_negative_survival)}건이다.

## 개업 기간 표본의 사건과 표본 구성

| 지표 | 전체 | 개인 카페 | 프랜차이즈 |
| --- | ---: | ---: | ---: |
| 인허가(운영자) 수 | {format_int(len(cohort))} | {format_int(int((~cohort['is_franchise']).sum()))} | {format_int(int(cohort['is_franchise'].sum()))} |
| 사건(폐업일 ≤ 관측 종료일) | {format_int(int(cohort['event'].sum()))} | {format_int(int(cohort.loc[~cohort['is_franchise'], 'event'].sum()))} | {format_int(int(cohort.loc[cohort['is_franchise'], 'event'].sum()))} |
| 중도절단 | {format_int(int((1 - cohort['event']).sum()))} | {format_int(int((1 - cohort.loc[~cohort['is_franchise'], 'event']).sum()))} | {format_int(int((1 - cohort.loc[cohort['is_franchise'], 'event']).sum()))} |

분석 대상은 `is_franchise == False`인 개인 카페다. 프랜차이즈 행은 향후 경쟁 수 계산을 위해 `cafes_licenses.parquet`에 보존했다.

## 연도별 개업·폐업 (개업 기간 표본)

{markdown_table(annual)}

## 좌표 처리와 좌표계 판단

원문 X/Y의 앞뒤 공백을 제거한 후 숫자로 변환했다. 전체 유지 표본 중 X 또는 Y가 없거나 숫자가 아닌 행은 {format_int(coordinate_missing)}건이며, 이 행은 제외하지 않고 `has_coordinates=False`, 경위도 결측으로 표시했다. 서울 근사 경계는 경도 126.76~127.19, 위도 37.42~37.71이다.

{markdown_table(coordinate_summary)}

세 후보의 서울 근사 경계 내 행 수는 같아 경계 검사만으로는 판별할 수 없었다. 아래 알려진 기준점 주소의 중앙거리 평균이 가장 작은 **EPSG:{selected_epsg}**를 선택해 `longitude`·`latitude`를 저장했다. 이는 공식 좌표계 필드의 확인을 대체하지 않는다.

알려진 기준점 두 곳인 서울시청(세종대로 110)과 서울역(한강대로 405)을 사용해, 같은 도로명주소를 가진 커피숍 행의 변환 위치와 중앙거리를 추가 점검했다. 한 주소에 여러 입점 시설이 있을 수 있어 보조 근거로만 사용한다.

{markdown_table(reference_summary)}

**좌표계 확인 필요:** EPSG:{selected_epsg}가 세 기준점 검증에서 가장 가까웠으나, 데이터 제공처의 CRS 명세를 확보해 최종 확인해야 한다. 공간 결합 전에는 이 공식 확인을 마친다.

## 프랜차이즈 매칭

브랜드 사전 `src/franchise_brands.csv`에 {format_int(len(brands))}개 브랜드를 넣었다. 사업장명을 NFKC 정규화하고 공백·일부 문장부호를 제거한 뒤, 사전 순서대로 첫 매칭 브랜드를 부여했다. 짧거나 일반적인 이름의 오탐 위험은 사전 비고와 검수 파일로 표시했다.

{markdown_table(brand_counts)}

검수용 파일:

- `outputs/02_franchise_review.csv`: 브랜드별 무작위 최대 10건(시드 고정)
- `outputs/02_unmatched_top.csv`: 미매칭 상호 빈도 상위 100개

## 산출물

- `data/processed/cafes_licenses.parquet`: 전체 커피숍 유지 표본(프랜차이즈 포함). 현재 준비된 환경에서 Parquet 엔진은 {parquet_engine_status}.
- `outputs/02_preprocess_report.md`: 이 보고서
- `outputs/02_franchise_review.csv`, `outputs/02_unmatched_top.csv`: 프랜차이즈 판별 검수용

## 결정 필요·한계

1. 프랜차이즈 매칭은 현재 상호명 기반 사전 분류다. 동명 개인점·상호 변경·신규/누락 체인이 있을 수 있으므로 검수 CSV를 확인해 사전을 승인·보완해야 한다. 특히 `달콤`, `더리터`, `더카페`, `커피나무`, `카페봄봄` 등은 오탐 위험이 있다.
2. 원자료에는 운영자 변경(양수도)을 식별하는 전용 필드가 없다. 승인된 정의에 따라 인허가(운영자) 단위로 유지했다.
3. 관측 종료일 이후 폐업일은 중도절단으로 처리했다. 원자료의 최신성이 2026-09-30까지 완전하다는 별도 근거는 이 작업에서 검증하지 않았다.
4. 좌표계는 공식 명세로 확정되지 않았으므로, 위의 임시 EPSG:{selected_epsg} 경위도는 향후 공간 결합 전에 재검증해야 한다.{parquet_limitation}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")

    write_parquet(included, PROCESSED_PATH)

    print(f"Wrote {PROCESSED_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {REVIEW_PATH}")
    print(f"Wrote {UNMATCHED_PATH}")
    print(f"Study cohort: {len(cohort)}; events: {int(cohort['event'].sum())}; censored: {int((1 - cohort['event']).sum())}")


if __name__ == "__main__":
    main()
