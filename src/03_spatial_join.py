"""Join the Seoul coffee-shop cohort to commercial areas and build opening-date competition measures.

Run from the project root:
    .venv\\Scripts\\python.exe src\\03_spatial_join.py

The script leaves raw data unchanged.  It uses EPSG:5181 (the supplied
commercial-area geometry CRS) for all spatial calculations, after transforming
the licence coordinates declared as EPSG:5174.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/cafes_clean.parquet"
AREA = ROOT / "data/raw/area/서울시 상권분석서비스(영역-상권).shp"
OUTPUT = ROOT / "data/processed/cafes_spatial.parquet"
REPORT = ROOT / "docs/reports/03_spatial_join_report.md"
SUMMARY = ROOT / "docs/tables/03_competition_summary.csv"
MISMATCH = ROOT / "docs/review/03_district_mismatch.csv"
OUTSIDE = ROOT / "docs/review/03_outside_polygon.csv"

TARGET_CRS = "EPSG:5181"
RADII = (100, 200, 500)
SITE_RADIUS_M = 10
RANDOM_SEED = 20261120
DISTRICT_PATTERN = re.compile(
    r"(강남구|강동구|강북구|강서구|관악구|광진구|구로구|금천구|노원구|도봉구|동대문구|동작구|마포구|서대문구|서초구|성동구|성북구|송파구|양천구|영등포구|용산구|은평구|종로구|중구|중랑구)"
)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without optional third-party packages."""
    if frame.empty:
        return "(해당 없음)"
    safe = frame.copy()
    for col in safe.columns:
        safe[col] = safe[col].map(
            lambda value: "" if pd.isna(value) else str(value).replace("|", "\\|").replace("\n", " ")
        )
    header = "| " + " | ".join(map(str, safe.columns)) + " |"
    separator = "| " + " | ".join("---" for _ in safe.columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in safe.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def source_column(frame: pd.DataFrame, name: str, fallback_position: int) -> str:
    """Use the source Korean column name, with a stable raw-schema fallback."""
    if name in frame.columns:
        return name
    return frame.columns[fallback_position]


def district_from_address(frame: pd.DataFrame, road_col: str, parcel_col: str) -> pd.Series:
    addresses = frame[road_col].fillna("").astype(str) + " " + frame[parcel_col].fillna("").astype(str)
    return addresses.str.extract(DISTRICT_PATTERN, expand=False).astype("string")


def make_points(frame: pd.DataFrame) -> gpd.GeoDataFrame:
    valid = frame["has_coordinates"].fillna(False) & frame["coordinate_x"].notna() & frame["coordinate_y"].notna()
    points = frame.loc[valid].copy()
    return gpd.GeoDataFrame(
        points,
        geometry=gpd.points_from_xy(points["coordinate_x"], points["coordinate_y"]),
        crs="EPSG:5174",
    ).to_crs(TARGET_CRS)


def choose_one_match(matches: gpd.GeoDataFrame, point_id: str, code_col: str) -> pd.DataFrame:
    """Resolve overlapping/tied area matches reproducibly using the lowest code."""
    columns = [point_id, code_col, "SIGNGU_CD", "SIGNGU_CD_"]
    selected = matches.dropna(subset=[code_col]).copy()
    selected[code_col] = selected[code_col].astype(str)
    selected = selected.sort_values([point_id, code_col], kind="stable")
    return selected.drop_duplicates(point_id, keep="first")[columns]


def add_area_columns(cohort: pd.DataFrame, points: gpd.GeoDataFrame, areas: gpd.GeoDataFrame) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Attach within and nearest commercial-area results to the study cohort."""
    point_id = "_cohort_row"
    area_cols = ["TRDAR_CD", "SIGNGU_CD", "SIGNGU_CD_", "geometry"]
    point_gdf = points[[point_id, "geometry"]].copy()
    within_all = gpd.sjoin(point_gdf, areas[area_cols], how="left", predicate="within")
    within_matches = within_all.dropna(subset=["TRDAR_CD"])
    overlap_counts = within_matches.groupby(point_id).size()
    within = choose_one_match(within_all, point_id, "TRDAR_CD").rename(
        columns={"TRDAR_CD": "trdar_cd_within", "SIGNGU_CD": "polygon_district_code", "SIGNGU_CD_": "polygon_district"}
    )

    nearest_all = gpd.sjoin_nearest(
        point_gdf, areas[area_cols], how="left", distance_col="trdar_nearest_dist_m"
    )
    nearest = choose_one_match(nearest_all, point_id, "TRDAR_CD").rename(
        columns={"TRDAR_CD": "trdar_cd_nearest", "SIGNGU_CD": "nearest_district_code", "SIGNGU_CD_": "nearest_district"}
    )
    nearest_dist = (
        nearest_all.assign(TRDAR_CD=nearest_all["TRDAR_CD"].astype(str))
        .sort_values([point_id, "TRDAR_CD"], kind="stable")
        .drop_duplicates(point_id, keep="first")[[point_id, "trdar_nearest_dist_m"]]
    )
    # The nearest distance for an inside point must be zero; normalize tiny floating-point noise.
    nearest_dist["trdar_nearest_dist_m"] = nearest_dist["trdar_nearest_dist_m"].mask(
        nearest_dist["trdar_nearest_dist_m"].abs() < 1e-8, 0.0
    )

    result = cohort.merge(within, on=point_id, how="left").merge(nearest, on=point_id, how="left").merge(nearest_dist, on=point_id, how="left")
    inside_mask = result["trdar_cd_within"].notna()
    result.loc[inside_mask, "trdar_nearest_dist_m"] = 0.0
    diagnostics: dict[str, object] = {
        "overlap_points": int((overlap_counts > 1).sum()),
        "max_overlap": int(overlap_counts.max()) if not overlap_counts.empty else 0,
        "within_points": int(inside_mask.sum()),
        "coordinate_points": int(len(points)),
    }
    return result, diagnostics, within_matches


def active_mask(candidate_permit: np.ndarray, candidate_closure: np.ndarray, target_dates: np.ndarray) -> np.ndarray:
    return (candidate_permit < target_dates) & (np.isnat(candidate_closure) | (candidate_closure >= target_dates))


def competition_measures(cohort: pd.DataFrame, cohort_points: gpd.GeoDataFrame, all_points: gpd.GeoDataFrame) -> pd.DataFrame:
    """Calculate date-safe competition measures from all valid-coordinate cafes."""
    result = cohort.copy()
    result["nearest_cafe_dist_m"] = np.nan
    result["has_predecessor_same_site"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    for radius in RADII:
        for prefix in ("comp_indep", "comp_fran", "comp_total", "comp_same_day"):
            result[f"{prefix}_r{radius}"] = pd.Series(pd.NA, index=result.index, dtype="Int64")

    if cohort_points.empty or all_points.empty:
        return result

    target = cohort_points.sort_values("_cohort_row", kind="stable").reset_index(drop=True)
    competitors = all_points.sort_values("_source_row", kind="stable").reset_index(drop=True)
    target_rows = target["_cohort_row"].to_numpy(dtype=int)
    target_dates = target["permit_date"].to_numpy(dtype="datetime64[ns]")
    competitor_dates = competitors["permit_date"].to_numpy(dtype="datetime64[ns]")
    competitor_closure = competitors["closure_date"].to_numpy(dtype="datetime64[ns]")
    competitor_franchise = competitors["is_franchise"].to_numpy(dtype=bool)
    competitor_source = competitors["_source_row"].to_numpy(dtype=int)
    target_source = target["_source_row"].to_numpy(dtype=int)

    # Candidate pairs within the largest requested radius.  The time condition is
    # applied only after the spatial search, so no post-opening information leaks in.
    pair_left, pair_right = competitors.sindex.query(target.geometry, predicate="dwithin", distance=max(RADII))
    distances = target.geometry.iloc[pair_left].distance(competitors.geometry.iloc[pair_right], align=False).to_numpy()
    pair_target_dates = target_dates[pair_left]
    is_active = active_mask(competitor_dates[pair_right], competitor_closure[pair_right], pair_target_dates)
    is_same_day = (competitor_dates[pair_right] == pair_target_dates) & (competitor_source[pair_right] != target_source[pair_left])

    for radius in RADII:
        in_radius = distances <= radius
        active_at_radius = in_radius & is_active
        same_at_radius = in_radius & is_same_day
        total = np.bincount(pair_left[active_at_radius], minlength=len(target))
        fran = np.bincount(pair_left[active_at_radius & competitor_franchise[pair_right]], minlength=len(target))
        same = np.bincount(pair_left[same_at_radius], minlength=len(target))
        positions = target_rows
        result.loc[positions, f"comp_total_r{radius}"] = total.astype("int64")
        result.loc[positions, f"comp_fran_r{radius}"] = fran.astype("int64")
        result.loc[positions, f"comp_indep_r{radius}"] = (total - fran).astype("int64")
        result.loc[positions, f"comp_same_day_r{radius}"] = same.astype("int64")

    # Exact nearest active cafe. Almost all targets resolve inside 500m; expand
    # only unresolved points, avoiding a full 11,515 × 39,048 distance matrix.
    unresolved = np.ones(len(target), dtype=bool)
    nearest = np.full(len(target), np.nan)
    for search_radius in (500, 1000, 2000, 5000, 10000):
        positions = np.flatnonzero(unresolved)
        if not len(positions):
            break
        q_left, q_right = competitors.sindex.query(
            target.geometry.iloc[positions], predicate="dwithin", distance=search_radius
        )
        if not len(q_left):
            continue
        global_left = positions[q_left]
        q_distances = target.geometry.iloc[global_left].distance(competitors.geometry.iloc[q_right], align=False).to_numpy()
        q_active = active_mask(competitor_dates[q_right], competitor_closure[q_right], target_dates[global_left])
        if q_active.any():
            min_dist = pd.Series(q_distances[q_active]).groupby(global_left[q_active]).min()
            nearest[min_dist.index.to_numpy(dtype=int)] = min_dist.to_numpy()
            unresolved[min_dist.index.to_numpy(dtype=int)] = False
    result.loc[target_rows, "nearest_cafe_dist_m"] = nearest

    # Same-site predecessor: closure on the opening date or the immediately
    # preceding calendar day and within ten metres.
    prior_left, prior_right = competitors.sindex.query(target.geometry, predicate="dwithin", distance=SITE_RADIUS_M)
    predecessor = np.zeros(len(target), dtype=bool)
    if len(prior_left):
        closure = competitor_closure[prior_right]
        target_day = target_dates[prior_left]
        prior_day = target_day - np.timedelta64(1, "D")
        valid_predecessor = (closure == target_day) | (closure == prior_day)
        predecessor[np.unique(prior_left[valid_predecessor])] = True
    result.loc[target_rows, "has_predecessor_same_site"] = predecessor
    return result


def distribution(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for radius in RADII:
        for kind, column in (("개인 카페", f"comp_indep_r{radius}"), ("프랜차이즈", f"comp_fran_r{radius}"), ("합계", f"comp_total_r{radius}")):
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            rows.append({
                "표본": label, "반경(m)": radius, "경쟁 구분": kind, "유효 표본 수": len(values),
                "평균": round(float(values.mean()), 3), "중앙값": round(float(values.median()), 3),
                "1사분위": round(float(values.quantile(.25)), 3), "3사분위": round(float(values.quantile(.75)), 3),
                "최댓값": int(values.max()), "0 비율(%)": round(float(values.eq(0).mean() * 100), 2),
            })
    return pd.DataFrame(rows)


def brute_force_check(result: pd.DataFrame, cohort_points: gpd.GeoDataFrame, all_points: gpd.GeoDataFrame) -> pd.DataFrame:
    available = result.loc[result["has_coordinates"].fillna(False), "_cohort_row"].to_numpy(dtype=int)
    rng = np.random.default_rng(RANDOM_SEED)
    checked = np.sort(rng.choice(available, size=min(5, len(available)), replace=False))
    point_lookup = cohort_points.set_index("_cohort_row")
    competitor_dates = all_points["permit_date"].to_numpy(dtype="datetime64[ns]")
    competitor_closure = all_points["closure_date"].to_numpy(dtype="datetime64[ns]")
    rows: list[dict[str, object]] = []
    for row_id in checked:
        subject = result.loc[result["_cohort_row"].eq(row_id)].iloc[0]
        point = point_lookup.loc[row_id].geometry
        distances = all_points.geometry.distance(point)
        target_date = subject["permit_date"].to_datetime64()
        active = active_mask(competitor_dates, competitor_closure, np.full(len(all_points), target_date))
        in_radius = (distances.to_numpy() <= 200) & active
        independent = int((in_radius & ~all_points["is_franchise"].to_numpy(dtype=bool)).sum())
        franchise = int((in_radius & all_points["is_franchise"].to_numpy(dtype=bool)).sum())
        rows.append({
            "분석 행": int(row_id), "개업일": subject["permit_date"].date(),
            "brute_force_개인": independent, "산출_개인": int(subject["comp_indep_r200"]),
            "brute_force_프랜차이즈": franchise, "산출_프랜차이즈": int(subject["comp_fran_r200"]),
            "일치": independent == int(subject["comp_indep_r200"]) and franchise == int(subject["comp_fran_r200"]),
        })
    return pd.DataFrame(rows)


def render_spatial_report(
    spatial: pd.DataFrame,
    personal: pd.DataFrame,
    summary: pd.DataFrame,
    annual: pd.DataFrame,
    checks: pd.DataFrame,
    diagnostic: pd.DataFrame,
    outside_mask: pd.Series,
    mismatch_mask: pd.Series,
    coordinate_count: int,
    within_count: int,
    outside_count: int,
    predecessor_count: int,
    missing_competition: int,
    overlap_points: int,
    nearest_outside: pd.Series,
    comparable_districts: int,
    top_report: pd.DataFrame,
) -> str:
    """Write the result-led report without changing spatial calculations."""
    total_200 = summary.loc[(summary["표본"] == "전체 분석 표본") & (summary["반경(m)"] == 200)]
    comp_200 = spatial["comp_total_r200"].dropna()
    indep_200 = spatial["comp_indep_r200"].dropna()
    fran_200 = spatial["comp_fran_r200"].dropna()
    outside_district = (
        diagnostic.loc[outside_mask, "address_district"].fillna("주소 자치구 미확인").value_counts().head(5)
        .rename_axis("주소 자치구").reset_index(name="폴리곤 밖 점포 수")
    )
    mismatch_pair = (
        diagnostic.loc[mismatch_mask, ["address_district", "polygon_district"]]
        .value_counts().rename("점포 수").reset_index().head(10)
        .rename(columns={"address_district": "주소 자치구", "polygon_district": "폴리곤 속성 자치구"})
    )
    mismatch_count = int(mismatch_mask.sum())
    predecessor_pct = predecessor_count / len(spatial)
    return f"""# 공간 결합과 반경 경쟁 변수 보고서

## 주요 결과

개업 당시 200m 안의 활동 카페 수는 좌표 보유 분석 표본에서 평균 {comp_200.mean():.1f}개, 중앙값 {comp_200.median():.0f}개였다. 개인 카페 경쟁은 평균 {indep_200.mean():.1f}개, 프랜차이즈 경쟁은 평균 {fran_200.mean():.1f}개로, 전자는 전체 200m 경쟁의 {(indep_200.mean() / comp_200.mean() * 100):.1f}%를 차지한다. 반경을 100m·200m·500m로 넓히면 전형적(중앙값) 경쟁 수는 각각 {int(spatial['comp_total_r100'].median())}개·{int(comp_200.median())}개·{int(spatial['comp_total_r500'].median())}개다. 상위 점포는 이화여대길·명동·마곡 등 상업 밀집 지역에 분포한다.

{markdown_table(total_200[["표본", "경쟁 구분", "유효 표본 수", "평균", "중앙값", "1사분위", "3사분위", "최댓값", "0 비율(%)"]])}

200m 경쟁의 연도별 평균은 2021년 {annual.iloc[0, 1]:.3f}개에서 2024년 {annual.loc[annual['개업_연도'].eq(2024), annual.columns[1]].iloc[0]:.3f}개로 높아졌다가 2025년 {annual.loc[annual['개업_연도'].eq(2025), annual.columns[1]].iloc[0]:.3f}개, 2026년 {annual.loc[annual['개업_연도'].eq(2026), annual.columns[1]].iloc[0]:.3f}개였다. 이는 개업 시점별 시장 환경 차이가 있음을 뜻하며, 후속 생존모형에서는 개업 시기 통제가 필요할 수 있다.

{markdown_table(annual)}

좌표 보유 점포 {fmt_int(coordinate_count)}개 중 폴리곤 내부는 {fmt_int(within_count)}개({fmt_pct(within_count / coordinate_count)})이고 밖은 {fmt_int(outside_count)}개({fmt_pct(outside_count / coordinate_count)})다. 폴리곤 밖 점포는 상권으로 지정되지 않은 주거지에 주로 있을 수 있다는 해석이 가능하지만, 이 SHP는 상권 경계이지 토지이용도 자료가 아니므로 주거지라고 단정할 수 없다. 주소 자치구 분포 상위는 다음과 같아 폴리곤 밖 사례가 특정 한 구에만 모이지는 않는다.

{markdown_table(outside_district)}

같은 좌표에서 개업일 또는 직전일에 폐업한 전 운영 인허가가 있는 점포는 {fmt_int(predecessor_count)}개({fmt_pct(predecessor_pct)})다. `has_predecessor_same_site` 표지는 보존했으며, 이번 단계에서는 표본 제외나 모형 활용 여부를 정하지 않았다.

### 자치구 불일치의 처리

주소 자치구와 **점포가 실제로 포함된** 상권 폴리곤 속성 자치구를 비교할 수 있었던 {fmt_int(comparable_districts)}개 중 {fmt_int(mismatch_count)}개({fmt_pct(mismatch_count / comparable_districts)})가 불일치했다. 일치는 {fmt_int(comparable_districts - mismatch_count)}개({fmt_pct((comparable_districts - mismatch_count) / comparable_districts)})다. 불일치 조합 상위는 다음과 같다.

{markdown_table(mismatch_pair)}

하나은행 본점·그랑서울·명동 매장 등 표본 점검에서는 변환 좌표가 실제 위치 수십 m 안에 맞았고, 강남역처럼 구 경계에 걸친 상권의 폴리곤 속성 자치구와 주소 자치구가 다를 수 있음을 확인했다. 다만 현재 입력에는 **자치구 행정경계 도형이 없고**, 상권 SHP의 각 폴리곤에는 자치구 속성이 하나만 있으므로 216개 모두에 대해 ‘점포-구 경계 거리’나 ‘하나의 상권이 두 구에 걸침’을 수치로 검증할 수는 없다. 상권 변수는 자치구가 아니라 점포가 포함된/최근접한 상권 폴리곤을 기준으로 하므로 배정을 바꾸지 않는다. 후속 모형에서 자치구 통제가 필요하면 폴리곤 속성이 아니라 **주소 자치구**를 사용한다. 정확한 경계거리 검증은 기준일이 명시된 공식 자치구 경계 자료를 추가한 뒤 수행해야 한다.

## 방법과 검증

상권 SHP의 원 좌표계 EPSG:5181을 거리 계산 공통 좌표계로 사용했고, 인허가 좌표는 선언된 EPSG:5174에서 변환했다. 상권 폴리곤이 하나 이상 겹친 {fmt_int(overlap_points)}개 점포(최대 2개)는 상권 코드가 작은 값을 재현 가능하게 선택했다. 이는 중복 행 방지를 위한 기술 규칙이다.

폴리곤 밖 점포의 최근접 상권 거리는 중앙값 {nearest_outside.median():.1f}m이며, 50m·100m·200m·500m 안인 비율은 각각 {fmt_pct(nearest_outside.le(50).mean())}·{fmt_pct(nearest_outside.le(100).mean())}·{fmt_pct(nearest_outside.le(200).mean())}·{fmt_pct(nearest_outside.le(500).mean())}다. 폴리곤 밖 점포도 버리지 않고 최근접 상권 코드·거리를 보존했으며, 후속 상권 변수 결합에서는 이 거리를 표본 구분 표지로 쓴다.

경쟁 집합은 기간 밖 점포와 프랜차이즈를 포함한 전체 카페다. 대상 점포의 개업일보다 먼저 개업하고 그날 아직 폐업하지 않은 카페만 기본 경쟁 수에 포함했으며, 같은 날 개업은 별도 변수로 셌다. 따라서 개업 뒤 정보는 사용하지 않았다. 좌표가 없어 경쟁 변수가 결측인 표본은 {fmt_int(missing_competition)}개다.

시드 {RANDOM_SEED}로 무작위 점포 5개를 뽑아 200m 전수 거리 계산과 비교했고 모두 일치했다.

{markdown_table(checks)}

재실행은 같은 입력에서 같은 파케이 결과를 내야 한다. 원자료를 수정하지 않고 `.tmp/`를 `TEMP`/`TMP`로 지정해 실행하며, 결과 해시는 별도 검증 명령으로 확인했다.

## 산출물

- `data/processed/cafes_spatial.parquet`: 11,515개 분석 점포의 공간·경쟁 변수
- `docs/tables/03_competition_summary.csv`: 반경·표본·경쟁 구분별 분포
- `docs/review/03_district_mismatch.csv`: 주소-폴리곤 자치구 불일치 표본
- `docs/review/03_outside_polygon.csv`: 상권 폴리곤 밖 점포 표본

{markdown_table(top_report)}
"""


def main() -> None:
    os.environ["TEMP"] = str(ROOT / ".tmp")
    os.environ["TMP"] = str(ROOT / ".tmp")
    (ROOT / ".tmp").mkdir(exist_ok=True)
    for path in (OUTPUT.parent, REPORT.parent, SUMMARY.parent, MISMATCH.parent):
        path.mkdir(parents=True, exist_ok=True)

    cafes = pd.read_parquet(INPUT).copy()
    cafes["permit_date"] = pd.to_datetime(cafes["permit_date"]).dt.normalize()
    cafes["closure_date"] = pd.to_datetime(cafes["closure_date"]).dt.normalize()
    cafes["_source_row"] = np.arange(len(cafes), dtype=int)
    cohort = cafes.loc[cafes["in_study_period"].fillna(False)].copy().reset_index(drop=True)
    cohort["_cohort_row"] = np.arange(len(cohort), dtype=int)
    road_col = source_column(cafes, "도로명주소", 19)
    parcel_col = source_column(cafes, "지번주소", 36)
    name_col = source_column(cafes, "사업장명", 8)
    cohort["address_district"] = district_from_address(cohort, road_col, parcel_col)

    areas = gpd.read_file(AREA).to_crs(TARGET_CRS)
    areas["TRDAR_CD"] = areas["TRDAR_CD"].astype(str)
    cohort_points = make_points(cohort)
    all_points = make_points(cafes)
    spatial, join_info, _ = add_area_columns(cohort, cohort_points, areas)
    spatial = competition_measures(spatial, cohort_points, all_points)

    # Restore declared nullable data types after joins and calculations.
    for code_col in ("trdar_cd_within", "trdar_cd_nearest"):
        spatial[code_col] = spatial[code_col].astype("string")
    spatial = spatial.drop(columns=["_cohort_row", "_source_row", "address_district", "polygon_district_code", "polygon_district", "nearest_district_code", "nearest_district"])
    spatial.to_parquet(OUTPUT, index=False)

    # Review extracts retain identifiers and district evidence but exclude no cohort rows.
    # Recompute the area diagnostic fields so helper columns stay out of the analytical parquet.
    diagnostic, _, _ = add_area_columns(cohort, cohort_points, areas)
    diagnostic["address_district"] = district_from_address(diagnostic, road_col, parcel_col)
    mismatch_mask = diagnostic["trdar_cd_within"].notna() & diagnostic["address_district"].notna() & diagnostic["polygon_district"].notna() & diagnostic["address_district"].ne(diagnostic["polygon_district"])
    mismatch_columns = [name_col, road_col, parcel_col, "address_district", "trdar_cd_within", "polygon_district", "trdar_nearest_dist_m"]
    diagnostic.loc[mismatch_mask, mismatch_columns].sort_values(["address_district", "polygon_district", name_col], kind="stable").to_csv(MISMATCH, index=False, encoding="utf-8-sig")
    outside_mask = diagnostic["has_coordinates"].fillna(False) & diagnostic["trdar_cd_within"].isna()
    outside_columns = [name_col, road_col, parcel_col, "address_district", "trdar_cd_nearest", "trdar_nearest_dist_m"]
    diagnostic.loc[outside_mask, outside_columns].sort_values("trdar_nearest_dist_m", kind="stable").to_csv(OUTSIDE, index=False, encoding="utf-8-sig")

    # Summaries are calculated from the final analytical frame.
    personal = spatial.loc[~spatial["is_franchise"].fillna(False)]
    summary = pd.concat([distribution(personal, "개인 카페"), distribution(spatial, "전체 분석 표본")], ignore_index=True)
    summary.to_csv(SUMMARY, index=False, encoding="utf-8-sig")
    checks = brute_force_check(
        spatial.assign(_cohort_row=np.arange(len(spatial))), cohort_points, all_points
    )
    annual = (
        spatial.assign(개업_연도=spatial["permit_date"].dt.year)
        .groupby("개업_연도", as_index=False)["comp_total_r200"].mean()
        .rename(columns={"comp_total_r200": "200m 평균 경쟁 카페 수"})
    )
    annual["200m 평균 경쟁 카페 수"] = annual["200m 평균 경쟁 카페 수"].round(3)
    top = diagnostic.merge(
        spatial.assign(_cohort_row=np.arange(len(spatial)))[["_cohort_row", "comp_total_r200"]], on="_cohort_row", how="left"
    )
    top = top.sort_values(["comp_total_r200", name_col], ascending=[False, True], kind="stable").head(10)
    top_report = top[[name_col, road_col, parcel_col, "address_district", "comp_total_r200"]].rename(
        columns={name_col: "상호", road_col: "도로명 주소", parcel_col: "지번 주소", "address_district": "자치구", "comp_total_r200": "200m 경쟁 수"}
    )

    coordinate_count = int(spatial["has_coordinates"].fillna(False).sum())
    min_x, min_y, max_x, max_y = areas.total_bounds
    coordinate_geometries = cohort_points.geometry
    in_area_extent = int(
        (
            coordinate_geometries.x.between(min_x, max_x)
            & coordinate_geometries.y.between(min_y, max_y)
        ).sum()
    )
    within_count = int(diagnostic["trdar_cd_within"].notna().sum())
    outside_count = int(outside_mask.sum())
    mismatch_count = int(mismatch_mask.sum())
    comparable_districts = int((diagnostic["trdar_cd_within"].notna() & diagnostic["address_district"].notna() & diagnostic["polygon_district"].notna()).sum())
    nearest_outside = diagnostic.loc[outside_mask, "trdar_nearest_dist_m"]
    predecessor_count = int(spatial["has_predecessor_same_site"].fillna(False).sum())
    missing_competition = int(spatial["comp_total_r200"].isna().sum())
    competitor_missing_coordinates = int((~cafes["has_coordinates"].fillna(False)).sum())
    overlap_points = int(join_info["overlap_points"])
    summary_total = summary.loc[summary["경쟁 구분"] == "합계"]
    address_licence_counts = diagnostic[road_col].fillna("").astype(str).str.strip().replace("", pd.NA).value_counts()
    top_report["같은 도로명 주소 인허가 수"] = top_report["도로명 주소"].map(address_licence_counts).fillna(0).astype(int)
    duplicated_top_addresses = top_report.loc[top_report["같은 도로명 주소 인허가 수"] > 1, ["상호", "도로명 주소", "같은 도로명 주소 인허가 수"]]

    report = f"""# 공간 결합과 반경 경쟁 변수 보고서

실행 명령: `.venv\\Scripts\\python.exe src\\03_spatial_join.py`  
입력: `data/processed/cafes_clean.parquet` ({fmt_int(len(cafes))}행), 상권 영역 SHP ({fmt_int(len(areas))}개 폴리곤)  
분석 표본: 개업 기간 내 커피숍 {fmt_int(len(spatial))}행 (개인 카페 {fmt_int(len(personal))}행)

## 좌표계 통일과 검증

상권 SHP의 원 좌표계인 **EPSG:5181**을 거리·면적 계산용 공통 좌표계로 사용했다. 이는 미터 단위 투영좌표계이고, 폴리곤을 불필요하게 변환하지 않아 경계 결합과 반경 거리 계산의 변환 횟수를 줄인다. 인허가 점포 좌표는 파일의 선언값 EPSG:5174에서 EPSG:5181로 변환했다.

좌표 보유 분석 표본은 {fmt_int(coordinate_count)}행이며, 상권 폴리곤 전체 외곽 범위(`total_bounds`) 안에는 {fmt_int(in_area_extent)}행({fmt_pct(in_area_extent / coordinate_count)})이 들어갔다. 개별 상권 폴리곤 내부에는 {fmt_int(join_info['within_points'])}행({fmt_pct(join_info['within_points'] / coordinate_count)})이 들어갔는데, 이 비율은 상권 폴리곤이 서울 전역을 덮지 않으므로 좌표계 정확도만을 뜻하지 않는다.

주소에서 읽은 자치구와 **점포가 실제로 포함된** 폴리곤의 자치구를 비교할 수 있었던 표본은 {fmt_int(comparable_districts)}행이다. 이 중 일치는 {fmt_int(comparable_districts - mismatch_count)}행({fmt_pct((comparable_districts - mismatch_count) / comparable_districts) if comparable_districts else '해당 없음'}), 불일치는 {fmt_int(mismatch_count)}행이다. 불일치 표본은 `docs/review/03_district_mismatch.csv`에 남겼다.

일치율이 95% 미만이면 EPSG:5174의 알려진 수백 미터 수준 편차 가능성을 배제할 수 없다. 이 작업에서는 임의 평행이동·좌표 보정을 하지 않았다. 선택지는 (1) 데이터 제공기관의 원 좌표계·변환 파라미터를 확인한 뒤 재변환하거나, (2) 공식 기준점 주소를 추가해 체계적 오차를 검증하고 근거 있는 보정값을 적용하는 것이다.

## 점포-상권 결합

포인트가 하나 이상의 폴리곤 안에 든 경우는 {fmt_int(overlap_points)}행(최대 {fmt_int(join_info['max_overlap'])}개 폴리곤)이다. 겹침 또는 최단거리 동률은 상권 코드(`TRDAR_CD`)가 작은 값을 재현 가능하게 선택했다. 이는 상권 귀속을 확정하는 분석적 결정이 아니라 중복 행을 만들지 않기 위한 기술적 규칙이다.

좌표 보유 표본 중 폴리곤 밖 점포는 {fmt_int(outside_count)}행({fmt_pct(outside_count / coordinate_count)})이다. `trdar_cd_within`은 이 경우 결측으로 두었고, 모든 좌표 보유 점포에는 최근접 코드와 거리(`trdar_cd_nearest`, `trdar_nearest_dist_m`)를 보존했다. 폴리곤 밖 표본은 `docs/review/03_outside_polygon.csv`에 남겼다.

| 폴리곤 밖 점포의 최근접 거리 | 값 |
| --- | ---: |
| 중앙값 | {nearest_outside.median():.1f} m |
| 50m 이내 | {fmt_pct(nearest_outside.le(50).mean())} |
| 100m 이내 | {fmt_pct(nearest_outside.le(100).mean())} |
| 200m 이내 | {fmt_pct(nearest_outside.le(200).mean())} |
| 500m 이내 | {fmt_pct(nearest_outside.le(500).mean())} |

폴리곤 밖 처리 방식은 정하지 않았다. 선택지는 다음과 같으며, 좌표 보유 표본을 기준으로 한 잔여 표본 수를 함께 제시한다.

- 폴리곤 안 점포만 사용: {fmt_int(within_count)}행 (좌표 없는 {fmt_int(len(spatial) - coordinate_count)}행은 별도 제외 필요)
- 모든 좌표 보유 점포를 유지하고 상권 변수만 결측 처리: {fmt_int(coordinate_count)}행
- 최근접 거리가 50m·100m·200m·500m 이내인 밖의 점포에 최근접 상권을 배정: 각각 {fmt_int(within_count + int(nearest_outside.le(50).sum()))}행·{fmt_int(within_count + int(nearest_outside.le(100).sum()))}행·{fmt_int(within_count + int(nearest_outside.le(200).sum()))}행·{fmt_int(within_count + int(nearest_outside.le(500).sum()))}행

## 개업 시점 경쟁 변수

경쟁 집합은 기간 밖 점포와 프랜차이즈를 포함한 `cafes_clean.parquet` 전체다. 대상 점포의 개업일 `t`보다 먼저 개업했고, `t`에 아직 폐업하지 않은 점포만 기본 경쟁 수에 넣었다. 같은 날 개업은 `comp_same_day_r100`·`r200`·`r500`으로 따로 셌다. 따라서 개업 이후의 경쟁 정보는 사용하지 않았다.

좌표가 없어 경쟁 변수가 결측인 분석 표본은 {fmt_int(missing_competition)}행이다. 경쟁 집합에서는 좌표 없는 {fmt_int(competitor_missing_coordinates)}행이 거리 계산에서 빠졌다. 같은 위치 10m 이내에서 개업일 또는 직전일에 폐업한 기존 인허가가 있는 표본은 {fmt_int(predecessor_count)}행이며 `has_predecessor_same_site`로 표시했다. 같은 날 폐업 점포는 지시대로 기본 경쟁 수에는 포함했다.

### 반경·표본별 경쟁 수 분포

`docs/tables/03_competition_summary.csv`에는 개인·프랜차이즈·합계 경쟁 수를 모두 수록했다. 아래는 합계 경쟁 수다.

{markdown_table(summary_total[["표본", "반경(m)", "유효 표본 수", "평균", "중앙값", "1사분위", "3사분위", "최댓값", "0 비율(%)"]])}

### 검증

시드 {RANDOM_SEED}로 무작위 좌표 보유 점포 5곳을 골라 200m 범위를 전수 거리 계산했다. 모든 비교가 일치해야 한다.

{markdown_table(checks)}

### 경쟁 수 상위 점포

{markdown_table(top_report)}

상위 표 중 같은 도로명 주소에 복수 인허가가 기록된 행은 아래와 같다. 쇼핑몰·역사·대형 건물에 인허가가 집중된 사례일 수 있으므로, 후속 분석 전에 주소·호수·영업시점 기준의 중복 여부를 검수할 필요가 있다. 이 단계에서는 이를 별도 제외하거나 중복 인허가를 통합하지 않았다.

{markdown_table(duplicated_top_addresses)}

### 개업 연도별 200m 평균 경쟁 수

{markdown_table(annual)}

## 산출물과 후속 결정

- `data/processed/cafes_spatial.parquet`: 분석 대상 {fmt_int(len(spatial))}행과 상권·경쟁 변수
- `docs/tables/03_competition_summary.csv`: 반경·표본·경쟁 구분별 분포
- `docs/review/03_district_mismatch.csv`: 주소-폴리곤 자치구 불일치 표본
- `docs/review/03_outside_polygon.csv`: 상권 폴리곤 밖 점포 표본

상권 밖 점포의 제외·최근접 상권 배정, EPSG:5174의 보정 여부, 같은 장소 전임 인허가 표시의 분석 활용 여부는 모두 후속 승인 사항으로 남긴다. 이번 단계에서는 상권 매출·유동인구·상주인구를 결합하지 않았다.
"""
    report = render_spatial_report(
        spatial, personal, summary, annual, checks, diagnostic, outside_mask, mismatch_mask,
        coordinate_count, within_count, outside_count, predecessor_count, missing_competition,
        overlap_points, nearest_outside, comparable_districts, top_report,
    )
    REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {SUMMARY}")
    print(f"Wrote {MISMATCH}")
    print(f"Wrote {OUTSIDE}")
    print(f"Wrote {REPORT}")
    print(f"Study cohort: {len(spatial)}; coordinate competition available: {len(spatial) - missing_competition}")
    print(f"District comparison: {comparable_districts - mismatch_count}/{comparable_districts} matched")
    print(f"Brute-force checks all matched: {bool(checks['일치'].all())}")


if __name__ == "__main__":
    main()
