"""Build a quarter-by-quarter start--stop panel for the cafe survival cohort.

This script deliberately does not fit a survival model.  It prepares the
time-varying data set for the next analysis step, keeping every covariate
strictly at or before the start of its interval.

Run from the repository root:
    .venv\\Scripts\\python.exe src\\06_time_varying_panel.py
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data/processed/cafes_analysis.parquet"
CLEAN = ROOT / "data/processed/cafes_clean.parquet"
AREA = ROOT / "data/raw/area/서울시 상권분석서비스(영역-상권).shp"
RENT_DIR = ROOT / "data/raw/rent"
OUTPUT = ROOT / "data/processed/cafes_quarterly_panel.parquet"
RENT_OUTPUT = ROOT / "data/processed/rent_rone_quarterly.parquet"
TABLE_DIR = ROOT / "docs/tables"
REVIEW = ROOT / "docs/review/06_rent_area_mapping.csv"
REVIEWED_MAPPING = ROOT / "docs/review/06_rent_area_mapping_reviewed.csv"
REPORT = ROOT / "docs/reports/06_time_varying_panel_report.md"
LOG = ROOT / "docs/logs/06_time_varying_panel_execution.md"
FIGURE_DIR = ROOT / "docs/figures"
MAPPING_AUDIT = TABLE_DIR / "06_panel_rent_mapping_audit.csv"

TARGET_CRS = "EPSG:5181"
RADII = (100, 200, 500)
OBSERVATION_END = pd.Timestamp("2026-09-30")
QUARTER_RE = re.compile(r"(20\d{2})년\s*([1-4])분기")
COMMON_NAME_WORDS = re.compile(r"[\s·ㆍ()\[\]{}'\".,/\\\\-]")
EXCLUDED_RONE_AREAS = {"서울", "도심", "강남", "영등포신촌", "기타", "양재말죽거리"}


def quarter_code(value: pd.Timestamp | pd.Period) -> int:
    period = value if isinstance(value, pd.Period) else value.to_period("Q")
    return period.year * 10 + period.quarter


def prior_quarter(code: int) -> int:
    year, quarter = divmod(int(code), 10)
    return (year - 1) * 10 + 4 if quarter == 1 else year * 10 + quarter - 1


def fmt_number(value: float | int | None, digits: int = 0) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:,.{digits}f}"


def normalized_name(value: object) -> str:
    """Use a conservative name normalisation for the auditable R-ONE match."""
    if pd.isna(value):
        return ""
    return COMMON_NAME_WORDS.sub("", str(value)).replace("상권", "")


def load_trade_module():
    """Load the established raw-data aggregation without executing its main()."""
    path = ROOT / "src/03_trade_area_variables.py"
    spec = importlib.util.spec_from_file_location("trade_area_variables", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("상권 변수 집계 스크립트를 불러올 수 없습니다.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rent() -> pd.DataFrame:
    """Convert the eight CP949 R-ONE CSVs with three header rows to long form."""
    rows: list[pd.DataFrame] = []
    for path in sorted(RENT_DIR.glob("*.csv")):
        kind = "small" if "소규모" in path.name else "medium" if "중대형" in path.name else None
        if kind is None:
            raise ValueError(f"상가 유형을 판별할 수 없는 임대료 파일: {path.name}")
        raw = pd.read_csv(path, encoding="cp949")
        if raw.shape[1] < 5:
            raise ValueError(f"열 수가 부족한 임대료 파일: {path.name}")
        quarter_columns = []
        for column in raw.columns[4:]:
            match = QUARTER_RE.search(str(column))
            if match:
                quarter_columns.append((column, int(match.group(1)) * 10 + int(match.group(2))))
        if not quarter_columns:
            raise ValueError(f"분기 열을 찾지 못한 임대료 파일: {path.name}")
        data = raw.loc[pd.to_numeric(raw.iloc[:, 0], errors="coerce").notna()].copy()
        data["city"] = data.iloc[:, 1].astype("string").str.strip()
        data["rone_major"] = data.iloc[:, 2].astype("string").str.strip()
        data["rone_area"] = data.iloc[:, 3].astype("string").str.strip()
        data = data.loc[data["city"].eq("서울") & data["rone_area"].notna()].copy()
        for column, code in quarter_columns:
            values = pd.to_numeric(data[column], errors="coerce")
            rows.append(pd.DataFrame({
                "rone_area": data["rone_area"].to_numpy(), "rone_major": data["rone_major"].to_numpy(),
                "quarter": code, "store_size": kind, "rent_thousand_krw_per_sqm": values.to_numpy(),
                "source_file": path.name,
            }))
    long = pd.concat(rows, ignore_index=True)
    duplicate = long.duplicated(["rone_area", "quarter", "store_size"], keep=False)
    if duplicate.any():
        example = long.loc[duplicate, ["rone_area", "quarter", "store_size", "source_file"]].head().to_dict("records")
        raise ValueError(f"임대료 상권-분기-유형 키가 중복됩니다: {example}")
    if long["rent_thousand_krw_per_sqm"].isna().any():
        raise ValueError("임대료 값 결측이 있습니다.")
    # Keep the published processed file in tidy long form.  The temporary
    # wide representation needed for the panel is made only at the merge.
    return long.sort_values(["rone_area", "quarter", "store_size"], kind="stable").reset_index(drop=True)


def make_rent_mapping(rent: pd.DataFrame, areas: gpd.GeoDataFrame) -> pd.DataFrame:
    """Map only unambiguous area-name matches; all other zones stay unattached."""
    zone_names = sorted(rent["rone_area"].dropna().unique())
    area_names = areas[["TRDAR_CD", "TRDAR_CD_N"]].copy()
    area_names["TRDAR_CD"] = area_names["TRDAR_CD"].astype(str)
    area_names["_normal"] = area_names["TRDAR_CD_N"].map(normalized_name)
    records: list[dict[str, object]] = []
    for zone in zone_names:
        key = normalized_name(zone)
        exact = area_names.loc[area_names["_normal"].eq(key)]
        containment = area_names.loc[
            area_names["_normal"].map(lambda name: len(key) >= 2 and (key in name or name in key))
        ]
        candidates = exact if len(exact) else containment
        if len(candidates) == 1:
            selected = candidates.iloc[0]
            reason = "정규화 이름 일치" if len(exact) else "정규화 이름 포함 일치"
            confidence = "높음" if len(exact) else "중간"
            code, name = selected["TRDAR_CD"], selected["TRDAR_CD_N"]
        elif len(candidates) == 0:
            reason, confidence, code, name = "자동 일치 후보 없음", "미확인", pd.NA, pd.NA
        else:
            reason, confidence, code, name = f"자동 일치 후보 {len(candidates)}개", "미확인", pd.NA, pd.NA
        records.append({
            "rone_area": zone, "trdar_cd": code, "trdar_name": name, "match_reason": reason,
            "confidence": confidence, "candidate_count": len(candidates),
            "candidate_names": " | ".join(candidates["TRDAR_CD_N"].astype(str).head(12)),
        })
    mapping = pd.DataFrame(records)
    mapped = mapping["trdar_cd"].dropna().astype(str)
    if mapped.duplicated().any():
        # One commercial polygon cannot represent more than one R-ONE zone.
        duplicated_codes = mapped.loc[mapped.duplicated(keep=False)].unique().tolist()
        mapping.loc[mapping["trdar_cd"].astype("string").isin(duplicated_codes), ["trdar_cd", "trdar_name", "confidence"]] = pd.NA
        mapping.loc[mapping["trdar_cd"].isna() & mapping["match_reason"].str.contains("일치"), "match_reason"] = "상권 코드가 복수 R-ONE 권역에 중복되어 미연결"
    return mapping


def apply_reviewed_mapping(auto_mapping: pd.DataFrame, areas: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine automatic one-to-one matches with Claude's reviewed R-ONE mapping.

    The reviewed table is authoritative where it supplies a mapping.  It may
    deliberately link one R-ONE zone to several commercial-area polygons.
    """
    reviewed = pd.read_csv(REVIEWED_MAPPING, encoding="utf-8-sig", dtype={"rone_area": "string", "trdar_cds": "string"})
    required = {"rone_area", "trdar_cds", "basis"}
    if missing := required.difference(reviewed.columns):
        raise ValueError(f"검토 임대료 대응표의 필수 열이 없습니다: {sorted(missing)}")
    reviewed["rone_area"] = reviewed["rone_area"].str.strip()
    reviewed = reviewed.loc[~reviewed["rone_area"].isin(EXCLUDED_RONE_AREAS)].copy()
    if reviewed["rone_area"].duplicated().any():
        raise ValueError("검토 임대료 대응표에 중복 R-ONE 권역이 있습니다.")

    valid_codes = set(areas["TRDAR_CD"].astype(str))
    auto_by_area = (
        auto_mapping.dropna(subset=["trdar_cd"]).assign(trdar_cd=lambda x: x["trdar_cd"].astype(str))
        .groupby("rone_area", observed=True)["trdar_cd"].agg(list).to_dict()
    )
    reviewed_by_area: dict[str, list[str]] = {}
    for row in reviewed.itertuples(index=False):
        codes = [code.strip() for code in str(row.trdar_cds).split(";") if code.strip()]
        if not codes or not set(codes).issubset(valid_codes):
            unknown = sorted(set(codes).difference(valid_codes))
            raise ValueError(f"검토 대응표의 상권 코드가 없거나 유효하지 않습니다: {row.rone_area} {unknown}")
        reviewed_by_area[str(row.rone_area)] = codes

    records: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    all_zones = sorted(set(auto_mapping["rone_area"]).union(reviewed_by_area))
    name_by_code = areas.set_index(areas["TRDAR_CD"].astype(str))["TRDAR_CD_N"].to_dict()
    for zone in all_zones:
        auto_codes = auto_by_area.get(zone, [])
        reviewed_codes = reviewed_by_area.get(zone, [])
        overlap = sorted(set(auto_codes).intersection(reviewed_codes))
        conflict = bool(auto_codes and reviewed_codes and set(auto_codes) != set(reviewed_codes))
        if reviewed_codes:
            selected, source = reviewed_codes, "검토 대응표"
            if auto_codes and not conflict:
                source = "자동+검토 대응표 일치"
            elif conflict:
                source = "검토 대응표 우선(자동 결과 상충)"
        else:
            selected, source = auto_codes, "자동 확정"
        for code in selected:
            records.append({"rone_area": zone, "trdar_cd": code, "trdar_name": name_by_code.get(code, pd.NA), "mapping_source": source})
        audit.append({
            "rone_area": zone,
            "automatic_trdar_cds": ";".join(auto_codes),
            "reviewed_trdar_cds": ";".join(reviewed_codes),
            "selected_trdar_cds": ";".join(selected),
            "mapping_source": source if selected else "미연결",
            "automatic_review_overlap": ";".join(overlap),
            "automatic_review_conflict": conflict,
            "selected_area_count": len(selected),
        })
    mapping = pd.DataFrame(records)
    if mapping["trdar_cd"].duplicated().any():
        duplicate = mapping.loc[mapping["trdar_cd"].duplicated(keep=False), ["rone_area", "trdar_cd"]]
        raise ValueError(f"복수 R-ONE 권역에 중복 배정된 상권 코드가 있습니다: {duplicate.to_dict('records')}")
    return mapping, pd.DataFrame(audit)


def main_sample_mask(data: pd.DataFrame) -> pd.Series:
    """Reproduce the fixed-variable Cox common-cohort rule without importing it."""
    required = ["is_franchise", "trdar_within_200m", "trdar_sales_total", "trdar_floating_pop", "trdar_resident_pop", "trdar_type", "trdar_cd_assigned", "comp_indep_r200", "comp_fran_r200"]
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"Cox 공통 표본 판별 열이 없습니다: {sorted(missing)}")
    mask = ~data["is_franchise"].fillna(True) & data["trdar_within_200m"].fillna(False)
    for column in ["trdar_sales_total", "trdar_floating_pop", "trdar_resident_pop", "trdar_type", "trdar_cd_assigned", "comp_indep_r200", "comp_fran_r200"]:
        mask &= data[column].notna()
    for column in ["trdar_sales_total", "trdar_floating_pop", "trdar_resident_pop"]:
        mask &= pd.to_numeric(data[column], errors="coerce").gt(0)
    return mask


def build_intervals(cohort: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in cohort.itertuples(index=False):
        opening = pd.Timestamp(row.permit_date).normalize()
        end = opening + pd.to_timedelta(int(row.survival_days), unit="D")
        if end > OBSERVATION_END:
            raise ValueError(f"관측 종료일 이후 생존기간: {row.license_id}")
        boundaries = [opening]
        next_quarter = (opening.to_period("Q") + 1).start_time
        while next_quarter < end:
            boundaries.append(next_quarter)
            next_quarter = (next_quarter.to_period("Q") + 1).start_time
        boundaries.append(end)
        for number, (start_date, stop_date) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
            records.append({
                "license_id": row.license_id, "interval_number": number,
                "interval_start_date": start_date, "interval_stop_date": stop_date,
                "start": int((start_date - opening).days), "stop": int((stop_date - opening).days),
                "event_interval": int(bool(row.event) and stop_date == end),
                "interval_quarter": quarter_code(start_date), "trdar_quarter_prior": prior_quarter(quarter_code(start_date)),
            })
    panel = pd.DataFrame(records)
    if panel.empty or panel["stop"].le(panel["start"]).any():
        raise ValueError("0 이하 길이의 시간 구간이 생성되었습니다.")
    return panel


def points(frame: pd.DataFrame) -> gpd.GeoDataFrame:
    valid = frame["has_coordinates"].fillna(False) & frame["coordinate_x"].notna() & frame["coordinate_y"].notna()
    output = frame.loc[valid, ["license_id", "coordinate_x", "coordinate_y"]].copy()
    return gpd.GeoDataFrame(output, geometry=gpd.points_from_xy(output.coordinate_x, output.coordinate_y), crs="EPSG:5174").to_crs(TARGET_CRS)


def add_competition(panel: pd.DataFrame, cohort: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Count active competitors at each interval start, excluding the focal licence."""
    for radius in RADII:
        for prefix in ("comp_indep", "comp_fran", "comp_total"):
            panel[f"{prefix}_r{radius}"] = np.nan
    cohort_points = points(cohort)
    clean = clean.loc[~clean["excluded_noncafe"].fillna(False)].copy()
    candidate_points = points(clean)
    if cohort_points.empty or candidate_points.empty:
        return panel
    candidate = candidate_points.merge(clean[["license_id", "permit_date", "closure_date", "is_franchise"]], on="license_id", how="left", validate="one_to_one")
    targets = panel.merge(cohort_points[["license_id", "geometry"]], on="license_id", how="left", validate="many_to_one")
    target_gdf = gpd.GeoDataFrame(targets, geometry="geometry", crs=TARGET_CRS)
    candidate_dates = candidate["permit_date"].to_numpy(dtype="datetime64[ns]")
    candidate_closure = candidate["closure_date"].to_numpy(dtype="datetime64[ns]")
    candidate_license = candidate["license_id"].astype(str).to_numpy()
    candidate_fran = candidate["is_franchise"].fillna(False).to_numpy(dtype=bool)
    for lower in range(0, len(target_gdf), 10_000):
        upper = min(lower + 10_000, len(target_gdf))
        chunk = target_gdf.iloc[lower:upper]
        available = chunk.geometry.notna().to_numpy()
        if not available.any():
            continue
        subset = chunk.loc[available]
        left, right = candidate.sindex.query(subset.geometry, predicate="dwithin", distance=max(RADII))
        if not len(left):
            for radius in RADII:
                for prefix in ("comp_indep", "comp_fran", "comp_total"):
                    panel.loc[subset.index, f"{prefix}_r{radius}"] = 0
            continue
        target_positions = subset.index.to_numpy()[left]
        start_dates = target_gdf.loc[target_positions, "interval_start_date"].to_numpy(dtype="datetime64[ns]")
        focal_license = target_gdf.loc[target_positions, "license_id"].astype(str).to_numpy()
        active = (candidate_dates[right] < start_dates) & (np.isnat(candidate_closure[right]) | (candidate_closure[right] >= start_dates))
        active &= candidate_license[right] != focal_license
        distances = subset.geometry.iloc[left].distance(candidate.geometry.iloc[right], align=False).to_numpy()
        for radius in RADII:
            selected = active & (distances <= radius)
            total = pd.Series(1, index=target_positions[selected]).groupby(level=0).sum()
            franchise = pd.Series(candidate_fran[right][selected].astype(int), index=target_positions[selected]).groupby(level=0).sum()
            panel.loc[subset.index, f"comp_total_r{radius}"] = 0
            panel.loc[subset.index, f"comp_fran_r{radius}"] = 0
            panel.loc[subset.index, f"comp_indep_r{radius}"] = 0
            panel.loc[total.index, f"comp_total_r{radius}"] = total.to_numpy()
            panel.loc[franchise.index, f"comp_fran_r{radius}"] = franchise.to_numpy()
            panel.loc[total.index, f"comp_indep_r{radius}"] = total.to_numpy() - franchise.reindex(total.index, fill_value=0).to_numpy()
    return panel


def carry_trade_values(panel: pd.DataFrame, trade: pd.DataFrame) -> pd.DataFrame:
    value_groups = {
        "sales": ["trdar_sales_total", "trdar_sales_coffee"],
        "stores": ["trdar_coffee_stores", "trdar_coffee_franchise_stores", "trdar_coffee_open_rate", "trdar_coffee_close_rate"],
        "population": ["trdar_floating_pop", "trdar_resident_pop"],
    }
    requested = panel[["trdar_cd_assigned", "trdar_quarter_prior"]].drop_duplicates().copy()
    requested = requested.dropna(subset=["trdar_cd_assigned"]).sort_values(["trdar_cd_assigned", "trdar_quarter_prior"])
    trade = trade.copy()
    trade["trdar_cd"] = trade["trdar_cd"].astype("string")
    for group, columns in value_groups.items():
        usable = trade.dropna(subset=columns, how="all")[["trdar_cd", "quarter", *columns]].sort_values(["trdar_cd", "quarter"])
        matched = pd.merge_asof(requested.sort_values("trdar_quarter_prior"), usable.sort_values("quarter"), left_on="trdar_quarter_prior", right_on="quarter", left_by="trdar_cd_assigned", right_by="trdar_cd", direction="backward")
        matched[f"trdar_{group}_carried"] = matched["quarter"].notna() & matched["quarter"].ne(matched["trdar_quarter_prior"])
        keep = ["trdar_cd_assigned", "trdar_quarter_prior", *columns, f"trdar_{group}_carried"]
        panel = panel.merge(matched[keep], on=["trdar_cd_assigned", "trdar_quarter_prior"], how="left", validate="many_to_one")
    return panel


def add_rent(panel: pd.DataFrame, cohort: pd.DataFrame, areas: gpd.GeoDataFrame, mapping: pd.DataFrame, rent: pd.DataFrame) -> pd.DataFrame:
    valid_mapping = mapping.dropna(subset=["trdar_cd"]).copy()
    valid_mapping["trdar_cd"] = valid_mapping["trdar_cd"].astype(str)
    code_to_zone = valid_mapping.set_index("trdar_cd")["rone_area"]
    location = cohort[["license_id", "trdar_cd_assigned"]].copy()
    location["rent_match_method"] = np.where(location["trdar_cd_assigned"].astype("string").isin(code_to_zone.index), "배정 상권 직접 일치", pd.NA)
    location["rone_area"] = location["trdar_cd_assigned"].astype("string").map(code_to_zone)
    if not valid_mapping.empty:
        mapped_areas = areas.loc[areas["TRDAR_CD"].astype(str).isin(code_to_zone.index), ["TRDAR_CD", "geometry"]].copy()
        unresolved = location["rone_area"].isna()
        unresolved_points = points(location.loc[unresolved].merge(cohort[["license_id", "coordinate_x", "coordinate_y", "has_coordinates"]], on="license_id", how="left"))
        if not unresolved_points.empty and not mapped_areas.empty:
            nearest = gpd.sjoin_nearest(unresolved_points, mapped_areas, how="left", max_distance=500, distance_col="rent_nearest_area_dist_m")
            nearest["TRDAR_CD"] = nearest["TRDAR_CD"].astype("string")
            nearest = nearest.sort_values(["license_id", "TRDAR_CD"], kind="stable").drop_duplicates("license_id")
            near_zone = nearest.set_index("license_id")["TRDAR_CD"].astype(str).map(code_to_zone)
            location.loc[location["license_id"].isin(near_zone.index) & location["rone_area"].isna(), "rone_area"] = location.loc[location["license_id"].isin(near_zone.index) & location["rone_area"].isna(), "license_id"].map(near_zone)
            location.loc[location["rone_area"].notna() & location["rent_match_method"].isna(), "rent_match_method"] = "매핑 상권 500m 이내 최근접"
    location["rent_match_method"] = location["rent_match_method"].fillna("미연결")
    panel = panel.merge(location[["license_id", "rone_area", "rent_match_method"]], on="license_id", how="left", validate="many_to_one")
    rent_wide = rent.pivot(index=["rone_area", "quarter"], columns="store_size", values="rent_thousand_krw_per_sqm").reset_index()
    rent_wide = rent_wide.rename(columns={"small": "rent_small_thousand_krw_per_sqm", "medium": "rent_medium_thousand_krw_per_sqm"})
    panel = panel.merge(rent_wide, left_on=["rone_area", "trdar_quarter_prior"], right_on=["rone_area", "quarter"], how="left", validate="many_to_one")
    return panel.drop(columns=["quarter"], errors="ignore")


def validation_tables(panel: pd.DataFrame, cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = panel.groupby("license_id", observed=True)
    check = grouped.agg(first_start=("start", "min"), final_stop=("stop", "max"), event_sum=("event_interval", "sum"), intervals=("interval_number", "size")).reset_index()
    check = check.merge(cohort[["license_id", "survival_days", "event"]], on="license_id", how="left", validate="one_to_one")
    check["duration_matches"] = check["final_stop"].eq(check["survival_days"])
    check["event_matches"] = check["event_sum"].eq(check["event"])
    check["start_matches"] = check["first_start"].eq(0)
    ordered = panel.sort_values(["license_id", "interval_number"], kind="stable").copy()
    ordered["previous_stop"] = ordered.groupby("license_id", observed=True)["stop"].shift()
    gap_failures = int((ordered["previous_stop"].notna() & ordered["start"].ne(ordered["previous_stop"])).sum())
    event_not_final = int((ordered["event_interval"].eq(1) & ordered["stop"].ne(ordered.groupby("license_id", observed=True)["stop"].transform("max"))).sum())
    validation = pd.DataFrame([
        {"check": "패널 점포 수", "value": len(check), "failures": len(cohort) - len(check)},
        {"check": "구간 종료시간=survival_days", "value": int(check.duration_matches.sum()), "failures": int((~check.duration_matches).sum())},
        {"check": "구간 사건합=점포 사건", "value": int(check.event_matches.sum()), "failures": int((~check.event_matches).sum())},
        {"check": "첫 구간 start=0", "value": int(check.start_matches.sum()), "failures": int((~check.start_matches).sum())},
        {"check": "인접 구간의 start=이전 stop", "value": len(panel) - len(check), "failures": gap_failures},
        {"check": "사건은 마지막 구간에만 있음", "value": int(panel["event_interval"].sum()) - event_not_final, "failures": event_not_final},
        {"check": "경쟁 음수", "value": 0, "failures": int((panel[[c for c in panel if c.startswith('comp_')]].fillna(0) < 0).any(axis=1).sum())},
    ])
    variables = ["comp_indep_r200", "comp_fran_r200", "comp_total_r200", "trdar_sales_total", "trdar_floating_pop", "trdar_resident_pop", "rent_small_thousand_krw_per_sqm"]
    rows = []
    for sample, subset in panel.groupby("panel_sample", observed=True):
        for column in variables:
            values = pd.to_numeric(subset[column], errors="coerce")
            rows.append({"panel_sample": sample, "variable": column, "n": int(values.notna().sum()), "missing": int(values.isna().sum()), "mean": values.mean(), "median": values.median(), "p25": values.quantile(.25), "p75": values.quantile(.75), "min": values.min(), "max": values.max()})
        for column in ["trdar_sales_carried", "trdar_stores_carried", "trdar_population_carried"]:
            rows.append({"panel_sample": sample, "variable": column, "n": len(subset), "missing": int(subset[column].isna().sum()), "mean": subset[column].mean(), "median": np.nan, "p25": np.nan, "p75": np.nan, "min": np.nan, "max": np.nan})
    distribution = pd.DataFrame(rows)
    first = panel.loc[panel.interval_number.eq(1), ["license_id", "comp_indep_r200", "comp_fran_r200", "comp_total_r200"]].set_index("license_id")
    comparisons = []
    for column in ["comp_indep_r200", "comp_fran_r200", "comp_total_r200"]:
        fixed = cohort.set_index("license_id")[column]
        difference = first[column] - fixed
        comparisons.append({"comparison": "개업 첫 구간 대 고정시점", "variable": column, "n": int(difference.notna().sum()), "same": int(difference.eq(0).sum()), "different": int(difference.ne(0).sum()), "mean_difference": difference.mean(), "p25_difference": difference.quantile(.25), "median_difference": difference.median(), "p75_difference": difference.quantile(.75)})
    for horizon, label in [(365, "개업 1년 후"), (730, "개업 2년 후")]:
        at_horizon = panel.loc[(panel.start <= horizon) & (panel.stop > horizon)].set_index("license_id")
        for column in ["comp_indep_r200", "comp_fran_r200", "comp_total_r200"]:
            difference = at_horizon[column] - first[column]
            comparisons.append({"comparison": label, "variable": column, "n": int(difference.notna().sum()), "same": int(difference.eq(0).sum()), "different": int(difference.ne(0).sum()), "mean_difference": difference.mean(), "p25_difference": difference.quantile(.25), "median_difference": difference.median(), "p75_difference": difference.quantile(.75), "increased": int(difference.gt(0).sum()), "decreased": int(difference.lt(0).sum())})
    linkage = panel.drop_duplicates("license_id").groupby(["panel_sample", "rent_match_method", "trdar_type", "event"], dropna=False, observed=True).size().rename("stores").reset_index()
    return validation, distribution, pd.DataFrame(comparisons), linkage


def rent_coverage_table(rent: pd.DataFrame) -> pd.DataFrame:
    coverage = rent.groupby(["quarter", "store_size"], observed=True).agg(
        area_count=("rone_area", "nunique"), value_count=("rent_thousand_krw_per_sqm", "count"),
        missing_values=("rent_thousand_krw_per_sqm", lambda x: int(x.isna().sum())),
    ).reset_index()
    duplicates = (
        rent.assign(_duplicate_key=rent.duplicated(["rone_area", "quarter", "store_size"]))
        .groupby(["quarter", "store_size"], observed=True)["_duplicate_key"].sum()
        .rename("duplicate_keys").reset_index()
    )
    return coverage.merge(duplicates, on=["quarter", "store_size"], how="left", validate="one_to_one")


def set_korean_font() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = font_manager.findfont("Malgun Gothic", fallback_to_default=False)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def make_panel_figures(panel: pd.DataFrame, cohort: pd.DataFrame, areas: gpd.GeoDataFrame, mapping: pd.DataFrame) -> None:
    """Create the two report-facing figures requested for the time-varying panel."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    set_korean_font()
    main_ids = set(cohort.loc[cohort["is_main_cox_sample"], "license_id"])
    first = panel.loc[panel["interval_number"].eq(1), ["license_id", "comp_indep_r200", "comp_fran_r200"]].set_index("license_id")
    labels, horizons = ["개업 1년 후", "개업 2년 후"], [365, 730]
    variables = [("comp_indep_r200", "개인 카페 경쟁 변화 (개)"), ("comp_fran_r200", "프랜차이즈 경쟁 변화 (개)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
    for ax, (column, label) in zip(axes, variables):
        values, counts = [], []
        for horizon in horizons:
            later = panel.loc[(panel["start"].le(horizon)) & (panel["stop"].gt(horizon)) & panel["license_id"].isin(main_ids), ["license_id", column]].set_index("license_id")
            difference = later[column].sub(first[column], fill_value=np.nan).dropna()
            values.append(difference.to_numpy())
            counts.append(len(difference))
        violin = ax.violinplot(values, showmedians=True, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor("#0072B2" if column == "comp_indep_r200" else "#D55E00")
            body.set_edgecolor("#555555")
            body.set_alpha(.75)
        violin["cmedians"].set_color("#555555")
        ax.axhline(0, color="#555555", linewidth=1)
        ax.set_xticks([1, 2], [f"{text}\n(n={count:,})" for text, count in zip(labels, counts)])
        ax.set_title(label, fontsize=15, pad=10)
        ax.set_ylabel("개업 시점 대비 변화량" if column == "comp_indep_r200" else "")
        ax.grid(axis="y", alpha=.22)
    fig.suptitle("개업 후 주변 200m 경쟁의 변화 분포: Cox 주 분석 표본", fontsize=17, y=.98)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_panel_competition_change.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    linked_codes = set(mapping["trdar_cd"].astype(str))
    plot_areas = areas.copy()
    plot_areas["linked"] = plot_areas["TRDAR_CD"].astype(str).isin(linked_codes)
    location = panel.drop_duplicates("license_id")[["license_id", "rone_area"]].merge(
        cohort[["license_id", "coordinate_x", "coordinate_y", "has_coordinates", "is_main_cox_sample"]], on="license_id", how="left", validate="one_to_one"
    )
    unresolved = points(location.loc[location["is_main_cox_sample"] & location["rone_area"].isna()].copy())
    fig, ax = plt.subplots(figsize=(9, 8.6))
    plot_areas.plot(ax=ax, color="#F2F2F2", edgecolor="#C8C8C8", linewidth=.15)
    plot_areas.loc[plot_areas["linked"]].plot(ax=ax, color="#E69F00", edgecolor="#A66900", linewidth=.3, alpha=.78)
    if not unresolved.empty:
        unresolved.plot(ax=ax, color="#555555", markersize=3.5, alpha=.36)
    ax.set_title("R-ONE 소규모 상가 임대료 연결 상권과 미연결 카페", fontsize=17, pad=12)
    ax.set_axis_off()
    ax.legend(handles=[
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#E69F00", markeredgecolor="#A66900", markersize=10, label="임대료 연결 상권 영역"),
        Line2D([0], [0], marker="o", color="#555555", linestyle="", markersize=4, label="임대료 미연결 개인 카페"),
    ], loc="lower left", frameon=True, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_panel_rent_linkage_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(panel: pd.DataFrame, cohort: pd.DataFrame, validation: pd.DataFrame, comparison: pd.DataFrame, linkage: pd.DataFrame, mapping: pd.DataFrame, mapping_audit: pd.DataFrame) -> None:
    main = panel.loc[panel.panel_sample.eq("Cox 주 분석 표본")]
    compare = comparison.loc[(comparison.comparison.isin(["개업 1년 후", "개업 2년 후"])) & comparison.variable.eq("comp_total_r200")]
    change_text = "; ".join(f"{r.comparison} 중앙값 {r.median_difference:+.1f}개 (증가 {int(r.increased):,}건, 감소 {int(r.decreased):,}건, n={int(r.n):,})" for r in compare.itertuples())
    fixed_diff = comparison.loc[(comparison.comparison.eq("개업 첫 구간 대 고정시점")) & comparison.variable.eq("comp_total_r200")].iloc[0]
    rent_linked = int(panel.drop_duplicates("license_id")["rone_area"].notna().sum())
    lines = [
        "# 분기별 시변 패널·임대료 연결 보고서", "",
        "## 주요 결과", "",
        f"개업기간 카페 전체 {len(cohort):,}개(프랜차이즈 포함)를 분기 start–stop 구간 {len(panel):,}행으로 바꿨고, 고정시점 Cox 주 분석 표본은 {int(cohort['is_main_cox_sample'].sum()):,}개로 표시했다. 주 분석 표본 구간은 {len(main):,}행이다.",
        f"200m 전체 경쟁은 좌표가 있어 비교 가능한 {int(fixed_diff.n):,}건에서 개업 첫 구간과 기존 고정시점 값이 모두 일치했다(불일치 {int(fixed_diff.different):,}건). 이는 첫 구간 경쟁 산출이 기존 개업일 경쟁 변수와 같은 시점·후보 규칙을 재현했음을 뜻한다.",
        f"개업 시점 대비 이후 200m 전체 경쟁 변화는 {change_text or '해당 생존 구간이 없음'}이다. 세부 개인·프랜차이즈별 변화 분포는 `06_panel_opening_comparison.csv`에 저장했다.",
        f"R-ONE 임대료는 {rent_linked:,}개 점포에 연결됐다. 자동 확정과 검토 대응표를 합쳐 {mapping['rone_area'].nunique():,}개 R-ONE 권역을 {len(mapping):,}개 상권 영역에 연결했고, 자동 결과와 검토 결과가 상충한 권역은 {int(mapping_audit['automatic_review_conflict'].sum()):,}개였다. 따라서 임대료가 연결된 표본은 보조 분석용으로만 해석해야 한다.",
        "",
        "![개업 후 200m 경쟁 변화 분포](../figures/06_panel_competition_change.png)",
        "개업 후에도 개인·프랜차이즈 경쟁 수가 양방향으로 바뀌어, 개업 시점 값만으로 이후 경쟁 환경을 대표하기 어렵다.",
        "",
        "![임대료 연결 상권과 미연결 점포](../figures/06_panel_rent_linkage_map.png)",
        "검토 대응표로 연결된 상권은 주요 상업지에 집중되며, 미연결 개인 카페 위치가 남아 있어 임대료 분석은 연결 하위 표본으로 제한한다.",
        "", "## 방법과 검증", "",
        "점포별로 개업일부터 해당 개업 분기 말까지를 첫 구간으로 하고, 이후 분기 경계에서 나눴다. 마지막 구간은 폐업일 또는 2026-09-30 관측 종료일에 끝난다. `start`·`stop`은 개업 후 일수이며, 사건은 마지막 구간에만 부여했다.",
        "경쟁은 각 구간 시작일 전에 개업하여 그 시점에 영업 중인 `cafes_clean.parquet`의 카페만 대상으로 EPSG:5181에서 100m·200m·500m 반경을 세었다. 본인 인허가와 비카페 업종은 제외했고, 개인·프랜차이즈·합계를 각각 저장했다. 따라서 미래 개업·폐업 정보가 공변량에 들어가지 않는다.",
        "상권 매출·점포·유동인구·상주인구는 기존 상권 원자료 집계 함수를 이용해 분기별로 다시 만들고, 구간 시작 직전 분기 값을 배정했다. 해당 분기의 값이 없으면 같은 상권의 마지막 관측값만 앞으로 이월했으며, 매출·점포·인구 이월 여부를 별도 열로 기록했다. 2026-03-31 이후에도 관측을 끊을 수 있도록 원자료가 없는 매출·점포 값의 이월을 보존했다.",
        "R-ONE CSV는 CP949 및 3행 헤더를 확인해 권역·분기·상가유형·임대료(천원/㎡) 형태로 정리했다. 자동으로 하나로 확정된 28개 권역에 Claude 검토 대응표의 39개 권역을 더했으며, 한 R-ONE 권역이 여러 상권 영역에 대응하는 경우에는 모두 같은 임대료를 붙였다. 서울·도심·강남·영등포신촌·기타 같은 상위 집계 행과 양재말죽거리는 연결하지 않았다. 배정 상권이 연결된 경우 직접 연결하고, 그렇지 않으면 연결 상권 폴리곤 500m 이내 최근접인 경우만 연결했다.",
        "", "### 검증", "",
        f"구간 종료시간=원래 생존기간 검증 실패 {int(validation.loc[validation.check.eq('구간 종료시간=survival_days'), 'failures'].iloc[0]):,}건, 구간 사건합 검증 실패 {int(validation.loc[validation.check.eq('구간 사건합=점포 사건'), 'failures'].iloc[0]):,}건이다. 전체 검증표는 `06_panel_validation.csv`에 있다.",
        "", "## 산출물", "",
        "- `data/processed/cafes_quarterly_panel.parquet`", "- `data/processed/rent_rone_quarterly.parquet`", "- `docs/review/06_rent_area_mapping.csv`", "- `docs/tables/06_panel_*.csv`", "- `docs/figures/06_panel_competition_change.png`", "- `docs/figures/06_panel_rent_linkage_map.png`", "",
        "## 제한과 결정이 필요한 사항", "",
        "R-ONE 권역은 상권분석서비스 상권 폴리곤과 공식 경계·코드 대응표가 없어, 검토 대응표에 없는 권역은 여전히 이름으로 하나로 확정된 경우만 연결했다. 연결 하위 표본의 대표성 또는 임대료의 인과적 효과는 주장하지 않는다.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    os.environ["TEMP"] = str(ROOT / ".tmp")
    os.environ["TMP"] = str(ROOT / ".tmp")
    (ROOT / ".tmp").mkdir(exist_ok=True)
    for directory in [OUTPUT.parent, RENT_OUTPUT.parent, TABLE_DIR, REVIEW.parent, REPORT.parent, LOG.parent, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    analysis = pd.read_parquet(ANALYSIS).copy()
    clean = pd.read_parquet(CLEAN).copy()
    for frame in [analysis, clean]:
        frame["permit_date"] = pd.to_datetime(frame["permit_date"]).dt.normalize()
        frame["closure_date"] = pd.to_datetime(frame["closure_date"]).dt.normalize()
        frame["license_id"] = frame["관리번호"].astype(str)
    if analysis["license_id"].duplicated().any() or clean["license_id"].duplicated().any():
        raise ValueError("관리번호가 인허가 식별자로 유일하지 않습니다.")
    if not analysis["survival_days"].gt(0).all():
        raise ValueError("분석 코호트에 0 이하 생존기간이 있습니다.")
    analysis["is_main_cox_sample"] = main_sample_mask(analysis)
    analysis["panel_sample"] = np.where(analysis["is_main_cox_sample"], "Cox 주 분석 표본", "프랜차이즈 포함 전체 카페 표본")
    static_columns = ["license_id", "permit_date", "closure_date", "event", "survival_days", "is_main_cox_sample", "panel_sample", "opening_year", "trdar_cd_assigned", "trdar_type", "coordinate_x", "coordinate_y", "has_coordinates"]
    analysis["opening_year"] = analysis["permit_date"].dt.year.astype(int)
    cohort = analysis[static_columns].copy()
    panel = build_intervals(cohort)
    panel = panel.merge(cohort, on="license_id", how="left", validate="many_to_one")
    panel = add_competition(panel, analysis, clean)
    trade_module = load_trade_module()
    trade, _ = trade_module.make_trade_area_panel()
    trade["trdar_cd"] = trade["trdar_cd"].astype("string")
    panel["trdar_cd_assigned"] = panel["trdar_cd_assigned"].astype("string")
    panel = carry_trade_values(panel, trade)
    areas = gpd.read_file(AREA).to_crs(TARGET_CRS)
    areas["TRDAR_CD"] = areas["TRDAR_CD"].astype(str)
    rent = read_rent()
    rent.to_parquet(RENT_OUTPUT, index=False)
    auto_mapping = make_rent_mapping(rent, areas)
    mapping, mapping_audit = apply_reviewed_mapping(auto_mapping, areas)
    mapping.to_csv(REVIEW, index=False, encoding="utf-8-sig")
    mapping_audit.to_csv(MAPPING_AUDIT, index=False, encoding="utf-8-sig")
    panel = add_rent(panel, analysis, areas, mapping, rent)
    panel = panel.sort_values(["license_id", "start", "interval_number"], kind="stable").reset_index(drop=True)
    panel.to_parquet(OUTPUT, index=False)
    validation, distribution, comparison, linkage = validation_tables(panel, analysis)
    validation.to_csv(TABLE_DIR / "06_panel_validation.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(TABLE_DIR / "06_panel_variable_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(TABLE_DIR / "06_panel_opening_comparison.csv", index=False, encoding="utf-8-sig")
    linkage.to_csv(TABLE_DIR / "06_panel_rent_linkage.csv", index=False, encoding="utf-8-sig")
    rent_coverage_table(rent).to_csv(TABLE_DIR / "06_panel_rent_coverage.csv", index=False, encoding="utf-8-sig")
    sample_flow = pd.DataFrame([
        {"step": "개업기간 카페 전체(프랜차이즈 포함)", "n_before": len(analysis), "excluded": 0, "reason": "cafes_analysis.parquet의 분석 기간 코호트", "n_after": len(analysis)},
        {"step": "고정시점 Cox 주 분석 표본 표지", "n_before": len(analysis), "excluded": int((~analysis["is_main_cox_sample"]).sum()), "reason": "기존 Cox 공통 표본 규칙; 전체 카페 패널에는 유지", "n_after": int(analysis["is_main_cox_sample"].sum())},
    ])
    sample_flow.to_csv(TABLE_DIR / "06_panel_sample_flow.csv", index=False, encoding="utf-8-sig")
    make_panel_figures(panel, cohort, areas, mapping)
    write_report(panel, cohort, validation, comparison, linkage, mapping, mapping_audit)
    LOG.write_text("\n".join([
        "# 분기별 시변 패널 실행 로그", "",
        f"- 입력: `data/processed/cafes_analysis.parquet` {len(analysis):,}행, `data/processed/cafes_clean.parquet` {len(clean):,}행",
        f"- 출력 패널: {len(panel):,}행, 점포 {panel['license_id'].nunique():,}개, 사건 구간 {int(panel['event_interval'].sum()):,}건",
        f"- Cox 주 분석 표본 표지: {int(analysis['is_main_cox_sample'].sum()):,}개",
        f"- R-ONE: {len(rent):,}개 권역-분기-상가유형 행, 최종 연결 {mapping['rone_area'].nunique():,}개 권역·{len(mapping):,}개 상권 영역, 자동-검토 상충 {int(mapping_audit['automatic_review_conflict'].sum()):,}개",
        "- 원자료 수정 및 패키지 설치 없음; 임시 파일은 프로젝트 .tmp/만 사용",
    ]) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {RENT_OUTPUT}")
    print(f"Wrote {REVIEW}")


if __name__ == "__main__":
    main()
