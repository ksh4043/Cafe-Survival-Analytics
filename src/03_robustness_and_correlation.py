"""Create the re-licensing robustness cohort and covariate diagnostics.

Run after ``03_trade_area_variables.py``.  The main licence-level analysis is
never replaced: this script only appends robustness columns and writes review
and diagnostic artifacts.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "data/processed/cafes_analysis.parquet"
CLEAN = ROOT / "data/processed/cafes_clean.parquet"
MERGED_REVIEW = ROOT / "docs/review/03_relicense_merged.csv"
SAMPLE_REVIEW = ROOT / "docs/review/03_cafes_analysis_sample.csv"
PEARSON = ROOT / "docs/tables/03_covariate_correlation_pearson.csv"
SPEARMAN = ROOT / "docs/tables/03_covariate_correlation_spearman.csv"
VIF = ROOT / "docs/tables/03_covariate_vif.csv"
COFFEE_MISSING = ROOT / "docs/tables/03_coffee_store_missingness_correlation.csv"
REPORT = ROOT / "docs/reports/03_robustness_and_correlation_report.md"
STUDY_START = pd.Timestamp("2021-04-01")
STUDY_END = pd.Timestamp("2026-03-31")
OBSERVATION_END = pd.Timestamp("2026-09-30")
PREVIOUS_EXCLUSION_BASELINE = {
    "전체": 11_515, "개인 카페": 8_054, "사건": 4_253,
    "200m 전체 경쟁 평균": 10.912, "200m 개인 경쟁 평균": 7.461, "200m 프랜차이즈 경쟁 평균": 3.452,
}


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(해당 없음)"
    safe = frame.copy()
    for column in safe:
        safe[column] = safe[column].map(lambda x: "" if pd.isna(x) else str(x).replace("|", "\\|"))
    return "\n".join([
        "| " + " | ".join(safe.columns) + " |",
        "| " + " | ".join("---" for _ in safe.columns) + " |",
        *("| " + " | ".join(row) + " |" for row in safe.itertuples(index=False, name=None)),
    ])


def fmt(value: int | float) -> str:
    return f"{value:,}"


def normalize_relicense_name(value: object, mode: str = "standard") -> str:
    """Make deterministic conservative keys for the same-name test.

    ``strict`` only removes spacing and punctuation.  ``standard`` additionally
    drops parenthetical labels and legal-entity notation; it is the approved
    robustness rule.  ``loose`` also removes common branch suffixes and is
    reported only as a sensitivity count, never used to form the cohort.
    """
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    if mode in {"standard", "loose"}:
        text = re.sub(r"\([^)]*\)|\[[^]]*\]|【[^】]*】", "", text)
        text = re.sub(r"주식회사|유한회사|합자회사|합명회사|\(주\)|㈜", "", text)
    text = re.sub(r"[\s\-_\.,·'`/\\&]+", "", text)
    if mode == "loose":
        text = re.sub(r"(?:\d+호?|직영|본|\d*)점$|지점$", "", text)
    return text


def link_edges(clean: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return one deterministic A→B candidate per later licence where possible."""
    work = clean.loc[
        clean["closure_date"].notna() & clean["has_coordinates"].fillna(False),
        ["관리번호", "사업장명", "permit_date", "closure_date", "coordinate_x", "coordinate_y", "is_franchise"],
    ].copy()
    work["name_key"] = clean.loc[work.index, "사업장명"].map(lambda x: normalize_relicense_name(x, mode))
    work = work.loc[work["name_key"].ne("")].copy()
    later = clean.loc[
        clean["has_coordinates"].fillna(False),
        ["관리번호", "사업장명", "permit_date", "closure_date", "coordinate_x", "coordinate_y", "is_franchise"],
    ].copy()
    later["name_key"] = clean.loc[later.index, "사업장명"].map(lambda x: normalize_relicense_name(x, mode))
    later = later.loc[later["name_key"].ne("")].copy()
    pairs = work.merge(later, on="name_key", suffixes=("_a", "_b"), how="inner")
    timing = pairs["permit_date_b"].sub(pairs["closure_date_a"]).dt.days
    pairs = pairs.loc[timing.isin([0, 1])].copy()
    pairs["distance_m"] = np.hypot(
        pairs["coordinate_x_a"] - pairs["coordinate_x_b"], pairs["coordinate_y_a"] - pairs["coordinate_y_b"]
    )
    pairs = pairs.loc[pairs["distance_m"].le(10)].copy()
    pairs = pairs.sort_values(["관리번호_b", "distance_m", "closure_date_a", "관리번호_a"], kind="stable")
    return pairs.drop_duplicates("관리번호_b", keep="first")


def build_spell_map(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    standard_edges = link_edges(clean, "standard")
    strict_edges = link_edges(clean, "strict")
    loose_edges = link_edges(clean, "loose")
    ids = clean["관리번호"].astype(str).tolist()
    parent = {identifier: identifier for identifier in ids}

    def find(identifier: str) -> str:
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge in standard_edges.itertuples(index=False):
        union(str(edge.관리번호_a), str(edge.관리번호_b))
    mapping = clean[["관리번호", "permit_date", "closure_date", "event", "is_franchise"]].copy()
    mapping["_root"] = mapping["관리번호"].astype(str).map(find)
    mapping = mapping.sort_values(["_root", "permit_date", "관리번호"], kind="stable")
    mapping["_position"] = mapping.groupby("_root", sort=False).cumcount()
    mapping["_size"] = mapping.groupby("_root", sort=False)["관리번호"].transform("size")
    first_ids = mapping.groupby("_root", sort=False)["관리번호"].transform("first").astype(str)
    mapping["relicense_spell_id"] = "relicense_" + first_ids
    mapping["relicense_role"] = np.where(mapping["_size"].eq(1), "단독", np.where(mapping["_position"].eq(0), "첫", "뒤"))
    spell_last = mapping.groupby("_root", sort=False).tail(1).set_index("_root")
    spell_first = mapping.groupby("_root", sort=False).head(1).set_index("_root")
    mapping["_spell_first_date"] = mapping["_root"].map(spell_first["permit_date"])
    mapping["_spell_last_end"] = mapping["_root"].map(spell_last["closure_date"].where(spell_last["event"].eq(1), OBSERVATION_END))
    mapping["_spell_event"] = mapping["_root"].map(spell_last["event"]).astype("int8")
    mapping["robust_relicense_in_sample"] = (
        mapping["relicense_role"].isin(["단독", "첫"])
        & mapping["_spell_first_date"].between(STUDY_START, STUDY_END, inclusive="both")
    )
    mapping["robust_relicense_survival_days"] = pd.Series(pd.NA, index=mapping.index, dtype="Int64")
    in_sample = mapping["robust_relicense_in_sample"]
    mapping.loc[in_sample, "robust_relicense_survival_days"] = (
        mapping.loc[in_sample, "_spell_last_end"] - mapping.loc[in_sample, "_spell_first_date"]
    ).dt.days.astype("Int64")
    mapping["robust_relicense_event"] = pd.Series(pd.NA, index=mapping.index, dtype="Int8")
    mapping.loc[in_sample, "robust_relicense_event"] = mapping.loc[in_sample, "_spell_event"].astype("Int8")
    merged = mapping.loc[mapping["_size"].gt(1), ["관리번호", "relicense_spell_id", "relicense_role", "_position", "_size"]].merge(
        clean[["관리번호", "사업장명", "도로명주소", "지번주소", "permit_date", "closure_date", "is_franchise"]], on="관리번호", how="left", validate="one_to_one"
    ).sort_values(["relicense_spell_id", "_position"], kind="stable")
    merged = merged.rename(columns={"_position": "인허가 순서", "_size": "합친 인허가 수", "사업장명": "상호", "도로명주소": "도로명주소", "지번주소": "지번주소", "permit_date": "개업일", "closure_date": "폐업일", "is_franchise": "프랜차이즈"})
    diagnostics = {"strict_edges": len(strict_edges), "standard_edges": len(standard_edges), "loose_edges": len(loose_edges), "merged_spells": int(mapping.loc[mapping["_size"].gt(1), "_root"].nunique())}
    return mapping.drop(columns=["_root", "_position", "_size", "_spell_first_date", "_spell_last_end", "_spell_event"]), merged, diagnostics


def correlation_inputs(frame: pd.DataFrame, coffee_zero: bool = False) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = pd.DataFrame(index=frame.index)
    labels = {
        "comp_indep_r100": "100m 개인 카페 수", "comp_fran_r100": "100m 프랜차이즈 수",
        "comp_indep_r200": "200m 개인 카페 수", "comp_fran_r200": "200m 프랜차이즈 수",
        "comp_indep_r500": "500m 개인 카페 수", "comp_fran_r500": "500m 프랜차이즈 수",
        "nearest_cafe_dist_m": "가장 가까운 카페 거리(m)", "trdar_coffee_stores": "상권 커피 점포 수",
        "trdar_coffee_franchise_stores": "상권 커피 프랜차이즈 점포 수", "trdar_coffee_open_rate": "상권 커피 개업률",
        "trdar_coffee_close_rate": "상권 커피 폐업률",
    }
    for source, label in labels.items():
        values = pd.to_numeric(frame[source], errors="coerce")
        if coffee_zero and source in {"trdar_coffee_stores", "trdar_coffee_franchise_stores", "trdar_coffee_open_rate", "trdar_coffee_close_rate"}:
            values = values.fillna(0)
        out[label] = values
    for source, label in (("trdar_sales_total", "로그 상권 총매출"), ("trdar_floating_pop", "로그 유동인구"), ("trdar_resident_pop", "로그 상주인구"), ("trdar_area_sqm", "로그 상권 면적")):
        values = pd.to_numeric(frame[source], errors="coerce")
        out[label] = np.where(values.ge(0), np.log1p(values), np.nan)
    types = frame["trdar_type"].fillna("결측").astype(str)
    dummies = pd.get_dummies(types, prefix="상권 구분", dtype=float)
    out = pd.concat([out, dummies], axis=1)
    numeric = list(labels.values()) + ["로그 상권 총매출", "로그 유동인구", "로그 상주인구", "로그 상권 면적"]
    return out, numeric, list(dummies.columns)


def vif_table(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.dropna().astype(float)
    rows = []
    for column in values:
        y = values[column].to_numpy()
        others = values.drop(columns=column).to_numpy()
        design = np.column_stack([np.ones(len(values)), others])
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        total = float(((y - y.mean()) ** 2).sum())
        residual = float(((y - fitted) ** 2).sum())
        r_squared = 1 - residual / total if total else np.nan
        vif = np.inf if pd.notna(r_squared) and r_squared >= 1 - 1e-12 else 1 / (1 - r_squared)
        rows.append({"변수": column, "유효 표본 수": len(values), "R²": round(r_squared, 6), "VIF": round(float(vif), 3) if np.isfinite(vif) else "무한대"})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False, key=lambda s: pd.to_numeric(s, errors="coerce").fillna(np.inf), kind="stable")


def main() -> None:
    os.environ["TEMP"] = str(ROOT / ".tmp")
    os.environ["TMP"] = str(ROOT / ".tmp")
    (ROOT / ".tmp").mkdir(exist_ok=True)
    for path in (MERGED_REVIEW.parent, PEARSON.parent, REPORT.parent):
        path.mkdir(parents=True, exist_ok=True)

    analysis = pd.read_parquet(ANALYSIS).copy()
    clean = pd.read_parquet(CLEAN).copy()
    for frame in (analysis, clean):
        frame["permit_date"] = pd.to_datetime(frame["permit_date"])
        frame["closure_date"] = pd.to_datetime(frame["closure_date"])
    spell_map, merged, spell_diagnostics = build_spell_map(clean)
    add = spell_map[["관리번호", "relicense_spell_id", "relicense_role", "robust_relicense_in_sample", "robust_relicense_survival_days", "robust_relicense_event"]]
    existing = [column for column in add if column in analysis]
    analysis = analysis.drop(columns=[column for column in existing if column != "관리번호"])
    analysis = analysis.merge(add, on="관리번호", how="left", validate="one_to_one")
    analysis["robust_relicense_in_sample"] = analysis["robust_relicense_in_sample"].fillna(False).astype("boolean")
    analysis["relicense_role"] = analysis["relicense_role"].astype("string")
    analysis.to_parquet(ANALYSIS, index=False)
    merged.to_csv(MERGED_REVIEW, index=False, encoding="utf-8-sig")
    analysis.head(200).to_csv(SAMPLE_REVIEW, index=False, encoding="utf-8-sig")

    main_personal = analysis.loc[~analysis["is_franchise"].fillna(False)].copy()
    robust = analysis.loc[analysis["robust_relicense_in_sample"].fillna(False)].copy()
    robust_personal = robust.loc[~robust["is_franchise"].fillna(False)].copy()
    stats = pd.DataFrame([
        {"표본": "주 분석 인허가 단위", "전체": len(analysis), "개인 카페": len(main_personal), "프랜차이즈": int(analysis["is_franchise"].sum()), "사건": int(analysis["event"].sum()), "개인 카페 사건": int(main_personal["event"].sum()), "개인 카페 생존기간 중앙값(일)": int(main_personal["survival_days"].median())},
        {"표본": "재인허가 통합 강건성", "전체": len(robust), "개인 카페": len(robust_personal), "프랜차이즈": int(robust["is_franchise"].sum()), "사건": int(robust["robust_relicense_event"].sum()), "개인 카페 사건": int(robust_personal["robust_relicense_event"].sum()), "개인 카페 생존기간 중앙값(일)": int(robust_personal["robust_relicense_survival_days"].median())},
    ])
    first_permit_by_spell = spell_map.groupby("relicense_spell_id")["permit_date"].min()
    analysis["_relicense_first_permit"] = analysis["relicense_spell_id"].map(first_permit_by_spell)
    later = analysis.loc[analysis["relicense_role"].eq("뒤")]
    excluded_after = len(later)
    later_first_in_period = int(later["_relicense_first_permit"].between(STUDY_START, STUDY_END, inclusive="both").sum())
    later_first_before_period = int(later["_relicense_first_permit"].lt(STUDY_START).sum())
    retained_first = int(analysis["relicense_role"].eq("첫").sum())

    exclusion_change = pd.DataFrame([
        {"지표": "인허가 단위 분석 표본", "이전": PREVIOUS_EXCLUSION_BASELINE["전체"], "현재": len(analysis)},
        {"지표": "개인 카페", "이전": PREVIOUS_EXCLUSION_BASELINE["개인 카페"], "현재": len(main_personal)},
        {"지표": "사건", "이전": PREVIOUS_EXCLUSION_BASELINE["사건"], "현재": int(analysis["event"].sum())},
        {"지표": "200m 전체 경쟁 평균", "이전": PREVIOUS_EXCLUSION_BASELINE["200m 전체 경쟁 평균"], "현재": round(float(analysis["comp_total_r200"].mean()), 3)},
        {"지표": "200m 개인 경쟁 평균", "이전": PREVIOUS_EXCLUSION_BASELINE["200m 개인 경쟁 평균"], "현재": round(float(analysis["comp_indep_r200"].mean()), 3)},
        {"지표": "200m 프랜차이즈 경쟁 평균", "이전": PREVIOUS_EXCLUSION_BASELINE["200m 프랜차이즈 경쟁 평균"], "현재": round(float(analysis["comp_fran_r200"].mean()), 3)},
    ])
    exclusion_change["증감"] = exclusion_change["현재"] - exclusion_change["이전"]
    exclusion_display = exclusion_change.copy()
    mean_rows = exclusion_display["지표"].str.contains("평균", regex=False)
    for column in ("이전", "현재", "증감"):
        exclusion_display[column] = np.where(
            mean_rows,
            exclusion_display[column].map(lambda value: f"{float(value):.3f}"),
            exclusion_display[column].map(lambda value: f"{int(value):,}"),
        )

    base = analysis.loc[analysis["trdar_within_200m"].fillna(False) & ~analysis["is_franchise"].fillna(False)].copy()
    covariates, numeric_columns, type_columns = correlation_inputs(base)
    covariates.corr(method="pearson").round(4).to_csv(PEARSON, encoding="utf-8-sig")
    covariates.corr(method="spearman").round(4).to_csv(SPEARMAN, encoding="utf-8-sig")
    vif_columns = ["200m 개인 카페 수", "200m 프랜차이즈 수", "로그 상권 총매출", "로그 유동인구", "로그 상주인구"] + type_columns[1:]
    vifs = vif_table(covariates[vif_columns])
    vifs.to_csv(VIF, index=False, encoding="utf-8-sig")
    zero_covariates, _, _ = correlation_inputs(base, coffee_zero=True)
    comparison_rows = []
    for metric, matrix in (("결측 유지", covariates), ("0 대체", zero_covariates)):
        for target in ("200m 개인 카페 수", "200m 프랜차이즈 수", "로그 상권 총매출"):
            paired = matrix[["상권 커피 점포 수", target]].dropna()
            comparison_rows.append({"처리": metric, "비교 변수": target, "유효 표본 수": len(paired), "Pearson r": round(float(paired.corr().iloc[0, 1]), 4) if len(paired) > 1 else np.nan, "Spearman ρ": round(float(paired.corr(method="spearman").iloc[0, 1]), 4) if len(paired) > 1 else np.nan})
    missingness = pd.DataFrame(comparison_rows)
    missingness.to_csv(COFFEE_MISSING, index=False, encoding="utf-8-sig")
    top_pairs = covariates[numeric_columns].corr().where(np.triu(np.ones((len(numeric_columns), len(numeric_columns)), dtype=bool), 1)).stack().abs().sort_values(ascending=False).head(5)
    top_pair_table = pd.DataFrame([{"변수 1": left, "변수 2": right, "|Pearson r|": round(float(value), 3)} for (left, right), value in top_pairs.items()])
    demand_pair = covariates[["로그 상권 총매출", "로그 유동인구", "로그 상주인구"]].corr().round(3)
    raw_log = pd.DataFrame([
        {"변수": "상권 총매출", "원값 Pearson r(200m 개인 카페 수)": round(float(base[["trdar_sales_total", "comp_indep_r200"]].corr().iloc[0, 1]), 3), "로그값 Pearson r": round(float(covariates[["로그 상권 총매출", "200m 개인 카페 수"]].corr().iloc[0, 1]), 3)},
        {"변수": "유동인구", "원값 Pearson r(200m 개인 카페 수)": round(float(base[["trdar_floating_pop", "comp_indep_r200"]].corr().iloc[0, 1]), 3), "로그값 Pearson r": round(float(covariates[["로그 유동인구", "200m 개인 카페 수"]].corr().iloc[0, 1]), 3)},
        {"변수": "상주인구", "원값 Pearson r(200m 개인 카페 수)": round(float(base[["trdar_resident_pop", "comp_indep_r200"]].corr().iloc[0, 1]), 3), "로그값 Pearson r": round(float(covariates[["로그 상주인구", "200m 개인 카페 수"]].corr().iloc[0, 1]), 3)},
    ])
    report = f"""# 재인허가 강건성 표본과 공변량 상관 점검 보고서

## 주요 결과

같은 상호·같은 자리에서 폐업일과 같은 날 또는 다음 날에 재개업한 인허가를 합쳐도, **주 분석 인허가 단위 표본은 변경하지 않았다.** 강건성 표본에서는 합쳐진 영업기간의 첫 인허가만 남긴 결과, 개인 카페는 {fmt(len(main_personal))}개에서 {fmt(len(robust_personal))}개, 개인 카페 폐업 사건은 {fmt(int(main_personal['event'].sum()))}건에서 {fmt(int(robust_personal['robust_relicense_event'].sum()))}건으로 바뀌었다. 개인 카페 생존기간 중앙값은 {fmt(int(main_personal['survival_days'].median()))}일에서 {fmt(int(robust_personal['robust_relicense_survival_days'].median()))}일이다. 이는 운영자 식별자가 없어 확정할 수 없는 재인허가를 보수적으로 점검하는 결과이며, 주 분석을 대체하지 않는다.

{markdown_table(stats)}

놀이·상담형 카페 제외로 갱신된 주요 값은 다음과 같다. 비교 기준은 이번 승인 전 체인 사전 보완 결과이며, 제외 점포는 분석 표본과 경쟁 집합에서 함께 뺐다.

{markdown_table(exclusion_display)}

200m 이내 상권이 배정된 개인 카페 {fmt(len(base))}개에서 반경별 카페 수는 서로 매우 유사할 수 있고, 특히 반경 경쟁 수와 상권 커피 점포 수는 함께 증가할 가능성이 있다. 수요 변수끼리는 아래 상관행렬처럼 겹침을 확인해야 한다. 따라서 후보 조합을 모두 함께 넣기 전에는 반경 경쟁 수와 상권 점포 수 중 어느 지표를 쓸지, 그리고 매출·유동·상주인구를 동시에 쓸지의 선택지를 검토해야 한다. 이 작업에서는 변수를 선택·제외하지 않았다.

가장 큰 절대 Pearson 상관(상권 구분 더미 제외)은 다음과 같다.

{markdown_table(top_pair_table)}

수요 지표 간 Pearson 상관은 다음과 같다.

{markdown_table(demand_pair.reset_index().rename(columns={'index': '변수'}))}

## 재인허가 합산 결과

표준 규칙으로 연결된 인허가 쌍은 {fmt(spell_diagnostics['standard_edges'])}건, 합쳐진 영업기간은 {fmt(spell_diagnostics['merged_spells'])}개다. 첫 인허가가 개업 기간 안인 합친 영업기간은 {fmt(retained_first)}개이며, 각 영업기간의 첫 인허가 한 건만 강건성 표본에 남겼다. 개업 기간 안의 뒤 인허가 {fmt(excluded_after)}건은 새 개업으로 세지 않았다. 이 중 {fmt(later_first_in_period)}건은 첫 인허가가 개업 기간 안인 영업기간의 뒤 인허가이고, {fmt(later_first_before_period)}건은 첫 인허가가 기간 전이라 강건성 표본 전체가 빠지는 뒤 인허가다. 프랜차이즈·개인 카페 결과는 위 표에 분리했다. 합쳐진 모든 인허가의 상호·주소·개업·폐업일은 `docs/review/03_relicense_merged.csv`에서 검수할 수 있다.

상호 정규화는 Unicode NFKC와 소문자화 뒤 공백·구두점을 제거하고, 표준 규칙에서는 괄호 안 표기와 `㈜`·`(주)`·`주식회사` 등 법인 표기도 제거했다. 동일 정규화 상호, 폐업일과 개업일 간격 0~1일, EPSG:5174 좌표 직선거리 10m 이하를 모두 만족해야 연결했다. 괄호·법인 표기를 보존한 엄격 규칙은 {fmt(spell_diagnostics['strict_edges'])}건, 지점·본점 같은 접미사도 지운 느슨한 규칙은 {fmt(spell_diagnostics['loose_edges'])}건을 연결했다. 표준 규칙만 강건성 표본에 사용했다.

## 공변량 상관과 다중공선성

Pearson·Spearman 전체 행렬은 각각 `docs/tables/03_covariate_correlation_pearson.csv`, `docs/tables/03_covariate_correlation_spearman.csv`에 저장했다. 상권 구분은 수치형 상관을 만들 수 없어 구분별 더미로 포함했다. 매출·유동인구·상주인구·면적은 `log(1+x)`로 변환했다. 원값과 로그값의 200m 개인 카페 수 상관은 다음과 같아, 로그 변환이 긴 오른쪽 꼬리의 영향만 완화하는지 확인할 수 있다.

{markdown_table(raw_log)}

주 분석 후보 조합(200m 개인·프랜차이즈 수, 로그 총매출·유동인구·상주인구, 상권 구분 더미)의 VIF는 다음과 같다. 상권 구분 더미는 한 범주를 기준으로 빼고 계산했다.

{markdown_table(vifs)}

원자료에 상권-업종 행이 없는 커피 점포 수는 0과 결측을 구별할 근거가 없어 기본 행렬에서는 결측으로 유지했다. 이를 0으로 대체했을 때의 민감도는 아래와 같으며, 전체 결과는 `docs/tables/03_coffee_store_missingness_correlation.csv`에 저장했다.

{markdown_table(missingness)}

## 방법과 검증

재인허가 연결은 `cafes_clean.parquet` 전체 기간을 대상으로 만든 뒤 개업 기간 표본에 표지를 붙였다. A→B→C 연쇄는 union-find로 하나의 영업기간으로 합쳤고, 시작일은 첫 인허가 개업일, 사건·종료일은 마지막 인허가 기준으로 계산했다. 첫 인허가가 개업 기간 안이면 첫 행만 강건성 표본에 남겼고, 기간 전이면 기간 안의 뒤 인허가도 넣지 않았다. 공변량은 기존 `cafes_analysis.parquet`의 첫 인허가 행에 이미 결합된 개업 직전 분기·개업 당시 경쟁 값이므로 합산으로 새 시점 정보를 끌어오지 않는다.

VIF는 절편을 포함한 최소제곱 회귀의 R²로 `1/(1−R²)`를 계산했다. `statsmodels` 없이 numpy만 사용했다. 상관은 쌍별 결측을 제외해 계산했고, VIF는 후보 변수가 모두 있는 완전 사례에 한정했다. Cox·KM 모형과 변수 선택은 수행하지 않았다.

## 산출물과 결정이 필요한 사항

- `data/processed/cafes_analysis.parquet`: `relicense_spell_id`, `relicense_role`, `robust_relicense_in_sample`, `robust_relicense_survival_days`, `robust_relicense_event` 추가
- `docs/review/03_relicense_merged.csv`: 합쳐진 재인허가 사례 전체
- 상관행렬·VIF·결측 민감도 표: `docs/tables/03_covariate_*.csv`

운영자 식별자가 없으므로 연결은 동일 상호·시점·좌표가 모두 맞는 사례에 국한했다. 느슨한 상호 규칙에서 추가되는 연결은 자동 반영하지 않았으며, 필요하면 검수 CSV를 바탕으로 사용자 승인을 받아야 한다.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote {ANALYSIS}")
    print(f"Wrote {MERGED_REVIEW}")
    print(f"Wrote {PEARSON}")
    print(f"Wrote {SPEARMAN}")
    print(f"Wrote {VIF}")
    print(f"Wrote {COFFEE_MISSING}")
    print(f"Wrote {REPORT}")
    print(f"Robust rows: {len(robust):,}; merged spells: {spell_diagnostics['merged_spells']:,}")


if __name__ == "__main__":
    main()
