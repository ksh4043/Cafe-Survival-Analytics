"""Cox analysis splitting franchise competition by brand type and distance.

This script is deliberately separate from the main Cox analysis.  It recreates
opening-date-safe competition counts from the established spatial inputs, checks
them against the published 200m counts, then reuses the main-analysis sample
preparation and Cox fitting functions.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from importlib import import_module
cox = import_module("05_cox")
spatial = import_module("03_spatial_join")

INPUT_ANALYSIS = ROOT / "data/processed/cafes_analysis.parquet"
INPUT_SPATIAL = ROOT / "data/processed/cafes_spatial.parquet"
INPUT_CLEAN = ROOT / "data/processed/cafes_clean.parquet"
BRANDS = ROOT / "src/franchise_brands.csv"
BRAND_TYPES = ROOT / "src/franchise_brand_types.csv"
TABLE = ROOT / "docs/tables"
FIG = ROOT / "docs/figures"
REVIEW = ROOT / "docs/review/05_franchise_brand_types_review.csv"
REPORT = ROOT / "docs/reports/05_cox_franchise_types_report.md"
LOG = ROOT / "docs/logs/05_cox_franchise_types_execution.md"

TYPE_ORDER = ["저가 커피", "대형 커피", "음료·디저트", "기타 커피"]
TYPE_KEYS = {"저가 커피": "lowcost", "대형 커피": "major", "음료·디저트": "drink", "기타 커피": "other"}
TYPE_COLORS = {"개인 카페": "#0072B2", "저가 커피": "#D55E00", "대형 커피": "#E69F00", "음료·디저트": "#CC79A7", "기타 커피": "#8C564B"}

# The prompt's specified lists are retained verbatim where possible.  Remaining
# brands are classified only by their stated main product/name; uncertain names
# stay in a review extract rather than being promoted to a price segment.
LOWCOST = {"메가MGC커피", "컴포즈커피", "매머드커피", "빽다방", "더벤티", "우지커피", "바나프레소", "백억커피", "텐퍼센트커피", "커스텀커피", "달리는 커피", "벤티프레소", "하이오커피", "이공커피", "하삼동커피", "커피에반하다", "감성커피", "토프레소", "더리터", "카페일리터", "만랩커피"}
MAJOR = {"스타벅스", "투썸플레이스", "이디야", "할리스", "폴바셋", "파스쿠찌", "커피빈", "엔제리너스", "탐앤탐스", "카페베네", "커피베이", "블루보틀", "테라로사", "팀홀튼", "커피명가", "달콤", "아티제"}
DRINK = {"공차", "쥬씨", "디저트39", "요거프레소", "설빙", "배스킨라빈스", "던킨", "시나본", "백미당", "아마스빈", "팔공티", "차백도", "요아정", "과일에반하다 프루타", "고망고", "마이쥬스", "잠바주스", "고디바", "오설록", "밀크티", "크로플덕", "쉬즈베이글", "곤트란쉐리에", "요거트월드", "요거트퍼플", "요거트아이스크림의 정석", "요거트맨", "요고 프로즌 요거트", "아오스요거트", "요미스라이스×요거트", "빌리엔젤", "스무디킹", "마노핀", "국민우유집", "오가다", "마리웨일마카롱", "모노치즈", "사과당", "베러댄와플", "빙동댕", "스위트밀", "아메리칸트레일러", "쥬스식스", "치키차카초코", "텍사스파파크레페", "커피식스쥬스식스", "고디바", "시나본", "오와케이크", "제주스", "차지", "카페차이", "카페오아시아", "타르타르", "행복한베이커리카페", "바나타이거", "빈스앤베리즈"}
AMBIGUOUS = {"하우", "요커", "다시스", "타래퀸", "만월경", "유캔두잇", "카페홈즈", "달다곰이", "떼루와", "모리셔스브라운", "와이블랙", "푸르파파", "피에스타7커피", "카페글렌", "카페일마지오", "카페미뇽", "킷사텐"}


def brand_type_frame() -> pd.DataFrame:
    brands = pd.read_csv(BRANDS, encoding="utf-8-sig").copy()
    name = "브랜드명"
    def classify(value: str) -> tuple[str, str]:
        if value in LOWCOST:
            return "저가 커피", "지시문 저가 커피 목록"
        if value in MAJOR:
            return "대형 커피", "지시문 대형 커피 목록"
        if value in DRINK:
            return "음료·디저트", "지시문 목록 또는 상호상 주력 음료·디저트"
        if value in AMBIGUOUS:
            return "기타 커피", "상호만으로 주력 상품 판단이 애매함; 검수 필요"
        return "기타 커피", "지시문 목록 외 커피 체인; 가격대 추정 없이 기타 커피"
    classified = brands[name].astype(str).map(classify)
    result = pd.DataFrame({"브랜드명": brands[name], "유형": [x[0] for x in classified], "근거": [x[1] for x in classified]})
    if set(result["브랜드명"]) != set(brands[name]) or len(result) != len(brands):
        raise ValueError("Every franchise dictionary brand must receive exactly one type.")
    return result


def typed_competition(cohort: pd.DataFrame, all_cafes: pd.DataFrame, type_map: dict[str, str]) -> pd.DataFrame:
    """Count active competitors using the same temporal rule as competition_measures."""
    cohort = cohort.copy().reset_index(drop=True)
    all_cafes = all_cafes.copy().reset_index(drop=True)
    cohort["_cohort_row"] = np.arange(len(cohort), dtype=int)
    all_cafes["_source_row"] = np.arange(len(all_cafes), dtype=int)
    cohort_points = spatial.make_points(cohort)
    all_points = spatial.make_points(all_cafes)
    result = pd.DataFrame(index=cohort.index)
    columns = ["comp_fran_r0_100", "comp_fran_r100_200", "comp_indep_r0_100", "comp_indep_r100_200"]
    columns += [f"comp_fran_{TYPE_KEYS[t]}_r{r}" for t in TYPE_ORDER for r in (200, 500)]
    for col in columns:
        result[col] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    if cohort_points.empty or all_points.empty:
        return result
    target = cohort_points.sort_values("_cohort_row", kind="stable").reset_index(drop=True)
    competitors = all_points.sort_values("_source_row", kind="stable").reset_index(drop=True)
    left, right = competitors.sindex.query(target.geometry, predicate="dwithin", distance=500)
    distance = target.geometry.iloc[left].distance(competitors.geometry.iloc[right], align=False).to_numpy()
    td = target["permit_date"].to_numpy(dtype="datetime64[ns]")
    cd = competitors["permit_date"].to_numpy(dtype="datetime64[ns]")
    closed = competitors["closure_date"].to_numpy(dtype="datetime64[ns]")
    active = spatial.active_mask(cd[right], closed[right], td[left])
    franchise = competitors["is_franchise"].fillna(False).to_numpy(dtype=bool)[right]
    competitor_type = competitors["brand"].astype("string").map(type_map).fillna("기타 커피").to_numpy()[right]
    n = len(target)
    target_rows = target["_cohort_row"].to_numpy(dtype=int)
    def count(mask: np.ndarray) -> np.ndarray:
        return np.bincount(left[mask], minlength=n).astype("int64")
    # Using >100 for the outer band makes the two bands a non-overlapping partition.
    fran = active & franchise
    indep = active & ~franchise
    result.loc[target_rows, "comp_fran_r0_100"] = count(fran & (distance <= 100))
    result.loc[target_rows, "comp_fran_r100_200"] = count(fran & (distance > 100) & (distance <= 200))
    result.loc[target_rows, "comp_indep_r0_100"] = count(indep & (distance <= 100))
    result.loc[target_rows, "comp_indep_r100_200"] = count(indep & (distance > 100) & (distance <= 200))
    for radius in (200, 500):
        for typ in TYPE_ORDER:
            result.loc[target_rows, f"comp_fran_{TYPE_KEYS[typ]}_r{radius}"] = count(fran & (competitor_type == typ) & (distance <= radius))
    return result


def summary_distribution(sample: pd.DataFrame, columns: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for group, col in columns:
        v = pd.to_numeric(sample[col], errors="coerce").dropna()
        rows.append({"경쟁 구분": group, "변수": col, "유효 표본 수": len(v), "0개 비율": float(v.eq(0).mean()), "평균": float(v.mean()), "중앙값": float(v.median()), "IQR": float(v.quantile(.75)-v.quantile(.25)), "최대": int(v.max())})
    return pd.DataFrame(rows)


def extract(model, variable: str, sample: pd.DataFrame, model_name: str) -> dict[str, object]:
    s = model.summary.loc[variable]
    beta, lo, hi = float(s["coef"]), float(s["coef lower 95%"]), float(s["coef upper 95%"])
    iqr = float(sample[variable].quantile(.75) - sample[variable].quantile(.25))
    return {"모형": model_name, "변수": variable, "계수": beta, "군집 강건 표준오차": float(s["se(coef)"]), "위험비": float(np.exp(beta)), "신뢰구간 하한": float(np.exp(lo)), "신뢰구간 상한": float(np.exp(hi)), "p값": float(s["p"]), "IQR": iqr, "IQR 위험비": float(np.exp(beta*iqr)), "IQR 신뢰구간 하한": float(np.exp(lo*iqr)), "IQR 신뢰구간 상한": float(np.exp(hi*iqr))}


def cluster_robust_covariance(model, frame: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Recover lifelines' cluster sandwich covariance, including off-diagonals.

    lifelines exposes the robust diagonal in ``summary`` but keeps
    ``variance_matrix_`` as the model-based covariance.  The off-diagonals are
    needed for a coefficient-difference Wald test, so this recreates the same
    sandwich estimator used for the displayed cluster-robust standard errors.
    """
    fitted = model._model
    indexed = frame.loc[fitted.durations.index, ["opening_stratum", *covariates]].copy()
    x = (indexed[covariates].astype(float) - fitted._norm_mean) / fitted._norm_std
    strata_index = indexed["opening_stratum"]
    x.index = strata_index
    duration, event, weight = fitted.durations.copy(), fitted.event_observed.copy(), fitted.weights.copy()
    duration.index = strata_index; event.index = strata_index; weight.index = strata_index
    covariance = fitted._compute_sandwich_estimator(x, duration, event, weight)
    return pd.DataFrame(covariance, index=covariates, columns=covariates)


def wald(model, a: str, b: str, label: str, frame: pd.DataFrame, covariates: list[str]) -> dict[str, object]:
    diff = float(model.params_[a] - model.params_[b])
    cov = cluster_robust_covariance(model, frame, covariates)
    var = float(cov.loc[a,a] + cov.loc[b,b] - 2*cov.loc[a,b])
    stat = diff**2 / var if var > 0 else np.nan
    return {"검정": label, "계수 차이": diff, "Wald χ²": stat, "자유도": 1, "p값": float(chi2.sf(stat, 1)) if np.isfinite(stat) else np.nan}


def plot_forest(coefs: pd.DataFrame, out: Path) -> None:
    plot = coefs.loc[coefs["모형"].eq("유형별 개수") & coefs["변수"].isin(["comp_indep_r200", *[f"comp_fran_{TYPE_KEYS[x]}_r200" for x in TYPE_ORDER]])].copy()
    labels = ["개인 카페", *TYPE_ORDER]
    plot["표시"] = labels
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for i, row in plot.reset_index(drop=True).iterrows():
        ax.errorbar(row["위험비"], i, xerr=[[row["위험비"]-row["신뢰구간 하한"]], [row["신뢰구간 상한"]-row["위험비"]]], fmt="o", color=TYPE_COLORS[row["표시"]], capsize=3, markersize=7)
    ax.axvline(1, color="#555555", lw=1)
    ax.set_yticks(y, plot["표시"])
    ax.set_xlabel("폐업 위험비 (주변 점포 1개 증가)")
    ax.set_title("200m 안 경쟁 점포 유형별 폐업 위험비")
    ax.set_xscale("log")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def plot_distance(coefs: pd.DataFrame, out: Path) -> None:
    plot = coefs.loc[coefs["모형"].eq("거리 구간")].copy().reset_index(drop=True)
    labels = ["개인 0–100m", "개인 100–200m", "프랜차이즈 0–100m", "프랜차이즈 100–200m"]
    plot["표시"] = labels
    y = np.arange(len(plot)); fig, ax = plt.subplots(figsize=(9, 5.0))
    for i, row in plot.iterrows():
        color = "#0072B2" if "개인" in row["표시"] else "#D55E00"
        ax.errorbar(row["위험비"], i, xerr=[[row["위험비"]-row["신뢰구간 하한"]], [row["신뢰구간 상한"]-row["위험비"]]], fmt="o", color=color, capsize=3, markersize=7)
    ax.axvline(1, color="#555555", lw=1); ax.set_yticks(y, plot["표시"]); ax.set_xscale("log")
    ax.set_xlabel("폐업 위험비 (주변 점포 1개 증가)"); ax.set_title("거리 구간별 경쟁과 폐업 위험비"); ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def plot_percent(coefs: pd.DataFrame, out: Path) -> None:
    wanted = ["comp_indep_r200", "comp_fran_lowcost_r200", "comp_fran_major_r200", "comp_fran_drink_r200"]
    plot = coefs.loc[(coefs["모형"] == "유형별 개수") & coefs["변수"].isin(wanted)].copy().set_index("변수").loc[wanted].reset_index()
    labels = ["개인 카페", "저가 커피", "대형 커피", "음료·디저트"]
    vals = (plot["위험비"] - 1) * 100; low = (plot["신뢰구간 하한"]-1)*100; high=(plot["신뢰구간 상한"]-1)*100
    fig, ax = plt.subplots(figsize=(10, 5.5)); x=np.arange(len(labels)); colors=[TYPE_COLORS[x] for x in labels]
    ax.bar(x, vals, color=colors, width=.62)
    ax.errorbar(x, vals, yerr=[vals-low, high-vals], fmt="none", color="#333333", capsize=4)
    ax.axhline(0, color="#555555", lw=1); ax.set_xticks(x, labels); ax.set_ylabel("폐업 위험 변화 (%)")
    ax.set_title("주변에 점포 하나 더 있을 때 폐업 위험 변화")
    for i, value in enumerate(vals): ax.text(i, value + (1.2 if value >= 0 else -2), f"{value:+.1f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=11)
    ax.set_ylim(min(low.min()-4, -8), max(high.max()+5, 8)); fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def main() -> None:
    os.environ["TEMP"] = str(ROOT / ".tmp"); os.environ["TMP"] = str(ROOT / ".tmp")
    (ROOT / ".tmp").mkdir(exist_ok=True); TABLE.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True); REVIEW.parent.mkdir(parents=True, exist_ok=True)
    cox.configure_matplotlib()
    types = brand_type_frame(); types.to_csv(BRAND_TYPES, index=False, encoding="utf-8-sig")
    type_map = types.set_index("브랜드명")["유형"].to_dict()
    analysis = pd.read_parquet(INPUT_ANALYSIS); spatial_frame = pd.read_parquet(INPUT_SPATIAL); clean = pd.read_parquet(INPUT_CLEAN)
    # cafes_analysis preserves the spatial cohort order.  Refuse to continue if that invariant is broken.
    identity = ["permit_date", "coordinate_x", "coordinate_y", "is_franchise", "brand"]
    if len(analysis) != len(spatial_frame) or not analysis[identity].reset_index(drop=True).equals(spatial_frame[identity].reset_index(drop=True)):
        raise ValueError("Analysis and spatial cohorts are not identically ordered; competition counts cannot be safely attached.")
    extra = typed_competition(spatial_frame, clean, type_map)
    enriched = analysis.copy()
    for col in extra: enriched[col] = extra[col].array
    checks = pd.DataFrame({
        "검증": ["유형별 200m 합계 = 기존 프랜차이즈 200m", "프랜차이즈 거리 구간 합계 = 기존 프랜차이즈 200m", "개인 카페 거리 구간 합계 = 기존 개인 카페 200m"],
        "불일치 점포 수": [int((extra[[f"comp_fran_{TYPE_KEYS[t]}_r200" for t in TYPE_ORDER]].sum(axis=1).astype("Int64") != analysis["comp_fran_r200"]).sum()), int(((extra["comp_fran_r0_100"]+extra["comp_fran_r100_200"]).astype("Int64") != analysis["comp_fran_r200"]).sum()), int(((extra["comp_indep_r0_100"]+extra["comp_indep_r100_200"]).astype("Int64") != analysis["comp_indep_r200"]).sum())]
    })
    checks.to_csv(TABLE / "05_cox_franchise_types_count_validation.csv", index=False, encoding="utf-8-sig")
    if checks["불일치 점포 수"].any(): raise ValueError(f"Competition-count validation failed: {checks.to_dict('records')}")
    sample, flow, scales = cox.prepare_sample(enriched)
    types_cols = scales["type_columns"]
    base = ["trdar_coffee_stores", cox.COFFEE_MISSING, *cox.DEMAND, "log_resident_pop", *types_cols]
    type_vars = [f"comp_fran_{TYPE_KEYS[t]}_r200" for t in TYPE_ORDER]
    model_type = cox.fit_cox(sample, [*base, "comp_indep_r200", *type_vars])
    presence_vars = []
    for typ, var in zip(TYPE_ORDER, type_vars):
        present = f"has_fran_{TYPE_KEYS[typ]}_r200"; sample[present] = sample[var].gt(0).astype(float); presence_vars.append(present)
    model_presence = cox.fit_cox(sample, [*base, "comp_indep_r200", *presence_vars])
    distance_vars = ["comp_indep_r0_100", "comp_indep_r100_200", "comp_fran_r0_100", "comp_fran_r100_200"]
    model_distance = cox.fit_cox(sample, [*base, *distance_vars])
    model_low_other = cox.fit_cox(sample, [*base, "comp_indep_r200", "comp_fran_lowcost_r200", "comp_fran_major_r200", "comp_fran_drink_r200", "comp_fran_other_r200"])
    sample["comp_fran_lowcost_plus_other_r200"] = sample["comp_fran_lowcost_r200"] + sample["comp_fran_other_r200"]
    sample["comp_fran_major_plus_other_r200"] = sample["comp_fran_major_r200"] + sample["comp_fran_other_r200"]
    model_low_plus = cox.fit_cox(sample, [*base, "comp_indep_r200", "comp_fran_lowcost_plus_other_r200", "comp_fran_major_r200", "comp_fran_drink_r200"])
    model_major_plus = cox.fit_cox(sample, [*base, "comp_indep_r200", "comp_fran_lowcost_r200", "comp_fran_major_plus_other_r200", "comp_fran_drink_r200"])
    type_500 = [f"comp_fran_{TYPE_KEYS[t]}_r500" for t in TYPE_ORDER]
    model_500 = cox.fit_cox(sample, [*base, "comp_indep_r200", *type_500])
    sample["lowcost_x_log_floating"] = sample["comp_fran_lowcost_r200"] * sample["log_floating_z"]
    sample["major_x_log_floating"] = sample["comp_fran_major_r200"] * sample["log_floating_z"]
    model_interaction = cox.fit_cox(sample, [*base, "comp_indep_r200", *type_vars, "lowcost_x_log_floating", "major_x_log_floating"])
    groups = [("개인 카페 200m", "comp_indep_r200"), *[(f"{t} 200m", f"comp_fran_{TYPE_KEYS[t]}_r200") for t in TYPE_ORDER], ("프랜차이즈 0–100m", "comp_fran_r0_100"), ("프랜차이즈 100–200m", "comp_fran_r100_200"), ("개인 카페 0–100m", "comp_indep_r0_100"), ("개인 카페 100–200m", "comp_indep_r100_200")]
    dist = summary_distribution(sample, groups); dist.to_csv(TABLE / "05_cox_franchise_types_count_distribution.csv", index=False, encoding="utf-8-sig")
    coef_rows=[]
    for var in ["comp_indep_r200", *type_vars]: coef_rows.append(extract(model_type,var,sample,"유형별 개수"))
    for var in ["comp_indep_r200", *presence_vars]: coef_rows.append(extract(model_presence,var,sample,"유형 있음·없음"))
    for var in distance_vars: coef_rows.append(extract(model_distance,var,sample,"거리 구간"))
    coefs=pd.DataFrame(coef_rows); coefs.to_csv(TABLE / "05_cox_franchise_types_coefficients.csv",index=False,encoding="utf-8-sig")
    type_covariates = [*base, "comp_indep_r200", *type_vars]
    distance_covariates = [*base, *distance_vars]
    walds=pd.DataFrame([wald(model_type,"comp_fran_lowcost_r200","comp_fran_major_r200","저가 커피와 대형 커피 계수 동일",sample,type_covariates), wald(model_distance,"comp_fran_r0_100","comp_fran_r100_200","프랜차이즈: 0–100m와 100–200m 계수 동일",sample,distance_covariates), wald(model_distance,"comp_indep_r0_100","comp_indep_r100_200","개인 카페: 0–100m와 100–200m 계수 동일",sample,distance_covariates)]); walds.to_csv(TABLE / "05_cox_franchise_types_wald_tests.csv",index=False,encoding="utf-8-sig")
    robust=[]
    for name, model, vars_ in [("기타 커피→저가 커피",model_low_plus,["comp_fran_lowcost_plus_other_r200","comp_fran_major_r200","comp_fran_drink_r200"]),("기타 커피→대형 커피",model_major_plus,["comp_fran_lowcost_r200","comp_fran_major_plus_other_r200","comp_fran_drink_r200"]),("500m 유형별",model_500,type_500),("유동인구 상호작용",model_interaction,["lowcost_x_log_floating","major_x_log_floating"])]:
        for var in vars_: robust.append(extract(model,var,sample,name))
    pd.DataFrame(robust).to_csv(TABLE / "05_cox_franchise_types_robustness.csv",index=False,encoding="utf-8-sig")
    main_old=pd.read_csv(TABLE / "05_cox_coefficients.csv",encoding="utf-8-sig"); old=float(main_old.loc[(main_old.model_key=="store_competition")&(main_old.variable=="comp_indep_r200"),"coefficient"].iloc[0]); new=float(model_type.params_["comp_indep_r200"])
    judgments=[]
    personal=coefs.loc[(coefs["모형"]=="유형별 개수")&(coefs["변수"]=="comp_indep_r200")].iloc[0]
    for typ,var in zip(TYPE_ORDER,type_vars):
        r=coefs.loc[(coefs["모형"]=="유형별 개수")&(coefs["변수"]==var)].iloc[0]
        conclusion="위험 증가와 연관" if r["신뢰구간 하한"]>1 else ("위험 감소와 연관" if r["신뢰구간 상한"]<1 else "연관 방향을 판단하기 어려움")
        if r["신뢰구간 상한"]>1 and personal["위험비"]>1: equivalent=np.log(r["신뢰구간 상한"])/np.log(personal["위험비"])
        else: equivalent=np.nan
        judgments.append({"유형":typ,"판단":conclusion,"1개 위험비":r["위험비"],"95% 신뢰구간":f"{r['신뢰구간 하한']:.3f}–{r['신뢰구간 상한']:.3f}","개인 카페 1개 대비 최악 경우 배수":equivalent})
    judgment=pd.DataFrame(judgments); judgment.to_csv(TABLE / "05_cox_franchise_types_judgment.csv",index=False,encoding="utf-8-sig")
    analysis_start, analysis_end = pd.Timestamp("2021-04-01"), pd.Timestamp("2026-03-31")
    active_during_period = clean["permit_date"].lt(analysis_end) & (clean["closure_date"].isna() | clean["closure_date"].ge(analysis_start))
    franchise_counts=clean.loc[clean["is_franchise"].fillna(False) & active_during_period,"brand"].astype("string").value_counts()
    review=types.copy(); review["분석 기간 경쟁 점포 수"] = review["브랜드명"].map(franchise_counts).fillna(0).astype(int); review=review.loc[review["근거"].str.contains("검수 필요") | review["분석 기간 경쟁 점포 수"].rank(method="min",ascending=False).le(30)].sort_values(["분석 기간 경쟁 점포 수","브랜드명"],ascending=[False,True]); review.to_csv(REVIEW,index=False,encoding="utf-8-sig")
    plot_forest(coefs,FIG / "05_cox_franchise_types_forest.png"); plot_distance(coefs,FIG / "05_cox_franchise_types_distance.png"); plot_percent(coefs,FIG / "05_cox_franchise_types_presentation.png")
    t=coefs.loc[coefs["모형"]=="유형별 개수"].copy().set_index("변수")
    type_text="; ".join([f"{typ} {t.loc[var,'위험비']:.3f} ({t.loc[var,'신뢰구간 하한']:.3f}–{t.loc[var,'신뢰구간 상한']:.3f})" for typ,var in zip(TYPE_ORDER,type_vars)])
    low_major_wald = walds.iloc[0]
    fran_distance_wald = walds.iloc[1]
    indep_distance_wald = walds.iloc[2]
    presence_other = coefs.loc[(coefs["모형"] == "유형 있음·없음") & (coefs["변수"] == "has_fran_other_r200")].iloc[0]
    interaction_major = pd.DataFrame(robust).loc[lambda x: x["변수"].eq("major_x_log_floating")].iloc[0]
    report=f"""# 프랜차이즈 유형·거리별 경쟁 Cox 분석

## 무엇을 알 수 있나

200m 안 프랜차이즈를 저가 커피·대형 커피·음료·디저트·기타 커피로 나누어도, 각 유형과 개인 카페 폐업 위험의 연관은 신뢰구간만으로 확정하기 어려웠다. 유형별 1개당 위험비(95% 신뢰구간)는 {type_text}이다. 따라서 기존에 프랜차이즈 전체로 연관이 뚜렷하지 않았던 이유를 **유형별 효과가 서로 지워졌기 때문이라고 확인할 근거는 없다.**

예비 창업자가 이 자료만으로 프랜차이즈 근처를 일괄적으로 피해야 한다고 말할 수는 없다. 반대로 안전하다고도 말할 수 없다. 이 분석은 같은 시점·입지 특성이 비슷한 점포들에서의 연관이며, 프랜차이즈가 폐업을 일으킨다는 인과 결론은 아니다.

개인 카페 경쟁의 계수는 주 분석 {old:.4f}에서 유형 모형 {new:.4f}로 변했고, 1개당 위험비는 1.012(1.004–1.020)였다. 저가 커피와 대형 커피의 계수가 같다는 가설도 기각하지 못했다(Wald p={low_major_wald['p값']:.3f}). 즉 두 유형 사이의 차이 역시 이 자료에서는 뚜렷하지 않다.

0개인 비율은 음료·디저트 72.3%, 기타 커피 70.6%로 높았다. 그래서 유형별로 한 곳이라도 있는지를 비교한 모형도 함께 적합했다. 이 보조 모형에서는 기타 커피가 하나 이상 있는 경우만 위험비 0.856(0.770–0.952)로 나타났지만, 개수 모형의 신뢰구간은 1을 포함한다. 다른 유형·거리·500m 반경 결과와 함께 볼 때, 이를 특정 유형이 위험을 낮춘다는 확정 결론으로 해석하지 않았다.

프랜차이즈는 0–100m의 위험비 1.022(0.990–1.055), 100–200m의 위험비 0.994(0.968–1.020)였고 구간 간 차이도 뚜렷하지 않았다(Wald p={fran_distance_wald['p값']:.3f}). 개인 카페는 가까운 0–100m에서만 1.028(1.006–1.051)로 연관됐으며, 두 거리 구간의 차이는 관찰됐다(p={indep_distance_wald['p값']:.3f}).

![유형별 위험비](../figures/05_cox_franchise_types_forest.png)

그림은 200m 안의 경쟁 점포가 한 곳 더 있을 때의 추정 위험비와 불확실성을 보여 준다.

![거리 구간 위험비](../figures/05_cox_franchise_types_distance.png)

그림은 바로 옆과 100–200m 구간의 경쟁 연관을 비교한다.

![발표용 경쟁 유형 그림](../figures/05_cox_franchise_types_presentation.png)

그림은 주변 점포가 하나 늘어날 때의 폐업 위험 변화 추정치를 쉽게 비교한다.

## 방법과 검증

주 분석과 동일하게 개인 카페·200m 이내 상권 배정의 완전 사례 표본을 사용하고, 개업연도 층화·배정 상권 군집 강건 표준오차·상권 커피 점포 수와 결측 표시·로그 총매출·로그 유동인구·로그 상주인구·상권 구분을 통제했다. 최종 표본은 {len(sample):,}개, 사건은 {int(sample['event'].sum()):,}건이다.

프랜차이즈 경쟁은 개업일보다 앞서 개업했고 그날 영업 중인 점포만 세어 미래 정보를 쓰지 않았다. 네 유형 200m 합계, 프랜차이즈 거리 구간 합계, 개인 카페 거리 구간 합계는 기존 200m 변수와 점포별로 모두 일치했다. 검증 표는 `../tables/05_cox_franchise_types_count_validation.csv`에 있다.

브랜드 분류는 지시문 명시 목록을 우선 적용했다. 목록 밖 커피 체인은 가격대를 추측하지 않고 기타 커피로 두었고, 상호만으로 주력 상품 판단이 애매한 브랜드는 검수 필요으로 표시했다. 전체 분류와 검수 대상은 `src/franchise_brand_types.csv`, `../review/05_franchise_brand_types_review.csv`에 있다.

기타 커피를 저가 또는 대형 커피에 합친 민감도 분석과 500m 반경 분석에서도 어떤 유형의 95% 신뢰구간도 1보다 완전히 크거나 작지 않았다. 유형별 경쟁과 로그 유동인구의 상호작용에서는 대형 커피 상호작용 항의 위험비 0.950(0.905–0.998)이었으나, 저가 커피 항은 0.997(0.961–1.035)이었다. 이는 보조 분석의 한 계수이며, 다중 비교와 관찰자료의 한계 때문에 유동인구가 대형 체인의 연관을 바꾼다고 단정하지 않았다.

상세 수치: 개수 분포 `../tables/05_cox_franchise_types_count_distribution.csv`, 계수 `../tables/05_cox_franchise_types_coefficients.csv`, Wald 검정 `../tables/05_cox_franchise_types_wald_tests.csv`, 강건성 결과 `../tables/05_cox_franchise_types_robustness.csv`, 창업자 판단 요약 `../tables/05_cox_franchise_types_judgment.csv`.
"""
    REPORT.write_text(report,encoding="utf-8")
    LOG.write_text("\n".join(["# 프랜차이즈 유형·거리 Cox 실행 로그","",f"- 입력: cafes_analysis {len(analysis):,}행, cafes_clean {len(clean):,}행",f"- 최종 표본: {len(sample):,}행, 사건 {int(sample['event'].sum()):,}건","- 검증: 유형 및 거리 구간 합계가 기존 200m 경쟁 변수와 점포별 일치","- Wald 검정: lifelines의 군집 샌드위치 공분산(대각·비대각)을 재구성해 사용","- 패키지 설치·원본 데이터 수정 없음","- 실행: .venv\\Scripts\\python.exe src\\05_cox_franchise_types.py",""]),encoding="utf-8")

if __name__ == "__main__": main()
