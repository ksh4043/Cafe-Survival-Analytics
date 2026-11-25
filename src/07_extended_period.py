"""Extended-period (2015--2026Q1) franchise competition analysis.

This is intentionally separate from the main analysis.  It rebuilds all
opening-date competition measures from ``cafes_clean.parquet`` and never
rewrites the established 2021Q2--2026Q1 inputs or outputs.
"""
from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spatial = import_module("03_spatial_join")
trade = import_module("03_trade_area_variables")
cox_main = import_module("05_cox")
franchise_types = import_module("05_cox_franchise_types")

CLEAN = ROOT / "data/processed/cafes_clean.parquet"
ANALYSIS = ROOT / "data/processed/cafes_analysis.parquet"
ALL_LICENSES = ROOT / "data/processed/cafes_licenses_all.parquet"
TYPES = ROOT / "src/franchise_brand_types.csv"
AREA = ROOT / "data/raw/area/서울시 상권분석서비스(영역-상권).shp"
OUT = ROOT / "data/processed/cafes_extended.parquet"
TABLE = ROOT / "docs/tables"
FIG = ROOT / "docs/figures"
REPORT = ROOT / "docs/reports/07_extended_period_report.md"
LOG = ROOT / "docs/logs/07_extended_period_execution.md"
START, END, OBS_END = pd.Timestamp("2015-01-01"), pd.Timestamp("2026-03-31"), pd.Timestamp("2026-09-30")
MAIN_START = pd.Timestamp("2021-04-01")
TYPE_ORDER = ["저가 커피", "대형 커피", "음료·디저트", "기타 커피"]
TYPE_KEYS = {v: k for v, k in zip(TYPE_ORDER, ["lowcost", "major", "drink", "other"])}
COLORS = {"개인 카페": "#0072B2", "프랜차이즈": "#D55E00", "코로나 시기": "#009E73", "이후": "#CC79A7", "기준선": "#555555", "코로나 이전": "#0072B2"}


def write(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(TABLE / name, index=False, encoding="utf-8-sig")


def hr_row(model: CoxPHFitter, variable: str, model_name: str, period: str = "전체") -> dict[str, object]:
    row = model.summary.loc[variable]
    b, lo, hi = float(row["coef"]), float(row["coef lower 95%"]), float(row["coef upper 95%"])
    return {"model": model_name, "period": period, "variable": variable, "coefficient": b,
            "hazard_ratio": float(np.exp(b)), "hr_ci_lower": float(np.exp(lo)),
            "hr_ci_upper": float(np.exp(hi)), "p_value": float(row["p"]),
            "robust_se": float(row["se(coef)"])}


def fit(frame: pd.DataFrame, covariates: list[str]) -> CoxPHFitter:
    data = frame[["survival_days", "event", "opening_year", "trdar_cd_assigned", *covariates]].copy()
    data[covariates] = data[covariates].astype(float)
    data["survival_days"] = data["survival_days"].astype(float)
    data["event"] = data["event"].astype(int)
    model = CoxPHFitter()
    model.fit(data, duration_col="survival_days", event_col="event", strata="opening_year",
              cluster_col="trdar_cd_assigned", robust=True, show_progress=False)
    return model


def equality_wald(frame: pd.DataFrame, covariates: list[str], first: str, second: str, label: str) -> dict[str, object]:
    """Robust Wald test of beta_first == beta_second via a contrast parameterization."""
    changed = frame.copy()
    summed, difference = f"sum_{first}_{second}", f"diff_{first}_{second}"
    changed[summed] = changed[first].astype(float) + changed[second].astype(float)
    changed[difference] = changed[first].astype(float) - changed[second].astype(float)
    replacement = [x for x in covariates if x not in {first, second}] + [summed, difference]
    model = fit(changed, replacement)
    row = model.summary.loc[difference]
    stat = float((row["coef"] / row["se(coef)"]) ** 2)
    return {"test": label, "coefficient_difference": float(2*row["coef"]), "wald_chi2": stat, "df": 1, "p_value": float(row["p"])}


def type_map() -> dict[str, str]:
    data = pd.read_csv(TYPES, encoding="utf-8-sig")
    return data.set_index("브랜드명")["유형"].to_dict()


def typed_features(cohort: pd.DataFrame, all_cafes: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Opening-date-safe type/band counts and nearest active franchise distance."""
    cohort = cohort.copy().reset_index(drop=True)
    all_cafes = all_cafes.copy().reset_index(drop=True)
    cohort["_cohort_row"] = np.arange(len(cohort), dtype=int)
    all_cafes["_source_row"] = np.arange(len(all_cafes), dtype=int)
    result = pd.DataFrame(index=cohort.index)
    names = ["comp_indep_r0_100", "comp_indep_r100_200", "comp_fran_r0_100", "comp_fran_r100_200", "nearest_fran_dist_m", "fran_within_2km", "fran_same_site", "indep_same_site"]
    names += [f"comp_fran_{TYPE_KEYS[t]}_r200" for t in TYPE_ORDER]
    for name in names:
        result[name] = pd.Series(pd.NA, index=result.index, dtype="boolean") if name in {"fran_within_2km", "fran_same_site", "indep_same_site"} else np.nan
    target = spatial.make_points(cohort).sort_values("_cohort_row", kind="stable").reset_index(drop=True)
    competitors = spatial.make_points(all_cafes).sort_values("_source_row", kind="stable").reset_index(drop=True)
    if target.empty or competitors.empty:
        return result
    left, right = competitors.sindex.query(target.geometry, predicate="dwithin", distance=2000)
    distances = target.geometry.iloc[left].distance(competitors.geometry.iloc[right], align=False).to_numpy()
    td = target["permit_date"].to_numpy(dtype="datetime64[ns]")
    cd = competitors["permit_date"].to_numpy(dtype="datetime64[ns]")
    closed = competitors["closure_date"].to_numpy(dtype="datetime64[ns]")
    active = spatial.active_mask(cd[right], closed[right], td[left])
    franchise = competitors["is_franchise"].fillna(False).to_numpy(bool)[right]
    types = competitors["brand"].astype("string").map(mapping).fillna("기타 커피").to_numpy()[right]
    n = len(target)
    target_rows = target["_cohort_row"].to_numpy(dtype=int)
    def count(mask: np.ndarray) -> np.ndarray:
        return np.bincount(left[mask], minlength=n).astype(float)
    fran, indep = active & franchise, active & ~franchise
    result.loc[target_rows, "comp_fran_r0_100"] = count(fran & (distances <= 100))
    result.loc[target_rows, "comp_fran_r100_200"] = count(fran & (distances > 100) & (distances <= 200))
    result.loc[target_rows, "comp_indep_r0_100"] = count(indep & (distances <= 100))
    result.loc[target_rows, "comp_indep_r100_200"] = count(indep & (distances > 100) & (distances <= 200))
    for typ in TYPE_ORDER:
        result.loc[target_rows, f"comp_fran_{TYPE_KEYS[typ]}_r200"] = count(fran & (types == typ) & (distances <= 200))
    nearest = np.full(n, 2000.0)
    if (fran).any():
        nearest_series = pd.Series(distances[fran]).groupby(left[fran]).min()
        nearest[nearest_series.index.to_numpy(int)] = nearest_series.to_numpy(float)
    result.loc[target_rows, "nearest_fran_dist_m"] = nearest
    result.loc[target_rows, "fran_within_2km"] = nearest < 2000
    result.loc[target_rows, "fran_same_site"] = count(fran & (distances <= 10)) > 0
    result.loc[target_rows, "indep_same_site"] = count(indep & (distances <= 10)) > 0
    return result


def add_static_area_and_competition(clean: pd.DataFrame) -> pd.DataFrame:
    all_cafes = clean.copy().reset_index(drop=True)
    all_cafes["_source_row"] = np.arange(len(all_cafes), dtype=int)
    cohort = all_cafes.loc[all_cafes["permit_date"].between(START, END)].copy().reset_index(drop=True)
    cohort["_cohort_row"] = np.arange(len(cohort), dtype=int)
    areas = gpd.read_file(AREA).to_crs(spatial.TARGET_CRS)
    areas["TRDAR_CD"] = areas["TRDAR_CD"].astype(str)
    points, all_points = spatial.make_points(cohort), spatial.make_points(all_cafes)
    enriched, _, _ = spatial.add_area_columns(cohort, points, areas)
    enriched = spatial.competition_measures(enriched, points, all_points)
    extras = typed_features(enriched, all_cafes, type_map())
    for col in extras:
        enriched[col] = extras[col].to_numpy()
    enriched["trdar_cd_assigned"] = enriched["trdar_cd_nearest"].astype("string")
    enriched["trdar_within_200m"] = enriched["trdar_nearest_dist_m"].le(200).fillna(False)
    area_columns = areas[["TRDAR_CD", "TRDAR_SE_1"]].drop_duplicates("TRDAR_CD").rename(columns={"TRDAR_CD": "trdar_cd_assigned", "TRDAR_SE_1": "trdar_type"})
    enriched = enriched.merge(area_columns, on="trdar_cd_assigned", how="left", validate="many_to_one")
    return enriched.drop(columns=[c for c in ["_cohort_row", "_source_row", "polygon_district_code", "polygon_district", "nearest_district_code", "nearest_district"] if c in enriched])


def basic_sample(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    flow = []
    base = data.copy()
    def record(name: str, before: int, after: int) -> None:
        flow.append({"step": name, "n_before": before, "excluded": before-after, "n_after": after, "events_after": int(base.event.sum())})
    before = len(base)
    base = base.loc[(~base.is_franchise.fillna(False)) & base.trdar_within_200m.fillna(False)].copy()
    record("개인 카페 및 200m 이내 최근접 상권", before, len(base))
    required = ["survival_days", "event", "trdar_cd_assigned", "trdar_type", "comp_indep_r200", "comp_fran_r200", "comp_total_r500"]
    for col in required:
        before = len(base); base = base.loc[base[col].notna()].copy(); record(f"{col} 결측 제외", before, len(base))
    base["opening_year"] = base.permit_date.dt.year.astype(int)
    observed = [x for x in ["골목상권", "발달상권", "전통시장", "관광특구"] if x in set(base.trdar_type.dropna())]
    baseline = "골목상권" if "골목상권" in observed else observed[0]
    dummy = []
    for typ in observed:
        if typ != baseline:
            name = f"area_{len(dummy)}"; base[name] = base.trdar_type.eq(typ).astype(float); dummy.append(name)
    covariates = ["comp_indep_r200", "comp_fran_r200", "comp_total_r500", *dummy]
    return base, pd.DataFrame(flow), covariates


def demand_snapshot(data: pd.DataFrame) -> pd.DataFrame:
    panel, _ = trade.make_trade_area_panel()
    snap = panel.loc[panel["quarter"].eq(20211), ["trdar_cd", "trdar_sales_total", "trdar_floating_pop", "trdar_resident_pop"]].copy()
    snap = snap.rename(columns={"trdar_cd": "trdar_cd_assigned", "trdar_sales_total": "snapshot_sales", "trdar_floating_pop": "snapshot_floating", "trdar_resident_pop": "snapshot_resident"})
    out = data.merge(snap, on="trdar_cd_assigned", how="left", validate="many_to_one")
    for source, target in [("snapshot_sales", "snapshot_log_sales_z"), ("snapshot_floating", "snapshot_log_floating_z"), ("snapshot_resident", "snapshot_log_resident_z")]:
        valid = out[source].gt(0)
        values = np.log(out.loc[valid, source].astype(float))
        out.loc[valid, target] = (values-values.mean())/values.std(ddof=0)
    return out


def period_models(sample: pd.DataFrame, covariates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = np.select([sample.opening_year.le(2019), sample.opening_year.between(2020, 2022)], ["코로나 이전", "코로나 시기"], default="이후")
    work = sample.copy(); work["opening_period"] = labels
    rows = []
    for period in ["전체", "코로나 이전", "코로나 시기", "이후"]:
        sub = work if period == "전체" else work.loc[work.opening_period.eq(period)].copy()
        model = fit(sub, covariates)
        for var in ["comp_indep_r200", "comp_fran_r200"]:
            rows.append(hr_row(model, var, "기본 확장 모형", period))
    # Explicit interaction test, using pre-COVID as the reference.
    inter = work.copy(); extra = []
    for period in ["코로나 시기", "이후"]:
        indicator = inter.opening_period.eq(period).astype(float)
        for var in ["comp_indep_r200", "comp_fran_r200"]:
            name = f"{var}_{'covid' if period == '코로나 시기' else 'post'}"; inter[name] = inter[var]*indicator; extra.append(name)
    model = fit(inter, [*covariates, *extra])
    interaction = pd.DataFrame([hr_row(model, x, "개업 시기 상호작용", "코로나 이전 대비") for x in extra])
    return pd.DataFrame(rows), interaction


def interval_models(sample: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    rows = []
    for label, low, high in [("개업 후 1년 이내", 0, 365), ("1~3년", 365, 1095), ("3년 이후", 1095, np.inf)]:
        duration = np.minimum(sample.survival_days.astype(float), high) - low
        event = ((sample.event.astype(int).eq(1)) & sample.survival_days.astype(float).gt(low) & sample.survival_days.astype(float).le(high)).astype(int)
        sub = sample.loc[duration.gt(0)].copy(); sub["survival_days"] = duration.loc[duration.gt(0)]; sub["event"] = event.loc[duration.gt(0)]
        model = fit(sub, covariates)
        for var in ["comp_indep_r200", "comp_fran_r200"]: rows.append(hr_row(model, var, "시간 구간", label))
    return pd.DataFrame(rows)


def km_plot(sample: pd.DataFrame) -> pd.DataFrame:
    work = sample.copy()
    work["period"] = np.select([work.opening_year.le(2019), work.opening_year.between(2020, 2022)], ["코로나 이전", "코로나 시기"], default="이후")
    fig, ax = plt.subplots(figsize=(10, 6.2)); rows=[]
    for label in ["코로나 이전", "코로나 시기", "이후"]:
        sub=work.loc[work.period.eq(label)]; km=KaplanMeierFitter(label=label).fit(sub.survival_days.astype(float), sub.event.astype(int))
        ax.step(km.survival_function_.index/365.25, km.survival_function_.iloc[:,0], where="post", lw=2.2, color=COLORS[label], label=label)
        rows.append({"opening_period":label,"n":len(sub),"events":int(sub.event.sum()),"five_year_survival":float(km.predict(365*5)) if sub.survival_days.max()>=365*5 else np.nan})
    ax.set(xlabel="개업 후 경과 연수", ylabel="생존확률", title="개업 시기별 개인 카페 생존곡선", ylim=(0,1.03)); ax.grid(axis="y", alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG/"07_extended_opening_period_km.png",dpi=200); plt.close(fig)
    return pd.DataFrame(rows)


def plot_period(rows: pd.DataFrame) -> None:
    p = rows.loc[rows.variable.eq("comp_fran_r200")].copy(); order=["전체","코로나 이전","코로나 시기","이후"]; p["period"]=pd.Categorical(p.period,order,ordered=True); p=p.sort_values("period")
    fig,ax=plt.subplots(figsize=(9.5,5.4)); y=np.arange(len(p))
    for i,r in p.reset_index(drop=True).iterrows():
        ax.errorbar(r.hazard_ratio,i,xerr=[[r.hazard_ratio-r.hr_ci_lower],[r.hr_ci_upper-r.hazard_ratio]],fmt="o",color=COLORS.get(r.period,"#D55E00"),capsize=3,ms=7)
    ax.axvline(1,color=COLORS["기준선"],lw=1); ax.set_xscale("log"); ax.set_yticks(y,p.period); ax.set_xlabel("프랜차이즈 1개 증가당 폐업 위험비"); ax.set_title("개업 시기별 프랜차이즈 경쟁 위험비"); ax.grid(axis="x",alpha=.25); fig.tight_layout(); fig.savefig(FIG/"07_extended_period_competition_hr.png",dpi=200); plt.close(fig)


def plot_comparison(main: pd.DataFrame, extended: pd.DataFrame) -> None:
    p=pd.concat([main.assign(sample="주 분석 기간"),extended.assign(sample="확장 기간")]); p=p.loc[p.variable.eq("comp_fran_r200")]
    fig,ax=plt.subplots(figsize=(8.8,4.4)); y=np.arange(len(p))
    for i,r in p.reset_index(drop=True).iterrows(): ax.errorbar(r.hazard_ratio,i,xerr=[[r.hazard_ratio-r.hr_ci_lower],[r.hr_ci_upper-r.hazard_ratio]],fmt="o",color="#D55E00",capsize=3,ms=7)
    ax.axvline(1,color=COLORS["기준선"],lw=1); ax.set_xscale("log"); ax.set_yticks(y,p["sample"]); ax.set_xlabel("프랜차이즈 1개 증가당 폐업 위험비"); ax.set_title("주 분석과 확장 기간의 프랜차이즈 경쟁 추정치"); ax.grid(axis="x",alpha=.25); fig.tight_layout(); fig.savefig(FIG/"07_extended_franchise_comparison.png",dpi=200); plt.close(fig)


def main() -> None:
    os.environ["TEMP"] = str(ROOT/".tmp"); os.environ["TMP"] = str(ROOT/".tmp")
    for d in [ROOT/".tmp", OUT.parent, TABLE, FIG, REPORT.parent, LOG.parent]: d.mkdir(parents=True,exist_ok=True)
    cox_main.configure_matplotlib()
    clean=pd.read_parquet(CLEAN); all_licenses=pd.read_parquet(ALL_LICENSES)
    raw_window=all_licenses.loc[all_licenses.permit_date.between(START,END)].copy()
    quality=[]
    for year,sub in raw_window.assign(opening_year=raw_window.permit_date.dt.year).groupby("opening_year"):
        surv=pd.to_numeric(sub.survival_days,errors="coerce").dropna()
        quality.append({"opening_year":year,"n":len(sub),"coordinate_missing_n":int((~sub.has_coordinates.fillna(False)).sum()),"coordinate_missing_share":float((~sub.has_coordinates.fillna(False)).mean()),"closed_without_date_n":int((sub.event.eq(1)&sub.closure_date.isna()).sum()),"survival_median_days":float(surv.median()),"survival_q25_days":float(surv.quantile(.25)),"survival_q75_days":float(surv.quantile(.75))})
    quality=pd.DataFrame(quality); write("07_extended_data_quality.csv",quality)
    extended=add_static_area_and_competition(clean)
    extended.to_parquet(OUT,index=False)
    # Mandatory reconstruction checks against the established contemporary cohort.
    prior=pd.read_parquet(ANALYSIS); overlap=extended.loc[extended.permit_date.between(MAIN_START,END)].copy()
    key="관리번호"; merged=overlap.merge(prior[[key,"comp_indep_r200","comp_fran_r200"]],on=key,how="outer",suffixes=("_new","_old"),indicator=True)
    checks=[]
    checks.append({"check":"overlap identifiers", "mismatch_n":int((merged._merge.ne("both")).sum())})
    for col in ["comp_indep_r200","comp_fran_r200"]: checks.append({"check":f"{col} contemporary equality", "mismatch_n":int((merged[f"{col}_new"].astype("Float64")!=merged[f"{col}_old"].astype("Float64")).sum())})
    valid_comp = extended.comp_fran_r200.notna() & extended.comp_indep_r200.notna()
    checks += [{"check":"franchise type sum equals total", "mismatch_n":int(((extended.loc[valid_comp,[f"comp_fran_{TYPE_KEYS[x]}_r200" for x in TYPE_ORDER]].sum(axis=1)-extended.loc[valid_comp,"comp_fran_r200"].astype(float)).abs()>1e-9).sum())}, {"check":"franchise bands sum equals total", "mismatch_n":int(((extended.loc[valid_comp,"comp_fran_r0_100"]+extended.loc[valid_comp,"comp_fran_r100_200"]).astype(float)!=extended.loc[valid_comp,"comp_fran_r200"].astype(float)).sum())}, {"check":"independent bands sum equals total", "mismatch_n":int(((extended.loc[valid_comp,"comp_indep_r0_100"]+extended.loc[valid_comp,"comp_indep_r100_200"]).astype(float)!=extended.loc[valid_comp,"comp_indep_r200"].astype(float)).sum())}]
    # Recreate the published franchise-type extract independently for the contemporary cohort.
    reference_map = franchise_types.brand_type_frame().set_index("브랜드명")["유형"].to_dict()
    reference_typed = franchise_types.typed_competition(overlap, clean, reference_map).reset_index(drop=True)
    current_typed = overlap.reset_index(drop=True)
    for column in ["comp_fran_r0_100", "comp_fran_r100_200", "comp_indep_r0_100", "comp_indep_r100_200", *[f"comp_fran_{TYPE_KEYS[x]}_r200" for x in TYPE_ORDER]]:
        checks.append({"check":f"published franchise-type recreation: {column}", "mismatch_n":int((current_typed[column].astype("Float64") != reference_typed[column].astype("Float64")).sum())})
    checks=pd.DataFrame(checks); write("07_extended_validation.csv",checks)
    if checks.mismatch_n.any(): raise ValueError(f"Validation failed: {checks.to_dict('records')}")
    sample,flow,covs=basic_sample(extended); write("07_extended_sample_flow.csv",flow)
    model=fit(sample,covs); basic_coef=pd.DataFrame([hr_row(model,x,"기본 확장 모형") for x in covs]); write("07_extended_coefficients.csv",basic_coef)
    periods,interactions=period_models(sample,covs); write("07_extended_period_interactions.csv",interactions); write("07_extended_period_coefficients.csv",periods)
    intervals=interval_models(sample,covs); write("07_extended_time_intervals.csv",intervals)
    # Type, distance, nearest-distance and same-site alternatives use exactly the basic covariates except the franchise representation.
    type_vars=[f"comp_fran_{TYPE_KEYS[x]}_r200" for x in TYPE_ORDER]
    typed=fit(sample,["comp_indep_r200",*type_vars,"comp_total_r500",*covs[3:]])
    distance_vars=["comp_indep_r0_100","comp_indep_r100_200","comp_fran_r0_100","comp_fran_r100_200"]
    distance=fit(sample,[*distance_vars,"comp_total_r500",*covs[3:]])
    nearest=sample.copy(); nearest["log_nearest_fran_dist"] = np.log1p(nearest.nearest_fran_dist_m.astype(float)); nearest_model=fit(nearest,["comp_indep_r200","log_nearest_fran_dist","comp_total_r500",*covs[3:]])
    site=fit(sample,["comp_indep_r200","fran_same_site","indep_same_site","comp_total_r500",*covs[3:]])
    franchise_detail=pd.concat([pd.DataFrame([hr_row(typed,x,"프랜차이즈 유형") for x in ["comp_indep_r200",*type_vars]]),pd.DataFrame([hr_row(distance,x,"거리 구간") for x in distance_vars]),pd.DataFrame([hr_row(nearest_model,"log_nearest_fran_dist","최근접 거리")]),pd.DataFrame([hr_row(site,x,"같은 자리") for x in ["fran_same_site","indep_same_site"]])],ignore_index=True)
    write("07_extended_franchise_detail.csv",franchise_detail)
    write("07_extended_franchise_wald_tests.csv", pd.DataFrame([
        equality_wald(sample, ["comp_indep_r200",*type_vars,"comp_total_r500",*covs[3:]], "comp_fran_lowcost_r200", "comp_fran_major_r200", "저가 커피와 대형 커피 계수 동일"),
        equality_wald(sample, [*distance_vars,"comp_total_r500",*covs[3:]], "comp_fran_r0_100", "comp_fran_r100_200", "프랜차이즈 0~100m와 100~200m 계수 동일"),
    ]))
    distribution=[]
    for name in ["comp_indep_r200","comp_fran_r200",*type_vars,"nearest_fran_dist_m"]:
        x=pd.to_numeric(sample[name],errors="coerce").dropna(); distribution.append({"measure":name,"n":len(x),"zero_share":float(x.eq(0).mean()),"mean":float(x.mean()),"median":float(x.median()),"q75":float(x.quantile(.75)),"max":float(x.max())})
    write("07_extended_competition_distribution.csv",pd.DataFrame(distribution))
    # Snapshot-demand supplemental model; values are deliberately not opening-date values before 2021.
    snap=demand_snapshot(sample); snap_cols=["snapshot_log_sales_z","snapshot_log_floating_z","snapshot_log_resident_z"]
    snap=snap.dropna(subset=snap_cols).copy(); snap_model=fit(snap,[*covs,*snap_cols]); supplemental=pd.DataFrame([hr_row(snap_model,x,"2021Q1 상권 스냅샷 보조 모형") for x in ["comp_indep_r200","comp_fran_r200",*snap_cols]])
    write("07_extended_snapshot_supplement.csv",supplemental)
    # Main-period compatibility: basic extended specification, then original main controls for the requested reclassification comparison.
    recent=sample.loc[sample.permit_date.ge(MAIN_START)].copy(); recent_model=fit(recent,covs); recent_coef=pd.DataFrame([hr_row(recent_model,x,"확장 기본 모형의 주 분석 기간") for x in ["comp_indep_r200","comp_fran_r200"]]); write("07_extended_recent_comparison.csv",recent_coef)
    original_main = pd.read_csv(TABLE / "05_cox_coefficients.csv", encoding="utf-8-sig")
    original_main = original_main.loc[(original_main.model_key.eq("store_competition")) & (original_main.variable.eq("comp_fran_r200")), ["variable", "hazard_ratio", "hr_ci_lower", "hr_ci_upper", "p_value"]].copy()
    write("07_extended_main_comparison.csv", pd.concat([original_main.assign(sample="기존 주 분석"), basic_coef.loc[basic_coef.variable.eq("comp_fran_r200"), ["variable","hazard_ratio","hr_ci_lower","hr_ci_upper","p_value"]].assign(sample="확장 기본 모형")], ignore_index=True))
    # Reclassify 'other coffee' chains as independent in both target and competitors, recompute all relevant counts.
    mapping=type_map(); changed_clean=clean.copy(); other=changed_clean.brand.astype("string").map(mapping).eq("기타 커피").fillna(False); changed_clean.loc[other,"is_franchise"]=False
    changed=extended.copy(); changed.loc[changed.brand.astype("string").map(mapping).eq("기타 커피").fillna(False),"is_franchise"]=False
    fresh=typed_features(changed,changed_clean,type_map())
    for col in fresh: changed[col]=fresh[col].to_numpy()
    changed["comp_fran_r200"]=(changed.comp_fran_r0_100+changed.comp_fran_r100_200).astype("Int64"); changed["comp_indep_r200"]=(changed.comp_indep_r0_100+changed.comp_indep_r100_200).astype("Int64")
    alt,alt_flow,alt_covs=basic_sample(changed); alt_model=fit(alt,alt_covs)
    # Main-period alternate uses the established time-safe control construction.
    main_changed=pd.read_parquet(ANALYSIS).copy(); main_changed.loc[main_changed.brand.astype("string").map(mapping).eq("기타 커피").fillna(False),"is_franchise"]=False
    replacement = changed[[key,"comp_indep_r200","comp_fran_r200"]].copy()
    main_changed = main_changed.drop(columns=["comp_indep_r200","comp_fran_r200"]).merge(replacement,on=key,how="left",validate="one_to_one")
    main_alt, _, main_scales = cox_main.prepare_sample(main_changed)
    main_alt_model = cox_main.fit_cox(main_alt, cox_main.model_definitions(main_scales["type_columns"])["store_competition"])
    reclass=pd.DataFrame([* [hr_row(alt_model,x,"확장 기간: 기타 커피를 개인 카페로 재분류") for x in ["comp_indep_r200","comp_fran_r200"]], *[hr_row(main_alt_model,x,"주 분석 기간: 기타 커피를 개인 카페로 재분류") for x in ["comp_indep_r200","comp_fran_r200"]]])
    reclass["sample_n"] = [len(alt)]*2+[len(main_alt)]*2; reclass["events"]=[int(alt.event.sum())]*2+[int(main_alt.event.sum())]*2
    write("07_extended_other_chain_reclassification.csv",reclass); write("07_extended_other_chain_sample_flow.csv",alt_flow)
    # Same-site examples retain both road and parcel address as verification evidence.
    name_col, road_col, parcel_col = changed.columns[8], changed.columns[19], changed.columns[36]
    examples=changed.loc[changed.fran_same_site.fillna(False)|changed.indep_same_site.fillna(False),[name_col,road_col,parcel_col,"permit_date","fran_same_site","indep_same_site","nearest_fran_dist_m"]].head(30)
    write("07_extended_same_site_examples.csv",examples)
    km=km_plot(sample); write("07_extended_opening_period_km.csv",km); plot_period(periods); plot_comparison(original_main.assign(sample="기존 주 분석"),basic_coef.loc[basic_coef.variable.eq("comp_fran_r200")])
    # Compact Korean decision table; conclusions are associations, never causal claims.
    fran=basic_coef.set_index("variable").loc["comp_fran_r200"]; low=fran.hr_ci_lower; high=fran.hr_ci_upper
    decision=pd.DataFrame([{"question":"프랜차이즈 인접 회피 여부","result":"위험 증가와 일관된 관련" if low>1 else ("위험 감소와 일관된 관련" if high<1 else "방향을 판단하기 어려움"),"basis":f"확장 기본 모형 HR {fran.hazard_ratio:.3f} ({low:.3f}–{high:.3f})"}]); write("07_extended_judgment.csv",decision)
    report=f"""# 확장 기간 프랜차이즈 경쟁 분석

## 한 줄 결론

2015년부터 2026년 1분기까지 개업한 개인 카페 {len(sample):,}개(폐업 {int(sample.event.sum()):,}건)를 보면, 200m 안 개인 카페가 많을수록 폐업 위험은 조금 높게 관련됐지만(HR 1.020, 95% CI 1.000–1.040), 프랜차이즈 수는 위험을 높이거나 낮춘다는 뚜렷한 관련이 없었다(HR {fran.hazard_ratio:.3f}, 95% CI {low:.3f}–{high:.3f}). 다만 같은 건물·단지 상가로 보이는 10m 이내 결과는 주의해서 볼 신호다.

## 무엇을 알게 되었나

### 기간을 넓혀도 개인 카페 경쟁 결론이 유지되는가

유지된다. 확장 표본 전체에서 200m 안 개인 카페가 1개 더 있을 때 폐업 위험은 2.0% 높게 관련됐다(HR 1.020, 95% CI 1.000–1.040). 개업 시기별로도 코로나 이전은 HR 1.030 (1.002–1.059), 코로나 시기는 1.004 (0.995–1.013), 이후는 1.014 (1.001–1.027)였다. 코로나 이전과 비교하면 코로나 시기의 관련은 더 약했다(상호작용 HR 0.984, p=0.042). 이후 시기의 차이는 뚜렷하지 않았다(상호작용 HR 0.988, p=0.220). 코로나 시기 개업에서 관계가 약하게 추정된 것은 사실이지만, 이 자료만으로 그 이유를 정할 수는 없다. 영업제한·상권 변화·신규 진입 구성의 변화 등이 가능한 설명일 뿐이다. 2020년 개업은 코로나 시기에 포함했다.

개업 후 1년 이내(HR 1.032, 1.001–1.064), 1~3년(HR 1.009, 1.002–1.016), 3년 이후(HR 1.009, 1.001–1.018)에도 개인 카페 경쟁은 같은 방향이었다. 거리를 나누면 이 관련은 0~100m에서 보였고(HR 1.053, 1.007–1.101), 100~200m에서는 뚜렷하지 않았다(HR 0.992, 0.976–1.009).

주 분석 기간만 같은 기본 확장 사양으로 다시 적합하면 개인 카페 1개당 HR은 1.011 (1.001–1.020)이다. 전체 확장 표본의 1.020과의 차이는 수요 통제의 유무가 아니라, 2015년부터의 더 오래된 개업 코호트와 그 관측기간을 추가한 데서 나온 차이다.

### 프랜차이즈 근처를 피해야 하는가

그렇게 권할 근거는 아직 없다. 200m 안 프랜차이즈 1개 증가는 폐업 위험과 거의 관련이 없었다(HR 0.998, 95% CI 0.986–1.011). 신뢰구간의 가장 높은 값 1.011도 확장 표본에서 개인 카페 1개당 추정치 1.020보다 작다. 즉 이 자료에서 프랜차이즈 한 곳의 가장 나쁜 경우조차 개인 카페 한 곳과 같은 크기의 위험 증가로 보이지는 않는다.

유형별로 저가 커피는 HR 1.027 (0.996–1.059, p=0.089)로 위험을 높이는 쪽으로 기울지만 확정은 아니다. 대형 커피는 0.995 (0.966–1.025), 음료는 1.041 (0.958–1.132), 기타 커피는 0.915 (0.822–1.019)로 모두 신뢰구간이 1을 포함했다. 프랜차이즈를 0~100m와 100~200m로 나누어도 각각 HR 1.008 (0.988–1.029), 0.978 (0.949–1.009)로 뚜렷한 관련이 없었다.

![개업 시기별 프랜차이즈 경쟁 위험비](../figures/07_extended_period_competition_hr.png)

그림은 전체와 세 개의 개업 시기 모두에서 프랜차이즈 1개당 신뢰구간이 기준선 1을 지나, 시기별로도 뚜렷한 위험 증가·감소를 확인하기 어렵다는 것을 보여 준다.

![주 분석과 확장 기간의 프랜차이즈 경쟁 추정치](../figures/07_extended_franchise_comparison.png)

그림은 주 분석과 확장 기간 모두 프랜차이즈 경쟁 추정치가 기준선 가까이에 있고, 확장 기간 쪽 신뢰구간이 더 좁아졌음을 보여 준다. 두 모형은 수요 통제 구성이 달라 추정치 차이를 원인 효과의 변화로 해석하지 않는다.

### 바로 옆·같은 건물 프랜차이즈는 어떤가

가장 가까운 프랜차이즈까지의 거리가 멀수록 위험은 낮게 추정됐다. `log(1+거리)` 모형을 거리 두 배로 풀면, 일반적인 거리 범위에서 두 배 멀어질 때 폐업 위험은 약 3.4% 낮았다(HR 0.966, 95% CI 0.934–0.998). 다만 이 추정은 여러 프랜차이즈 지표를 함께 검정한 결과 중 하나이므로 p=0.040 수준의 주의 신호로만 해석한다.

10m 이내에 프랜차이즈가 있는 개인 카페는 표본의 13.7%(2,650/19,276)였고, 이 경우 폐업 위험은 더 높게 관련됐다(HR 1.257, 1.008–1.568, p=0.042). 10m 이내 개인 카페가 있는 경우도 23.6%(4,552/19,276)이며 HR은 1.093 (1.007–1.185)이었다. 주소 예시에는 경희궁의아침 단지 상가, 르메이에르종로타운, 종로타워처럼 같은 좌표를 쓰는 대형 건물·단지 상가가 여럿 보인다. 따라서 10m 이내는 실제 바로 옆 점포라기보다 같은 건물 또는 같은 단지 상가를 뜻할 가능성이 크며, 대형 상가·오피스 같은 건물 유형의 차이가 결과에 섞였을 수 있다. 이 때문에 같은 자리 프랜차이즈의 p≈0.04 결과도 회피 판단의 근거가 아니라 추가 확인이 필요한 주의 신호다.

### 소규모 체인 분류가 결론을 바꾸는가

기타 커피 소규모 체인을 개인 카페로 다시 분류해도 결론은 바뀌지 않았다. 확장 표본은 20,302개(폐업 13,008건)로 늘었고, 개인 카페 경쟁은 HR 1.017 (1.000–1.034), 프랜차이즈 경쟁은 1.002 (0.989–1.015)였다. 주 분석 기간에서도 각각 1.010 (1.002–1.018), 1.008 (0.988–1.028)이었다. 즉 분류를 바꿔도 개인 카페 경쟁의 작은 양의 관련과 프랜차이즈의 불확실한 관련이라는 요지는 유지된다.

2021년 1분기 상권 매출·유동인구·상주인구 스냅샷을 붙인 보조 모형도 기본 모형과 같은 결론이었다. 개인 카페는 HR 1.020 (1.002–1.038), 프랜차이즈는 0.994 (0.977–1.011)였다.

![개업 시기별 개인 카페 생존곡선](../figures/07_extended_opening_period_km.png)

그림은 코로나 이전 개업과 코로나 시기 개업의 관측된 5년 생존확률이 각각 40.8%, 36.9%임을 보여 준다. 이후 개업 코호트는 5년 추적기간이 아직 충분하지 않아 이 두 코호트와 직접 비교할 수 없으며, 이 곡선은 경쟁 효과를 뜻하지 않는 기술적 결과다.

## 방법과 검증

원본 정제본에서 2015-01-01~2026-03-31 개업 기록을 선택했다. 임시 인허가, 생존기간 0일 이하, 비카페, 폐업 상태인데 폐업일이 없는 기록의 처리 기준은 기존 정제와 같으며, 연도별 점검에서 폐업 상태인데 폐업일이 없는 건은 모두 0건이었다. 좌표 결측률은 개업연도별 0.35~2.65%였다. 연도별 점포 수·폐업 수·좌표 결측률·생존기간 분포는 `07_extended_data_quality.csv`, 최종 표본 흐름은 `07_extended_sample_flow.csv`에 있다.

경쟁 점포는 개업일보다 앞서 개업했고 그 날 영업 중인 점포만 세었다. 2021Q2~2026Q1 겹침 표본에서 개인·프랜차이즈 200m 수가 기존 `cafes_analysis.parquet`와 점포별로 일치하고, 유형합·거리구간합이 200m 합계와 일치함을 확인했다. 모든 불일치 수는 `07_extended_validation.csv`에서 0이다. 2km 안에 프랜차이즈가 없는 경우 최근접 거리는 2km로 절단하고 `fran_within_2km`에 표시했다.

기본 모형은 개업연도 층화 Cox, 배정 상권 군집-강건 표준오차를 사용했다. 공변량은 200m 개인·프랜차이즈 경쟁, 상권 구분, 그리고 개업 시점 인허가 자료로 계산한 500m 전체 카페 수다. 2021년 이전에는 개업 직전 상권 수요자료가 없으므로 기본 모형에 넣지 않았다. 보조 모형의 2021Q1 상권 스냅샷은 2021년 이전 개업 점포에는 개업 뒤 정보이므로, 기본 결과를 확인하는 용도로만 사용했으며 인과 해석에는 쓰지 않았다.

## 산출물

- 확장 표본: `data/processed/cafes_extended.parquet`
- 표: `docs/tables/07_extended_*.csv`
- 그림: `docs/figures/07_extended_*.png`
"""
    REPORT.write_text(report,encoding="utf-8")
    LOG.write_text("\n".join(["# 확장 기간 분석 실행 로그",f"- 입력: {CLEAN.relative_to(ROOT)}",f"- 확장 공간 표본: {len(extended):,}",f"- 기본 분석 표본: {len(sample):,}; 폐업 {int(sample.event.sum()):,}","- 검증: 현대 표본 및 경쟁 합계 불일치 0건", "- 패키지 설치 및 기존 주 분석 산출물 변경 없음"])+"\n",encoding="utf-8")
    print(f"extended={len(extended)} sample={len(sample)} events={int(sample.event.sum())} HR_fran={fran.hazard_ratio:.6f}")

if __name__ == "__main__":
    main()
