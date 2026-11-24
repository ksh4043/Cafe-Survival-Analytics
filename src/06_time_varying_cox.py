"""Time-varying Cox analyses for the Seoul independent-cafe cohort.

Run from the repository root:
    .venv\\Scripts\\python.exe src\\06_time_varying_cox.py

Each quarterly start-stop interval is fitted as a delayed-entry row with
``CoxPHFitter(duration_col="stop", entry_col="start")``. This uses the same
partial likelihood as ``CoxTimeVaryingFitter`` (checked on the main model and
written to the log), and adds the assigned-trade-area clustered sandwich
covariance that the time-varying fitter does not provide.

The narrative report (docs/reports/06_time_varying_cox_report.md) is written
by hand from the tables this script produces.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.fitters.coxph_fitter import SemiParametricPHFitter
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/cafes_quarterly_panel.parquet"
TABLE_DIR = ROOT / "docs/tables"
FIGURE_DIR = ROOT / "docs/figures"
LOG = ROOT / "docs/logs/06_time_varying_cox_execution.md"
TYPE_BASELINE = "골목상권"
TYPE_ORDER = ["골목상권", "발달상권", "전통시장", "관광특구"]
CLUSTER = "trdar_cd_assigned"

MODEL_LABELS = {
    "time_varying_store": "시변 상권 + 점포 단위 경쟁",
    "time_varying_competition": "시변 경쟁만",
    "time_varying_competition_demand": "시변 경쟁 + 수요",
    "time_varying_interaction": "시변 상호작용",
    "opening_change": "개업 시점 경쟁 + 개업 후 변화량",
    "cutoff_store": "2026-03-31 절단: 시변 상권 + 점포 단위 경쟁",
    "cutoff_interaction": "2026-03-31 절단: 시변 상호작용",
    "radius_100_store": "100m: 시변 상권 + 점포 단위 경쟁",
    "radius_100_interaction": "100m: 시변 상호작용",
    "radius_500_store": "500m: 시변 상권 + 점포 단위 경쟁",
    "radius_500_interaction": "500m: 시변 상호작용",
    "carried_control": "이월 표시 통제: 시변 상권 + 점포 단위 경쟁",
    "rent_store_without": "임대료 연결 표본: 임대료 제외",
    "rent_store_with": "임대료 연결 표본: 로그 소규모 임대료 포함",
    "rent_interaction_without": "임대료 연결 표본: 상호작용·임대료 제외",
    "rent_interaction_with": "임대료 연결 표본: 상호작용·로그 소규모 임대료 포함",
}


def configure_matplotlib() -> None:
    font_path = Path(r"C:\Windows\Fonts\malgun.ttf")
    if not font_path.exists():
        raise RuntimeError("맑은 고딕 글꼴을 찾을 수 없습니다.")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["font.size"] = 12
    # Must follow the font setting: Malgun Gothic has no U+2212 glyph.
    plt.rcParams["axes.unicode_minus"] = False


def plain_log_axis(axis, lower: float, upper: float, which: str = "x") -> None:
    """Log-scaled hazard-ratio axis with plain decimal ticks (no mathtext)."""
    candidates = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0]
    ticks = [t for t in candidates if lower * 0.98 <= t <= upper * 1.02]
    target = axis.xaxis if which == "x" else axis.yaxis
    (axis.set_xscale if which == "x" else axis.set_yscale)("log")
    target.set_major_locator(FixedLocator(ticks))
    target.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    target.set_minor_locator(NullLocator())
    target.set_minor_formatter(NullFormatter())
    (axis.set_xlim if which == "x" else axis.set_ylim)(lower, upper)


def label_for(variable: str) -> str:
    labels = {
        "trdar_coffee_stores": "상권 커피 점포 수 (1개)",
        "trdar_coffee_stores_missing": "상권 커피 점포 수 결측 표시",
        "log_sales_z": "로그 총매출 (1표준편차)",
        "log_floating_z": "로그 유동인구 (1표준편차)",
        "log_resident_pop": "로그 상주인구 (1 단위)",
        "comp_indep_r200": "200m 개인 카페 경쟁 (1개)",
        "comp_fran_r200": "200m 프랜차이즈 경쟁 (1개)",
        "comp_indep_r100": "100m 개인 카페 경쟁 (1개)",
        "comp_fran_r100": "100m 프랜차이즈 경쟁 (1개)",
        "comp_indep_r500": "500m 개인 카페 경쟁 (1개)",
        "comp_fran_r500": "500m 프랜차이즈 경쟁 (1개)",
        "comp_total_x_sales": "전체 경쟁×로그 총매출",
        "comp_total_x_floating": "전체 경쟁×로그 유동인구",
        "comp_indep_open": "개업 시점 개인 카페 경쟁 (1개)",
        "comp_fran_open": "개업 시점 프랜차이즈 경쟁 (1개)",
        "comp_indep_change": "개업 후 개인 카페 경쟁 변화 (1개)",
        "comp_fran_change": "개업 후 프랜차이즈 경쟁 변화 (1개)",
        "log_rent_small": "직전 분기 로그 소규모 상가 임대료 (1 단위)",
        "type_발달상권": "발달상권 (골목상권 대비)",
        "type_전통시장": "전통시장 (골목상권 대비)",
        "type_관광특구": "관광특구 (골목상권 대비)",
        "trdar_sales_carried": "매출 이월 표시",
        "trdar_stores_carried": "점포 이월 표시",
        "trdar_population_carried": "인구 이월 표시",
    }
    for radius in (100, 500):
        labels[f"comp_total_r{radius}_x_sales"] = f"{radius}m 전체 경쟁×로그 총매출"
        labels[f"comp_total_r{radius}_x_floating"] = f"{radius}m 전체 경쟁×로그 유동인구"
    return labels.get(variable, variable)


def type_dummies(frame: pd.DataFrame) -> list[str]:
    observed = set(frame["trdar_type"].dropna())
    unexpected = observed.difference(TYPE_ORDER)
    if unexpected or TYPE_BASELINE not in observed:
        raise ValueError(f"상권 구분 기준 범주를 확인할 수 없습니다: {sorted(unexpected)}")
    columns = []
    for area_type in TYPE_ORDER[1:]:
        if area_type in observed:
            column = f"type_{area_type}"
            frame[column] = frame["trdar_type"].eq(area_type).astype(float)
            columns.append(column)
    return columns


def prepare_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], list[str]]:
    panel = pd.read_parquet(INPUT).copy()
    panel["license_id"] = panel["license_id"].astype(str)
    panel[CLUSTER] = panel[CLUSTER].astype(str)
    panel["interval_start_date"] = pd.to_datetime(panel["interval_start_date"])
    panel["interval_stop_date"] = pd.to_datetime(panel["interval_stop_date"])
    base = panel.loc[panel["is_main_cox_sample"].fillna(False)].copy()

    def events(frame: pd.DataFrame) -> int:
        return int(frame.groupby("license_id")["event_interval"].max().sum())

    flow = [{"step": "시변 패널의 Cox 주 분석 표본 표시", "stores_before": panel["license_id"].nunique(), "excluded_stores": panel["license_id"].nunique() - base["license_id"].nunique(), "reason": "개인 카페·200m 이내 상권·개업 시점 주 분석 조건", "stores_after": base["license_id"].nunique(), "events_after": events(base)}]
    required = ["comp_indep_r200", "comp_fran_r200", "trdar_sales_total", "trdar_floating_pop", "trdar_resident_pop", "trdar_type", CLUSTER, "opening_year"]
    missing_mask = base[required].isna().any(axis=1)
    excluded_detail = base.loc[missing_mask, ["license_id", "interval_number", "interval_quarter", "trdar_quarter_prior", CLUSTER, "event"]].copy()
    excluded_detail["missing_variables"] = base.loc[missing_mask, required].isna().apply(lambda row: ";".join(row.index[row]), axis=1)
    invalid_ids = excluded_detail["license_id"].unique()
    if len(invalid_ids):
        before = base["license_id"].nunique()
        base = base.loc[~base["license_id"].isin(invalid_ids)].copy()
        flow.append({"step": "구간별 필수 상권값 결측 점포 제외", "stores_before": before, "excluded_stores": len(invalid_ids), "reason": "후속 구간의 상권값 결측(세부는 06_tv_cox_excluded_intervals.csv)", "stores_after": base["license_id"].nunique(), "events_after": events(base)})
    for column in ["trdar_sales_total", "trdar_floating_pop", "trdar_resident_pop"]:
        invalid_ids = base.loc[pd.to_numeric(base[column], errors="coerce").le(0), "license_id"].unique()
        if len(invalid_ids):
            before = base["license_id"].nunique()
            base = base.loc[~base["license_id"].isin(invalid_ids)].copy()
            flow.append({"step": f"{column} 0 이하 점포 제외", "stores_before": before, "excluded_stores": len(invalid_ids), "reason": "자연로그 변환 불가", "stores_after": base["license_id"].nunique(), "events_after": events(base)})
    base["trdar_coffee_stores_missing"] = base["trdar_coffee_stores"].isna().astype(float)
    for column in ["trdar_coffee_stores", "trdar_coffee_franchise_stores"]:
        base[column] = pd.to_numeric(base[column], errors="coerce").fillna(0.0)
    opening = base.loc[base["interval_number"].eq(1)].copy()
    scales: dict[str, object] = {}
    for raw, target in [("trdar_sales_total", "log_sales_z"), ("trdar_floating_pop", "log_floating_z")]:
        logged = np.log(opening[raw].astype(float))
        mean, sd = float(logged.mean()), float(logged.std(ddof=0))
        if not np.isfinite(sd) or sd == 0:
            raise ValueError(f"{raw}의 개업 시점 표준편차가 유효하지 않습니다.")
        base[target] = (np.log(base[raw].astype(float)) - mean) / sd
        scales[f"{target}_mean"] = mean
        scales[f"{target}_sd"] = sd
    base["log_resident_pop"] = np.log(base["trdar_resident_pop"].astype(float))
    base["opening_stratum"] = base["opening_year"].astype(int).replace({2026: 2025})
    scales["stratum_rule"] = "2026년 사건이 30건 미만이어서 2025년과 합침"
    type_columns = type_dummies(base)
    for radius in [100, 200, 500]:
        base[f"comp_total_r{radius}"] = base[f"comp_indep_r{radius}"].astype(float) + base[f"comp_fran_r{radius}"].astype(float)
    base["comp_total_x_sales"] = base["comp_total_r200"] * base["log_sales_z"]
    base["comp_total_x_floating"] = base["comp_total_r200"] * base["log_floating_z"]
    for radius in [100, 500]:
        base[f"comp_total_r{radius}_x_sales"] = base[f"comp_total_r{radius}"] * base["log_sales_z"]
        base[f"comp_total_r{radius}_x_floating"] = base[f"comp_total_r{radius}"] * base["log_floating_z"]
    first = base.loc[base["interval_number"].eq(1), ["license_id", "comp_indep_r200", "comp_fran_r200"]].set_index("license_id")
    base["comp_indep_open"] = base["license_id"].map(first["comp_indep_r200"])
    base["comp_fran_open"] = base["license_id"].map(first["comp_fran_r200"])
    base["comp_indep_change"] = base["comp_indep_r200"] - base["comp_indep_open"]
    base["comp_fran_change"] = base["comp_fran_r200"] - base["comp_fran_open"]
    for column in ["trdar_sales_carried", "trdar_stores_carried", "trdar_population_carried"]:
        base[column] = base[column].astype("boolean").fillna(False).astype(float)
    flow.append({"step": "최종 시변 Cox 표본", "stores_before": base["license_id"].nunique(), "excluded_stores": 0, "reason": "개업 시점 기준값으로 로그·표준화 및 상권 구분 더미 생성", "stores_after": base["license_id"].nunique(), "events_after": events(base), "intervals": len(base)})
    return base.sort_values(["license_id", "start"]), pd.DataFrame(flow), excluded_detail, scales, type_columns


def definitions(type_columns: list[str], radius: int = 200) -> dict[str, list[str]]:
    competition = [f"comp_indep_r{radius}", f"comp_fran_r{radius}"]
    trade = ["trdar_coffee_stores", "trdar_coffee_stores_missing", "log_sales_z", "log_floating_z", "log_resident_pop", *type_columns]
    store = [*trade, *competition]
    suffix = "" if radius == 200 else f"_r{radius}"
    interaction = [*store, f"comp_total{suffix}_x_sales", f"comp_total{suffix}_x_floating"]
    return {
        "time_varying_store": store,
        "time_varying_competition": [*competition, *type_columns],
        "time_varying_competition_demand": [*competition, *type_columns, "log_sales_z", "log_floating_z", "log_resident_pop"],
        "time_varying_interaction": interaction,
        "opening_change": [*trade, "comp_indep_open", "comp_fran_open", "comp_indep_change", "comp_fran_change"],
    }


class RobustFit:
    """Delayed-entry stratified Cox with trade-area clustered sandwich covariance."""

    def __init__(self, frame: pd.DataFrame, covariates: list[str]):
        data = frame.loc[:, ["start", "stop", "event_interval", "opening_stratum", CLUSTER, *covariates]].copy()
        data[covariates] = data[covariates].astype(float)
        data[["start", "stop"]] = data[["start", "stop"]].astype(float)
        data["event_interval"] = data["event_interval"].astype(int)
        model = CoxPHFitter(penalizer=0.0)
        # lifelines exposes robust SEs but not the full sandwich matrix. Keep the
        # matrix it computes internally (for linear contrasts) instead of
        # recomputing it, and confirm that its diagonal matches lifelines.
        captured: list[np.ndarray] = []
        original = SemiParametricPHFitter._compute_sandwich_estimator

        def capture(fitter, *args, **kwargs):
            result = original(fitter, *args, **kwargs)
            captured.append(np.asarray(result))
            return result

        SemiParametricPHFitter._compute_sandwich_estimator = capture
        try:
            model.fit(data, duration_col="stop", entry_col="start", event_col="event_interval", strata="opening_stratum", cluster_col=CLUSTER, robust=True, show_progress=False)
            _ = model.standard_errors_
        finally:
            SemiParametricPHFitter._compute_sandwich_estimator = original
        if len(captured) != 1:
            raise AssertionError("군집 강건 공분산을 가져오지 못했습니다.")
        covariance = captured[0]
        if not np.allclose(np.sqrt(np.diag(covariance)), model.standard_errors_.to_numpy()):
            raise AssertionError("재구성한 군집 강건 공분산이 lifelines 표준오차와 다릅니다.")
        self.model = model
        self.params = model.params_
        self.covariance = pd.DataFrame(covariance, index=model.params_.index, columns=model.params_.index)
        self.concordance = float(model.concordance_index_)

    def contrast(self, weights: dict[str, float]) -> dict[str, float]:
        vector = pd.Series(0.0, index=self.params.index)
        for name, weight in weights.items():
            vector[name] = weight
        beta = float(vector @ self.params)
        se = float(np.sqrt(max(float(vector @ self.covariance @ vector), 0.0)))
        p_value = float(chi2.sf((beta / se) ** 2, 1)) if se > 0 else np.nan
        return {"coefficient": beta, "se": se, "hazard_ratio": float(np.exp(beta)), "hr_ci_lower": float(np.exp(beta - 1.96 * se)), "hr_ci_upper": float(np.exp(beta + 1.96 * se)), "p_value": p_value}


def equivalence_check(frame: pd.DataFrame, covariates: list[str], robust: RobustFit) -> tuple[float, float]:
    data = frame.loc[:, ["license_id", "start", "stop", "event_interval", "opening_stratum", *covariates]].copy()
    data[covariates] = data[covariates].astype(float)
    reference = CoxTimeVaryingFitter(penalizer=0.0)
    reference.fit(data, id_col="license_id", event_col="event_interval", start_col="start", stop_col="stop", strata="opening_stratum", show_progress=False)
    coefficient_gap = float((reference.params_ - robust.params.reindex(reference.params_.index)).abs().max())
    loglik_gap = float(abs(reference.log_likelihood_ - robust.model.log_likelihood_))
    return coefficient_gap, loglik_gap


def coefficient_rows(fits: dict[str, RobustFit], sample: pd.DataFrame, section: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, fitted in fits.items():
        summary = fitted.model.summary
        for variable in fitted.params.index:
            coefficient = float(fitted.params[variable])
            iqr = float(sample[variable].quantile(.75) - sample[variable].quantile(.25)) if variable.startswith("comp_") and "_x_" not in variable else np.nan
            rows.append({"section": section, "model": MODEL_LABELS[key], "model_key": key, "variable": variable, "variable_label": label_for(variable), "coefficient": coefficient, "se_cluster_robust": float(summary.loc[variable, "se(coef)"]), "hazard_ratio": float(np.exp(coefficient)), "hr_ci_lower": float(summary.loc[variable, "exp(coef) lower 95%"]), "hr_ci_upper": float(summary.loc[variable, "exp(coef) upper 95%"]), "p_value": float(summary.loc[variable, "p"]), "stores": sample["license_id"].nunique(), "events": int(sample["event_interval"].sum()), "intervals": len(sample), "concordance": fitted.concordance, "iqr": iqr, "hr_iqr": float(np.exp(coefficient * iqr)) if np.isfinite(iqr) else np.nan})
    return pd.DataFrame(rows)


def competition_change(fits: dict[str, RobustFit]) -> pd.DataFrame:
    rows = []
    for variable in ["comp_indep_r200", "comp_fran_r200"]:
        before = fits["time_varying_competition"].contrast({variable: 1.0})
        after = fits["time_varying_competition_demand"].contrast({variable: 1.0})
        rows.append({"variable": variable, "variable_label": label_for(variable), "hr_competition_only": before["hazard_ratio"], "hr_competition_only_ci_lower": before["hr_ci_lower"], "hr_competition_only_ci_upper": before["hr_ci_upper"], "hr_competition_demand": after["hazard_ratio"], "hr_competition_demand_ci_lower": after["hr_ci_lower"], "hr_competition_demand_ci_upper": after["hr_ci_upper"], "coefficient_change": after["coefficient"] - before["coefficient"], "percent_change": 100 * (after["coefficient"] - before["coefficient"]) / before["coefficient"]})
    return pd.DataFrame(rows)


def conditional_hr(fitted: RobustFit, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for demand, z_col, interaction, raw in [("총매출", "log_sales_z", "comp_total_x_sales", "trdar_sales_total"), ("유동인구", "log_floating_z", "comp_total_x_floating", "trdar_floating_pop")]:
        interaction_test = fitted.contrast({interaction: 1.0})
        for quantile, level in [(0.25, "하위 25%"), (.5, "중앙"), (.75, "상위 25%")]:
            value = float(frame[z_col].quantile(quantile))
            for competition in ["comp_indep_r200", "comp_fran_r200"]:
                effect = fitted.contrast({competition: 1.0, interaction: value})
                rows.append({"demand": demand, "demand_variable": raw, "demand_level": level, "quantile": quantile, "demand_standardized_value": value, "competition_variable": competition, "competition_label": label_for(competition), "hazard_ratio_per_store": effect["hazard_ratio"], "hr_ci_lower": effect["hr_ci_lower"], "hr_ci_upper": effect["hr_ci_upper"], "p_value": effect["p_value"], "interaction_hr": interaction_test["hazard_ratio"], "interaction_p_value": interaction_test["p_value"]})
    return pd.DataFrame(rows)


def fixed_comparison(coefficients: pd.DataFrame) -> pd.DataFrame:
    fixed = pd.read_csv(TABLE_DIR / "05_cox_coefficients.csv", encoding="utf-8-sig")
    fixed = fixed.loc[fixed["model_key"].eq("store_competition") & fixed["variable"].isin(["comp_indep_r200", "comp_fran_r200", "log_sales_z", "log_floating_z"]), ["variable", "hazard_ratio", "hr_ci_lower", "hr_ci_upper"]].rename(columns={"hazard_ratio": "fixed_hr", "hr_ci_lower": "fixed_hr_ci_lower", "hr_ci_upper": "fixed_hr_ci_upper"})
    time = coefficients.loc[coefficients["model_key"].eq("time_varying_store") & coefficients["variable"].isin(fixed["variable"]), ["variable", "variable_label", "hazard_ratio", "hr_ci_lower", "hr_ci_upper", "p_value"]].rename(columns={"hazard_ratio": "time_varying_hr", "hr_ci_lower": "time_varying_hr_ci_lower", "hr_ci_upper": "time_varying_hr_ci_upper", "p_value": "time_varying_p_value"})
    return fixed.merge(time, on="variable", how="inner", validate="one_to_one")


def truncate_at_cutoff(frame: pd.DataFrame) -> pd.DataFrame:
    cutoff = pd.Timestamp("2026-03-31")
    output = frame.loc[frame["interval_start_date"].lt(cutoff)].copy()
    opening = pd.to_datetime(output["permit_date"])
    cutoff_stop = (cutoff - opening).dt.days.clip(lower=0)
    crossed = output["stop"].gt(cutoff_stop)
    output.loc[crossed, "stop"] = cutoff_stop.loc[crossed]
    output.loc[crossed, "event_interval"] = 0
    return output.loc[output["stop"].gt(output["start"])].copy()


def fit_suite(frame: pd.DataFrame, specs: dict[str, list[str]], section: str) -> tuple[dict[str, RobustFit], pd.DataFrame]:
    fits = {key: RobustFit(frame, columns) for key, columns in specs.items()}
    return fits, coefficient_rows(fits, frame, section)


def robustness(frame: pd.DataFrame, type_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = definitions(type_columns)
    runs: list[tuple[str, pd.DataFrame, dict[str, list[str]]]] = [("2026-03-31 절단", truncate_at_cutoff(frame), {"cutoff_store": specs["time_varying_store"], "cutoff_interaction": specs["time_varying_interaction"]})]
    for radius in [100, 500]:
        radius_specs = definitions(type_columns, radius)
        runs.append((f"반경 {radius}m", frame, {f"radius_{radius}_store": radius_specs["time_varying_store"], f"radius_{radius}_interaction": radius_specs["time_varying_interaction"]}))
    carried_controls = [column for column in ["trdar_sales_carried", "trdar_stores_carried", "trdar_population_carried"] if frame[column].nunique(dropna=False) > 1]
    runs.append(("이월 표시 통제", frame, {"carried_control": [*specs["time_varying_store"], *carried_controls]}))
    all_rows, summary = [], []
    for label, candidate, run_specs in runs:
        _, rows = fit_suite(candidate, run_specs, "강건성")
        rows["robustness"] = label
        all_rows.append(rows)
        summary.append({"robustness": label, "stores": candidate["license_id"].nunique(), "events": int(candidate["event_interval"].sum()), "intervals": len(candidate), "models": ";".join(run_specs)})
    return pd.concat(all_rows, ignore_index=True), pd.DataFrame(summary)


def store_profile(frame: pd.DataFrame, label: str) -> dict[str, object]:
    first = frame.loc[frame["interval_number"].eq(frame.groupby("license_id")["interval_number"].transform("min"))]
    shares = first["trdar_type"].value_counts(normalize=True)
    row: dict[str, object] = {"sample": label, "stores": frame["license_id"].nunique(), "events": int(frame["event_interval"].sum()), "event_share": float(frame["event_interval"].sum() / frame["license_id"].nunique()), "trade_areas": first[CLUSTER].nunique()}
    for area_type in TYPE_ORDER:
        row[f"share_{area_type}"] = float(shares.get(area_type, 0.0))
    row["median_comp_indep_r200_open"] = float(first["comp_indep_r200"].median())
    row["median_comp_fran_r200_open"] = float(first["comp_fran_r200"].median())
    return row


def rent_analysis(frame: pd.DataFrame, primary_specs: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    linked = frame.loc[frame["rone_area"].notna() & pd.to_numeric(frame["rent_small_thousand_krw_per_sqm"], errors="coerce").gt(0)].copy()
    # Retain a store only when every one of its analytic intervals has an observed prior-quarter rent.
    full_counts = frame.groupby("license_id").size()
    observed_counts = linked.groupby("license_id").size()
    keep = observed_counts.index[observed_counts.eq(full_counts.reindex(observed_counts.index))]
    linked = linked.loc[linked["license_id"].isin(keep)].copy()
    linked["log_rent_small"] = np.log(linked["rent_small_thousand_krw_per_sqm"].astype(float))
    specs = {
        "rent_store_without": primary_specs["time_varying_store"],
        "rent_store_with": [*primary_specs["time_varying_store"], "log_rent_small"],
        "rent_interaction_without": primary_specs["time_varying_interaction"],
        "rent_interaction_with": [*primary_specs["time_varying_interaction"], "log_rent_small"],
    }
    _, rows = fit_suite(linked, specs, "임대료 보조 분석")
    unlinked = frame.loc[~frame["license_id"].isin(keep)]
    profile = pd.DataFrame([store_profile(frame, "시변 Cox 최종 표본"), store_profile(linked, "임대료 연결 하위 표본"), store_profile(unlinked, "임대료 미연결 점포")])
    rent_first = linked.loc[linked["interval_number"].eq(1), "rent_small_thousand_krw_per_sqm"].astype(float)
    profile.loc[1, "rent_small_open_median"] = float(rent_first.median())
    profile.loc[1, "log_rent_small_sd"] = float(linked["log_rent_small"].std())
    return rows, profile


def plot_fixed_vs_time(comparison: pd.DataFrame) -> None:
    order = ["comp_indep_r200", "comp_fran_r200", "log_sales_z", "log_floating_z"]
    data = comparison.set_index("variable").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    y = np.arange(len(data))[::-1]
    for offset, prefix, color, label in [(.16, "fixed", "#555555", "개업 시점 고정"), (-.16, "time_varying", "#0072B2", "분기별 시변")]:
        hr, lower, upper = (data[f"{prefix}_hr"], data[f"{prefix}_hr_ci_lower"], data[f"{prefix}_hr_ci_upper"])
        ax.errorbar(hr, y + offset, xerr=[hr - lower, upper - hr], fmt="o", color=color, capsize=3, markersize=7, label=label)
    ax.axvline(1, color="#555555", linestyle="--", linewidth=1)
    ax.set_yticks(y, data["variable_label"])
    plain_log_axis(ax, 0.85, 1.18)
    ax.set_xlabel("위험비 (로그 축, 95% 신뢰구간)")
    ax.set_title("개업 시점 고정 모형과 분기별 시변 모형의 경쟁·수요 연관")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_tv_cox_fixed_vs_time_varying.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_conditional(conditional: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    levels = ["하위 25%", "중앙", "상위 25%"]
    for ax, demand in zip(axes, ["총매출", "유동인구"]):
        subset = conditional.loc[conditional["demand"].eq(demand)]
        for variable, color, label, shift in [("comp_indep_r200", "#0072B2", "개인 카페", -.06), ("comp_fran_r200", "#D55E00", "프랜차이즈", .06)]:
            part = subset.loc[subset["competition_variable"].eq(variable)].set_index("demand_level").loc[levels]
            x = np.arange(len(levels)) + shift
            hr = part["hazard_ratio_per_store"]
            ax.errorbar(x, hr, yerr=[hr - part["hr_ci_lower"], part["hr_ci_upper"] - hr], fmt="o-", color=color, capsize=3, markersize=7, label=label)
        ax.axhline(1, color="#555555", linestyle="--", linewidth=1)
        p_value = subset["interaction_p_value"].iloc[0]
        ax.set_title(f"{demand} 수준별 (상호작용 p={p_value:.3f})")
        ax.set_xticks(range(len(levels)), levels)
        ax.set_xlabel(f"직전 분기 {demand} 수준")
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("200m 경쟁 1개당 위험비 (95% 신뢰구간)")
    axes[1].legend(frameon=False)
    fig.suptitle("분기별 시변 모형: 수요 수준에 따른 경쟁 연관", y=1.02, fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_tv_cox_competition_hr_by_demand.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_competition_robustness(coefficients: pd.DataFrame) -> None:
    """Personal-cafe competition HR across the main, robustness and rent models."""
    picks = [
        ("time_varying_store", "comp_indep_r200", "주 분석 (200m)"),
        ("cutoff_store", "comp_indep_r200", "2026-03-31 절단"),
        ("radius_100_store", "comp_indep_r100", "반경 100m"),
        ("radius_500_store", "comp_indep_r500", "반경 500m"),
        ("carried_control", "comp_indep_r200", "이월 표시 통제"),
        ("rent_store_without", "comp_indep_r200", "임대료 연결 표본: 임대료 제외"),
        ("rent_store_with", "comp_indep_r200", "임대료 연결 표본: 임대료 포함"),
    ]
    rows = []
    for key, variable, label in picks:
        for comp_var, kind in [(variable, "개인 카페"), (variable.replace("indep", "fran"), "프랜차이즈")]:
            match = coefficients.loc[coefficients["model_key"].eq(key) & coefficients["variable"].eq(comp_var)].iloc[0]
            rows.append({"label": label, "kind": kind, "hr": match["hazard_ratio"], "low": match["hr_ci_lower"], "high": match["hr_ci_upper"]})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(picks))[::-1]
    for kind, color, shift in [("개인 카페", "#0072B2", .15), ("프랜차이즈", "#D55E00", -.15)]:
        part = data.loc[data["kind"].eq(kind)]
        ax.errorbar(part["hr"], y + shift, xerr=[part["hr"] - part["low"], part["high"] - part["hr"]], fmt="o", color=color, capsize=3, markersize=7, label=f"{kind} 경쟁 1개")
    ax.axvline(1, color="#555555", linestyle="--", linewidth=1)
    ax.set_yticks(y, [label for _, _, label in picks])
    lower = min(0.95, float(data["low"].min()) * 0.99)
    upper = max(1.06, float(data["high"].max()) * 1.01)
    ax.set_xscale("log")
    ax.set_xlim(lower, upper)
    ticks = [t for t in [0.9, 0.95, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.15, 1.2] if lower <= t <= upper]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("경쟁 1개당 위험비 (로그 축, 95% 신뢰구간)")
    ax.set_title("분기별 시변 모형의 경쟁 연관: 강건성과 임대료 통제")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_tv_cox_competition_robustness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.environ["TEMP"] = str(ROOT / ".tmp")
    os.environ["TMP"] = str(ROOT / ".tmp")
    for directory in [TABLE_DIR, FIGURE_DIR, LOG.parent, ROOT / ".tmp"]:
        directory.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    frame, flow, excluded_detail, scales, type_columns = prepare_panel()
    primary_specs = definitions(type_columns)
    primary_fits, primary_rows = fit_suite(frame, primary_specs, "주 분석")
    max_difference, loglik_gap = equivalence_check(frame, primary_specs["time_varying_store"], primary_fits["time_varying_store"])
    # Both maximise the same partial likelihood; residual gaps are convergence tolerance.
    if max_difference > 1e-4 or loglik_gap > 1e-6:
        raise AssertionError(f"지연 진입 CoxPHFitter와 CoxTimeVaryingFitter 결과가 다릅니다: 계수 {max_difference:.3e}, 로그우도 {loglik_gap:.3e}")
    changes = competition_change(primary_fits)
    conditional = conditional_hr(primary_fits["time_varying_interaction"], frame)
    comparison = fixed_comparison(primary_rows)
    robust_rows, robust_summary = robustness(frame, type_columns)
    rent_rows, rent_profile = rent_analysis(frame, primary_specs)
    coefficients = pd.concat([primary_rows, robust_rows, rent_rows], ignore_index=True)
    outputs = {
        "06_tv_cox_coefficients.csv": coefficients,
        "06_tv_cox_competition_changes.csv": changes,
        "06_tv_cox_demand_competition_hr.csv": conditional,
        "06_tv_cox_fixed_comparison.csv": comparison,
        "06_tv_cox_sample_flow.csv": flow,
        "06_tv_cox_excluded_intervals.csv": excluded_detail,
        "06_tv_cox_robustness_coefficients.csv": robust_rows,
        "06_tv_cox_robustness_summary.csv": robust_summary,
        "06_tv_cox_rent_coefficients.csv": rent_rows,
        "06_tv_cox_rent_sample_profile.csv": rent_profile,
    }
    for name, table in outputs.items():
        table.to_csv(TABLE_DIR / name, index=False, encoding="utf-8-sig")
    plot_fixed_vs_time(comparison)
    plot_conditional(conditional)
    plot_competition_robustness(coefficients)
    LOG.write_text("\n".join([
        "# 시변 Cox와 임대료 보조 분석 실행 로그", "",
        "- 실행: `.venv\\Scripts\\python.exe src\\06_time_varying_cox.py`",
        f"- 입력 패널: {len(frame):,}행, 점포 {frame['license_id'].nunique():,}개, 사건 {int(frame['event_interval'].sum()):,}건",
        "- 적합: `CoxPHFitter(duration_col=\"stop\", entry_col=\"start\")`, 개업연도 층화, 배정 상권 군집 강건 샌드위치 분산. 이전 실행의 상권 클러스터 부트스트랩(5회·3회)은 반복 수가 부족해 폐기함",
        f"- 동일성 확인: 시변 상권+점포 단위 경쟁 모형에서 `CoxTimeVaryingFitter`와의 계수 최대 절대차 {max_difference:.2e}(상권 커피 점포 수 결측 표시처럼 우도면이 평평한 변수에서 나는 수렴 허용오차 수준), 부분 로그우도 차 {loglik_gap:.2e} → 같은 부분우도로 판단(기준: 계수 1e-4, 로그우도 1e-6 이하)",
        f"- 결측 제외: 점포 {excluded_detail['license_id'].nunique()}개, 구간 {len(excluded_detail)}개 (`docs/tables/06_tv_cox_excluded_intervals.csv`)",
        f"- 표준화: 로그 총매출 평균 {scales['log_sales_z_mean']:.6f}, 표준편차 {scales['log_sales_z_sd']:.6f}; 로그 유동인구 평균 {scales['log_floating_z_mean']:.6f}, 표준편차 {scales['log_floating_z_sd']:.6f}",
        f"- 층화: {scales['stratum_rule']}",
        "- 그림: 맑은 고딕 설정 뒤 `axes.unicode_minus=False`, 로그 축 눈금은 소수 형식으로 직접 지정",
        "- 보고서 `docs/reports/06_time_varying_cox_report.md`는 이 스크립트의 표를 보고 직접 작성함",
        "- 패키지 설치·원자료 수정 없음; 임시 파일은 프로젝트 .tmp/만 사용",
    ]) + "\n", encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
