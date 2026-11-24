"""Cox main analysis for Seoul independent coffee-shop survival.

All covariates are fixed at the opening quarter.  The script intentionally
reads the established analysis data only, uses one complete-case cohort for
every main model, and records every exclusion and diagnostic in CSV outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.statistics import proportional_hazard_test
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "cafes_analysis.parquet"
TABLE_DIR = ROOT / "docs" / "tables"
FIGURE_DIR = ROOT / "docs" / "figures"
REPORT_PATH = ROOT / "docs" / "reports" / "05_cox_main_report.md"
LOG_PATH = ROOT / "docs" / "logs" / "05_cox_execution.md"
COFFEE_MISSING = "trdar_coffee_stores_missing"
DEMAND = ["log_sales_z", "log_floating_z"]
COMPETITION = ["comp_indep_r200", "comp_fran_r200"]
CONTROL = ["log_resident_pop", "trdar_coffee_stores", COFFEE_MISSING]
TYPE_BASELINE = "골목상권"
MODEL_LABELS = {
    "trade_area": "상권 모형",
    "store_competition": "상권 + 점포 단위 경쟁",
    "competition_only": "경쟁만",
    "competition_demand": "경쟁 + 수요",
    "interaction": "상호작용",
    "franchise_nonlinear": "프랜차이즈 경쟁 구간",
}


def configure_matplotlib() -> None:
    font_path = Path(r"C:\Windows\Fonts\malgun.ttf")
    if not font_path.exists():
        raise RuntimeError("Malgun Gothic font was not found.")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=str(font_path)).get_name(),
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def fmt_number(value: float | int | None, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}"


def fmt_p(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def fmt_hr(value: float, lower: float, upper: float) -> str:
    return f"{value:.3f} ({lower:.3f}–{upper:.3f})"


def type_dummy_columns(frame: pd.DataFrame) -> list[str]:
    """Create stable treatment dummies, with 골목상권 as the documented baseline."""
    known_types = ["골목상권", "발달상권", "전통시장", "관광특구"]
    observed = set(frame["trdar_type"].dropna().astype(str))
    unknown = observed.difference(known_types)
    if unknown:
        raise ValueError(f"Unexpected trade-area types: {sorted(unknown)}")
    columns: list[str] = []
    for area_type in known_types:
        if area_type == TYPE_BASELINE or area_type not in observed:
            continue
        column = f"type_{area_type}"
        frame[column] = frame["trdar_type"].eq(area_type).astype(int)
        columns.append(column)
    if TYPE_BASELINE not in observed:
        raise ValueError("The designated 골목상권 baseline is absent.")
    return columns


def prepare_sample(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    required = {
        "is_franchise", "trdar_within_200m", "survival_days", "event", "permit_date",
        "comp_indep_r200", "comp_fran_r200", "trdar_coffee_stores",
        "trdar_coffee_franchise_stores", "trdar_sales_total", "trdar_floating_pop",
        "trdar_resident_pop", "trdar_type", "trdar_cd_assigned",
    }
    absent = required.difference(data.columns)
    if absent:
        raise ValueError(f"Required columns are absent: {sorted(absent)}")
    if not (data["survival_days"] > 0).all():
        raise ValueError("The analysis input contains non-positive survival days.")

    flow: list[dict[str, object]] = []
    base = data.loc[(~data["is_franchise"]) & data["trdar_within_200m"].fillna(False)].copy()
    flow.append({"step": "개인 카페·200m 이내 상권 배정", "n_before": len(data), "excluded": len(data) - len(base),
                 "reason": "주 분석 범위 밖(프랜차이즈 또는 200m 밖/배정 없음)", "n_after": len(base), "events_after": int(base["event"].sum())})

    coffee_missing = base["trdar_coffee_stores"].isna()
    base[COFFEE_MISSING] = coffee_missing.astype(int)
    for column in ["trdar_coffee_stores", "trdar_coffee_franchise_stores"]:
        base[column] = base[column].fillna(0.0).astype(float)
    flow.append({"step": "상권 커피 점포 수 결측 처리", "n_before": len(base), "excluded": 0,
                 "reason": f"결측 {int(coffee_missing.sum()):,}건을 0으로 채우고 결측 표시 변수 생성", "n_after": len(base), "events_after": int(base["event"].sum())})

    # Sequential exclusions make the recorded counts add up exactly.
    complete_columns = ["trdar_sales_total", "trdar_floating_pop", "trdar_resident_pop", "trdar_type", "trdar_cd_assigned", *COMPETITION]
    for column in complete_columns:
        missing = base[column].isna()
        before = len(base)
        base = base.loc[~missing].copy()
        flow.append({"step": f"{column} 결측 제외", "n_before": before, "excluded": int(missing.sum()),
                     "reason": "Cox 투입 변수 또는 군집·층화 식별자 결측", "n_after": len(base), "events_after": int(base["event"].sum())})

    positive_columns = ["trdar_sales_total", "trdar_floating_pop", "trdar_resident_pop"]
    for column in positive_columns:
        invalid = base[column].astype(float).le(0)
        before = len(base)
        base = base.loc[~invalid].copy()
        flow.append({"step": f"{column} 0 이하 제외", "n_before": before, "excluded": int(invalid.sum()),
                     "reason": "자연로그 변환 불가", "n_after": len(base), "events_after": int(base["event"].sum())})

    base["opening_year"] = pd.to_datetime(base["permit_date"]).dt.year.astype(int)
    base["analysis_row_id"] = np.arange(len(base), dtype=int)
    year_events = base.groupby("opening_year", observed=True)["event"].sum()
    if int(year_events.get(2026, 0)) < 30:
        base["opening_stratum"] = base["opening_year"].replace({2026: 2025})
        stratum_rule = "2026년 사건이 30건 미만이어서 2025년과 합침"
    else:
        base["opening_stratum"] = base["opening_year"]
        stratum_rule = "2026년 사건이 30건 이상이어서 개업연도를 그대로 층화"

    base["log_sales"] = np.log(base["trdar_sales_total"].astype(float))
    base["log_floating"] = np.log(base["trdar_floating_pop"].astype(float))
    base["log_resident_pop"] = np.log(base["trdar_resident_pop"].astype(float))
    scale_info: dict[str, float] = {}
    for raw, standardised in [("log_sales", "log_sales_z"), ("log_floating", "log_floating_z")]:
        mean = float(base[raw].mean())
        sd = float(base[raw].std(ddof=0))
        if not np.isfinite(sd) or sd == 0:
            raise ValueError(f"{raw} has zero or invalid standard deviation.")
        base[standardised] = (base[raw] - mean) / sd
        scale_info[f"{raw}_mean"] = mean
        scale_info[f"{raw}_sd"] = sd
    base["comp_total_r200"] = base["comp_indep_r200"].astype(float) + base["comp_fran_r200"].astype(float)
    base["comp_total_x_sales"] = base["comp_total_r200"] * base["log_sales_z"]
    base["comp_total_x_floating"] = base["comp_total_r200"] * base["log_floating_z"]
    base["fran_1_2"] = base["comp_fran_r200"].between(1, 2).astype(int)
    base["fran_3_5"] = base["comp_fran_r200"].between(3, 5).astype(int)
    base["fran_6_plus"] = base["comp_fran_r200"].ge(6).astype(int)
    type_columns = type_dummy_columns(base)
    scale_info["type_columns"] = type_columns  # type: ignore[assignment]
    scale_info["stratum_rule"] = stratum_rule  # type: ignore[assignment]
    flow.append({"step": "최종 공통 분석 표본", "n_before": len(base), "excluded": 0,
                 "reason": "모든 주 모형에 같은 완전 사례 표본 사용", "n_after": len(base), "events_after": int(base["event"].sum())})
    return base, pd.DataFrame(flow), scale_info


def fit_cox(frame: pd.DataFrame, covariates: list[str]) -> CoxPHFitter:
    columns = ["survival_days", "event", "opening_stratum", "trdar_cd_assigned", *covariates]
    fitting_frame = frame.loc[:, columns].copy()
    # lifelines calculates stratum-specific medians before fitting.  Keeping
    # binary covariates as pandas nullable integers can fail when a stratum
    # median is 0.5 in a sensitivity subset, so all model covariates are float.
    fitting_frame = fitting_frame.astype({column: float for column in covariates})
    fitting_frame["survival_days"] = fitting_frame["survival_days"].astype(float)
    fitting_frame["event"] = fitting_frame["event"].astype(int)
    model = CoxPHFitter()
    model.fit(
        fitting_frame, duration_col="survival_days", event_col="event",
        strata="opening_stratum", cluster_col="trdar_cd_assigned", robust=True,
        show_progress=False,
    )
    return model


def model_definitions(type_columns: list[str]) -> dict[str, list[str]]:
    trade_area = ["trdar_coffee_stores", COFFEE_MISSING, *DEMAND, "log_resident_pop", *type_columns]
    store_competition = [*trade_area, *COMPETITION]
    competition_only = [*COMPETITION, *type_columns]
    competition_demand = [*competition_only, *DEMAND, "log_resident_pop"]
    interaction = [*store_competition, "comp_total_x_sales", "comp_total_x_floating"]
    franchise_nonlinear = [*trade_area, "comp_indep_r200", "fran_1_2", "fran_3_5", "fran_6_plus"]
    return {
        "trade_area": trade_area, "store_competition": store_competition,
        "competition_only": competition_only, "competition_demand": competition_demand,
        "interaction": interaction, "franchise_nonlinear": franchise_nonlinear,
    }


def label_for(variable: str) -> str:
    labels = {
        "trdar_coffee_stores": "상권 커피 점포 수 (1개)",
        COFFEE_MISSING: "상권 커피 점포 수 결측 표시",
        "log_sales_z": "로그 총매출 (1표준편차)",
        "log_floating_z": "로그 유동인구 (1표준편차)",
        "log_resident_pop": "로그 상주인구 (1 단위)",
        "comp_indep_r200": "200m 개인 카페 경쟁 (1개)",
        "comp_fran_r200": "200m 프랜차이즈 경쟁 (1개)",
        "comp_total_x_sales": "전체 경쟁×로그 총매출(표준화)",
        "comp_total_x_floating": "전체 경쟁×로그 유동인구(표준화)",
        "type_발달상권": "발달상권 (골목상권 대비)",
        "type_전통시장": "전통시장 (골목상권 대비)",
        "type_관광특구": "관광특구 (골목상권 대비)",
        "fran_1_2": "프랜차이즈 1–2개 (0개 대비)",
        "fran_3_5": "프랜차이즈 3–5개 (0개 대비)",
        "fran_6_plus": "프랜차이즈 6개 이상 (0개 대비)",
    }
    return labels.get(variable, variable)


def coefficient_table(models: dict[str, CoxPHFitter], sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, model in models.items():
        summary = model.summary
        for variable, result in summary.iterrows():
            beta = float(result["coef"])
            lower_beta = float(result["coef lower 95%"])
            upper_beta = float(result["coef upper 95%"])
            row: dict[str, object] = {
                "model": MODEL_LABELS[key], "model_key": key, "variable": variable,
                "variable_label": label_for(variable), "coefficient": beta,
                "se_cluster_robust": float(result["se(coef)"]), "hazard_ratio": float(np.exp(beta)),
                "hr_ci_lower": float(np.exp(lower_beta)), "hr_ci_upper": float(np.exp(upper_beta)),
                "p_value": float(result["p"]), "scale": "1 단위",
            }
            if variable in COMPETITION:
                iqr = float(sample[variable].quantile(0.75) - sample[variable].quantile(0.25))
                row["iqr"] = iqr
                row["hr_iqr"] = float(np.exp(beta * iqr))
                row["hr_iqr_ci_lower"] = float(np.exp(lower_beta * iqr))
                row["hr_iqr_ci_upper"] = float(np.exp(upper_beta * iqr))
            else:
                row["iqr"] = np.nan
                row["hr_iqr"] = np.nan
                row["hr_iqr_ci_lower"] = np.nan
                row["hr_iqr_ci_upper"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def model_fit_table(models: dict[str, CoxPHFitter], n: int, events: int) -> pd.DataFrame:
    rows = []
    baseline = models["trade_area"]
    extended = models["store_competition"]
    lr_stat = 2 * (extended.log_likelihood_ - baseline.log_likelihood_)
    lr_df = len(extended.params_) - len(baseline.params_)
    lr_p = float(chi2.sf(lr_stat, lr_df))
    for key, model in models.items():
        rows.append({
            "model": MODEL_LABELS[key], "model_key": key, "n": n, "events": events,
            "n_parameters": len(model.params_), "partial_log_likelihood": model.log_likelihood_,
            "partial_aic": model.AIC_partial_, "harrell_c": model.concordance_index_,
            "lr_vs_trade_area_stat": lr_stat if key == "store_competition" else np.nan,
            "lr_vs_trade_area_df": lr_df if key == "store_competition" else np.nan,
            "lr_vs_trade_area_p": lr_p if key == "store_competition" else np.nan,
            "aic_change_vs_trade_area": model.AIC_partial_ - baseline.AIC_partial_,
            "c_change_vs_trade_area": model.concordance_index_ - baseline.concordance_index_,
        })
    return pd.DataFrame(rows)


def competition_change_table(models: dict[str, CoxPHFitter]) -> pd.DataFrame:
    rows = []
    before = models["competition_only"].params_
    after = models["competition_demand"].params_
    for variable in COMPETITION:
        beta_before, beta_after = float(before[variable]), float(after[variable])
        hr_before, hr_after = float(np.exp(beta_before)), float(np.exp(beta_after))
        rows.append({
            "variable": variable, "variable_label": label_for(variable),
            "coefficient_competition_only": beta_before, "coefficient_competition_demand": beta_after,
            "hazard_ratio_competition_only": hr_before, "hazard_ratio_competition_demand": hr_after,
            "coefficient_change_pct": (beta_after - beta_before) / abs(beta_before) * 100 if beta_before != 0 else np.nan,
            "hazard_ratio_change_pct": (hr_after - hr_before) / hr_before * 100,
        })
    return pd.DataFrame(rows)


def demand_competition_table(model: CoxPHFitter, sample: pd.DataFrame) -> pd.DataFrame:
    """Conditional one-store effects for both competition types in model 5."""
    params = model.params_
    variance = model.variance_matrix_
    rows: list[dict[str, object]] = []
    scenarios = [
        ("총매출", "log_sales_z", "comp_total_x_sales", "trdar_sales_total"),
        ("유동인구", "log_floating_z", "comp_total_x_floating", "trdar_floating_pop"),
    ]
    for demand_label, z_column, interaction, raw_column in scenarios:
        for quantile, scenario_label in [(0.25, "하위 25%"), (0.50, "중앙"), (0.75, "상위 25%")]:
            raw_value = float(sample[raw_column].quantile(quantile))
            z_value = float(sample[z_column].quantile(quantile))
            for competition in COMPETITION:
                beta = float(params[competition] + z_value * params[interaction])
                variance_effect = float(
                    variance.loc[competition, competition]
                    + z_value ** 2 * variance.loc[interaction, interaction]
                    + 2 * z_value * variance.loc[competition, interaction]
                )
                se = float(np.sqrt(max(variance_effect, 0)))
                lower, upper = beta - 1.96 * se, beta + 1.96 * se
                rows.append({
                    "demand": demand_label, "demand_variable": raw_column, "demand_level": scenario_label,
                    "quantile": quantile, "demand_raw_value": raw_value, "demand_standardized_value": z_value,
                    "competition_variable": competition, "competition_label": label_for(competition),
                    "hazard_ratio_per_store": float(np.exp(beta)), "hr_ci_lower": float(np.exp(lower)),
                    "hr_ci_upper": float(np.exp(upper)), "log_hr": beta, "se": se,
                })
    return pd.DataFrame(rows)


def vif_table(frame: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """VIF via least-squares R²; avoids introducing an unpinned package dependency."""
    values = frame.loc[:, covariates].astype(float).to_numpy()
    rows = []
    for index, name in enumerate(covariates):
        y = values[:, index]
        x = np.delete(values, index, axis=1)
        x = np.column_stack([np.ones(len(x)), x])
        fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        vif = 1 / (1 - r2) if np.isfinite(r2) and r2 < 1 else np.inf
        rows.append({"variable": name, "variable_label": label_for(name), "r_squared": r2, "vif": vif})
    return pd.DataFrame(rows)


def ph_table(models: dict[str, CoxPHFitter], sample: pd.DataFrame, definitions: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for key in ["store_competition", "interaction"]:
        model = models[key]
        columns = ["survival_days", "event", "opening_stratum", "trdar_cd_assigned", *definitions[key]]
        result = proportional_hazard_test(model, sample.loc[:, columns], time_transform="rank")
        summary = result.summary
        for variable, values in summary.iterrows():
            rows.append({"model": MODEL_LABELS[key], "model_key": key, "variable": variable,
                         "variable_label": label_for(variable), "test_statistic": float(values["test_statistic"]),
                         "p_value": float(values["p"]), "time_transform": "rank"})
    return pd.DataFrame(rows)


def time_interaction_table(sample: pd.DataFrame, definitions: dict[str, list[str]], ph: pd.DataFrame) -> pd.DataFrame:
    """Fit a counting-process sensitivity model only when target variables fail PH."""
    target = set(COMPETITION + DEMAND)
    violated = ph.loc[ph["variable"].isin(target) & ph["p_value"].lt(0.05), "variable"].unique().tolist()
    if not violated:
        return pd.DataFrame(columns=["status", "variable", "period", "hazard_ratio", "hr_ci_lower", "hr_ci_upper", "p_value"])
    covariates = definitions["store_competition"]
    early = sample.copy()
    early["start"] = 0.0
    early["stop"] = np.minimum(early["survival_days"].astype(float), 365.0)
    early["event_interval"] = ((early["event"] == 1) & (early["survival_days"] <= 365)).astype(int)
    late = sample.loc[sample["survival_days"] > 365].copy()
    late["start"] = 365.0
    late["stop"] = late["survival_days"].astype(float)
    late["event_interval"] = late["event"].astype(int)
    for variable in violated:
        early[f"after_year_{variable}"] = 0.0
        late[f"after_year_{variable}"] = late[variable].astype(float)
    long = pd.concat([early, late], ignore_index=True)
    expanded = [*covariates, *[f"after_year_{variable}" for variable in violated]]
    fitter = CoxTimeVaryingFitter()
    # lifelines does not implement robust variance for this fitter; this is a
    # diagnostic supplement, while the five requested main models are clustered.
    fitter.fit(long.loc[:, ["analysis_row_id", "event_interval", "start", "stop", "opening_stratum", *expanded]],
               id_col="analysis_row_id", event_col="event_interval", start_col="start", stop_col="stop",
               strata="opening_stratum", robust=False)
    rows = []
    for variable in violated:
        for period, coefficient in [("개업 후 1년 이내", variable), ("개업 후 1년 이후", None)]:
            beta = float(fitter.params_[coefficient]) if coefficient else float(fitter.params_[variable] + fitter.params_[f"after_year_{variable}"])
            if coefficient:
                variance = float(fitter.variance_matrix_.loc[variable, variable])
            else:
                variance = float(fitter.variance_matrix_.loc[variable, variable] + fitter.variance_matrix_.loc[f"after_year_{variable}", f"after_year_{variable}"] + 2 * fitter.variance_matrix_.loc[variable, f"after_year_{variable}"])
            se = np.sqrt(max(variance, 0))
            rows.append({"status": "PH 위반 변수의 구간 상호작용 보조모형(비군집 표준오차)", "variable": variable,
                         "period": period, "hazard_ratio": np.exp(beta), "hr_ci_lower": np.exp(beta - 1.96 * se),
                         "hr_ci_upper": np.exp(beta + 1.96 * se), "p_value": np.nan})
    return pd.DataFrame(rows)


def sensitivity_tables(sample: pd.DataFrame, definitions: dict[str, list[str]], main_models: dict[str, CoxPHFitter]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications = [
        ("관광특구 제외", sample.loc[~sample["trdar_type"].eq("관광특구")].copy(), ["store_competition", "interaction"], "type_관광특구"),
        ("상권 커피 점포 수 원결측 제외", sample.loc[sample[COFFEE_MISSING].eq(0)].copy(), ["store_competition"], COFFEE_MISSING),
    ]
    for sensitivity, subset, keys, constant_column in specifications:
        for key in keys:
            # A control becomes identically zero after each specified exclusion.
            # Removing that non-estimable column is algebraically necessary and
            # does not change the remaining covariates' definition.
            covariates = [column for column in definitions[key] if column != constant_column]
            model = fit_cox(subset, covariates)
            for variable in COMPETITION:
                result = model.summary.loc[variable]
                main_result = main_models[key].summary.loc[variable]
                rows.append({
                    "sensitivity": sensitivity, "model": MODEL_LABELS[key], "model_key": key,
                    "n": len(subset), "events": int(subset["event"].sum()), "variable": variable,
                    "hazard_ratio": float(np.exp(result["coef"])), "hr_ci_lower": float(np.exp(result["coef lower 95%"])),
                    "hr_ci_upper": float(np.exp(result["coef upper 95%"])), "p_value": float(result["p"]),
                    "main_hazard_ratio": float(np.exp(main_result["coef"])),
                })
    return pd.DataFrame(rows)


def plot_forest(coefficients: pd.DataFrame) -> None:
    selected = coefficients.loc[coefficients["model_key"].eq("store_competition")].copy()
    selected = selected.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.8, max(5.2, len(selected) * 0.48)))
    y = np.arange(len(selected))
    hr = selected["hazard_ratio"].to_numpy(float)
    low = selected["hr_ci_lower"].to_numpy(float)
    high = selected["hr_ci_upper"].to_numpy(float)
    ax.errorbar(hr, y, xerr=[hr - low, high - hr], fmt="o", color="#0072B2", ecolor="#0072B2", capsize=3)
    ax.axvline(1, color="#555555", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y, selected["variable_label"])
    ax.set_xlabel("위험비 (로그 축, 95% 신뢰구간)")
    ax.set_title("상권 + 점포 단위 경쟁 모형의 위험비")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_cox_store_competition_forest.png", bbox_inches="tight")
    plt.close(fig)


def plot_conditional_hr(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5), sharey=True)
    levels = ["하위 25%", "중앙", "상위 25%"]
    colors = {"200m 개인 카페 경쟁 (1개)": "#0072B2", "200m 프랜차이즈 경쟁 (1개)": "#D55E00"}
    for ax, demand in zip(axes, ["총매출", "유동인구"]):
        subset = table.loc[table["demand"].eq(demand)]
        for label, group in subset.groupby("competition_label", sort=False):
            group = group.set_index("demand_level").loc[levels].reset_index()
            x = np.arange(len(levels))
            hr = group["hazard_ratio_per_store"].to_numpy(float)
            low, high = group["hr_ci_lower"].to_numpy(float), group["hr_ci_upper"].to_numpy(float)
            ax.errorbar(x, hr, yerr=[hr - low, high - hr], marker="o", capsize=3, label=label, color=colors[label])
        ax.axhline(1, color="#555555", linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(len(levels)), levels)
        ax.set_title(f"{demand} 수준별 경쟁 1개당 위험비")
        ax.set_ylabel("위험비 (95% 신뢰구간)")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_cox_competition_hr_by_demand.png", bbox_inches="tight")
    plt.close(fig)


def write_report(sample: pd.DataFrame, flow: pd.DataFrame, scales: dict[str, float], coefficients: pd.DataFrame,
                 fit: pd.DataFrame, changes: pd.DataFrame, conditional: pd.DataFrame, ph: pd.DataFrame,
                 sensitivity: pd.DataFrame, vif: pd.DataFrame, time_interaction: pd.DataFrame) -> None:
    def result(key: str, variable: str) -> pd.Series:
        return coefficients.loc[(coefficients["model_key"] == key) & (coefficients["variable"] == variable)].iloc[0]
    m2_indep, m2_fran = result("store_competition", "comp_indep_r200"), result("store_competition", "comp_fran_r200")
    m4_sales, m4_float = result("competition_demand", "log_sales_z"), result("competition_demand", "log_floating_z")
    i_sales, i_float = result("interaction", "comp_total_x_sales"), result("interaction", "comp_total_x_floating")
    comparison = fit.loc[fit["model_key"].eq("store_competition")].iloc[0]
    ph_targets = ph.loc[ph["variable"].isin(COMPETITION + DEMAND)]
    max_vif = float(vif["vif"].replace(np.inf, np.nan).max())
    low_high = conditional.loc[conditional["demand_level"].isin(["하위 25%", "상위 25%"])].copy()
    conditional_text = "; ".join(
        f"{row.demand} {row.demand_level}·{row.competition_label.replace('200m ', '')}: {fmt_hr(row.hazard_ratio_per_store, row.hr_ci_lower, row.hr_ci_upper)}"
        for row in low_high.itertuples()
    )
    lines = [
        "# 개업 시점 고정 변수 Cox 주 분석 보고서",
        "",
        "## 주요 결과",
        "",
        f"개업 직전 상권 정보와 개업 시점의 반경 200m 경쟁을 함께 넣은 Cox 분석에서, 최종 공통 표본은 개인 카페 {len(sample):,}개(폐업 사건 {int(sample['event'].sum()):,}건)였다. 모든 결과는 개업연도로 층화하고 배정 상권 기준 군집 강건 표준오차를 사용했으며, 연관성으로만 해석한다.",
        "",
        "### 경쟁",
        "",
        f"상권 변수만 포함한 모형에 200m 점포 단위 경쟁을 더하면 부분 AIC는 {comparison.partial_aic - comparison.aic_change_vs_trade_area:.1f}에서 {comparison.partial_aic:.1f}로 {comparison.aic_change_vs_trade_area:.1f} 변했고, Harrell's C는 {comparison.harrell_c - comparison.c_change_vs_trade_area:.3f}에서 {comparison.harrell_c:.3f}로 {comparison.c_change_vs_trade_area:+.3f} 변했다. 우도비 검정은 χ²={comparison.lr_vs_trade_area_stat:.2f}, df={int(comparison.lr_vs_trade_area_df)}, p={fmt_p(comparison.lr_vs_trade_area_p)}였다.",
        f"점포 단위 경쟁을 함께 조정한 모형에서 개인 카페 경쟁 1개당 위험비는 {fmt_hr(m2_indep.hazard_ratio, m2_indep.hr_ci_lower, m2_indep.hr_ci_upper)} (p={fmt_p(m2_indep.p_value)}), 프랜차이즈 경쟁 1개당은 {fmt_hr(m2_fran.hazard_ratio, m2_fran.hr_ci_lower, m2_fran.hr_ci_upper)} (p={fmt_p(m2_fran.p_value)})였다. 사분위 범위(IQR)당 위험비는 계수 표에 함께 제시했다.",
        "개인 경쟁의 추정치는 1보다 크고 신뢰구간도 1을 넘어서, KM에서 개인·전체 경쟁이 많을수록 생존확률이 낮았던 방향과 일치한다. 반면 프랜차이즈 경쟁은 신뢰구간이 1을 포함해 KM의 구간별 차이가 독립적인 선형 연관으로 이어진다고 보기 어렵다. 단, 이 관찰자료는 입지 선택과 미관측 요인을 제거하지 못하므로 경쟁 자체의 효과로 단정하지 않는다.",
        "",
        "### 수요",
        "",
        f"경쟁과 상권 구분을 함께 넣은 모형에서 로그 총매출 1표준편차당 위험비는 {fmt_hr(m4_sales.hazard_ratio, m4_sales.hr_ci_lower, m4_sales.hr_ci_upper)} (p={fmt_p(m4_sales.p_value)}), 로그 유동인구 1표준편차당은 {fmt_hr(m4_float.hazard_ratio, m4_float.hr_ci_lower, m4_float.hr_ci_upper)} (p={fmt_p(m4_float.p_value)})였다.",
        "두 수요 지표의 신뢰구간이 모두 1을 포함해, 총매출·유동인구 사분위별 생존 차이가 작고 방향도 일관되지 않았던 KM 관찰과 대체로 같다. 수요가 높아 폐업 위험이 달라진다는 인과 해석은 하지 않는다.",
        "",
        "### 상쇄: 가려짐과 완충",
        "",
        f"수요 투입 전후 경쟁 계수 변화는 `05_cox_competition_changes.csv`에 기록했다. 개인 경쟁의 위험비 변화율은 {changes.loc[changes.variable.eq('comp_indep_r200'), 'hazard_ratio_change_pct'].iloc[0]:+.1f}%, 프랜차이즈 경쟁은 {changes.loc[changes.variable.eq('comp_fran_r200'), 'hazard_ratio_change_pct'].iloc[0]:+.1f}%로 매우 작았다. 따라서 KM에서 보인 수요·경쟁의 동반 분포가 경쟁 계수를 크게 가렸다는 근거는 이 모형에서 뚜렷하지 않다.",
        f"경쟁 합계×표준화 로그 총매출 상호작용의 위험비는 {fmt_hr(i_sales.hazard_ratio, i_sales.hr_ci_lower, i_sales.hr_ci_upper)} (p={fmt_p(i_sales.p_value)})로 완충의 근거가 뚜렷하지 않았다. 경쟁 합계×표준화 로그 유동인구는 {fmt_hr(i_float.hazard_ratio, i_float.hr_ci_lower, i_float.hr_ci_upper)} (p={fmt_p(i_float.p_value)})로, 유동인구가 높은 곳에서 경쟁과 높은 폐업 위험의 연관이 약해지는 방향이었다. 수요 수준별 조건부 경쟁 위험비는 다음과 같으며, 개인·프랜차이즈 경쟁에 상호작용 변화가 동일하게 더해진 값이다: {conditional_text}.",
        "이는 유동인구의 저수요 집단에서 고경쟁의 생존율이 더 낮고 고수요 집단에서는 차이가 작았던 KM 그림과 방향이 같다. 총매출에서는 같은 완충 패턴이 확인되지 않았다. 상호작용은 다른 입지 특성과 함께 관찰된 연관이므로, 수요가 경쟁의 결과를 완충한다고 인과적으로 해석하지 않는다.",
        "",
        "## 방법과 검증",
        "",
        "### 표본과 변수 처리",
        "",
        "주 분석은 프랜차이즈가 아닌 카페 중 최근접 상권 경계 안쪽 200m에 배정된 점포로 시작했다. 상권 커피 점포 수 및 프랜차이즈 점포 수의 행 부재는 0으로 채우고 `trdar_coffee_stores_missing`을 넣었다. 다른 투입 변수의 결측은 완전 사례에서 제외했으며, 모든 주 모형은 동일한 최종 표본을 사용했다. 표본 흐름은 `05_cox_sample_flow.csv`에 있다.",
        f"총매출·유동인구·상주인구에는 자연로그를 적용했다. 0 이하 값은 없었으므로 로그 변환을 위한 추가 제외는 없었다. 총매출과 유동인구는 평균 0·표준편차 1로 표준화했다(로그 총매출 평균 {scales['log_sales_mean']:.3f}, 표준편차 {scales['log_sales_sd']:.3f}; 로그 유동인구 평균 {scales['log_floating_mean']:.3f}, 표준편차 {scales['log_floating_sd']:.3f}). 상권 구분은 골목상권을 기준 범주로 더미 처리했다.",
        f"{scales['stratum_rule']}. 배정 상권 코드로 군집 강건 표준오차를 계산했고, 동일한 점포의 이전 운영 여부(`has_predecessor_same_site`)는 모형에 넣지 않았다.",
        "",
        "### 모형 비교와 진단",
        "",
        "상권 모형, 상권+점포 단위 경쟁, 경쟁만, 경쟁+수요, 상호작용 모형을 각각 적합했다. 상권+점포 단위 경쟁 모형과 상호작용 모형은 Schoenfeld 잔차의 rank 시간변환 검정을 했고 변수별 p값은 `05_cox_proportional_hazards.csv`에 저장했다.",
        f"경쟁·수요 네 변수의 PH 검정 p값은 " + ", ".join(f"{label_for(row.variable)}={fmt_p(row.p_value)}" for row in ph_targets.itertuples()) + ".",
        ("검정에서 p<0.05인 경쟁·수요 변수가 있어 1년 이내/이후 구간 상호작용 보조모형을 추가했고, 결과는 `05_cox_time_interaction.csv`에 저장했다. 해당 보조모형은 lifelines의 제약으로 군집 강건 표준오차를 제공하지 못하므로 진단용으로 한정한다." if not time_interaction.empty else "경쟁·수요 변수에서 p<0.05인 PH 위반이 없어 시간 구간 상호작용 모형은 추가하지 않았다."),
        f"프랜차이즈 경쟁은 0 / 1–2 / 3–5 / 6개 이상 더미로 바꾼 보조모형을 `05_cox_coefficients.csv`에 함께 수록했다. 수요 수준별 조건부 위험비는 다른 수요 변수를 평균(표준화 0)에 고정해 계산했다. 투입 변수 VIF의 최댓값은 {max_vif:.2f}이며 전체 값은 `05_cox_vif.csv`에 있다.",
        "",
        "### 민감도 분석",
        "",
        "관광특구를 제외한 표본에서 상권+점포 단위 경쟁 및 상호작용 모형을 다시 적합했고, 상권 커피 점포 수 원결측 점포를 제외한 상권+점포 단위 경쟁 모형도 적합했다. 해당 경쟁 계수 비교는 `05_cox_sensitivity.csv`에 저장했다. 이 점검은 특정 상권 유형이나 0 채움 처리에 결과가 좌우되는지 확인하기 위한 것이며, 주 분석 표본·정의를 바꾸지 않는다.",
        "",
        "## 산출물",
        "",
        "- 계수·위험비: `docs/tables/05_cox_coefficients.csv`",
        "- 모형 적합도·표본 흐름·VIF·비례위험·민감도: `docs/tables/05_cox_*.csv`",
        "- 상권+점포 단위 경쟁 forest plot: `docs/figures/05_cox_store_competition_forest.png`",
        "- 수요 수준별 경쟁 위험비: `docs/figures/05_cox_competition_hr_by_demand.png`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    data = pd.read_parquet(INPUT_PATH)
    sample, flow, scales = prepare_sample(data)
    type_columns = scales["type_columns"]  # type: ignore[assignment]
    definitions = model_definitions(type_columns)
    main_keys = ["trade_area", "store_competition", "competition_only", "competition_demand", "interaction"]
    models = {key: fit_cox(sample, definitions[key]) for key in main_keys}
    models["franchise_nonlinear"] = fit_cox(sample, definitions["franchise_nonlinear"])
    coefficients = coefficient_table(models, sample)
    fit = model_fit_table({key: models[key] for key in main_keys}, len(sample), int(sample["event"].sum()))
    changes = competition_change_table(models)
    conditional = demand_competition_table(models["interaction"], sample)
    vif = vif_table(sample, definitions["store_competition"])
    ph = ph_table(models, sample, definitions)
    time_interaction = time_interaction_table(sample, definitions, ph)
    sensitivity = sensitivity_tables(sample, definitions, models)
    outputs = {
        "05_cox_coefficients.csv": coefficients,
        "05_cox_model_fit.csv": fit,
        "05_cox_competition_changes.csv": changes,
        "05_cox_demand_competition_hr.csv": conditional,
        "05_cox_vif.csv": vif,
        "05_cox_proportional_hazards.csv": ph,
        "05_cox_time_interaction.csv": time_interaction,
        "05_cox_sensitivity.csv": sensitivity,
        "05_cox_sample_flow.csv": flow,
    }
    for filename, table in outputs.items():
        table.to_csv(TABLE_DIR / filename, index=False, encoding="utf-8-sig")
    plot_forest(coefficients)
    plot_conditional_hr(conditional)
    write_report(sample, flow, scales, coefficients, fit, changes, conditional, ph, sensitivity, vif, time_interaction)
    log_lines = [
        "# Cox 주 분석 실행 로그", "", f"- 입력: `{INPUT_PATH.relative_to(ROOT)}` ({len(data):,}행)",
        f"- 최종 공통 표본: {len(sample):,}행, 사건 {int(sample['event'].sum()):,}건",
        f"- 2026년 사건: {int(sample.loc[sample.opening_year.eq(2026), 'event'].sum()):,}건; {scales['stratum_rule']}",
        "- 패키지 설치·원본 데이터 수정 없음", "- 모든 주 모형: 개업연도 층화, 배정 상권 군집 강건 표준오차",
        "- 완료: 계수·적합도·진단·민감도 표, forest plot, 수요 수준별 경쟁 위험비 그림, 보고서 생성",
    ]
    LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
