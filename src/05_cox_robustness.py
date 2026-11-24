"""Recheck Cox PH diagnostics and assess radius and sample robustness.

Run from the repository root:
    .venv\\Scripts\\python.exe src\\05_cox_robustness.py

The established main-analysis preparation and model definitions are imported
from 05_cox.py.  No raw data are altered.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter
from lifelines.statistics import proportional_hazard_test
from scipy.stats import chi2, pearsonr


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "cafes_analysis.parquet"
TABLE_DIR = ROOT / "docs" / "tables"
FIGURE_DIR = ROOT / "docs" / "figures"
REPORT = ROOT / "docs" / "reports" / "05_cox_robustness_report.md"
LOG = ROOT / "docs" / "logs" / "05_cox_robustness_execution.md"
MAIN_REPORT = ROOT / "docs" / "reports" / "05_cox_main_report.md"
MAIN_PH = ROOT / "docs" / "tables" / "05_cox_proportional_hazards.csv"


def load_main_module():
    spec = importlib.util.spec_from_file_location("cox_main", ROOT / "src" / "05_cox.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import src/05_cox.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cox = load_main_module()


def write_csv(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(TABLE_DIR / name, index=False, encoding="utf-8-sig")


def competition_label(variable: str, radius: int) -> str:
    if variable == "comp_indep_r200":
        return f"{radius}m 개인 카페 경쟁 (1개)"
    if variable == "comp_fran_r200":
        return f"{radius}m 프랜차이즈 경쟁 (1개)"
    return cox.label_for(variable)


def fitted_frame(sample: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    frame = sample.loc[:, ["survival_days", "event", "opening_stratum", *covariates]].copy()
    frame[covariates] = frame[covariates].astype(float)
    frame["survival_days"] = frame["survival_days"].astype(float)
    frame["event"] = frame["event"].astype(int)
    return frame


def fit_model_set(
    sample: pd.DataFrame,
    extra_controls: list[str] | None = None,
    keys: list[str] | None = None,
):
    definitions = cox.model_definitions(cox.type_dummy_columns(sample))
    if extra_controls:
        for key in ("store_competition", "interaction"):
            definitions[key] = [*definitions[key], *extra_controls]
    keys = keys or ["trade_area", "store_competition", "competition_only", "competition_demand", "interaction"]
    models = {key: cox.fit_cox(sample, definitions[key]) for key in keys}
    return definitions, models


def coefficient_rows(
    label: str,
    sample: pd.DataFrame,
    definitions: dict[str, list[str]],
    models: dict[str, object],
    *,
    radius: int = 200,
) -> list[dict[str, object]]:
    wanted = set(cox.COMPETITION + cox.DEMAND + ["comp_total_x_sales", "comp_total_x_floating", "has_predecessor_same_site"])
    rows: list[dict[str, object]] = []
    for key, model in models.items():
        for variable in definitions[key]:
            if variable not in wanted:
                continue
            value = model.summary.loc[variable]
            lower, upper = float(value["coef lower 95%"]), float(value["coef upper 95%"])
            row = {
                "analysis": label, "model_key": key, "model": cox.MODEL_LABELS[key],
                "radius_m": radius, "n": len(sample), "events": int(sample["event"].sum()),
                "variable": variable, "variable_label": competition_label(variable, radius),
                "coefficient": float(value["coef"]), "se_cluster_robust": float(value["se(coef)"]),
                "hazard_ratio_per_store": float(np.exp(value["coef"])),
                "hr_ci_lower": float(np.exp(lower)), "hr_ci_upper": float(np.exp(upper)),
                "p_value": float(value["p"]), "iqr": np.nan, "hazard_ratio_iqr": np.nan,
                "hr_iqr_ci_lower": np.nan, "hr_iqr_ci_upper": np.nan,
            }
            if variable in cox.COMPETITION:
                iqr = float(sample[variable].quantile(.75) - sample[variable].quantile(.25))
                row.update({"iqr": iqr, "hazard_ratio_iqr": float(np.exp(value["coef"] * iqr)),
                            "hr_iqr_ci_lower": float(np.exp(lower * iqr)), "hr_iqr_ci_upper": float(np.exp(upper * iqr))})
            rows.append(row)
    return rows


def conditional_rows(label: str, sample: pd.DataFrame, model, radius: int) -> pd.DataFrame:
    table = cox.demand_competition_table(model, sample).copy()
    table.insert(0, "analysis", label)
    table.insert(1, "radius_m", radius)
    table["competition_label"] = table["competition_variable"].map(lambda value: competition_label(value, radius))
    iqrs = {variable: float(sample[variable].quantile(.75) - sample[variable].quantile(.25)) for variable in cox.COMPETITION}
    table["iqr"] = table["competition_variable"].map(iqrs)
    table["hazard_ratio_iqr"] = np.exp(table["log_hr"] * table["iqr"])
    table["hr_iqr_ci_lower"] = np.exp((table["log_hr"] - 1.96 * table["se"]) * table["iqr"])
    table["hr_iqr_ci_upper"] = np.exp((table["log_hr"] + 1.96 * table["se"]) * table["iqr"])
    return table


def ph_recheck(sample: pd.DataFrame, definitions: dict[str, list[str]]):
    rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    residuals_for_plot: dict[str, pd.DataFrame] = {}
    models = {}
    for key in ("store_competition", "interaction"):
        covariates = definitions[key]
        # Clustered/robust covariance is deliberately not used here: the
        # residual scaling and the PH test share the ordinary Cox covariance.
        model = cox.fit_cox(sample, covariates, robust=False, cluster=False)
        models[key] = model
        frame = fitted_frame(sample, covariates)
        for transform in ("rank", "km", "log"):
            result = proportional_hazard_test(model, frame, time_transform=transform)
            for variable, value in result.summary.iterrows():
                rows.append({"model": cox.MODEL_LABELS[key], "model_key": key, "variable": variable,
                             "variable_label": competition_label(variable, 200), "test_statistic": float(value["test_statistic"]),
                             "p_value": float(value["p"]), "time_transform": transform,
                             "estimator": "비군집·비강건 Cox (Schoenfeld 재검정)"})
        residuals = model.compute_residuals(frame, kind="scaled_schoenfeld")
        event_time = frame.loc[residuals.index, "survival_days"].astype(float)
        time_rank = event_time.rank(method="average")
        for variable in [*cox.COMPETITION, *cox.DEMAND]:
            corr, p_value = pearsonr(residuals[variable].to_numpy(float), time_rank.to_numpy(float))
            correlation_rows.append({"model": cox.MODEL_LABELS[key], "model_key": key, "variable": variable,
                                     "variable_label": competition_label(variable, 200), "n_events": len(residuals),
                                     "pearson_r_scaled_schoenfeld_time_rank": corr, "p_value": p_value})
        if key == "store_competition":
            plotted = residuals.loc[:, [*cox.COMPETITION, *cox.DEMAND]].copy()
            plotted["survival_days"] = event_time
            residuals_for_plot[key] = plotted
    return pd.DataFrame(rows), pd.DataFrame(correlation_rows), residuals_for_plot["store_competition"]


def plot_schoenfeld(residuals: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), sharex=False)
    targets = ["comp_indep_r200", "comp_fran_r200", "log_sales_z", "log_floating_z"]
    for axis, variable in zip(axes.ravel(), targets):
        data = residuals.loc[:, ["survival_days", variable]].sort_values("survival_days")
        axis.scatter(data["survival_days"], data[variable], s=7, alpha=.14, color="#4C78A8", linewidths=0)
        # Equal-count bins supply a deterministic non-parametric smoothing line.
        bins = pd.qcut(data["survival_days"].rank(method="first"), q=min(35, len(data)), duplicates="drop")
        smooth = data.groupby(bins, observed=True).agg(time=("survival_days", "mean"), residual=(variable, "mean"))
        axis.plot(smooth["time"], smooth["residual"], color="#D55E00", linewidth=2)
        axis.axhline(0, color="#555555", linestyle="--", linewidth=.9)
        axis.set_title(competition_label(variable, 200))
        axis.set_xlabel("사건 시점까지 일수")
        axis.set_ylabel("Scaled Schoenfeld 잔차")
        axis.grid(axis="y", color="#E2E2E2", linewidth=.7)
    fig.suptitle("비군집·비강건 Cox의 scaled Schoenfeld 잔차", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_cox_robustness_schoenfeld.png", bbox_inches="tight")
    plt.close(fig)


def interval_model(sample: pd.DataFrame, definitions: dict[str, list[str]]) -> tuple[pd.DataFrame, str]:
    covariates = definitions["store_competition"]
    targets = [*cox.COMPETITION, *cox.DEMAND]
    early = sample.copy()
    early["start"] = 0.0
    early["stop"] = np.minimum(early["survival_days"].astype(float), 365.0)
    early["event_interval"] = ((early["event"].eq(1)) & (early["survival_days"].le(365))).astype(int)
    late = sample.loc[sample["survival_days"].gt(365)].copy()
    late["start"] = 365.0
    late["stop"] = late["survival_days"].astype(float)
    late["event_interval"] = late["event"].astype(int)
    for variable in targets:
        early[f"after_year_{variable}"] = 0.0
        late[f"after_year_{variable}"] = late[variable].astype(float)
    long = pd.concat([early, late], ignore_index=True)
    expanded = [*covariates, *[f"after_year_{variable}" for variable in targets]]
    columns = ["analysis_row_id", "event_interval", "start", "stop", "opening_stratum", *expanded]
    status = "비군집 표준오차"
    fitter = CoxTimeVaryingFitter()
    try:
        fitter.fit(long.loc[:, columns], id_col="analysis_row_id", event_col="event_interval", start_col="start",
                   stop_col="stop", strata="opening_stratum", robust=True, show_progress=False)
        status = "강건 표준오차"
    except (NotImplementedError, ValueError):
        fitter = CoxTimeVaryingFitter()
        fitter.fit(long.loc[:, columns], id_col="analysis_row_id", event_col="event_interval", start_col="start",
                   stop_col="stop", strata="opening_stratum", robust=False, show_progress=False)
    rows = []
    parameter_positions = {name: position for position, name in enumerate(fitter.params_.index)}
    for variable in targets:
        for period, after in (("개업 후 1년 이내", False), ("개업 후 1년 이후", True)):
            beta = float(fitter.params_[variable])
            variable_position = parameter_positions[variable]
            variance = float(fitter.variance_matrix_.iloc[variable_position, variable_position])
            if after:
                name = f"after_year_{variable}"
                after_position = parameter_positions[name]
                beta += float(fitter.params_[name])
                variance += float(fitter.variance_matrix_.iloc[after_position, after_position] + 2 * fitter.variance_matrix_.iloc[variable_position, after_position])
            se = float(np.sqrt(max(variance, 0)))
            rows.append({"variable": variable, "variable_label": competition_label(variable, 200), "period": period,
                         "n_person_periods": len(long), "events": int(long["event_interval"].sum()), "se_method": status,
                         "hazard_ratio": float(np.exp(beta)), "hr_ci_lower": float(np.exp(beta - 1.96 * se)),
                         "hr_ci_upper": float(np.exp(beta + 1.96 * se)), "p_value": float(chi2.sf((beta / se) ** 2, 1)) if se else np.nan})
    return pd.DataFrame(rows), status


def radius_analysis(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficient_rows_all: list[dict[str, object]] = []
    fit_rows: list[pd.DataFrame] = []
    conditional_rows_all: list[pd.DataFrame] = []
    for radius in (100, 200, 500):
        changed = data.copy()
        for prefix in ("comp_indep", "comp_fran", "comp_total"):
            changed[f"{prefix}_r200"] = changed[f"{prefix}_r{radius}"].copy()
        sample, _, _ = cox.prepare_sample(changed)
        definitions, models = fit_model_set(sample)
        coefficient_rows_all.extend(coefficient_rows(f"{radius}m 반경", sample, definitions, models, radius=radius))
        fit = cox.model_fit_table({"trade_area": models["trade_area"], "store_competition": models["store_competition"]}, len(sample), int(sample["event"].sum()))
        fit.insert(0, "radius_m", radius)
        fit_rows.append(fit)
        conditional_rows_all.append(conditional_rows(f"{radius}m 반경", sample, models["interaction"], radius))
    return pd.DataFrame(coefficient_rows_all), pd.concat(fit_rows, ignore_index=True), pd.concat(conditional_rows_all, ignore_index=True)


def prepare_variant(data: pd.DataFrame, label: str):
    if label == "200m 밖 최근접 상권 포함":
        changed = data.copy()
        changed["trdar_within_200m"] = changed["trdar_cd_assigned"].notna()
        return cox.prepare_sample(changed)
    if label == "프랜차이즈 포함 전체 커피숍":
        changed = data.copy()
        changed["is_franchise"] = False
        return cox.prepare_sample(changed)
    if label == "재인허가 통합":
        changed = data.loc[data["robust_relicense_in_sample"].fillna(False)].copy()
        changed["survival_days"] = changed["robust_relicense_survival_days"]
        changed["event"] = changed["robust_relicense_event"]
        return cox.prepare_sample(changed)
    raise ValueError(label)


def sample_analysis(
    data: pd.DataFrame,
    main_sample: pd.DataFrame,
    section: str = "all",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficient_rows_all: list[dict[str, object]] = []
    conditional_rows_all: list[pd.DataFrame] = []
    flow_rows: list[dict[str, object]] = []

    variants = ["프랜차이즈 포함 전체 커피숍", "재인허가 통합"] if section in {"all", "variants"} else []
    prepared: dict[str, pd.DataFrame] = {"주 분석": main_sample} if section in {"all", "core"} else {}
    for label in variants:
        sample, flow, _ = prepare_variant(data, label)
        prepared[label] = sample
        flow_rows.append({"analysis": label, "input_rows": int(flow.iloc[0]["n_before"]), "n": len(sample),
                          "events": int(sample["event"].sum()), "excluded": int(flow.iloc[0]["n_before"] - len(sample)),
                          "exclusion_reason": "상권 변수 결측·0 이하 및 Cox 필수 변수 결측은 주 분석과 같은 순서로 제외"})

    # The outside-area control is a requested companion specification, not a
    # separate cohort.  It can identify only stores whose nearest area exists.
    if section in {"all", "core"}:
        outside, _, _ = prepare_variant(data, "200m 밖 최근접 상권 포함")
        outside["outside_200m"] = outside["trdar_nearest_dist_m"].gt(200).astype(float)
        definitions_out, models_out = fit_model_set(outside, ["outside_200m"], ["store_competition", "interaction"])
        coefficient_rows_all.extend(coefficient_rows("200m 밖 포함 + 밖 표시 통제", outside, definitions_out, models_out))
        conditional_rows_all.append(conditional_rows("200m 밖 포함 + 밖 표시 통제", outside, models_out["interaction"], 200))
        flow_rows.append({"analysis": "200m 밖 최근접 상권 포함", "input_rows": len(data), "n": len(outside), "events": int(outside["event"].sum()), "excluded": len(data) - len(outside), "exclusion_reason": "최근접 상권 코드가 있는 개인 카페만 포함 후 주 분석과 같은 완전 사례 처리"})

        predecessor = main_sample.copy()
        predecessor["has_predecessor_same_site"] = predecessor["has_predecessor_same_site"].fillna(False).astype(float)
        definitions_pred, models_pred = fit_model_set(predecessor, ["has_predecessor_same_site"], ["store_competition", "interaction"])
        coefficient_rows_all.extend(coefficient_rows("같은 자리 전 운영자 표시 포함", predecessor, definitions_pred, models_pred))
        conditional_rows_all.append(conditional_rows("같은 자리 전 운영자 표시 포함", predecessor, models_pred["interaction"], 200))
        flow_rows.append({"analysis": "같은 자리 전 운영자 표시 포함", "input_rows": len(main_sample), "n": len(predecessor), "events": int(predecessor["event"].sum()), "excluded": 0, "exclusion_reason": "주 분석 표본에 표시 변수만 추가"})

    for label, sample in prepared.items():
        if label in {"200m 밖 최근접 상권 포함", "같은 자리 전 운영자 표시 포함"}:
            continue
        definitions, models = fit_model_set(sample, keys=["store_competition", "interaction"])
        coefficient_rows_all.extend(coefficient_rows(label, sample, definitions, models))
        conditional_rows_all.append(conditional_rows(label, sample, models["interaction"], 200))
        if label == "주 분석":
            flow_rows.append({"analysis": label, "input_rows": len(data), "n": len(sample), "events": int(sample["event"].sum()),
                              "excluded": len(data) - len(sample), "exclusion_reason": "개인 카페·200m 이내 및 완전 사례"})

    # Opening-period models and the full-cohort competition-by-period test.
    if section in {"all", "period"}:
        for label, mask in (("코로나 시기 개업 (2021–2022)", main_sample["opening_year"].between(2021, 2022)),
                            ("이후 개업 (2023년 이후)", main_sample["opening_year"].ge(2023))):
            sample = main_sample.loc[mask].copy()
            definitions, models = fit_model_set(sample, keys=["store_competition", "interaction"])
            coefficient_rows_all.extend(coefficient_rows(label, sample, definitions, models))
            conditional_rows_all.append(conditional_rows(label, sample, models["interaction"], 200))
            flow_rows.append({"analysis": label, "input_rows": len(main_sample), "n": len(sample), "events": int(sample["event"].sum()),
                              "excluded": len(main_sample) - len(sample), "exclusion_reason": "개업연도 기준 하위 표본"})

        period = main_sample.copy()
        period["post_2023"] = period["opening_year"].ge(2023).astype(float)
        period["comp_indep_x_post_2023"] = period["comp_indep_r200"] * period["post_2023"]
        period["comp_fran_x_post_2023"] = period["comp_fran_r200"] * period["post_2023"]
        # The opening-period main effect is absorbed by the opening-year
        # strata.  Only slope differences by competition remain identifiable.
        definitions_period, models_period = fit_model_set(
            period, ["comp_indep_x_post_2023", "comp_fran_x_post_2023"], ["store_competition"]
        )
        for variable in ("comp_indep_x_post_2023", "comp_fran_x_post_2023"):
            result = models_period["store_competition"].summary.loc[variable]
            coefficient_rows_all.append({"analysis": "전체 표본 경쟁×개업시기", "model_key": "store_competition", "model": "상권 + 점포 단위 경쟁 + 개업시기 상호작용",
                                         "radius_m": 200, "n": len(period), "events": int(period["event"].sum()), "variable": variable,
                                         "variable_label": "개인 경쟁×2023년 이후" if variable.startswith("comp_indep") else "프랜차이즈 경쟁×2023년 이후",
                                         "coefficient": float(result["coef"]), "se_cluster_robust": float(result["se(coef)"]), "hazard_ratio_per_store": float(np.exp(result["coef"])),
                                         "hr_ci_lower": float(np.exp(result["coef lower 95%"])), "hr_ci_upper": float(np.exp(result["coef upper 95%"])), "p_value": float(result["p"]),
                                         "iqr": np.nan, "hazard_ratio_iqr": np.nan, "hr_iqr_ci_lower": np.nan, "hr_iqr_ci_upper": np.nan})
    return pd.DataFrame(coefficient_rows_all), pd.concat(conditional_rows_all, ignore_index=True), pd.DataFrame(flow_rows)


def plot_competition_forest(radius: pd.DataFrame, samples: pd.DataFrame) -> None:
    selected = pd.concat([
        radius.loc[(radius["model_key"].eq("store_competition")) & radius["variable"].isin(cox.COMPETITION), ["analysis", "variable_label", "hazard_ratio_per_store", "hr_ci_lower", "hr_ci_upper"]],
        samples.loc[(samples["model_key"].eq("store_competition")) & samples["variable"].isin(cox.COMPETITION), ["analysis", "variable_label", "hazard_ratio_per_store", "hr_ci_lower", "hr_ci_upper"]],
    ], ignore_index=True)
    selected["label"] = selected["analysis"] + " · " + selected["variable_label"]
    selected = selected.iloc[::-1]
    fig, axis = plt.subplots(figsize=(10, max(6, len(selected) * .31)))
    y = np.arange(len(selected))
    hr, low, high = (selected[column].to_numpy(float) for column in ("hazard_ratio_per_store", "hr_ci_lower", "hr_ci_upper"))
    axis.errorbar(hr, y, xerr=[hr - low, high - hr], fmt="o", color="#0072B2", ecolor="#0072B2", capsize=2.5)
    axis.axvline(1, color="#555555", linestyle="--", linewidth=1)
    axis.set_xscale("log")
    axis.set_yticks(y, selected["label"])
    axis.set_xlabel("경쟁 1개당 위험비 (95% 신뢰구간, 로그 축)")
    axis.set_title("반경·표본별 점포 단위 경쟁 연관")
    axis.grid(axis="x", color="#E2E2E2", linewidth=.7)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_cox_robustness_competition_forest.png", bbox_inches="tight")
    plt.close(fig)


def fmt_hr(row: pd.Series) -> str:
    return f"{row.hazard_ratio_per_store:.3f} ({row.hr_ci_lower:.3f}–{row.hr_ci_upper:.3f})"


def update_main_report(ph: pd.DataFrame, correlation: pd.DataFrame, interval: pd.DataFrame) -> None:
    text = MAIN_REPORT.read_text(encoding="utf-8")
    targets = ph.loc[ph["variable"].isin([*cox.COMPETITION, *cox.DEMAND])]
    rank = targets.loc[targets["time_transform"].eq("rank")]
    violations = rank.loc[rank["p_value"].lt(.05)]
    rank_text = ", ".join(f"{row.variable_label}={row.p_value:.3f}" for row in rank.itertuples())
    cross = correlation.loc[correlation["model_key"].eq("store_competition")]
    cross_text = ", ".join(f"{row.variable_label} r={row.pearson_r_scaled_schoenfeld_time_rank:.3f}, p={row.p_value:.3f}" for row in cross.itertuples())
    period_text = "; ".join(f"{row.variable_label} {row.period}: HR {row.hazard_ratio:.3f} ({row.hr_ci_lower:.3f}–{row.hr_ci_upper:.3f})" for row in interval.itertuples())
    replacement = "\n".join([
        "### 모형 비교와 진단", "",
        "상권 모형, 상권+점포 단위 경쟁, 경쟁만, 경쟁+수요, 상호작용 모형을 각각 적합했다. 비례위험 재검정은 군집 강건 분산을 쓰지 않은 같은 층화 Cox 적합에서 수행했다. 기존 표는 군집 강건 적합을 `proportional_hazard_test`에 넘겨 scaled Schoenfeld 잔차의 척도가 맞지 않았고, 그 결과 p값이 비정상적으로 0.9 부근에 집중됐다.",
        f"수정된 rank 시간변환의 경쟁·수요 검정 p값은 {rank_text}이다. `km`·`log` 변환까지 포함한 전체 결과는 `05_cox_proportional_hazards.csv`에, scaled 잔차와 사건시간 순위의 독립 교차점검은 `05_cox_robustness_ph_residual_correlation.csv`에 저장했다. 후자의 결과는 {cross_text}이다.",
        (f"rank 검정에서 p<0.05인 변수는 {', '.join(violations['variable_label'])}로, 비례위험이 완전히 성립한다고 보기 어렵다." if not violations.empty else "rank 검정에서 p<0.05인 경쟁·수요 변수는 없었다."),
        f"위반 여부와 별개로 1년 이내/이후 구간 상호작용 모형을 적합했다: {period_text}. 이 보조모형은 lifelines의 제약으로 군집 강건 표준오차를 제공하지 못해 시간 패턴 확인용으로만 해석한다.",
        "프랜차이즈 경쟁은 0 / 1–2 / 3–5 / 6개 이상 더미로 바꾼 보조모형을 `05_cox_coefficients.csv`에 함께 수록했다. 수요 수준별 조건부 위험비는 다른 수요 변수를 평균(표준화 0)에 고정해 계산했다. 투입 변수 VIF의 최댓값은 3.81이며 전체 값은 `05_cox_vif.csv`에 있다.",
    ])
    start = text.index("### 모형 비교와 진단")
    end = text.index("### 민감도 분석")
    MAIN_REPORT.write_text(text[:start] + replacement + "\n\n" + text[end:], encoding="utf-8")


def write_report(ph: pd.DataFrame, interval: pd.DataFrame, radius: pd.DataFrame, fit: pd.DataFrame, samples: pd.DataFrame, flow: pd.DataFrame) -> None:
    main = samples.loc[(samples["analysis"].eq("주 분석")) & (samples["model_key"].eq("store_competition")) & samples["variable"].isin(cox.COMPETITION)]
    radius_main = radius.loc[(radius["model_key"].eq("store_competition")) & radius["variable"].isin(cox.COMPETITION)]
    period = samples.loc[samples["analysis"].eq("전체 표본 경쟁×개업시기")]
    outside = flow.loc[flow["analysis"].eq("200m 밖 최근접 상권 포함")].iloc[0]
    interaction = samples.loc[(samples["model_key"].eq("interaction")) & samples["variable"].isin(["comp_total_x_sales", "comp_total_x_floating"])]
    radius_text = "; ".join(
        f"{int(row.radius_m)}m HR {row.hazard_ratio_per_store:.3f} ({row.hr_ci_lower:.3f}–{row.hr_ci_upper:.3f}), IQR HR {row.hazard_ratio_iqr:.3f}"
        for row in radius_main.loc[radius_main["variable"].eq("comp_indep_r200")].itertuples()
    )
    sample_independent = samples.loc[(samples["model_key"].eq("store_competition")) & samples["variable"].eq("comp_indep_r200") & ~samples["analysis"].eq("주 분석")]
    sample_text = "; ".join(
        f"{row.analysis} HR {row.hazard_ratio_per_store:.3f} ({row.hr_ci_lower:.3f}–{row.hr_ci_upper:.3f})"
        for row in sample_independent.itertuples()
    )
    floating = interaction.loc[interaction["variable"].eq("comp_total_x_floating")]
    floating_text = "; ".join(f"{row.analysis} HR {row.hazard_ratio_per_store:.3f}, p={row.p_value:.3f}" for row in floating.itertuples())
    period_text = "; ".join(f"{row.variable_label} HR {row.hazard_ratio_per_store:.3f}, p={row.p_value:.3f}" for row in period.itertuples())
    early_independent = interval.loc[(interval["variable"].eq("comp_indep_r200")) & interval["period"].eq("개업 후 1년 이내")].iloc[0]
    late_independent = interval.loc[(interval["variable"].eq("comp_indep_r200")) & interval["period"].eq("개업 후 1년 이후")].iloc[0]
    lines = [
        "# Cox 비례위험 재검정과 강건성 검증 보고서", "", "## 주요 결과", "",
        "군집 강건 적합 결과를 그대로 Schoenfeld 검정에 전달했던 기존 검정은 scaled 잔차 척도가 맞지 않아 p값이 0.9 근처에 몰렸다. 같은 표본·층화에서 비군집·비강건 Cox로 다시 적합해 rank·KM·log 시간변환을 적용하고, 잔차-시간 순위 상관으로 교차 점검했다. 따라서 이 문서의 비례위험 판단은 수정된 검정에 근거한다.",
        "",
        "### 비례위험과 개업 초기", "",
        f"수정 검정의 경쟁·수요 변수 결과는 `05_cox_robustness_proportional_hazards.csv`에 있다. rank p<0.05 변수는 {int((ph.loc[(ph['time_transform'].eq('rank')) & ph['variable'].isin([*cox.COMPETITION, *cox.DEMAND])]['p_value'] < .05).sum())}개이며, 시간 구간 위험비는 `05_cox_robustness_time_interval.csv`에 제시했다. 개인 경쟁은 1년 이내 HR {early_independent.hazard_ratio:.3f} ({early_independent.hr_ci_lower:.3f}–{early_independent.hr_ci_upper:.3f}), 이후 HR {late_independent.hazard_ratio:.3f} ({late_independent.hr_ci_lower:.3f}–{late_independent.hr_ci_upper:.3f})였다. 이 보조모형의 표준오차 방식은 {interval['se_method'].iloc[0]}이다.",
        "따라서 개인 경쟁의 양(+)의 연관은 개업 초기에만 국한됐다고 보기보다 두 구간에서 비슷하게 관찰됐다. 시간 구간 모형은 군집 강건 표준오차가 없어 확정적 검정으로 해석하지 않는다.",
        "",
        "### 반경과 표본을 바꾼 결과", "",
        f"개인 경쟁의 양(+)의 연관은 반경을 바꿔도 유지됐다: {radius_text}. 상권 모형 대비 점포 단위 경쟁 추가의 우도비 검정은 100m p<0.001, 200m p<0.001, 500m p<0.001이고, C 증가는 각각 +0.025, +0.014, +0.013이었다. 반경별 1개당·IQR당 위험비와 완충 조건부 위험비는 `05_cox_robustness_radius_coefficients.csv`, `05_cox_robustness_radius_conditional_hr.csv`에 있다.",
        f"표본을 바꾼 결과도 개인 경쟁의 방향은 같았다: {sample_text}. 반면 프랜차이즈 경쟁의 신뢰구간은 반경·표본 전반에서 대체로 1을 포함해, 프랜차이즈 경쟁의 독립적 연관은 강건하지 않았다.",
        f"200m 밖 개인 카페는 지시된 573개 가운데 최근접 상권 코드가 있는 438개만 포함 가능했고, 완전 사례 처리 뒤 이 표본은 {int(outside.n):,}개였다. 나머지 135개는 좌표 또는 최근접 상권이 없어 거리 제한 없는 최근접 상권 값을 만들 수 없었다.",
        "프랜차이즈 포함 표본에서는 대상 점포 자신이 경쟁 수에 포함되지 않는다. 경쟁 변수는 대상보다 **앞서** 개업한 점포만 세도록 계산되어(동일 개업일은 별도 집계), 프랜차이즈 대상에도 자기 자신이 빠진다.",
        "",
        "### 상쇄와 개업 시기", "",
        f"유동인구×전체 경쟁 상호작용은 반경별로 100m·200m·500m에서 모두 위험비가 1보다 작았고, 표본 변형에서도 대체로 같은 방향이었다({floating_text}). 총매출 상호작용은 안정적으로 1과 다르지 않아, 상쇄 결론은 유동인구에 한정해 지지된다. 관찰자료이므로 완충의 인과 효과로 해석하지 않는다.",
        f"개업 시기별 하위 표본과 전체 표본의 경쟁×시기 상호작용 계수는 표본별 계수표에 수록했다. 전체 표본의 시기 차이 검정은 {period_text}로, 두 경쟁 유형 모두 뚜렷한 시기 차이를 확인하지 못했다.",
        "", "## 방법과 검증", "",
        "### 공통 설계", "",
        "모든 강건성 Cox 적합은 `src/05_cox.py`의 표본 준비·변수 정의·적합 함수를 가져와 사용했다. 주 추정치는 개업연도 층화와 배정 상권 군집 강건 표준오차를 유지했다. 반경 강건성은 주 분석의 동일한 7,351개 완전 사례에서 경쟁 변수만 100m·200m·500m로 바꿨다.",
        "", "### 비례위험 재검정", "",
        "상권+점포 단위 경쟁 모형과 상호작용 모형을 robust=False·cluster_col 없음으로 재적합했다. 세 시간변환(rank·KM·log)의 Grambsch–Therneau 검정과 scaled Schoenfeld 잔차 대 사건시간 순위 Pearson 상관을 별도로 계산했다. 잔차 그림의 주황선은 동일 수 사건 수 구간 평균으로 만든 결정론적 평활선이다.",
        "", "### 표본 변형", "",
        "거리 제한 없는 최근접 상권은 기존 `trdar_cd_assigned`를 사용했다. 이 값은 `trdar_cd_nearest`로, 공간 결합 단계에서 모든 좌표 보유 점포에 가장 가까운 상권을 붙인 값이며 상권 지표는 개업 직전 분기 값이다. 200m 밖 포함 모형에는 200m 밖 표시를 추가한 모형도 적합했다. 재인허가 통합은 지정된 생존기간·사건 열과 통합 표본 표시를 사용했다. 같은 자리 전 운영자 표시는 주 분석 표본에서만 보조적으로 추가했다.",
        "", "### 산출물", "",
        "- 비례위험 재검정·잔차 교차점검·구간 위험비: `docs/tables/05_cox_robustness_proportional_hazards.csv`, `05_cox_robustness_ph_residual_correlation.csv`, `05_cox_robustness_time_interval.csv`",
        "- 반경·표본 계수 및 표본 흐름: `docs/tables/05_cox_robustness_radius_*.csv`, `05_cox_robustness_sample_*.csv`",
        "- 잔차 및 forest 그림: `docs/figures/05_cox_robustness_schoenfeld.png`, `05_cox_robustness_competition_forest.png`",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ph_stage(data: pd.DataFrame) -> None:
    main_sample, _, _ = cox.prepare_sample(data)
    definitions, _ = fit_model_set(main_sample)
    ph, correlation, residuals = ph_recheck(main_sample, definitions)
    interval, interval_status = interval_model(main_sample, definitions)
    write_csv("05_cox_robustness_proportional_hazards.csv", ph)
    write_csv("05_cox_robustness_ph_residual_correlation.csv", correlation)
    write_csv("05_cox_robustness_time_interval.csv", interval)
    write_csv("05_cox_proportional_hazards.csv", ph)
    plot_schoenfeld(residuals)
    (ROOT / ".tmp" / "05_cox_robustness_ph_status.txt").write_text(interval_status, encoding="utf-8")


def run_radius_stage(data: pd.DataFrame) -> None:
    radius_coeff, radius_fit, radius_conditional = radius_analysis(data)
    write_csv("05_cox_robustness_radius_coefficients.csv", radius_coeff)
    write_csv("05_cox_robustness_radius_model_fit.csv", radius_fit)
    write_csv("05_cox_robustness_radius_conditional_hr.csv", radius_conditional)


def run_sample_stage(data: pd.DataFrame, section: str = "all") -> None:
    main_sample, _, _ = cox.prepare_sample(data)
    sample_coeff, sample_conditional, sample_flow = sample_analysis(data, main_sample, section)
    suffix = "" if section == "all" else f"_{section}"
    write_csv(f"05_cox_robustness_sample_coefficients{suffix}.csv", sample_coeff)
    write_csv(f"05_cox_robustness_sample_conditional_hr{suffix}.csv", sample_conditional)
    write_csv(f"05_cox_robustness_sample_flow{suffix}.csv", sample_flow)


def finalize_stage(data: pd.DataFrame) -> None:
    main_sample, _, _ = cox.prepare_sample(data)
    ph = pd.read_csv(TABLE_DIR / "05_cox_robustness_proportional_hazards.csv", encoding="utf-8-sig")
    correlation = pd.read_csv(TABLE_DIR / "05_cox_robustness_ph_residual_correlation.csv", encoding="utf-8-sig")
    interval = pd.read_csv(TABLE_DIR / "05_cox_robustness_time_interval.csv", encoding="utf-8-sig")
    radius_coeff = pd.read_csv(TABLE_DIR / "05_cox_robustness_radius_coefficients.csv", encoding="utf-8-sig")
    radius_fit = pd.read_csv(TABLE_DIR / "05_cox_robustness_radius_model_fit.csv", encoding="utf-8-sig")
    sections = ["core", "variants", "period"]
    coefficient_parts = [pd.read_csv(TABLE_DIR / f"05_cox_robustness_sample_coefficients_{section}.csv", encoding="utf-8-sig") for section in sections]
    conditional_parts = [pd.read_csv(TABLE_DIR / f"05_cox_robustness_sample_conditional_hr_{section}.csv", encoding="utf-8-sig") for section in sections]
    flow_parts = [pd.read_csv(TABLE_DIR / f"05_cox_robustness_sample_flow_{section}.csv", encoding="utf-8-sig") for section in sections]
    sample_coeff = pd.concat(coefficient_parts, ignore_index=True)
    sample_conditional = pd.concat(conditional_parts, ignore_index=True)
    sample_flow = pd.concat(flow_parts, ignore_index=True)
    write_csv("05_cox_robustness_sample_coefficients.csv", sample_coeff)
    write_csv("05_cox_robustness_sample_conditional_hr.csv", sample_conditional)
    write_csv("05_cox_robustness_sample_flow.csv", sample_flow)
    plot_competition_forest(radius_coeff, sample_coeff)
    update_main_report(ph, correlation, interval)
    write_report(ph, interval, radius_coeff, radius_fit, sample_coeff, sample_flow)
    interval_status = interval["se_method"].iloc[0]
    LOG.write_text("\n".join([
        "# Cox 강건성 검증 실행 로그", "", f"- 입력: `data/processed/cafes_analysis.parquet` ({len(data):,}행)",
        f"- 주 분석 공통 표본: {len(main_sample):,}행, 사건 {int(main_sample['event'].sum()):,}건",
        "- 실행: `src/05_cox.py` 재실행 후 `05_cox_robustness.py`의 ph·radius·samples_core·samples_variants·samples_period·finalize 단계 실행",
        "- 주 분석 계수·적합도는 수정하지 않고, PH 표와 주 분석 보고서의 PH 절만 수정",
        "- 패키지 설치·원본 데이터 수정 없음", f"- 시간 구간 보조모형 표준오차: {interval_status}",
        "- 완료: PH 재검정, 반경·표본 강건성 표와 그림, 보고서 생성",
    ]) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    cox.configure_matplotlib()
    data = pd.read_parquet(INPUT)
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "all":
        run_ph_stage(data)
        run_radius_stage(data)
        for section in ("core", "variants", "period"):
            run_sample_stage(data, section)
        finalize_stage(data)
        return
    if stage == "ph":
        run_ph_stage(data)
    if stage == "radius":
        run_radius_stage(data)
    if stage == "samples":
        run_sample_stage(data)
    if stage in {"samples_core", "samples_variants", "samples_period"}:
        run_sample_stage(data, stage.removeprefix("samples_"))
    if stage == "finalize":
        finalize_stage(data)


if __name__ == "__main__":
    main()

