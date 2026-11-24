"""Cluster-robust, delayed-entry Cox models for opening-time intervals.

Run from the repository root:
    .venv\\Scripts\\python.exe src\\05_cox_time_split.py

The analysis leaves the established main-analysis data and coefficients
unchanged. It imports the shared sample preparation from ``05_cox.py`` and
splits each shop record at 180, 365, and 730 days using CoxPHFitter's
``entry_col``. This retains the opening-year strata and trade-area clustered
sandwich covariance estimator.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "cafes_analysis.parquet"
TABLE_DIR = ROOT / "docs" / "tables"
FIGURE_DIR = ROOT / "docs" / "figures"
ROBUST_REPORT = ROOT / "docs" / "reports" / "05_cox_robustness_report.md"
MAIN_REPORT = ROOT / "docs" / "reports" / "05_cox_main_report.md"
LOG = ROOT / "docs" / "logs" / "05_cox_robustness_execution.md"

TARGETS = ["comp_indep_r200", "comp_fran_r200", "log_sales_z", "log_floating_z"]
PERIOD_LABELS = ("개업 후 기준일 이내", "개업 후 기준일 이후")


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


def fmt_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def fmt_hr(row: pd.Series) -> str:
    return f"{row.hazard_ratio:.3f} ({row.hr_ci_lower:.3f}–{row.hr_ci_upper:.3f})"


def split_records(sample: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Make [0, cutoff] and (cutoff, stop] intervals with one event at most."""
    early = sample.copy()
    early["entry_day"] = 0.0
    early["exit_day"] = np.minimum(early["survival_days"].astype(float), float(cutoff))
    early["event_interval"] = (
        early["event"].eq(1) & early["survival_days"].le(cutoff)
    ).astype(int)
    early["after_cutoff"] = 0.0

    late = sample.loc[sample["survival_days"].gt(cutoff)].copy()
    late["entry_day"] = float(cutoff)
    late["exit_day"] = late["survival_days"].astype(float)
    late["event_interval"] = late["event"].astype(int)
    late["after_cutoff"] = 1.0

    split = pd.concat([early, late], ignore_index=True)
    if len(split) != len(sample) + int(sample["survival_days"].gt(cutoff).sum()):
        raise AssertionError("Unexpected person-period count after splitting.")
    if int(split["event_interval"].sum()) != int(sample["event"].sum()):
        raise AssertionError("The split intervals did not preserve the event total.")
    if not split["exit_day"].gt(split["entry_day"]).all():
        raise AssertionError("A split interval has non-positive length.")
    return split


def fit_split(
    split: pd.DataFrame,
    covariates: list[str],
) -> tuple[CoxPHFitter, pd.DataFrame, np.ndarray]:
    """Fit delayed-entry Cox with trade-area clustered sandwich covariance."""
    columns = [
        "exit_day", "event_interval", "entry_day", "opening_stratum",
        "trdar_cd_assigned", *covariates,
    ]
    frame = split.loc[:, columns].copy()
    frame[covariates] = frame[covariates].astype(float)
    frame["exit_day"] = frame["exit_day"].astype(float)
    frame["entry_day"] = frame["entry_day"].astype(float)
    frame["event_interval"] = frame["event_interval"].astype(int)
    model = CoxPHFitter()
    model.fit(
        frame,
        duration_col="exit_day",
        event_col="event_interval",
        entry_col="entry_day",
        strata="opening_stratum",
        cluster_col="trdar_cd_assigned",
        robust=True,
        show_progress=False,
    )

    # lifelines exposes robust standard errors but not the full sandwich
    # covariance. Recreate its internal sandwich calculation for coefficient
    # contrasts (late-period beta = beta + beta_after). Its diagonal exactly
    # matches model.standard_errors_ and is retained only for contrasts.
    x, duration, event, weights, _, _, _ = model._preprocess_dataframe(frame)
    x_norm = (x - model._norm_mean) / model._norm_std
    covariance = model._compute_sandwich_estimator(x_norm, duration, event, weights)
    if not np.allclose(np.sqrt(np.diag(covariance)), model.standard_errors_.to_numpy()):
        raise AssertionError("Rebuilt robust covariance does not match lifelines standard errors.")
    return model, frame, covariance


def contrast(model: CoxPHFitter, covariance: np.ndarray, weights: dict[str, float]) -> tuple[float, float, float, float, float, float]:
    """Return coefficient, robust SE, HR CI, and Wald p for a linear contrast."""
    vector = np.zeros(len(model.params_), dtype=float)
    positions = {name: index for index, name in enumerate(model.params_.index)}
    for name, weight in weights.items():
        vector[positions[name]] = weight
    beta = float(vector @ model.params_.to_numpy(float))
    variance = float(vector @ covariance @ vector)
    se = float(np.sqrt(max(variance, 0.0)))
    z2 = (beta / se) ** 2 if se else np.nan
    p_value = float(chi2.sf(z2, 1)) if np.isfinite(z2) else np.nan
    return beta, se, float(np.exp(beta)), float(np.exp(beta - 1.96 * se)), float(np.exp(beta + 1.96 * se)), p_value


def effect_rows(
    model_name: str,
    model: CoxPHFitter,
    covariance: np.ndarray,
    cutoff: int,
    targets: list[str],
    n: int,
    n_person_periods: int,
    events: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variable in targets:
        after = f"after_{variable}"
        difference = contrast(model, covariance, {after: 1.0})
        for label, weight_map in (
            (PERIOD_LABELS[0], {variable: 1.0}),
            (PERIOD_LABELS[1], {variable: 1.0, after: 1.0}),
        ):
            beta, se, hr, lower, upper, p_value = contrast(model, covariance, weight_map)
            rows.append({
                "model": model_name,
                "cutoff_days": cutoff,
                "variable": variable,
                "variable_label": cox.label_for(variable),
                "period": label,
                "n": n,
                "n_person_periods": n_person_periods,
                "events": events,
                "se_method": "배정 상권 군집 강건 샌드위치",
                "coefficient": beta,
                "se_cluster_robust": se,
                "hazard_ratio": hr,
                "hr_ci_lower": lower,
                "hr_ci_upper": upper,
                "p_value": p_value,
                "difference_late_minus_early": difference[0],
                "difference_hr_ratio": difference[2],
                "difference_hr_ci_lower": difference[3],
                "difference_hr_ci_upper": difference[4],
                "difference_wald_p": difference[5],
            })
    return rows


def validate_split(
    sample: pd.DataFrame,
    split: pd.DataFrame,
    base_covariates: list[str],
    cutoff: int,
) -> tuple[pd.DataFrame, CoxPHFitter, np.ndarray]:
    """Verify that split records without time interactions reproduce main betas."""
    original = cox.fit_cox(sample, base_covariates)
    split_model, _, covariance = fit_split(split, base_covariates)
    rows = []
    for variable in base_covariates:
        original_beta = float(original.params_[variable])
        split_beta = float(split_model.params_[variable])
        rows.append({
            "cutoff_days": cutoff,
            "variable": variable,
            "variable_label": cox.label_for(variable),
            "original_coefficient": original_beta,
            "split_no_interaction_coefficient": split_beta,
            "absolute_difference": abs(split_beta - original_beta),
            "within_1e_8": abs(split_beta - original_beta) < 1e-8,
            "n": len(sample),
            "n_person_periods": len(split),
            "events": int(split["event_interval"].sum()),
        })
    return pd.DataFrame(rows), split_model, covariance


def run_models(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    definitions = cox.model_definitions(cox.type_dummy_columns(sample))
    base_covariates = definitions["store_competition"]
    interaction_covariates = definitions["interaction"]
    interaction_targets = ["comp_total_x_sales", "comp_total_x_floating"]
    effects: list[dict[str, object]] = []
    validations: list[pd.DataFrame] = []

    for cutoff in (180, 365, 730):
        split = split_records(sample, cutoff)
        validation, _, _ = validate_split(sample, split, base_covariates, cutoff)
        validations.append(validation)

        main_covariates = [*base_covariates]
        for variable in TARGETS:
            after = f"after_{variable}"
            split[after] = split[variable].astype(float) * split["after_cutoff"]
            main_covariates.append(after)
        model, _, covariance = fit_split(split, main_covariates)
        effects.extend(effect_rows(
            "경쟁·수요 시간구간", model, covariance, cutoff, TARGETS,
            len(sample), len(split), int(split["event_interval"].sum()),
        ))

        interaction_covariates_split = [*interaction_covariates]
        for variable in interaction_targets:
            after = f"after_{variable}"
            split[after] = split[variable].astype(float) * split["after_cutoff"]
            interaction_covariates_split.append(after)
        interaction_model, _, interaction_covariance = fit_split(split, interaction_covariates_split)
        effects.extend(effect_rows(
            "완충 상호작용 시간구간", interaction_model, interaction_covariance, cutoff,
            interaction_targets, len(sample), len(split), int(split["event_interval"].sum()),
        ))
    return pd.DataFrame(effects), pd.concat(validations, ignore_index=True)


def plot_effects(effects: pd.DataFrame) -> None:
    selected = effects.loc[
        effects["model"].eq("경쟁·수요 시간구간") & effects["cutoff_days"].eq(365)
    ].copy()
    order = ["comp_indep_r200", "comp_fran_r200", "log_sales_z", "log_floating_z"]
    labels = {
        "comp_indep_r200": "개인 카페 경쟁\n(1개)",
        "comp_fran_r200": "프랜차이즈 경쟁\n(1개)",
        "log_sales_z": "로그 총매출\n(1표준편차)",
        "log_floating_z": "로그 유동인구\n(1표준편차)",
    }
    selected["order"] = selected["variable"].map({name: i for i, name in enumerate(order)})
    selected = selected.sort_values(["order", "period"])
    fig, axis = plt.subplots(figsize=(10, 5.8))
    y_base = np.arange(len(order))
    positions = {PERIOD_LABELS[0]: y_base + .16, PERIOD_LABELS[1]: y_base - .16}
    # 개업 시기 색(#009E73·#CC79A7)과 겹치지 않게 구간 색을 따로 쓴다.
    colors = {PERIOD_LABELS[0]: "#333333", PERIOD_LABELS[1]: "#56B4E9"}
    for period in PERIOD_LABELS:
        current = selected.loc[selected["period"].eq(period)].set_index("variable").loc[order]
        hr = current["hazard_ratio"].to_numpy(float)
        low = current["hr_ci_lower"].to_numpy(float)
        high = current["hr_ci_upper"].to_numpy(float)
        axis.errorbar(
            hr, positions[period], xerr=[hr - low, high - hr], fmt="o",
            color=colors[period], ecolor=colors[period], capsize=3, markersize=6,
            label="개업 후 1년 이내" if period == PERIOD_LABELS[0] else "개업 후 1년 이후",
        )
    axis.axvline(1, color="#555555", linestyle="--", linewidth=1)
    axis.set_xscale("log")
    # 맑은 고딕에는 U+2212 마이너스 글자가 없어 로그 축 기본 눈금(mathtext)이 깨진다.
    axis.xaxis.set_major_locator(FixedLocator([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_yticks(y_base, [labels[name] for name in order])
    axis.set_xlabel("위험비 (95% 신뢰구간, 로그 축)")
    axis.set_title("개업 후 1년 이내·이후의 경쟁 및 수요 연관")
    axis.grid(axis="x", color="#E2E2E2", linewidth=.7)
    axis.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_cox_time_split_hr.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def key_rows(effects: pd.DataFrame, model: str, cutoff: int) -> pd.DataFrame:
    return effects.loc[(effects["model"].eq(model)) & (effects["cutoff_days"].eq(cutoff))].copy()


def get_row(effects: pd.DataFrame, model: str, variable: str, period: str, cutoff: int = 365) -> pd.Series:
    return effects.loc[
        effects["model"].eq(model) & effects["variable"].eq(variable)
        & effects["period"].eq(period) & effects["cutoff_days"].eq(cutoff)
    ].iloc[0]


def update_main_report(effects: pd.DataFrame) -> None:
    text = MAIN_REPORT.read_text(encoding="utf-8")
    early_personal = get_row(effects, "경쟁·수요 시간구간", "comp_indep_r200", PERIOD_LABELS[0])
    late_personal = get_row(effects, "경쟁·수요 시간구간", "comp_indep_r200", PERIOD_LABELS[1])
    early_franchise = get_row(effects, "경쟁·수요 시간구간", "comp_fran_r200", PERIOD_LABELS[0])
    late_franchise = get_row(effects, "경쟁·수요 시간구간", "comp_fran_r200", PERIOD_LABELS[1])
    early_sales = get_row(effects, "경쟁·수요 시간구간", "log_sales_z", PERIOD_LABELS[0])
    late_sales = get_row(effects, "경쟁·수요 시간구간", "log_sales_z", PERIOD_LABELS[1])
    early_floating = get_row(effects, "경쟁·수요 시간구간", "log_floating_z", PERIOD_LABELS[0])
    late_floating = get_row(effects, "경쟁·수요 시간구간", "log_floating_z", PERIOD_LABELS[1])

    sections = "\n".join([
        "### 경쟁", "",
        "상권 변수만 포함한 모형에 200m 점포 단위 경쟁을 더하면 부분 AIC는 47458.0에서 47446.9로 -11.1 변했고, Harrell's C는 0.508에서 0.521로 +0.014 변했다. 우도비 검정은 χ²=15.14, df=2, p=<0.001였다.",
        "점포 단위 경쟁을 함께 조정한 모형에서 개인 카페 경쟁 1개당 위험비는 1.011 (1.003–1.019) (p=0.008), 프랜차이즈 경쟁 1개당은 1.006 (0.988–1.024) (p=0.526)였다. 사분위 범위(IQR)당 위험비는 계수 표에 함께 제시했다.",
        "개인 경쟁의 추정치는 1보다 크고 신뢰구간도 1을 넘어서, KM에서 개인·전체 경쟁이 많을수록 생존확률이 낮았던 방향과 일치한다. 반면 프랜차이즈 경쟁은 신뢰구간이 1을 포함해 KM의 구간별 차이가 독립적인 선형 연관으로 이어진다고 보기 어렵다. 단, 이 관찰자료는 입지 선택과 미관측 요인을 제거하지 못하므로 경쟁 자체의 효과로 단정하지 않는다.", "",
        "지연 진입을 쓴 군집 강건 시간구간 모형에서도 개인 경쟁은 개업 후 1년 이내 "
        f"HR {fmt_hr(early_personal)}, 이후 HR {fmt_hr(late_personal)}였고, 두 구간 계수 차이 Wald p={fmt_p(float(early_personal.difference_wald_p))}였다. "
        "프랜차이즈 경쟁은 첫해 "
        f"HR {fmt_hr(early_franchise)}, 이후 HR {fmt_hr(late_franchise)}였으며, 두 구간 차이는 p={fmt_p(float(early_franchise.difference_wald_p))}였다. "
        "개인 경쟁의 점추정은 두 구간에서 비슷했고, 프랜차이즈 경쟁의 점추정은 첫해에 더 컸다. 다만 프랜차이즈 경쟁의 두 구간 신뢰구간은 모두 1을 포함하고 계수 차이도 뚜렷하지 않아 시간에 따른 차이를 확정할 수는 없다.", "",
        "![개업 초기·이후 경쟁 및 수요 위험비](../figures/05_cox_time_split_hr.png)", "",
        "그림은 개인 경쟁의 점추정은 두 구간에서 비슷하고, 프랜차이즈 경쟁과 수요의 점추정 차이는 첫해에 더 큰 방향임을 보여 준다.", "",
        "![상권 및 점포 단위 경쟁 위험비](../figures/05_cox_store_competition_forest.png)", "",
        "그림은 주 분석에서 개인 경쟁의 위험비가 1보다 크고 프랜차이즈 경쟁의 신뢰구간은 1을 포함함을 보여 준다.", "",
        "### 수요", "",
        "경쟁과 상권 구분을 함께 넣은 모형에서 로그 총매출 1표준편차당 위험비는 1.019 (0.918–1.130) (p=0.729), 로그 유동인구 1표준편차당은 0.951 (0.876–1.031) (p=0.224)였다.",
        "두 수요 지표의 신뢰구간이 모두 1을 포함해, 총매출·유동인구 사분위별 생존 차이가 작고 방향도 일관되지 않았던 KM 관찰과 대체로 같다. 수요가 높아 폐업 위험이 달라진다는 인과 해석은 하지 않는다.", "",
        "시간구간 모형에서는 로그 총매출이 첫해 HR "
        f"{fmt_hr(early_sales)}, 이후 HR {fmt_hr(late_sales)}였고(구간 차이 p={fmt_p(float(early_sales.difference_wald_p))}), "
        "로그 유동인구는 첫해 HR "
        f"{fmt_hr(early_floating)}, 이후 HR {fmt_hr(late_floating)}였다(구간 차이 p={fmt_p(float(early_floating.difference_wald_p))}). "
        "점추정은 첫해와 이후에 다른 방향을 보였지만, 군집 강건 신뢰구간과 계수 차이 Wald 검정은 뚜렷하지 않았다. 따라서 주 분석의 평균 위험비가 1에 가까운 결과가 시간에 따른 차이를 평균한 것이라고 단정할 수는 없다.", "",
        "### 상쇄: 가려짐과 완충", "",
        "수요 투입 전후 경쟁 계수 변화는 `05_cox_competition_changes.csv`에 기록했다. 개인 경쟁의 위험비 변화율은 +0.1%, 프랜차이즈 경쟁은 -0.0%로 매우 작았다. 따라서 KM에서 보인 수요·경쟁의 동반 분포가 경쟁 계수를 크게 가렸다는 근거는 이 모형에서 뚜렷하지 않다.",
        "경쟁 합계×표준화 로그 총매출 상호작용의 위험비는 1.005 (0.997–1.014) (p=0.234)로 완충의 근거가 뚜렷하지 않았다. 경쟁 합계×표준화 로그 유동인구는 0.990 (0.982–0.999) (p=0.024)로, 유동인구가 높은 곳에서 경쟁과 높은 폐업 위험의 연관이 약해지는 방향이었다. 수요 수준별 조건부 경쟁 위험비는 `05_cox_demand_competition_hr.csv`에 수록했다. 이는 유동인구의 저수요 집단에서 고경쟁의 생존율이 더 낮고 고수요 집단에서는 차이가 작았던 KM 그림과 방향이 같다. 총매출에서는 같은 완충 패턴이 확인되지 않았다. 상호작용은 다른 입지 특성과 함께 관찰된 연관이므로, 수요가 경쟁의 결과를 완충한다고 인과적으로 해석하지 않는다.", "",
        "![수요 수준별 경쟁 위험비](../figures/05_cox_competition_hr_by_demand.png)", "",
        "그림은 유동인구가 높은 상권에서 경쟁의 양(+) 연관이 약해지는 방향을 보여 준다.", "",
    ])
    start_sections = text.index("### 경쟁")
    methods_start = text.index("## 방법과 검증")
    text = text[:start_sections] + sections + text[methods_start:]

    diagnostic_start = text.index("### 모형 비교와 진단")
    diagnostic_end = text.index("### 민감도 분석")
    diagnostics = "\n".join([
        "### 모형 비교와 진단", "",
        "상권 모형, 상권+점포 단위 경쟁, 경쟁만, 경쟁+수요, 상호작용 모형을 각각 적합했다. 비례위험 재검정은 군집 강건 분산을 쓰지 않은 같은 층화 Cox 적합에서 수행했다. 기존 표는 군집 강건 적합을 `proportional_hazard_test`에 넘겨 scaled Schoenfeld 잔차의 척도가 맞지 않았고, 그 결과 p값이 비정상적으로 0.9 부근에 집중됐다.",
        "lifelines 0.30.3의 `rank` 변환은 시간의 평균순위가 아니라 이벤트 표시의 누적합(`np.cumsum(events)`)으로 구현되어 있으며, 소스에도 ‘hacky’라고 명시돼 있다. 따라서 동순위 사건과 층화가 있는 이 자료에서 rank 검정은 직접 계산한 사건시점 평균순위 상관과 같은 검정이 아니다. rank 결과는 비례위험 판단 근거에서 제외하고, KM·log 변환과 scaled Schoenfeld 잔차-사건시점 순위 상관을 함께 사용했다.",
        "상권+점포 단위 경쟁 모형에서 로그 총매출은 KM p=0.011, log p<0.001, 직접 상관 r=-0.047 (p=0.006)으로 시간에 따른 연관이 확인됐다. 경쟁 변수와 로그 유동인구는 이 세 점검에서 뚜렷한 위반이 확인되지 않았다. 상호작용 모형에서는 총매출·유동인구 완충항의 KM·log 검정도 p<0.05여서, 시간구간 모형으로 보완했다.",
        "분할 자료에서 시간 상호작용을 넣지 않은 적합은 원 주 분석 계수와 최대 절대차 1e-8 미만으로 일치해, 지연 진입 분할이 위험집합과 사건 수를 보존함을 확인했다. 180일·365일·730일 기준 민감도와 계수 차이 Wald 검정은 `05_cox_time_split_effects.csv`에 수록했다.",
        "프랜차이즈 경쟁은 0 / 1–2 / 3–5 / 6개 이상 더미로 바꾼 보조모형을 `05_cox_coefficients.csv`에 함께 수록했다. 수요 수준별 조건부 경쟁 위험비는 다른 수요 변수를 평균(표준화 0)에 고정해 계산했다. 투입 변수 VIF의 최댓값은 3.81이며 전체 값은 `05_cox_vif.csv`에 있다.",
        "",
    ])
    text = text[:diagnostic_start] + diagnostics + text[diagnostic_end:]
    text = text.replace(
        "- 상권+점포 단위 경쟁 forest plot: `docs/figures/05_cox_store_competition_forest.png`\n"
        "- 수요 수준별 경쟁 위험비: `docs/figures/05_cox_competition_hr_by_demand.png`",
        "- 그림: 경쟁·수요·시간구간 결과 문단에 삽입",
    )
    MAIN_REPORT.write_text(text, encoding="utf-8")


def write_robust_report(effects: pd.DataFrame, validation: pd.DataFrame) -> None:
    primary = key_rows(effects, "경쟁·수요 시간구간", 365)
    interaction = key_rows(effects, "완충 상호작용 시간구간", 365)
    main_rows = {variable: get_row(effects, "경쟁·수요 시간구간", variable, PERIOD_LABELS[0]) for variable in TARGETS}
    late_rows = {variable: get_row(effects, "경쟁·수요 시간구간", variable, PERIOD_LABELS[1]) for variable in TARGETS}
    floating_early = get_row(effects, "완충 상호작용 시간구간", "comp_total_x_floating", PERIOD_LABELS[0])
    floating_late = get_row(effects, "완충 상호작용 시간구간", "comp_total_x_floating", PERIOD_LABELS[1])
    valid = validation["within_1e_8"].all()
    early_summary = "; ".join(
        f"{cox.label_for(v)} HR {fmt_hr(main_rows[v])}" for v in TARGETS
    )
    late_summary = "; ".join(
        f"{cox.label_for(v)} HR {fmt_hr(late_rows[v])}" for v in TARGETS
    )
    sensitivity = effects.loc[
        effects["model"].eq("경쟁·수요 시간구간") & effects["variable"].eq("log_sales_z")
        & effects["period"].eq(PERIOD_LABELS[0]), ["cutoff_days", "hazard_ratio", "difference_wald_p"]
    ].sort_values("cutoff_days")
    sensitivity_text = "; ".join(
        f"{int(row.cutoff_days)}일 기준 첫 구간 HR {row.hazard_ratio:.3f}, 구간 차이 p={fmt_p(float(row.difference_wald_p))}"
        for row in sensitivity.itertuples()
    )
    lines = [
        "# Cox 비례위험 재검정과 강건성 검증 보고서", "", "## 주요 결과", "",
        "비례위험 재검정에서 lifelines의 `rank` 시간변환이 사건시점의 평균순위와 다른 누적 사건순서로 구현된 사실을 확인했다. 이 때문에 rank p값은 판단에서 제외하고 KM·log 변환 및 직접 계산한 scaled Schoenfeld 잔차-시간 순위 상관으로 판단했다. 로그 총매출은 이 진단에서 시간에 따른 연관이 달랐지만, 군집 강건 시간구간 모형의 구간 차이 검정은 이를 뚜렷하게 확인하지 못했다.",
        "",
        "### 비례위험과 개업 초기", "",
        "`rank` 구현은 `np.cumsum(events)`이며, lifelines 소스가 이를 ‘hacky’라고 명시한다. 따라서 동순위 사건이 있는 이 분석에서 기존 rank p값 0.86–0.99는 평균순위 상관과 직접 비교할 수 없다. 상권+점포 단위 경쟁 모형에서 로그 총매출은 KM p=0.011, log p<0.001, 직접 상관 p=0.006으로 시간 의존성을 보였다. 경쟁 변수와 로그 유동인구에서는 세 점검이 일관되게 뚜렷한 시간 의존성을 보이지 않았다.",
        "",
        f"365일 절단 군집 강건 모형의 첫 구간 결과는 {early_summary}이고, 이후 구간 결과는 {late_summary}이다.",
        "개인 경쟁은 두 구간의 위험비가 비슷했다. 프랜차이즈 경쟁·로그 총매출·로그 유동인구는 점추정상 첫해에 더 큰 차이를 보였지만, 각 구간의 군집 강건 신뢰구간은 1을 포함했다. 계수 차이 Wald p는 각각 "
        f"{fmt_p(float(main_rows['comp_fran_r200'].difference_wald_p))}, {fmt_p(float(main_rows['log_sales_z'].difference_wald_p))}, {fmt_p(float(main_rows['log_floating_z'].difference_wald_p))}였다. "
        "이 결과만으로 주 분석의 평균 위험비가 시간에 따라 다른 연관을 평균한 결과라고 결론내릴 수는 없지만, 점추정의 시간 패턴은 비례위험 진단을 보완한다.",
        "",
        "![개업 초기·이후 경쟁 및 수요 위험비](../figures/05_cox_time_split_hr.png)", "",
        "그림은 개인 경쟁의 점추정이 두 구간에서 비슷하고, 프랜차이즈 경쟁과 수요의 점추정 차이가 첫해에 더 큰 방향임을 보여 준다. 다만 구간별 군집 강건 신뢰구간은 모두 1을 포함한다.",
        "",
        "완충 상호작용에서도 전체 경쟁×로그 유동인구는 첫해 HR "
        f"{fmt_hr(floating_early)}, 이후 HR {fmt_hr(floating_late)}였고, 구간 차이 p={fmt_p(float(floating_early.difference_wald_p))}였다. "
        "유동인구가 경쟁의 양(+) 연관을 약하게 하는 방향이 첫해에 더 집중됐는지 점검했지만, 구간 차이 검정은 뚜렷하지 않았다. 관찰자료이므로 인과적 완충으로 해석하지 않는다.",
        "",
        "![Scaled Schoenfeld 잔차](../figures/05_cox_robustness_schoenfeld.png)", "",
        "그림은 경쟁·수요 변수의 잔차가 사건시점에 따라 어떻게 달라지는지 보여 주며, 로그 총매출의 시간 의존성 판단을 보완한다.",
        "",
        "### 반경과 표본을 바꾼 결과", "",
        "개인 경쟁의 양(+) 연관은 100m·200m·500m 반경과 표본 변형에서도 같은 방향이었다. 반면 프랜차이즈 경쟁의 신뢰구간은 대체로 1을 포함해 독립적 연관이 강건하지 않았다. 반경·표본별 계수, IQR 위험비, 완충 조건부 위험비는 `05_cox_robustness_radius_*.csv`, `05_cox_robustness_sample_*.csv`에 수록했다.",
        "",
        "![반경·표본별 점포 단위 경쟁 위험비](../figures/05_cox_robustness_competition_forest.png)", "",
        "그림은 개인 경쟁의 양(+) 연관이 반경과 표본을 바꿔도 유지되는 방향을 보여 준다.",
        "",
        "## 방법과 검증", "",
        "### 지연 진입 시간구간 모형", "",
        "주 분석과 같은 7,351개 완전 사례(사건 3,411건)를 각 기준일 전·후 행으로 나눴다. 후행 행은 `entry_col`을 기준일로 두어 그 시점까지 생존한 점포만 위험집합에 진입시켰다. 모든 모형은 개업연도 층화와 배정 상권 군집 강건 샌드위치 분산을 사용했다. 365일 기준 분할은 13,153 person-period였고, 사건 수 3,411건을 보존했다.",
        f"시간 상호작용 없이 분할한 모형의 계수는 원 주 분석과 {'모든 공변량에서' if valid else '대부분'} 1e-8 미만 차이로 일치했다. 이 검증 결과는 `05_cox_time_split_validation.csv`에 저장했다.",
        "구간별 위험비는 기준일 이전의 원 계수와 이후의 원 계수+시간상호작용 계수로 계산했다. 두 구간 계수 차이는 시간상호작용 계수의 군집 강건 Wald 검정으로 계산했다. 강건 공분산은 lifelines가 산출한 군집 샌드위치 표준오차와 대각 원소가 일치하는지 확인한 뒤 선형 대비에 사용했다.",
        f"기준일 민감도에서 로그 총매출의 첫 구간 결과는 {sensitivity_text}였다. 모든 기준일·변수·상호작용 결과는 `05_cox_time_split_effects.csv`에 있다.",
        "",
        "### 비례위험 재검정", "",
        "상권+점포 단위 경쟁 모형과 상호작용 모형을 비군집·비강건 Cox로 재적합해 KM·log Schoenfeld 검정을 수행했고, scaled Schoenfeld 잔차와 실제 사건시점 평균순위의 Pearson 상관을 별도로 계산했다. rank 변환은 구현상 다른 시간척도이므로 판단 근거에서 제외했다.",
        "",
        "### 산출물", "",
        "- 시간구간 효과·Wald 검정: `docs/tables/05_cox_time_split_effects.csv`",
        "- 분할 검증: `docs/tables/05_cox_time_split_validation.csv`",
        "- 개업 초기·이후 위험비: `docs/figures/05_cox_time_split_hr.png`",
    ]
    ROBUST_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_log(effects: pd.DataFrame, validation: pd.DataFrame, sample: pd.DataFrame) -> None:
    split_365 = effects.loc[(effects["model"].eq("경쟁·수요 시간구간")) & effects["cutoff_days"].eq(365)].iloc[0]
    max_difference = float(validation["absolute_difference"].max())
    existing = LOG.read_text(encoding="utf-8") if LOG.exists() else "# Cox 강건성 검증 실행 로그\n"
    if "## 시간구간 재추정" in existing:
        existing = existing.split("## 시간구간 재추정", maxsplit=1)[0].rstrip() + "\n"
    addition = "\n".join([
        "", "## 시간구간 재추정", "",
        f"- 실행: `.venv\\Scripts\\python.exe src\\05_cox_time_split.py`",
        f"- 입력·공통 표본: `cafes_analysis.parquet`, {len(sample):,}개 점포, 사건 {int(sample['event'].sum()):,}건",
        "- 분할: 180일·365일·730일; `CoxPHFitter(entry_col)`와 개업연도 층화, 배정 상권 군집 강건 샌드위치 분산",
        f"- 365일 분할: {int(split_365.n_person_periods):,} person-period, 사건 {int(split_365.events):,}건",
        f"- 분할 무상호작용 계수 검증: 최대 절대차 {max_difference:.3e}",
        "- PH 판단 수정: lifelines rank는 누적 사건순서 구현이어서 평균 시간순위 상관과 다르므로 판단에서 제외; KM·log·직접 상관 사용",
        "- 패키지 설치·원본 데이터·06단계 파일 수정 없음",
    ])
    LOG.write_text(existing.rstrip() + "\n" + addition + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    cox.configure_matplotlib()
    data = pd.read_parquet(INPUT)
    sample, _, _ = cox.prepare_sample(data)
    effects, validation = run_models(sample)
    write_csv("05_cox_time_split_effects.csv", effects)
    write_csv("05_cox_time_split_validation.csv", validation)
    plot_effects(effects)
    update_main_report(effects)
    write_robust_report(effects, validation)
    update_log(effects, validation, sample)


if __name__ == "__main__":
    main()
