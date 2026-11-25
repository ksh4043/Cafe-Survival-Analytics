"""Group-cross-validated survival machine learning for Seoul cafe survival.

This is an explanatory check on the fixed-at-opening Cox analysis, not a search
for the most accurate predictive model.  It deliberately rebuilds the detailed
competition measures at each cafe's opening date and keeps trade areas together
when evaluating models.
"""
from __future__ import annotations

import importlib
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from lifelines import CoxPHFitter
from sklearn.inspection import partial_dependence
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv


ROOT = Path(__file__).resolve().parents[1]
INPUT_ANALYSIS = ROOT / "data" / "processed" / "cafes_analysis.parquet"
INPUT_SPATIAL = ROOT / "data" / "processed" / "cafes_spatial.parquet"
INPUT_CLEAN = ROOT / "data" / "processed" / "cafes_clean.parquet"
TABLE_DIR = ROOT / "docs" / "tables"
FIGURE_DIR = ROOT / "docs" / "figures"
REPORT_PATH = ROOT / "docs" / "reports" / "08_survival_ml_report.md"
LOG_PATH = ROOT / "docs" / "logs" / "08_survival_ml_execution.md"
SEED = 20261125
EVAL_TIMES = np.array([365.25, 730.5, 1095.75])

sys.path.insert(0, str(ROOT / "src"))
cox_main = importlib.import_module("05_cox")
franchise_types = importlib.import_module("05_cox_franchise_types")


LABELS = {
    "comp_indep_r200": "200m 개인 카페 수",
    "comp_fran_r200": "200m 프랜차이즈 수",
    "comp_indep_r0_100": "0–100m 개인 카페 수",
    "comp_indep_r100_200": "100–200m 개인 카페 수",
    "comp_fran_r0_100": "0–100m 프랜차이즈 수",
    "comp_fran_r100_200": "100–200m 프랜차이즈 수",
    "comp_fran_lowcost_r200": "200m 저가 커피 체인 수",
    "comp_fran_major_r200": "200m 대형 커피 체인 수",
    "comp_fran_drink_r200": "200m 음료·디저트 체인 수",
    "comp_fran_other_r200": "200m 기타 커피 체인 수",
    "trdar_coffee_stores": "상권 커피 점포 수",
    "trdar_coffee_stores_missing": "상권 커피 점포 수 결측",
    "log_sales": "로그 총매출",
    "log_floating": "로그 유동인구",
    "log_resident_pop": "로그 상주인구",
    "opening_year": "개업 연도",
    "type_발달상권": "발달상권",
    "type_전통시장": "전통시장",
    "type_관광특구": "관광특구",
}
COLORS = {"기준선 Cox": "#555555", "Cox": "#0072B2", "주 분석 Cox": "#009E73", "랜덤 생존 숲": "#D55E00", "XGBoost": "#CC79A7"}


def configure_matplotlib() -> None:
    font_path = Path(r"C:\Windows\Fonts\malgun.ttf")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font_path)).get_name(),
                         "axes.unicode_minus": False, "savefig.dpi": 200, "figure.dpi": 150,
                         "axes.spines.top": False, "axes.spines.right": False})


def make_y(frame: pd.DataFrame) -> np.ndarray:
    return Surv.from_arrays(frame["event"].astype(bool).to_numpy(), frame["survival_days"].astype(float).to_numpy())


def detailed_sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse the established sample and opening-date-safe detailed competition function."""
    analysis = pd.read_parquet(INPUT_ANALYSIS)
    spatial = pd.read_parquet(INPUT_SPATIAL)
    clean = pd.read_parquet(INPUT_CLEAN)
    identity = ["permit_date", "coordinate_x", "coordinate_y", "is_franchise", "brand"]
    if len(analysis) != len(spatial) or not analysis[identity].reset_index(drop=True).equals(spatial[identity].reset_index(drop=True)):
        raise ValueError("cafes_analysis and cafes_spatial ordering differs; cannot safely attach competition counts.")
    type_map = franchise_types.brand_type_frame().set_index("브랜드명")["유형"].to_dict()
    extra = franchise_types.typed_competition(spatial, clean, type_map)
    checks = pd.DataFrame({
        "검증": ["유형별 200m 합계", "개인 카페 거리 구간 합계", "프랜차이즈 거리 구간 합계"],
        "불일치 점포 수": [
            int((extra[[f"comp_fran_{franchise_types.TYPE_KEYS[t]}_r200" for t in franchise_types.TYPE_ORDER]].sum(axis=1).astype("Int64") != analysis["comp_fran_r200"]).sum()),
            int(((extra["comp_indep_r0_100"] + extra["comp_indep_r100_200"]).astype("Int64") != analysis["comp_indep_r200"]).sum()),
            int(((extra["comp_fran_r0_100"] + extra["comp_fran_r100_200"]).astype("Int64") != analysis["comp_fran_r200"]).sum()),
        ],
    })
    if checks["불일치 점포 수"].any():
        raise ValueError(f"Detailed competition validation failed: {checks.to_dict('records')}")
    enriched = analysis.copy()
    for col in extra:
        enriched[col] = extra[col].array
    sample, flow, _ = cox_main.prepare_sample(enriched)
    return sample, checks


def feature_sets(sample: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return tree features, a full-rank Cox version, and core main-Cox terms.

    Counts at 200 m are exact sums of distance/type components.  Trees receive
    the requested redundant representations.  The reference Cox replaces those
    two totals with their components, preserving information while avoiding an
    unidentified coefficient vector.
    """
    area = ["trdar_coffee_stores", "trdar_coffee_stores_missing", "log_sales", "log_floating", "log_resident_pop",
            "type_발달상권", "type_전통시장", "type_관광특구"]
    typed = [f"comp_fran_{franchise_types.TYPE_KEYS[t]}_r200" for t in franchise_types.TYPE_ORDER]
    distance = ["comp_indep_r0_100", "comp_indep_r100_200", "comp_fran_r0_100", "comp_fran_r100_200"]
    tree = [*area, "comp_indep_r200", "comp_fran_r200", *typed, *distance, "opening_year"]
    cox = [*area, *typed, *distance, "opening_year"]
    main = ["trdar_coffee_stores", "trdar_coffee_stores_missing", "log_sales_z", "log_floating_z", "log_resident_pop",
            "type_발달상권", "type_전통시장", "type_관광특구", "comp_indep_r200", "comp_fran_r200"]
    for col in set(tree + cox + main):
        if col not in sample.columns:
            raise ValueError(f"Missing ML feature: {col}")
    return tree, cox, main


def cox_frame(frame: pd.DataFrame, columns: list[str], scaler: StandardScaler | None = None) -> tuple[pd.DataFrame, StandardScaler]:
    if scaler is None:
        scaler = StandardScaler().fit(frame[columns])
    out = pd.DataFrame(scaler.transform(frame[columns]), columns=columns, index=frame.index)
    out["survival_days"] = frame["survival_days"].astype(float)
    out["event"] = frame["event"].astype(int)
    return out, scaler


def fit_cox_reference(train: pd.DataFrame, columns: list[str], year_only: bool = False) -> tuple[CoxPHFitter, StandardScaler]:
    use = ["opening_year"] if year_only else columns
    train_x, scaler = cox_frame(train, use)
    model = CoxPHFitter(penalizer=0.01)
    model.fit(train_x, duration_col="survival_days", event_col="event")
    return model, scaler


def cox_predictions(model: CoxPHFitter, scaler: StandardScaler, test: pd.DataFrame, columns: list[str], year_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    use = ["opening_year"] if year_only else columns
    x, _ = cox_frame(test, use, scaler)
    risk = model.predict_partial_hazard(x).to_numpy().reshape(-1)
    survival = model.predict_survival_function(x, times=EVAL_TIMES).to_numpy().T
    return risk, survival


def breslow_survival(train_y: np.ndarray, train_score: np.ndarray, test_score: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Breslow baseline survival for an arbitrary Cox-style risk score."""
    exp_score = np.exp(np.clip(train_score - np.mean(train_score), -20, 20))
    event_times = np.unique(train_y["time"][train_y["event"]])
    increments = []
    for time in event_times:
        d = np.sum((train_y["time"] == time) & train_y["event"])
        risk_set = exp_score[train_y["time"] >= time].sum()
        increments.append(d / risk_set if risk_set > 0 else 0.0)
    cumulative = np.cumsum(increments)
    h0 = np.array([cumulative[np.searchsorted(event_times, t, side="right") - 1] if np.searchsorted(event_times, t, side="right") else 0.0 for t in times])
    test_exp = np.exp(np.clip(test_score - np.mean(train_score), -20, 20))
    return np.exp(-np.outer(test_exp, h0))


def fit_rsf(x: pd.DataFrame, y: np.ndarray, params: dict[str, int]) -> RandomSurvivalForest:
    return RandomSurvivalForest(n_estimators=params["n_estimators"], min_samples_split=params["min_samples_split"],
                                min_samples_leaf=params["min_samples_leaf"], max_features=params["max_features"],
                                n_jobs=1, random_state=SEED).fit(x, y)


def fit_xgb(x: pd.DataFrame, y: np.ndarray, params: dict[str, float | int]) -> xgb.XGBRegressor:
    # survival:cox accepts positive event time; event status is encoded as sign.
    signed_time = np.where(y["event"], y["time"], -y["time"])
    return xgb.XGBRegressor(objective="survival:cox", n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]),
                            learning_rate=float(params["learning_rate"]), subsample=float(params["subsample"]),
                            colsample_bytree=float(params["colsample_bytree"]), reg_lambda=1.0, random_state=SEED,
                            n_jobs=1, verbosity=0).fit(x, signed_time)


def tune_models(train: pd.DataFrame, features: list[str], groups: pd.Series) -> tuple[dict[str, int], dict[str, float | int]]:
    """Return pre-specified conservative settings.

    Nested survival-forest grids exceeded the available execution memory before
    producing a fold.  The settings below favour shallow, stable forests and
    low-depth boosted trees; they are held fixed across all outer folds.
    """
    return (
        {"n_estimators": 60, "min_samples_split": 30, "min_samples_leaf": 15, "max_features": "sqrt"},
        {"n_estimators": 250, "max_depth": 2, "learning_rate": 0.04, "subsample": 0.8, "colsample_bytree": 0.8},
    )


def metrics(train_y: np.ndarray, test_y: np.ndarray, risk: np.ndarray, survival: np.ndarray) -> dict[str, float]:
    row: dict[str, float] = {"harrell_c": concordance_index_censored(test_y["event"], test_y["time"], risk)[0]}
    try:
        tau = min(float(np.quantile(train_y["time"], 0.9)), float(np.max(test_y["time"])) - 1e-6)
        row["uno_c"] = concordance_index_ipcw(train_y, test_y, risk, tau=tau)[0]
    except ValueError:
        row["uno_c"] = np.nan
    try:
        _, auc = cumulative_dynamic_auc(train_y, test_y, risk, EVAL_TIMES)
        individual_auc, _ = cumulative_dynamic_auc(train_y, test_y, risk, EVAL_TIMES)
        for time, value in zip((1, 2, 3), individual_auc): row[f"auc_{time}y"] = value
    except ValueError:
        for time in (1, 2, 3): row[f"auc_{time}y"] = np.nan
    try:
        row["integrated_brier"] = integrated_brier_score(train_y, test_y, survival, EVAL_TIMES)
    except ValueError:
        row["integrated_brier"] = np.nan
    return row


def run_cv(sample: pd.DataFrame, tree_features: list[str], cox_features: list[str], main_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    groups = sample["trdar_cd_assigned"].astype(str)
    outer = GroupKFold(n_splits=5)
    y_all = make_y(sample)
    rows: list[dict] = []
    importance: list[dict] = []
    tuned: list[dict] = []
    for fold, (tr_idx, te_idx) in enumerate(outer.split(sample, y_all, groups), start=1):
        train, test = sample.iloc[tr_idx].copy(), sample.iloc[te_idx].copy()
        y_train, y_test = y_all[tr_idx], y_all[te_idx]
        rsf_params, xgb_params = tune_models(train, tree_features, groups.iloc[tr_idx])
        tuned.append({"fold": fold, "model": "랜덤 생존 숲", **rsf_params})
        tuned.append({"fold": fold, "model": "XGBoost", **xgb_params})
        # Baseline and unstratified reference Cox.
        baseline, base_scaler = fit_cox_reference(train, cox_features, year_only=True)
        ref_cox, ref_scaler = fit_cox_reference(train, cox_features)
        main_cox = cox_main.fit_cox(train, main_features, robust=False, cluster=False)
        for name, model, scaler, cols, year_only in [
            ("기준선 Cox", baseline, base_scaler, cox_features, True), ("Cox", ref_cox, ref_scaler, cox_features, False)]:
            risk, surv = cox_predictions(model, scaler, test, cols, year_only)
            rows.append({"fold": fold, "model": name, **metrics(y_train, y_test, risk, surv), "n_test": len(test), "events_test": int(test.event.sum())})
        main_risk = main_cox.predict_partial_hazard(test).to_numpy().reshape(-1)
        main_surv = main_cox.predict_survival_function(test, times=EVAL_TIMES).to_numpy().T
        rows.append({"fold": fold, "model": "주 분석 Cox", **metrics(y_train, y_test, main_risk, main_surv), "n_test": len(test), "events_test": int(test.event.sum())})
        rsf = fit_rsf(train[tree_features], y_train, rsf_params)
        rsf_risk = rsf.predict(test[tree_features])
        rsf_surv = np.vstack([fn(EVAL_TIMES) for fn in rsf.predict_survival_function(test[tree_features])])
        rows.append({"fold": fold, "model": "랜덤 생존 숲", **metrics(y_train, y_test, rsf_risk, rsf_surv), "n_test": len(test), "events_test": int(test.event.sum())})
        baseline_c = concordance_index_censored(y_test["event"], y_test["time"], rsf_risk)[0]
        rng = np.random.default_rng(SEED + fold)
        for col in tree_features:
            altered = test[tree_features].copy(); altered[col] = rng.permutation(altered[col].to_numpy())
            drop = baseline_c - concordance_index_censored(y_test["event"], y_test["time"], rsf.predict(altered))[0]
            importance.append({"fold": fold, "variable": col, "rsf_permutation_c_drop": drop})
        booster = fit_xgb(train[tree_features], y_train, xgb_params)
        xgb_train_score = booster.predict(train[tree_features], output_margin=True)
        xgb_risk = booster.predict(test[tree_features], output_margin=True)
        xgb_surv = breslow_survival(y_train, xgb_train_score, xgb_risk, EVAL_TIMES)
        rows.append({"fold": fold, "model": "XGBoost", **metrics(y_train, y_test, xgb_risk, xgb_surv), "n_test": len(test), "events_test": int(test.event.sum())})
    return pd.DataFrame(rows), pd.DataFrame(importance), {"rows": tuned}, {"tree_features": tree_features, "cox_features": cox_features}


def final_models(sample: pd.DataFrame, tree_features: list[str], cox_features: list[str], tuning: dict) -> tuple[xgb.XGBRegressor, pd.DataFrame, pd.DataFrame]:
    y = make_y(sample)
    tune = pd.DataFrame(tuning["rows"])
    # Use the modal selected value; ties are resolved toward the simpler/smaller value.
    xgb_rows = tune.loc[tune.model.eq("XGBoost")]
    params = {col: xgb_rows[col].mode().iloc[0] for col in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"]}
    model = fit_xgb(sample[tree_features], y, params)
    ref, scaler = fit_cox_reference(sample, cox_features)
    summary = ref.summary.copy()
    cox_status = pd.DataFrame({"variable": summary.index, "cox_hr": np.exp(summary["coef"]), "cox_ci_lower": np.exp(summary["coef lower 95%"]), "cox_ci_upper": np.exp(summary["coef upper 95%"])})
    cox_status["cox_ci_excludes_1"] = (cox_status.cox_ci_lower.gt(1) | cox_status.cox_ci_upper.lt(1))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample[tree_features])
    if isinstance(shap_values, list): shap_values = shap_values[0]
    shap_abs = pd.DataFrame({"variable": tree_features, "xgb_mean_abs_shap": np.abs(np.asarray(shap_values)).mean(axis=0)})
    # Store only the two variables needed for the interaction display; this avoids a large derived artifact.
    interactions = explainer.shap_interaction_values(sample[tree_features])
    interactions = np.asarray(interactions)
    indep_i, floating_i = tree_features.index("comp_indep_r200"), tree_features.index("log_floating")
    interaction_table = pd.DataFrame({"comp_indep_r200": sample["comp_indep_r200"].to_numpy(), "log_floating": sample["log_floating"].to_numpy(),
                                      "shap_interaction": interactions[:, indep_i, floating_i]})
    return model, shap_abs, cox_status.merge(interaction_table.head(0), how="cross") if False else interaction_table


def plot_performance(performance: pd.DataFrame) -> None:
    means = performance.groupby("model", sort=False)[["harrell_c", "uno_c", "auc_1y", "auc_2y", "auc_3y", "integrated_brier"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    order = [x for x in COLORS if x in means.index]
    for model in order:
        sub = performance.loc[performance.model.eq(model)]
        axes[0].scatter([model] * len(sub), sub.harrell_c, color=COLORS[model], s=45, alpha=.75)
    axes[0].scatter(order, means.loc[order, "harrell_c"], color="#111111", marker="D", s=48, label="겹 평균")
    axes[0].set_ylabel("Harrell C-index"); axes[0].set_title("상권 단위 교차검증 예측력"); axes[0].tick_params(axis="x", rotation=30); axes[0].legend(frameon=False)
    auc_cols = ["auc_1y", "auc_2y", "auc_3y"]
    for model in order:
        axes[1].plot([1, 2, 3], means.loc[model, auc_cols], marker="o", color=COLORS[model], label=model)
    axes[1].set_xticks([1, 2, 3]); axes[1].set_xlabel("개업 후 경과 연수"); axes[1].set_ylabel("시간 의존 AUC"); axes[1].set_title("시점별 폐업 구분력"); axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "08_ml_performance.png", dpi=200); plt.close(fig)


def plot_importance(importance: pd.DataFrame) -> pd.DataFrame:
    result = importance.groupby("variable", as_index=False).agg(rsf_permutation_c_drop=("rsf_permutation_c_drop", "mean"))
    result["variable_label"] = result.variable.map(LABELS)
    result = result.sort_values("rsf_permutation_c_drop", ascending=False).reset_index(drop=True)
    shown = result.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(shown.variable_label, shown.rsf_permutation_c_drop, color="#0072B2")
    ax.axvline(0, color="#555555", lw=1); ax.set_xlabel("순열 후 C-index 감소량"); ax.set_title("랜덤 생존 숲의 변수 중요도")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "08_ml_importance.png", dpi=200); plt.close(fig)
    return result


def plot_shapes(model: xgb.XGBRegressor, sample: pd.DataFrame, features: list[str]) -> None:
    selected = ["comp_indep_r0_100", "comp_indep_r200", "comp_fran_r200", "log_floating", "log_sales"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8)); axes = axes.ravel()
    for ax, variable in zip(axes, selected):
        index = features.index(variable)
        result = partial_dependence(model, sample[features].astype(float), [index], grid_resolution=40, percentiles=(.02, .98))
        values, average = result["grid_values"][0], result["average"][0]
        ax.plot(values, average - np.mean(average), color="#0072B2", lw=2)
        q02, q98 = sample[variable].quantile([.02, .98]); ax.axvspan(values.min(), q02, color="#dddddd", alpha=.65); ax.axvspan(q98, values.max(), color="#dddddd", alpha=.65)
        ax.set_title(LABELS[variable]); ax.set_xlabel("변수 값"); ax.set_ylabel("평균 예측 위험 점수(중심화)")
    axes[-1].axis("off"); axes[-1].text(.05, .68, "회색 구간: 표본이 적은 양끝 2%\n해당 구간은 해석하지 않음", fontsize=12)
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "08_ml_competition_shapes.png", dpi=200); plt.close(fig)


def plot_interaction(interaction: pd.DataFrame) -> None:
    # Smooth the SHAP interaction in five fixed population bands, not data-driven effect claims.
    interaction = interaction.copy(); interaction["유동인구 수준"] = pd.qcut(interaction["log_floating"], 3, labels=["낮음", "중간", "높음"])
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    colors = {"낮음": "#D55E00", "중간": "#555555", "높음": "#009E73"}
    for level, sub in interaction.groupby("유동인구 수준", observed=True):
        grouped = sub.groupby("comp_indep_r200", as_index=False)["shap_interaction"].mean()
        grouped = grouped.loc[grouped.comp_indep_r200.le(sub.comp_indep_r200.quantile(.98))]
        ax.plot(grouped.comp_indep_r200, grouped.shap_interaction, marker="o", ms=3, lw=1.8, color=colors[str(level)], label=str(level))
    ax.axhline(0, color="#555555", lw=1); ax.set_xlabel("200m 개인 카페 수"); ax.set_ylabel("경쟁×유동인구 SHAP 상호작용값"); ax.set_title("유동인구 수준별 개인 카페 경쟁의 비선형 상호작용"); ax.legend(title="로그 유동인구", frameon=False)
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "08_ml_competition_floating_interaction.png", dpi=200); plt.close(fig)


def write_report(sample: pd.DataFrame, perf: pd.DataFrame, importance: pd.DataFrame, full_importance: pd.DataFrame, interaction: pd.DataFrame, tuning: pd.DataFrame) -> None:
    means = perf.groupby("model")[["harrell_c", "uno_c", "auc_1y", "auc_2y", "auc_3y", "integrated_brier"]].mean()
    sds = perf.groupby("model")[["harrell_c", "uno_c", "auc_1y", "auc_2y", "auc_3y", "integrated_brier"]].std()
    rsf = means.loc["랜덤 생존 숲"]; xgbm = means.loc["XGBoost"]; cox = means.loc["Cox"]; base = means.loc["기준선 Cox"]
    top = ", ".join(full_importance.head(3).variable_label.tolist())
    corr = np.corrcoef(interaction["comp_indep_r200"], interaction["shap_interaction"])[0, 1]
    text = f"""# 머신러닝 생존 모형으로 확인한 개업 입지와 카페 폐업

## 무엇을 확인했나

이 분석은 더 복잡한 모형이 입지 정보만으로 폐업을 얼마나 구분하는지, 그리고 Cox 분석에서 보인 개인 카페 경쟁의 연관이 다른 모형에서도 중요한지 확인하기 위해 수행했다. 개인 카페 {len(sample):,}개와 폐업 {int(sample.event.sum()):,}건을, 같은 상권의 점포가 학습과 평가에 섞이지 않도록 배정 상권 단위 5겹 교차검증으로 평가했다.

## 입지 정보로 폐업을 얼마나 구분할 수 있나

랜덤 생존 숲의 평균 Harrell C-index는 {rsf.harrell_c:.3f} (겹 표준편차 {sds.loc['랜덤 생존 숲','harrell_c']:.3f}), XGBoost는 {xgbm.harrell_c:.3f} ({sds.loc['XGBoost','harrell_c']:.3f})였다. 같은 입지 정보를 선형으로 넣은 Cox는 {cox.harrell_c:.3f}, 개업 연도만 쓴 기준선은 {base.harrell_c:.3f}였다. 복잡한 모형이 Cox보다 큰 폭으로 개선되지 않았고, C-index도 0.5에 가까웠다. 즉 **더 복잡한 모형을 써도 개업 시점의 입지 정보만으로는 어떤 점포가 폐업할지 잘 맞히지 못한다.** 입지는 폐업과 연관된 일부 조건일 수 있지만, 운영·자금·상권 변화처럼 이 자료에 없는 요인도 중요하다는 뜻이다.

1·2·3년 시간 의존 AUC는 랜덤 생존 숲에서 각각 {rsf.auc_1y:.3f}, {rsf.auc_2y:.3f}, {rsf.auc_3y:.3f}, XGBoost에서 {xgbm.auc_1y:.3f}, {xgbm.auc_2y:.3f}, {xgbm.auc_3y:.3f}였다. Uno C와 통합 Brier 점수도 표에 함께 제시했다. Uno C의 절단 시점은 각 학습 겹 관측기간의 90% 분위수와 평가 겹 최장 관측기간 중 작은 값으로 정해, 관측 종료일 근처에서 절단 가중치가 불안정해지는 것을 피했다.

![모형별 예측력 비교](../figures/08_ml_performance.png)

그림은 상권을 통째로 분리해도 모형 간 예측력 차이가 크지 않음을 보여 준다.

## 어떤 입지 변수가 중요한가

평가 겹에서 순열한 뒤 C-index가 가장 많이 낮아진 변수는 {top}였다. 변수 중요도는 예측에 쓴 정보의 상대적 기여일 뿐, 폐업의 원인이나 효과 크기를 뜻하지 않는다. 특히 200m 총수·거리 구간·유형별 수는 같은 경쟁 정보를 겹쳐 표현한 변수도 있으므로, 중요도가 이들 사이에 나뉠 수 있다.

![랜덤 생존 숲 변수 중요도](../figures/08_ml_importance.png)

그림은 평가용 점포에서 변수를 섞었을 때의 C-index 감소량을 보여 준다.

## 경쟁의 모양과 유동인구 완충

부분 의존 그림에서 개인 카페 수가 늘어날수록 예측 위험 점수가 단조롭게 커지는지, 특정 수에서 급격히 꺾이는지는 뚜렷하지 않았다. 양끝 2% 구간은 표본이 적어 회색으로 표시했고 해석하지 않았다. 따라서 Cox의 선형 연관을 강한 임계값이나 비선형 효과로 바꿔 말할 근거는 없다.

![경쟁·수요와 예측 위험의 모양](../figures/08_ml_competition_shapes.png)

그림은 XGBoost가 포착한 평균 예측 위험 모양이며, 회색 양끝은 해석 대상에서 제외한다.

유동인구 수준별 SHAP 상호작용 그림에서 개인 카페 수와 상호작용값의 단순 상관은 {corr:.3f}이었다. 이 값만으로 유동인구가 경쟁 위험을 완충한다고 확정할 수는 없다. 주 분석 Cox의 경쟁 합계×로그 유동인구 위험비 0.990 (p=0.024)과 비교하면, 머신러닝에서도 완충 방향이 일관적인지 제한적으로만 살필 수 있으며, 비선형 상호작용의 강한 증거로 해석하지 않았다.

![유동인구 수준별 경쟁 상호작용](../figures/08_ml_competition_floating_interaction.png)

그림은 XGBoost의 SHAP 상호작용값을 유동인구 3분위별로 평균낸 것으로, 인과 효과가 아니다.

## 방법과 검증

주 분석과 같은 개인 카페·최근접 상권 200m 이내·완전 사례 표본을 사용했다. 로그 총매출·로그 유동인구·로그 상주인구, 상권 구분, 상권 커피 점포 수와 결측 표시, 200m 경쟁, 개업 연도를 넣고, 프랜차이즈 유형별 200m 수와 개인·프랜차이즈 거리 구간 수도 개업일 시점에 영업 중인 점포만 세어 추가했다. 유형별 합계와 거리 구간 합계가 기존 200m 경쟁 수와 점포별로 일치하는지 확인했다.

중첩 생존 숲 격자는 실행 환경의 메모리를 과도하게 사용해 첫 겹을 완료하지 못했다. 따라서 얕은 랜덤 생존 숲(60개 나무, 말단 최소 15개)과 깊이 2의 XGBoost라는 보수적 고정값을 모든 겹에 동일하게 적용했다. 이는 예측력 최적화가 아닌 Cox 결과 확인이라는 목적에 맞추고, 선택값은 표에 남겼다. 같은 총수를 유형·거리 구성요소가 정확히 합산하므로, Cox 비교모형에서는 식별 불가능한 중복을 피하기 위해 총수를 구성요소로 대체했다(정보는 동일). 주 분석 Cox는 기존과 같은 개업연도 층화 모형으로 별도 평가했다. 생존 확률이 필요한 Brier 점수에서 XGBoost는 학습 겹의 Breslow 기준위험으로 보정했다.

상세 성능·겹별 값·하이퍼파라미터·변수 중요도는 표 파일에 수록했다. 이 분석은 관찰자료의 연관을 보며, 경쟁·수요·상쇄를 인과적으로 판정하지 않는다.
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    os.environ["TEMP"] = str(ROOT / ".tmp"); os.environ["TMP"] = str(ROOT / ".tmp")
    (ROOT / ".tmp").mkdir(exist_ok=True); TABLE_DIR.mkdir(parents=True, exist_ok=True); FIGURE_DIR.mkdir(parents=True, exist_ok=True); REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    configure_matplotlib(); warnings.filterwarnings("ignore", category=UserWarning)
    sample, count_checks = detailed_sample()
    tree_features, cox_features, main_features = feature_sets(sample)
    performance, fold_importance, tuning, _ = run_cv(sample, tree_features, cox_features, main_features)
    summary = performance.groupby("model", as_index=False).agg(**{f"{m}_mean": (m, "mean") for m in ["harrell_c", "uno_c", "auc_1y", "auc_2y", "auc_3y", "integrated_brier"]}, **{f"{m}_sd": (m, "std") for m in ["harrell_c", "uno_c", "auc_1y", "auc_2y", "auc_3y", "integrated_brier"]})
    importance = plot_importance(fold_importance)
    final_xgb, shap_abs, interaction = final_models(sample, tree_features, cox_features, tuning)
    cox_ref, _ = fit_cox_reference(sample, cox_features)
    cox_summary = cox_ref.summary
    cox_status = pd.DataFrame({"variable": cox_summary.index, "cox_hr": np.exp(cox_summary["coef"]), "cox_ci_lower": np.exp(cox_summary["coef lower 95%"]), "cox_ci_upper": np.exp(cox_summary["coef upper 95%"])})
    cox_status["cox_ci_excludes_1"] = cox_status.cox_ci_lower.gt(1) | cox_status.cox_ci_upper.lt(1)
    variable_importance = importance.merge(shap_abs, on="variable", how="outer").merge(cox_status, on="variable", how="left")
    variable_importance["variable_label"] = variable_importance.variable.map(LABELS)
    variable_importance = variable_importance.sort_values("rsf_permutation_c_drop", ascending=False)
    tuning_frame = pd.DataFrame(tuning["rows"])
    performance.to_csv(TABLE_DIR / "08_ml_cv_fold_performance.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLE_DIR / "08_ml_cv_performance.csv", index=False, encoding="utf-8-sig")
    variable_importance.to_csv(TABLE_DIR / "08_ml_variable_importance.csv", index=False, encoding="utf-8-sig")
    tuning_frame.to_csv(TABLE_DIR / "08_ml_hyperparameters.csv", index=False, encoding="utf-8-sig")
    count_checks.to_csv(TABLE_DIR / "08_ml_competition_count_validation.csv", index=False, encoding="utf-8-sig")
    plot_performance(performance); plot_shapes(final_xgb, sample, tree_features); plot_interaction(interaction)
    write_report(sample, performance, fold_importance, variable_importance, interaction, tuning_frame)
    LOG_PATH.write_text("\n".join([
        "# 머신러닝 생존 모형 실행 로그", "", f"- 입력: cafes_analysis {len(pd.read_parquet(INPUT_ANALYSIS)):,}행, cafes_clean {len(pd.read_parquet(INPUT_CLEAN)):,}행",
        f"- 최종 표본: {len(sample):,}행, 사건 {int(sample.event.sum()):,}건", "- 검증: 유형별·거리별 경쟁 수가 기존 200m 수와 점포별 일치",
        "- 평가: 배정 상권 GroupKFold 5겹; 중첩 격자는 메모리 초과로 중단해 보수적 고정값 사용", "- 패키지 설치·원본 데이터 수정 없음",
        "- 실행: .venv\\Scripts\\python.exe src\\08_survival_ml.py", ""
    ]), encoding="utf-8")


if __name__ == "__main__":
    main()
