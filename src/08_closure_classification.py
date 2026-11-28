"""Group-cross-validated fixed-horizon cafe closure classification.

Primary thresholds use each outer training fold's closure prevalence: the
1-prevalence quantile of grouped OOF probabilities.  Thus every classifier is
compared at approximately the same predicted-closure share.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR, FIGURE_DIR = ROOT / "docs/tables", ROOT / "docs/figures"
REPORT_PATH, LOG_PATH = ROOT / "docs/reports/08_closure_classification_report.md", ROOT / "docs/logs/08_closure_classification_execution.md"
SEED, N_SPLITS = 20261128, 5
sys.path.insert(0, str(ROOT / "src"))
survival_ml = importlib.import_module("08_survival_ml")
LABELS = survival_ml.LABELS
BASELINE, YEAR, LOGIT, TREE, FOREST, XGB = "다수 클래스 기준선", "개업 연도 로지스틱", "전체 로지스틱", "의사결정나무", "랜덤 포레스트", "XGBoost"
ORDER = [BASELINE, YEAR, LOGIT, TREE, FOREST, XGB]
COLORS = {BASELINE: "#555555", YEAR: "#56B4E9", LOGIT: "#0072B2", TREE: "#E69F00", FOREST: "#D55E00", XGB: "#CC79A7"}


def configure_matplotlib() -> None:
    path = Path(r"C:\Windows\Fonts\malgun.ttf")
    if not path.exists(): raise RuntimeError("맑은 고딕 글꼴을 찾지 못했습니다.")
    font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(path)).get_name(), "axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})


def specs() -> dict[str, object]:
    return {
        LOGIT: Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED))]),
        TREE: DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=SEED),
        FOREST: RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=20, class_weight="balanced", n_jobs=1, random_state=SEED),
        XGB: xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=1, random_state=SEED, verbosity=0),
    }


def fit(estimator: object, x: pd.DataFrame, y: np.ndarray, weighted: bool = False) -> object:
    model = clone(estimator)
    if weighted:
        pos, neg = int(y.sum()), len(y) - int(y.sum())
        if not pos or not neg: raise ValueError("XGBoost 학습 분할에는 두 결과 범주가 필요합니다.")
        model.set_params(scale_pos_weight=neg / pos)
    return model.fit(x, y)


def oof(estimator: object, x: pd.DataFrame, y: np.ndarray, groups: pd.Series, weighted: bool = False) -> np.ndarray:
    result = np.empty(len(y))
    splitter = GroupKFold(n_splits=min(3, groups.nunique()))
    for tr, te in splitter.split(x, y, groups):
        result[te] = fit(estimator, x.iloc[tr], y[tr], weighted).predict_proba(x.iloc[te])[:, 1]
    return result


def f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    values = np.unique(np.r_[0., .5, p])
    scores = [f1_score(y, p >= v, zero_division=0) for v in values]
    return float(values[np.flatnonzero(np.asarray(scores) == max(scores))[0]])


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = p >= threshold
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0), "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "predicted_closure_share": float(pred.mean())}


def horizon_sample(base: pd.DataFrame, days: int, cutoff: str) -> pd.DataFrame:
    out = base.loc[pd.to_datetime(base.permit_date).le(pd.Timestamp(cutoff))].copy()
    out["closed_within_horizon"] = (out.event.astype(bool) & out.survival_days.le(days)).astype(int)
    if out.closed_within_horizon.nunique() != 2: raise ValueError("분류 결과 범주가 하나뿐입니다.")
    return out.reset_index(drop=True)


def evaluate(sample: pd.DataFrame, features: list[str], period: str, detail: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    x, y = sample[features].astype(float), sample.closed_within_horizon.to_numpy(int)
    groups = sample.trdar_cd_assigned.astype(str).reset_index(drop=True)
    primary, f1max, fixed, importance = [], [], [], []
    pooled: dict[str, list[np.ndarray]] = {"y": [], "fold": []}
    for name in ORDER: pooled[name], pooled[f"threshold_{name}"] = [], []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=N_SPLITS).split(x, y, groups), 1):
        xt, xv, yt, yv, gt = x.iloc[tr], x.iloc[te], y[tr], y[te], groups.iloc[tr]
        models, thresholds, f1s = {}, {}, {}
        models[BASELINE] = DummyClassifier(strategy="most_frequent").fit(xt, yt)
        thresholds[BASELINE] = f1s[BASELINE] = .5
        year_est = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))])
        models[YEAR] = fit(year_est, xt[["opening_year"]], yt)
        year_oof = oof(year_est, xt[["opening_year"]], yt, gt)
        thresholds[YEAR], f1s[YEAR] = float(np.quantile(year_oof, 1 - yt.mean())), f1_threshold(yt, year_oof)
        for name, est in specs().items():
            weighted = name == XGB
            models[name] = fit(est, xt, yt, weighted)
            prob = oof(est, xt, yt, gt, weighted)
            thresholds[name], f1s[name] = float(np.quantile(prob, 1 - yt.mean())), f1_threshold(yt, prob)
        for name in ORDER:
            test_x = xv[["opening_year"]] if name == YEAR else xv
            prob = models[name].predict_proba(test_x)[:, 1]
            base = {"기간": period, "fold": fold, "model": name, "n_test": len(te), "closures_test": int(yv.sum())}
            primary.append({**base, "threshold": thresholds[name], "threshold_rule": "학습 OOF 폐업비율 분위수", **metrics(yv, prob, thresholds[name])})
            f1max.append({**base, "threshold": f1s[name], "threshold_rule": "학습 OOF F1 최대", **metrics(yv, prob, f1s[name])})
            fixed.append({**base, "threshold": .5, "threshold_rule": "고정 0.5", **metrics(yv, prob, .5)})
            pooled[name].append(prob); pooled[f"threshold_{name}"].append(np.repeat(thresholds[name], len(te)))
        pooled["y"].append(yv); pooled["fold"].append(np.repeat(fold, len(te)))
        if detail:
            for name in [FOREST, XGB]:
                pi = permutation_importance(models[name], xv, yv, scoring="roc_auc", n_repeats=10, random_state=SEED + fold, n_jobs=1)
                for variable, mean, sd in zip(features, pi.importances_mean, pi.importances_std): importance.append({"fold": fold, "method": f"{name} 순열 ROC-AUC 감소", "variable": variable, "importance": mean, "within_fold_sd": sd})
            values = shap.TreeExplainer(models[XGB]).shap_values(xv)
            values = values[1] if isinstance(values, list) else values
            for variable, value in zip(features, np.abs(np.asarray(values)).mean(axis=0)): importance.append({"fold": fold, "method": "XGBoost 평균 |SHAP|", "variable": variable, "importance": float(value), "within_fold_sd": np.nan})
    return pd.DataFrame(primary), pd.DataFrame(f1max), pd.DataFrame(fixed), pd.DataFrame(importance), {k: np.concatenate(v) for k, v in pooled.items()}


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "predicted_closure_share"]
    result = frame.groupby(["기간", "model"], sort=False)[columns].agg(["mean", "std"])
    result.columns = [f"{a}_{b}" for a, b in result.columns]
    return result.reset_index()


def importance_table(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["method", "variable"], sort=False).importance.agg(["mean", "std"]).reset_index()
    wide = grouped.pivot(index="variable", columns="method", values=["mean", "std"])
    wide.columns = [f"{method}_{stat}" for stat, method in wide.columns]
    out = wide.reset_index(); out.insert(1, "variable_label", out.variable.map(LABELS))
    return out.sort_values("XGBoost 순열 ROC-AUC 감소_mean", ascending=False)


def plot_performance(frame: pd.DataFrame) -> None:
    stats = [("accuracy_mean", "정확도"), ("precision_mean", "정밀도"), ("recall_mean", "재현율"), ("f1_mean", "F1"), ("roc_auc_mean", "ROC-AUC")]
    pos, width = np.arange(len(frame)), .15
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (col, label) in enumerate(stats):
        vals, err = frame[col].to_numpy(), frame[col.replace("_mean", "_std")].fillna(0).to_numpy()
        bars = ax.bar(pos + (i - 2) * width, vals, width, yerr=err, capsize=3, label=label)
        for bar, val, sd in zip(bars, vals, err): ax.text(bar.get_x() + bar.get_width()/2, min(1.02, val + sd + .018), f"{val:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set(xticks=pos, xticklabels=frame.model, ylim=(0, 1.11), ylabel="평가 지표 (5개 폴드 평균, 막대=표준편차)", title="개업 2년 내 폐업 분류 성능 비교")
    ax.tick_params(axis="x", rotation=15); ax.legend(ncol=5, loc="upper center"); fig.tight_layout(); fig.savefig(FIGURE_DIR / "08_clf_performance.png", dpi=200); plt.close(fig)


def plot_confusion(pooled: dict[str, np.ndarray], model: str) -> None:
    matrix = confusion_matrix(pooled["y"], pooled[model] >= pooled[f"threshold_{model}"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5)); image = ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2): ax.text(j, i, f"{matrix[i,j]:,}", ha="center", va="center", fontsize=18, color="white" if matrix[i,j] > matrix.max()/2 else "#222222")
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["생존 예측", "폐업 예측"], yticklabels=["실제 생존", "실제 폐업"], xlabel="예측", ylabel="실제", title=f"{model}: 2년 내 폐업 분류")
    fig.colorbar(image, ax=ax, label="점포 수"); fig.tight_layout(); fig.savefig(FIGURE_DIR / "08_clf_confusion_matrix.png", dpi=200); plt.close(fig)


def plot_roc(pooled: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for name in ORDER:
        fpr, tpr, _ = roc_curve(pooled["y"], pooled[name]); ax.plot(fpr, tpr, lw=2.2, color=COLORS[name], label=f"{name} (AUC={roc_auc_score(pooled['y'], pooled[name]):.3f})")
    ax.plot([0,1], [0,1], color="#999999", linestyle="--", label="무작위")
    ax.set(xlim=(0,1), ylim=(0,1), xlabel="거짓 양성 비율", ylabel="참 양성 비율", title="개업 2년 내 폐업 예측 ROC 곡선"); ax.legend(loc="lower right", fontsize=10); fig.tight_layout(); fig.savefig(FIGURE_DIR / "08_clf_roc.png", dpi=200); plt.close(fig)


def plot_importance(frame: pd.DataFrame) -> None:
    col, sd = "XGBoost 순열 ROC-AUC 감소_mean", "XGBoost 순열 ROC-AUC 감소_std"
    top = frame.sort_values(col, ascending=False).head(10).iloc[::-1]; values, errors = top[col].to_numpy(), top[sd].fillna(0).to_numpy()
    fig, ax = plt.subplots(figsize=(11, 7)); bars = ax.barh(top.variable_label, values, xerr=errors, color="#CC79A7", capsize=3)
    for bar, value, error in zip(bars, values, errors): ax.text(value + error + max(abs(values).max(), .001)*.015, bar.get_y()+bar.get_height()/2, f"{value:.3f}", va="center", fontsize=10)
    ax.set(xlim=(min(0, float((values-errors).min())), max(values+errors)*1.25), xlabel="순열 후 ROC-AUC 평균 감소 (막대=폴드 표준편차)", title="XGBoost 변수 중요도: 순열 ROC-AUC 감소 상위 10개"); fig.tight_layout(); fig.savefig(FIGURE_DIR / "08_clf_variable_importance.png", dpi=200); plt.close(fig)


def ms(row: pd.Series, name: str) -> str: return f"{row[f'{name}_mean']:.3f}±{row[f'{name}_std']:.3f}"


def write_report(sample: pd.DataFrame, primary: pd.DataFrame, f1: pd.DataFrame, importance: pd.DataFrame, best: str, folds: pd.DataFrame) -> None:
    row, f1row = primary.loc[primary.model.eq(best)].iloc[0], f1.loc[f1.model.eq(best)].iloc[0]
    rank = importance.sort_values("XGBoost 순열 ROC-AUC 감소_mean", ascending=False); first, second = rank.iloc[0], rank.iloc[1]
    unstable = (rank["XGBoost 순열 ROC-AUC 감소_mean"].abs() < rank["XGBoost 순열 ROC-AUC 감소_std"]).mean() > .5
    REPORT_PATH.write_text(f"""# 개업 2년 내 폐업 분류 모형 비교

## 분석으로 알게 된 점

개업 시점의 입지·경쟁 정보를 사용해 개인 카페의 **개업 후 2년 이내 폐업 여부**를 분류했다. 관찰 창이 완결된 표본은 {len(sample):,}개이며, 2년 내 폐업은 {int(sample.closed_within_horizon.sum()):,}개({sample.closed_within_horizon.mean():.1%})다.

주 결과에서는 각 학습 폴드의 실제 폐업 비율만큼 예측 점수가 높은 점포를 폐업으로 분류했다. 따라서 모든 모형이 대체로 같은 예측 폐업 비율에서 정확도·정밀도·재현율·F1을 비교한다. ROC-AUC 평균이 가장 높은 모형은 **{best}**이며, 정확도 {ms(row,'accuracy')}, 정밀도 {ms(row,'precision')}, 재현율 {ms(row,'recall')}, F1 {ms(row,'f1')}, ROC-AUC {ms(row,'roc_auc')}다. ROC-AUC가 0.5에 가까워 개업 시점 정보만으로 개별 점포의 2년 내 폐업을 강하게 구분한다고 보기는 어렵다.

부록의 OOF F1 최대 기준에서 {best}의 F1은 {ms(f1row,'f1')}이다. 이 기준은 낮은 임계값에서 거의 모든 점포를 폐업으로 분류하는 방향으로 치우칠 수 있어 주 비교 기준으로 쓰지 않았다. 고정 0.5 기준도 제공하며, XGBoost에는 매 학습 분할에서 `scale_pos_weight=생존 수/폐업 수`를 적용했다.

변수 중요도는 순열 ROC-AUC 감소를 주 지표로 제시한다. 평균 감소가 가장 큰 변수는 **{first['variable_label']}**({first['XGBoost 순열 ROC-AUC 감소_mean']:.3f}), 다음은 **{second['variable_label']}**({second['XGBoost 순열 ROC-AUC 감소_mean']:.3f})다. {"대부분의 변수에서 평균 감소가 폴드 간 표준편차보다 작아 모형의 구분력이 낮고 변수 순위도 불안정하다." if unstable else "폴드 간 표준편차를 함께 제시하며, 낮은 ROC-AUC를 고려해 순위를 강한 영향의 증거로 해석하지 않는다."} 평균 |SHAP|은 보조 지표로 표에만 수록했다.

![모형별 분류 성능](../figures/08_clf_performance.png)

그림은 학습 폴드의 폐업 비율을 맞춘 기준값에서의 5개 검증 폴드 평균 성능과 표준편차다.

![혼동 행렬](../figures/08_clf_confusion_matrix.png)

그림은 ROC-AUC가 가장 높은 모형의 모든 검증 폴드 예측을 합친 결과다.

![ROC 곡선](../figures/08_clf_roc.png)

그림은 임계값과 무관한 점수 순위의 구분력을 비교한다.

![변수 중요도](../figures/08_clf_variable_importance.png)

그림은 XGBoost에서 변수를 섞었을 때의 ROC-AUC 평균 감소와 폴드 간 표준편차를 보여 준다.

## 방법과 검증

`05_cox.py`의 표본 준비 함수와 `08_survival_ml.py`, `05_cox_franchise_types.py`의 개업 시점 경쟁 변수 계산을 재사용했다. 모든 공변량은 개업 시점 또는 직전 분기의 값이며 개업 후 정보는 사용하지 않았다. 상권 코드가 같은 점포가 학습과 평가에 나뉘지 않도록 5개 GroupKFold를 사용했다.

각 외부 평가 폴드의 주 임계값은 훈련 자료의 그룹 OOF 예측확률에서 `1 − 훈련 폐업비율` 분위수로 정해 평가 폴드에 적용했다. 동점 확률이 많은 모형은 정확히 같은 예측 비율을 맞추지 못할 수 있으므로 실제 예측 폐업비율도 표에 기록했다. 다수 클래스 기준선은 그대로 최빈 클래스(생존)를 예측한다.

하이퍼파라미터는 사전 점검에서 중첩 격자 탐색이 과도하게 오래 걸려 고정했다. 전체 로지스틱(C=1), 의사결정나무(깊이 4·말단 최소 20), 랜덤 포레스트(150개·깊이 8·말단 최소 20), XGBoost(200개·깊이 3·학습률 0.05)를 적용했다. XGBoost 가중치는 외부 학습 분할과 OOF의 내부 학습 분할마다 독립적으로 다시 계산했다.

랜덤 포레스트와 XGBoost의 순열 ROC-AUC 감소를 중요도의 주 지표로, XGBoost 평균 |SHAP|을 보조 지표로 저장했다. 상세 성능·폴드별 임계값·F1 최대 및 0.5 부록·표본 흐름은 `docs/tables/08_clf_*.csv`에 수록했다.
""", encoding="utf-8")


def main() -> None:
    for directory in [TABLE_DIR, FIGURE_DIR, REPORT_PATH.parent, LOG_PATH.parent]: directory.mkdir(parents=True, exist_ok=True)
    configure_matplotlib(); base, validation = survival_ml.detailed_sample(); features, _, _ = survival_ml.feature_sets(base)
    periods = {"1년": (365, "2025-09-30"), "2년": (730, "2024-09-30"), "3년": (1095, "2023-09-30")}
    primaries, f1s, fixeds, main_sample, main_raw, pooled = [], [], [], None, None, None
    for name, (days, cutoff) in periods.items():
        sample = horizon_sample(base, days, cutoff); primary, f1, fixed, raw, pool = evaluate(sample, features, name, name == "2년")
        primaries.append(primary); f1s.append(f1); fixeds.append(fixed)
        if name == "2년": main_sample, main_raw, pooled = sample, raw, pool
    folds, f1folds, fixedfolds = pd.concat(primaries), pd.concat(f1s), pd.concat(fixeds)
    performance, f1performance, importance = summary(folds), summary(f1folds), importance_table(main_raw)
    main, f1main = performance.query("기간 == '2년'").copy(), f1performance.query("기간 == '2년'").copy()
    performance.to_csv(TABLE_DIR / "08_clf_performance.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(TABLE_DIR / "08_clf_fold_performance.csv", index=False, encoding="utf-8-sig")
    f1performance.to_csv(TABLE_DIR / "08_clf_threshold_f1max_performance.csv", index=False, encoding="utf-8-sig")
    fixedfolds.to_csv(TABLE_DIR / "08_clf_threshold_05_performance.csv", index=False, encoding="utf-8-sig")
    performance.loc[performance.기간.isin(["1년","3년"])].to_csv(TABLE_DIR / "08_clf_horizon_sensitivity.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(TABLE_DIR / "08_clf_variable_importance.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(TABLE_DIR / "08_clf_competition_count_validation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"model": LOGIT, "설정": "C=1.0, class_weight=balanced"}, {"model": TREE, "설정": "max_depth=4, min_samples_leaf=20, class_weight=balanced"}, {"model": FOREST, "설정": "n_estimators=150, max_depth=8, min_samples_leaf=20, class_weight=balanced"}, {"model": XGB, "설정": "n_estimators=200, max_depth=3, learning_rate=0.05; scale_pos_weight=각 학습 분할 생존 수/폐업 수"}, {"model": "공통", "설정": "주 결과=학습 OOF 1−폐업비율 분위수; 부록=OOF F1 최대 및 고정 0.5"}]).to_csv(TABLE_DIR / "08_clf_hyperparameters.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"기간": n, "관찰 완결 개업일": c, "표본 수": len(horizon_sample(base,d,c)), "폐업 수": int(horizon_sample(base,d,c).closed_within_horizon.sum())} for n,(d,c) in periods.items()]).to_csv(TABLE_DIR / "08_clf_sample_flow.csv", index=False, encoding="utf-8-sig")
    best = main.assign(order=main.model.map({n:i for i,n in enumerate(ORDER)})).sort_values(["roc_auc_mean","order"], ascending=[False,True]).iloc[0].model
    plot_performance(main); plot_confusion(pooled, best); plot_roc(pooled); plot_importance(importance); write_report(main_sample, main, f1main, importance, best, folds.query("기간 == '2년'"))
    LOG_PATH.write_text(f"# 개업 후 폐업 분류 실행 로그\n\n- 실행: `src/08_closure_classification.py`\n- 시드: {SEED}\n- 평가: 상권 단위 GroupKFold 5개, 주 임계값=학습 OOF 폐업비율 분위수\n- XGBoost: 매 학습 분할 scale_pos_weight=생존 수/폐업 수\n- 2년 표본: {len(main_sample):,}개, 폐업 {int(main_sample.closed_within_horizon.sum()):,}개({main_sample.closed_within_horizon.mean():.1%})\n", encoding="utf-8")


if __name__ == "__main__": main()
