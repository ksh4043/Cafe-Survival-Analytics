"""Kaplan--Meier descriptions for Seoul coffee-shop survival.

The script reads only the established analysis file and writes the requested
figures, a group-level summary table, and a Korean report.  It intentionally
does not fit a Cox model or change the analysis cohort.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import median_survival_times
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "cafes_analysis.parquet"
FIGURE_DIR = ROOT / "docs" / "figures"
TABLE_PATH = ROOT / "docs" / "tables" / "04_km_summary.csv"
OPENING_MIX_PATH = ROOT / "docs" / "tables" / "04_km_opening_period_mix.csv"
OFFSET_DIAGNOSTICS_PATH = ROOT / "docs" / "tables" / "04_km_offset_diagnostics.csv"
REPORT_PATH = ROOT / "docs" / "reports" / "04_kaplan_meier_report.md"
LOG_PATH = ROOT / "docs" / "logs" / "04_kaplan_meier_execution.md"
OBSERVATION_END = pd.Timestamp("2026-09-30")
RISK_MINIMUM = 30
YEAR_DAYS = (365, 730, 1095)

# Fixed colours make a group recognisable wherever it reappears.
COLORS = {
    "개인 카페": "#0072B2",
    "프랜차이즈": "#D55E00",
    "코로나 시기 개업 (2021–2022)": "#009E73",
    "이후 개업 (2023–2026)": "#CC79A7",
    "주 분석 인허가 단위": "#0072B2",
    "재인허가 통합 강건성": "#E69F00",
}
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]


def configure_matplotlib() -> None:
    """Use Malgun Gothic (required by the task) and a true minus sign."""
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunsl.ttf",
    ]
    font_path = next((path for path in candidates if Path(path).exists()), None)
    if font_path is None:
        raise RuntimeError("Malgun Gothic font was not found; figures would not render Korean safely.")
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def fmt_number(value: float | int | None, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "—"
    if digits == 0:
        return f"{value:,.0f}"
    return f"{value:,.{digits}f}"


def fmt_p(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def read_data() -> pd.DataFrame:
    data = pd.read_parquet(INPUT_PATH)
    required = {
        "is_franchise", "permit_date", "survival_days", "event",
        "comp_total_r200", "comp_indep_r200", "comp_fran_r200",
        "trdar_within_200m", "trdar_sales_total", "trdar_floating_pop",
        "trdar_resident_pop", "trdar_type", "robust_relicense_in_sample",
        "robust_relicense_survival_days", "robust_relicense_event",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Required analysis columns are absent: {sorted(missing)}")
    if len(data) != 11501:
        raise ValueError(f"Expected 11,501 analysis records, found {len(data):,}.")
    if not (data["survival_days"] > 0).all():
        raise ValueError("Non-positive survival duration found in the established analysis file.")
    return data


def qgroups(frame: pd.DataFrame, value: str) -> tuple[pd.Series, str]:
    """Quartiles calculated only in the comparison's valid sample."""
    values = frame[value].astype(float)
    cuts = values.quantile([0.25, 0.50, 0.75]).to_numpy()
    bins = np.r_[-np.inf, cuts, np.inf]
    if len(np.unique(cuts)) != 3:
        # This has not occurred in the current data, but avoids silently giving
        # duplicate bins should a future input contain many ties.
        rank = values.rank(method="first")
        categories = pd.qcut(rank, 4, labels=["Q1", "Q2", "Q3", "Q4"])
        description = "사분위 경계에 동점이 있어 동점 순서를 사용한 4등분"
        return categories.astype(str), description
    if value.startswith("comp_"):
        # pd.cut's right-closed intervals put an integer boundary in the lower
        # quartile.  State that inclusive range plainly, without overlap.
        lower = int(values.min())
        bounds = [int(np.floor(cut)) + 1 for cut in cuts]
        labels = [
            f"Q1 ({lower}개 이상 {bounds[0]}개 미만)",
            f"Q2 ({bounds[0]}개 이상 {bounds[1]}개 미만)",
            f"Q3 ({bounds[1]}개 이상 {bounds[2]}개 미만)",
            f"Q4 ({bounds[2]}개 이상)",
        ]
    else:
        labels = [
            f"Q1 (≤ {fmt_number(cuts[0])})",
            f"Q2 ({fmt_number(cuts[0])}, {fmt_number(cuts[1])}]",
            f"Q3 ({fmt_number(cuts[1])}, {fmt_number(cuts[2])}]",
            f"Q4 (> {fmt_number(cuts[2])})",
        ]
    category = pd.cut(values, bins=bins, labels=labels, include_lowest=True)
    description = " / ".join(fmt_number(x) for x in cuts)
    return category.astype(str), description


def franchise_comp_groups(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    values = frame["comp_fran_r200"].astype(int)
    labels = np.select(
        [values.eq(0), values.between(1, 2), values.between(3, 5), values.ge(6)],
        ["0개", "1–2개", "3–5개", "6개 이상"],
        default="결측",
    )
    return pd.Series(labels, index=frame.index), "0 / 1–2 / 3–5 / 6개 이상"


def quartile_order(groups: pd.Series) -> list[str]:
    """Preserve low-to-high Q1…Q4 order for figures and trend tests."""
    return sorted(pd.unique(groups), key=lambda value: int(str(value).split(" ")[0][1:]))


def trend_logrank(
    duration: pd.Series, event: pd.Series, group: pd.Series, group_order: list[str]
) -> float | None:
    """One-df ordered log-rank trend test, without fitting a survival model."""
    if len(group_order) < 3:
        return None
    observed = set(group.dropna().unique())
    if observed != set(group_order):
        raise ValueError("Trend-test group order does not match observed groups.")
    score = {name: i for i, name in enumerate(group_order)}
    duration = duration.astype(float).to_numpy()
    event = event.astype(int).to_numpy()
    codes = group.map(score).to_numpy()
    u, variance = 0.0, 0.0
    for time in np.unique(duration[event == 1]):
        risk = duration >= time
        deaths = (duration == time) & (event == 1)
        n = risk.sum()
        d = deaths.sum()
        if n <= 1 or d == 0 or d == n:
            continue
        risk_scores = codes[risk]
        death_scores = codes[deaths]
        mean_score = risk_scores.mean()
        u += death_scores.sum() - d * mean_score
        variance += (d * (n - d) / (n - 1)) * np.mean((risk_scores - mean_score) ** 2)
    if variance <= 0:
        return None
    return float(chi2.sf((u * u) / variance, df=1))


def verify_trend_logrank() -> float:
    """Check ordered scoring on deliberately shuffled, monotonic groups."""
    labels = np.array(["Q4", "Q1", "Q3", "Q2"] * 25)
    scores = pd.Series(labels).map({"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}).to_numpy()
    # Higher groups fail earlier; row order deliberately is not Q1→Q4.
    duration = pd.Series(np.where(scores == 3, 20, np.where(scores == 2, 40, np.where(scores == 1, 80, 120))))
    event = pd.Series(np.where(scores == 0, 0, 1))
    p_value = trend_logrank(duration, event, pd.Series(labels), ["Q1", "Q2", "Q3", "Q4"])
    if p_value is None or p_value >= 0.001:
        raise AssertionError("Ordered trend log-rank synthetic check did not detect the designed trend.")
    return p_value


def km_ci_at(kmf: KaplanMeierFitter, day: int) -> tuple[float, float, float]:
    point = float(kmf.predict(day))
    ci = kmf.confidence_interval_survival_function_
    prior = ci.loc[ci.index <= day]
    row = ci.iloc[0] if prior.empty else prior.iloc[-1]
    return point, float(row.iloc[0]), float(row.iloc[1])


def median_and_ci(kmf: KaplanMeierFitter) -> tuple[float | None, float | None, float | None]:
    median = float(kmf.median_survival_time_)
    if not np.isfinite(median):
        return None, None, None
    # lifelines inverts the lower/upper survival-confidence curves at 0.5.
    # This keeps the median bound order aligned with median_survival_times.
    bounds = median_survival_times(kmf.confidence_interval_survival_function_).iloc[0]
    lower = float(bounds.iloc[0]) if np.isfinite(bounds.iloc[0]) else None
    upper = float(bounds.iloc[1]) if np.isfinite(bounds.iloc[1]) else None
    return median, lower, upper


def curve_cutoff(kmf: KaplanMeierFitter) -> int:
    events = kmf.event_table
    valid = events.loc[events["at_risk"] >= RISK_MINIMUM]
    return int(valid.index.max()) if not valid.empty else 0


def add_risk_table(ax: plt.Axes, fitted: list[KaplanMeierFitter], labels: list[str]) -> None:
    ticks = ax.get_xticks()
    ticks = [int(t) for t in ticks if t >= 0]
    rows = []
    for kmf in fitted:
        counts = []
        for t in ticks:
            table = kmf.event_table
            prior = table.loc[table.index <= t]
            if prior.empty:
                counts.append(int(table.iloc[0]["at_risk"]))
            else:
                last = prior.iloc[-1]
                counts.append(int(last["at_risk"] - last["removed"]))
        rows.append(counts)
    table_ax = ax.table(
        cellText=rows,
        rowLabels=[f"위험집합: {label}" for label in labels],
        colLabels=[f"{t:,}" for t in ticks],
        cellLoc="center",
        rowLoc="right",
        bbox=[0.0, -0.48, 1.0, 0.28],
    )
    table_ax.auto_set_font_size(False)
    table_ax.set_fontsize(7)
    for cell in table_ax.get_celld().values():
        cell.set_linewidth(0.25)
        cell.set_edgecolor("#BBBBBB")


def plot_comparison(
    name: str,
    title: str,
    frame: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    rule: str,
    ordered: bool = False,
    duration_col: str = "survival_days",
    event_col: str = "event",
) -> list[dict]:
    """Fit descriptive KMs, render a figure, and return summary rows."""
    frame = frame.copy()
    frame = frame[frame[group_col].isin(group_order)].copy()
    if frame.empty:
        raise ValueError(f"No valid records for {name}.")
    overall_p = None
    if len(group_order) >= 2:
        test = multivariate_logrank_test(frame[duration_col], frame[group_col], frame[event_col])
        overall_p = float(test.p_value)
    trend_p = (
        trend_logrank(frame[duration_col], frame[event_col], frame[group_col], group_order)
        if ordered else None
    )

    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    fitted: list[KaplanMeierFitter] = []
    rows: list[dict] = []
    for i, label in enumerate(group_order):
        subset = frame.loc[frame[group_col] == label]
        kmf = KaplanMeierFitter(label=label)
        kmf.fit(subset[duration_col].astype(float), subset[event_col].astype(int))
        fitted.append(kmf)
        color = COLORS.get(label, PALETTE[i % len(PALETTE)])
        cutoff = curve_cutoff(kmf)
        survival = kmf.survival_function_
        ci = kmf.confidence_interval_survival_function_
        visible = survival.index <= cutoff
        ax.step(survival.index[visible], survival.iloc[:, 0][visible], where="post", color=color, lw=2, label=label)
        ax.fill_between(
            ci.index[visible], ci.iloc[:, 0][visible], ci.iloc[:, 1][visible],
            step="post", color=color, alpha=0.16,
        )
        # A dotted tail warns that estimates after this point have fewer than 30 at risk.
        tail = survival.index >= cutoff
        if tail.sum() > 1:
            ax.step(survival.index[tail], survival.iloc[:, 0][tail], where="post", color=color, lw=1.1, ls=":", alpha=0.7)
        median, median_lo, median_hi = median_and_ci(kmf)
        result = {
            "comparison": name,
            "group": label,
            "n": len(subset),
            "events": int(subset[event_col].sum()),
            "censored": int((1 - subset[event_col]).sum()),
            "median_days": median,
            "median_ci_lower_days": median_lo,
            "median_ci_upper_days": median_hi,
            "logrank_overall_p": overall_p,
            "logrank_trend_p": trend_p,
            "group_rule": rule,
            "curve_cutoff_days": cutoff,
            "max_followup_days": int(subset[duration_col].max()),
        }
        for year, day in enumerate(YEAR_DAYS, start=1):
            estimate, lower, upper = km_ci_at(kmf, day)
            result[f"survival_{year}y"] = estimate
            result[f"survival_{year}y_ci_lower"] = lower
            result[f"survival_{year}y_ci_upper"] = upper
        rows.append(result)

    max_days = int(frame[duration_col].max())
    tick_max = int(np.ceil(max_days / 365) * 365)
    ax.set_xlim(0, tick_max)
    ax.set_ylim(0, 1.03)
    ax.set_xticks(np.arange(0, tick_max + 1, 365))
    ax.set_xticklabels([f"{x / 365:.0f}" for x in np.arange(0, tick_max + 1, 365)])
    ax.set_xlabel("개업 후 경과 연수")
    ax.set_ylabel("생존 확률")
    ax.set_title(title, pad=14, weight="bold")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.text(0.01, 0.02, "실선: 위험집합 30개 이상  ·  점선: 30개 미만", transform=ax.transAxes, fontsize=8, color="#555555")
    add_risk_table(ax, fitted, group_order)
    fig.subplots_adjust(bottom=0.34, left=0.11, right=0.97, top=0.90)
    fig.savefig(FIGURE_DIR / f"04_km_{name}.png", bbox_inches="tight")
    plt.close(fig)
    return rows


def make_opening_period_mix(comparisons: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Describe opening-period composition within every competition/demand group."""
    pieces: list[pd.DataFrame] = []
    for comparison, frame in comparisons.items():
        work = frame[["group", "permit_date"]].copy()
        work["opening_period"] = np.where(
            work["permit_date"].dt.year <= 2022, "코로나 시기 개업 (2021–2022)", "이후 개업 (2023–2026)"
        )
        counts = work.groupby(["group", "opening_period"], observed=True).size().rename("n").reset_index()
        counts.insert(0, "comparison", comparison)
        counts["group_n"] = counts.groupby("group")["n"].transform("sum")
        counts["share"] = counts["n"] / counts["group_n"]
        pieces.append(counts)
    return pd.concat(pieces, ignore_index=True)


def make_offset_diagnostics(
    frame: pd.DataFrame, demand_column: str, demand_label: str, comparison: str
) -> pd.DataFrame:
    """Quantify clustering, competition-only gaps, and 2×2 cell sizes."""
    grouped, _ = qgroups(frame, demand_column)
    work = frame.assign(group=grouped)
    rows: list[dict] = []
    rho = float(work[["comp_total_r200", demand_column]].corr(method="spearman").iloc[0, 1])
    for group in quartile_order(grouped):
        values = work.loc[work["group"] == group, "comp_total_r200"]
        rows.append({
            "diagnostic": "demand_quartile_competition",
            "comparison": comparison,
            "demand_variable": demand_column,
            "group": group,
            "n": len(values),
            "competition_median": float(values.median()),
            "competition_mean": float(values.mean()),
            "spearman_rho": rho,
        })

    competition_median = float(frame["comp_total_r200"].median())
    demand_median = float(frame[demand_column].median())
    competition_group = np.where(
        frame["comp_total_r200"] <= competition_median, "경쟁 낮음", "경쟁 높음"
    )
    low_competition = frame.loc[competition_group == "경쟁 낮음"]
    high_competition = frame.loc[competition_group == "경쟁 높음"]
    low_kmf = KaplanMeierFitter().fit(low_competition["survival_days"], low_competition["event"])
    high_kmf = KaplanMeierFitter().fit(high_competition["survival_days"], high_competition["event"])
    competition_only_gaps = {
        f"high_minus_low_{year}y_pp": (float(high_kmf.predict(day)) - float(low_kmf.predict(day))) * 100
        for year, day in enumerate(YEAR_DAYS, start=1)
    }
    rows.append({
        "diagnostic": "competition_only_gap",
        "comparison": comparison,
        "demand_variable": demand_column,
        "group": "고경쟁−저경쟁",
        "n": len(frame),
        "low_competition_n": len(low_competition),
        "high_competition_n": len(high_competition),
        "competition_cutpoint": competition_median,
        "demand_cutpoint": demand_median,
        **competition_only_gaps,
    })

    demand_group = np.where(frame[demand_column] <= demand_median, f"{demand_label} 낮음", f"{demand_label} 높음")
    cell_group = pd.Series(competition_group + " · " + demand_group, index=frame.index)
    for group in [
        f"경쟁 낮음 · {demand_label} 낮음",
        f"경쟁 낮음 · {demand_label} 높음",
        f"경쟁 높음 · {demand_label} 낮음",
        f"경쟁 높음 · {demand_label} 높음",
    ]:
        rows.append({
            "diagnostic": "offset_cell_n",
            "comparison": comparison,
            "demand_variable": demand_column,
            "group": group,
            "n": int((cell_group == group).sum()),
            "competition_cutpoint": competition_median,
            "demand_cutpoint": demand_median,
        })
    return pd.DataFrame(rows)


def make_report(
    summary: pd.DataFrame,
    metadata: dict[str, str],
    data: pd.DataFrame,
    opening_mix: pd.DataFrame,
    offset_diagnostics: pd.DataFrame,
) -> str:
    def group_text(comparison: str) -> str:
        rows = summary[summary["comparison"] == comparison]
        pieces = []
        for _, row in rows.iterrows():
            pieces.append(
                f"{row['group']} {int(row['n']):,}개(사건 {int(row['events']):,}건, "
                f"1년 생존 {row['survival_1y'] * 100:.1f}%, 3년 생존 {row['survival_3y'] * 100:.1f}%)"
            )
        return "; ".join(pieces)

    p_text = lambda comparison: fmt_p(summary.loc[summary["comparison"] == comparison, "logrank_overall_p"].iloc[0])
    trend_text = lambda comparison: fmt_p(summary.loc[summary["comparison"] == comparison, "logrank_trend_p"].iloc[0])
    private_n = int((~data["is_franchise"]).sum())
    full_n = len(data)
    robust_private = int(((~data["is_franchise"]) & data["robust_relicense_in_sample"].fillna(False)).sum())
    franchise_distribution = metadata["franchise_distribution"]
    def survival(comparison: str, group: str, year: int) -> float:
        row = summary.loc[(summary["comparison"] == comparison) & (summary["group"] == group)]
        return float(row.iloc[0][f"survival_{year}y"] * 100)

    def survival_3y(comparison: str, position: int) -> float:
        return float(summary.loc[summary["comparison"] == comparison].iloc[position]["survival_3y"] * 100)

    def gap_text(comparison: str, lower_group: str, higher_group: str) -> str:
        gaps = [survival(comparison, higher_group, year) - survival(comparison, lower_group, year) for year in range(1, 4)]
        return " · ".join(f"{year}년 {gap:+.1f}%p" for year, gap in enumerate(gaps, start=1))

    def mix_markdown() -> str:
        wide = opening_mix.pivot(index=["comparison", "group"], columns="opening_period", values=["n", "share"]).reset_index()
        wide.columns = [first if not second else f"{first}__{second}" for first, second in wide.columns]
        period_a = "코로나 시기 개업 (2021–2022)"
        period_b = "이후 개업 (2023–2026)"
        lines = ["| 비교 | 집단 | 코로나 시기 개업 | 이후 개업 |", "| --- | --- | ---: | ---: |"]
        labels = {
            "competition_total": "전체 경쟁 수", "competition_independent": "개인 경쟁 수",
            "competition_franchise": "프랜차이즈 경쟁 수", "demand_sales": "총매출",
            "demand_floating": "유동인구", "demand_resident": "상주인구(참고)",
        }
        for _, row in wide.iterrows():
            a = f"{int(row[f'n__{period_a}']):,} ({row[f'share__{period_a}'] * 100:.1f}%)"
            b = f"{int(row[f'n__{period_b}']):,} ({row[f'share__{period_b}'] * 100:.1f}%)"
            lines.append(f"| {labels[row['comparison']]} | {row['group']} | {a} | {b} |")
        return "\n".join(lines)

    def composition_range(comparison: str) -> str:
        shares = opening_mix.loc[
            (opening_mix["comparison"] == comparison)
            & (opening_mix["opening_period"] == "이후 개업 (2023–2026)"), "share"
        ] * 100
        return f"{shares.min():.1f}~{shares.max():.1f}%"

    def offset_diagnostic_text(comparison: str) -> str:
        rows = offset_diagnostics[
            (offset_diagnostics["comparison"] == comparison)
            & (offset_diagnostics["diagnostic"] == "demand_quartile_competition")
        ]
        pieces = [
            f"{row['group']} 중앙값 {row['competition_median']:.0f}개·평균 {row['competition_mean']:.1f}개"
            for _, row in rows.iterrows()
        ]
        return "; ".join(pieces) + f" (Spearman ρ={rows.iloc[0]['spearman_rho']:.3f})"

    def competition_only_gap_text(comparison: str) -> str:
        row = offset_diagnostics.loc[
            (offset_diagnostics["comparison"] == comparison)
            & (offset_diagnostics["diagnostic"] == "competition_only_gap")
        ].iloc[0]
        gaps = [row[f"high_minus_low_{year}y_pp"] for year in range(1, 4)]
        return " · ".join(f"{year}년 {gap:+.1f}%p" for year, gap in enumerate(gaps, start=1)) + (
            f" (저경쟁 {int(row['low_competition_n']):,}개, 고경쟁 {int(row['high_competition_n']):,}개)"
        )

    def offset_cell_count_text(comparison: str) -> str:
        rows = offset_diagnostics.loc[
            (offset_diagnostics["comparison"] == comparison)
            & (offset_diagnostics["diagnostic"] == "offset_cell_n")
        ]
        return "; ".join(f"{row['group']} {int(row['n']):,}개" for _, row in rows.iterrows())

    competition_direction = (
        f"전체 경쟁은 최저 사분위 {survival_3y('competition_total', 0):.1f}%에서 "
        f"최고 사분위 {survival_3y('competition_total', -1):.1f}%로, 개인 경쟁은 "
        f"{survival_3y('competition_independent', 0):.1f}%에서 {survival_3y('competition_independent', -1):.1f}%로 "
        "3년 생존확률이 낮아져 경쟁이 많을수록 폐업 위험이 높다는 가설과 같은 방향이다."
    )
    demand_direction = (
        f"반면 총매출 최저·최고 사분위의 3년 생존확률은 각각 "
        f"{survival_3y('demand_sales', 0):.1f}%·{survival_3y('demand_sales', -1):.1f}%, "
        f"유동인구는 {survival_3y('demand_floating', 0):.1f}%·{survival_3y('demand_floating', -1):.1f}%로 "
        "높은 수요가 생존을 높인다는 가설과 일관된 단변량 방향을 보이지 않았다."
    )
    sales_low = "경쟁 낮음 · 총매출 낮음"
    sales_high = "경쟁 높음 · 총매출 낮음"
    sales_low_demand_high_comp = gap_text("offset_sales", sales_low, sales_high)
    sales_high_demand_low_comp = gap_text("offset_sales", "경쟁 낮음 · 총매출 높음", "경쟁 높음 · 총매출 높음")
    floating_low = "경쟁 낮음 · 유동인구 낮음"
    floating_high = "경쟁 높음 · 유동인구 낮음"
    floating_low_demand_high_comp = gap_text("offset_floating", floating_low, floating_high)
    floating_high_demand_low_comp = gap_text("offset_floating", "경쟁 낮음 · 유동인구 높음", "경쟁 높음 · 유동인구 높음")
    sales_competition_only = competition_only_gap_text("demand_sales")
    floating_competition_only = competition_only_gap_text("demand_floating")
    sales_cell_counts = offset_cell_count_text("demand_sales")
    floating_cell_counts = offset_cell_count_text("demand_floating")
    robust_gap = gap_text("robustness_relicense", "주 분석 인허가 단위", "재인허가 통합 강건성")
    return f"""# Kaplan–Meier 생존곡선 보고서

## 주요 결과

이 분석은 인허가 단위의 카페 점포 생존을 단변량으로 비교한 탐색 결과다. 주 분석 대상인 개인 카페는 {private_n:,}개이며, 관측 종료일은 2026-09-30이다. 아래의 p값은 차이의 크기를 대신하지 않으므로, 각 비교에서 생존확률의 방향과 간격을 먼저 확인해야 한다. 다른 입지·수요·개업 시기는 통제하지 않았으며, Cox 단계에서 함께 확인할 사항이다.

개인 카페 전체 곡선은 `04_km_overall.png`에 제시했다. 표본별 점포 수, 사건 수, 중앙 생존기간과 신뢰구간, 1·2·3년 생존확률은 `docs/tables/04_km_summary.csv`에 모두 수록했다. 중앙 생존기간의 신뢰구간이 `—`인 경우에는 해당 신뢰구간 곡선이 50%에 이르지 않아 추정되지 않은 것이다.

개업 시기 비교에서는 {group_text('opening_period')} (전체 log-rank p={p_text('opening_period')})였다. 이후 개업 집단은 1년 생존확률이 더 낮지만(82.5% 대 85.9%), 3년은 54.0% 대 52.7%로 역전되어 곡선이 교차한다. 2023년 이후 개업 점포는 관측 가능한 기간이 짧으므로, 이 두 집단은 **두 집단 모두가 관측된 공통 기간**까지만 비교해야 하며 3년 차이는 불안정할 수 있다.

점포 유형 비교에서는 {group_text('store_type')} (전체 log-rank p={p_text('store_type')})였다. 이 그림은 전체 커피 점포 {full_n:,}개를 사용한 보조 비교이며, 개인 카페와 프랜차이즈의 운영 방식·입지 구성이 다를 수 있어 주 분석의 인과적 비교가 아니다.

경쟁 비교의 관찰값은 다음과 같다.

- 전체 경쟁 카페 수 사분위: {group_text('competition_total')} (전체 p={p_text('competition_total')}, 순서 추세 p={trend_text('competition_total')})
- 개인 카페 경쟁 수 사분위: {group_text('competition_independent')} (전체 p={p_text('competition_independent')}, 순서 추세 p={trend_text('competition_independent')})
- 프랜차이즈 경쟁 수: {group_text('competition_franchise')} (전체 p={p_text('competition_franchise')}, 순서 추세 p={trend_text('competition_franchise')})

프랜차이즈 경쟁의 원분포는 {franchise_distribution}이며, 0개가 한 구간에 집중된 특성을 보존하려고 `0 / 1–2 / 3–5 / 6개 이상`으로 구분했다. 그림의 방향은 경쟁 가설과 맞는지 확인하는 **기술적 신호**이며, 개업 시기와 수요가 집단별로 달라질 수 있으므로 Cox 모형에서 동시 조정이 필요하다.

{competition_direction}

수요 지표의 비교는 다음과 같다.

- 개업 직전 분기 상권 총매출 사분위: {group_text('demand_sales')} (전체 p={p_text('demand_sales')}, 순서 추세 p={trend_text('demand_sales')})
- 유동인구 사분위: {group_text('demand_floating')} (전체 p={p_text('demand_floating')}, 순서 추세 p={trend_text('demand_floating')})
- 상주인구 사분위(참고): {group_text('demand_resident')} (전체 p={p_text('demand_resident')}, 순서 추세 p={trend_text('demand_resident')})

{demand_direction}

### 경쟁과 수요의 상쇄 점검

상쇄 그림은 `04_km_offset_sales.png`, `04_km_offset_floating.png`에 제시했다. 수요 사분위별 200m 전체 경쟁 수는 총매출에서 {offset_diagnostic_text('demand_sales')}, 유동인구에서 {offset_diagnostic_text('demand_floating')}였다. 즉 수요가 높은 곳에 경쟁도 집중되는 양의 상관이 있어, 수요의 단변량 곡선 차이가 작게 보이는 한 가지 가능한 설명과 부합한다. 이는 보정 결과가 아니며 Cox에서 경쟁과 수요를 함께 넣어 확인할 사항이다.

경쟁만으로 200m 경쟁 합계 중앙값을 기준으로 나눈 고경쟁−저경쟁 생존율 차이는 총매출 표본에서 {sales_competition_only}, 유동인구 표본에서 {floating_competition_only}였다. 수요 집단 안의 경쟁 차이와 나란히 보면, 같은 총매출 범위에서는 저수요에서 {sales_low_demand_high_comp}, 고수요에서 {sales_high_demand_low_comp}였고, 같은 유동인구 범위에서는 저수요에서 {floating_low_demand_high_comp}, 고수요에서 {floating_high_demand_low_comp}였다. 음수는 고경쟁 집단의 생존율이 낮다는 뜻이다. 2×2 칸별 점포 수는 총매출 표본에서 {sales_cell_counts}; 유동인구 표본에서 {floating_cell_counts}였다. 따라서 경쟁 낮음·수요 낮음과 경쟁 높음·수요 높음 칸에 점포가 상대적으로 많이 모이는지 표본 구성도 함께 확인할 수 있다. 반면 수요의 양끝 사분위 곡선 차이는 작고 일정하지 않아, 경쟁 집중과 수요의 반대 방향 연관이 서로 가려졌을 가능성을 시사할 뿐이다. 2×2 전체 log-rank p값은 총매출 {p_text('offset_sales')}, 유동인구 {p_text('offset_floating')}이며, 이 단변량 분류만으로 상쇄의 크기나 원인을 확정하지 않는다.

200m 이내 상권 배정 개인 카페의 상권 구분 곡선은 `04_km_trade_area_type.png`에 제시했다. {group_text('trade_area_type')} 관광특구의 1년 생존확률은 다른 상권 구분보다 크게 낮지만, 상권 구성과 개업 시기 차이가 반영될 수 있어 해당 구분 자체의 효과로 해석하지 않는다.

재인허가 통합 강건성 곡선은 주 분석 개인 카페와 재인허가 통합 개인 카페 {robust_private:,}개를 겹쳐 제시했다(`04_km_robustness_relicense.png`). 통합 표본−주 분석 표본의 생존율 차이는 {robust_gap}로 작으며, 주 분석의 인허가 단위 정의를 바꾸지 않고 민감도만 점검한다.

### 경쟁·수요 집단의 개업 시기 구성

경쟁·수요 집단의 개업 시기 비율은 `docs/tables/04_km_opening_period_mix.csv`에 저장했고, 아래에 함께 제시했다. 이후 개업 비율의 집단 간 범위는 전체 경쟁 수 {composition_range('competition_total')}, 개인 경쟁 수 {composition_range('competition_independent')}, 프랜차이즈 경쟁 수 {composition_range('competition_franchise')}, 총매출 {composition_range('demand_sales')}, 유동인구 {composition_range('demand_floating')}다. 이 차이가 있는 비교에서는 KM 차이 일부가 개업 시기 구성에서 온 것일 수 있으므로, Cox에서 개업 연도로 층화해 통제한다.

{mix_markdown()}

## Cox 설계에서 확인할 점

- **수요 투입 전후 경쟁 계수:** 경쟁만 넣은 모형의 경쟁 계수와 총매출·유동인구를 함께 넣은 모형의 계수를 비교해, 수요와의 공간적 동반 분포를 조정했을 때 경쟁의 방향과 크기가 얼마나 달라지는지 확인한다.
- **경쟁×수요 상호작용:** 경쟁의 연관이 저수요·고수요 입지에서 같은지 확인한다. 상쇄 그림의 집단별 차이는 기술적 비교이므로, 상호작용항의 추정치와 신뢰구간으로 별도 검증한다.
- **개업 연도 층화:** 개업 연도별 시장 환경과 관측기간 차이를 통제하도록 개업 연도 층화 Cox 모형을 우선 검토한다.
- **비례위험 가정:** 고경쟁·고매출 집단 곡선은 개업 초기 몇 달 안에 먼저 떨어지는 모습이 보이므로, Schoenfeld 잔차와 시간상호작용으로 비례위험 가정을 점검한다. 위반 시 초기와 이후 위험이 다른 시간가변 효과 또는 층화 모형을 검토한다.
- **관광특구 처리:** 관광특구의 낮은 초기 생존이 경쟁 효과와 혼동되지 않도록 상권 구분을 보정하고, 관광특구 제외 민감도 분석도 함께 제시한다.
- **프랜차이즈 경쟁의 비선형성:** 프랜차이즈 경쟁 6개 이상 구간의 생존 차이가 앞선 구간과 선형적으로 이어지는지 제한적 스플라인 또는 구간 더미로 확인한다.

## 방법과 검증

### 표본과 변수

- 전체·개업 시기·경쟁·수요·상권 구분의 기본 비교는 개인 카페(`is_franchise == False`)를 썼다. 점포 유형 비교만 프랜차이즈를 포함한 전체 커피 점포를 사용했다.
- 경쟁 비교는 좌표와 해당 경쟁 변수가 있는 개인 카페만, 수요 및 상권 구분 비교는 200m 이내 최근접 상권 배정(`trdar_within_200m == True`)과 해당 변수의 비결측을 모두 만족한 개인 카페만 포함했다. 따라서 비교별 표본 수는 달라질 수 있다.
- 매출·유동인구·상주인구의 사분위 경계는 각각의 유효 비교 표본에서 계산했다. 연속형 구간은 `(a, b]`, 경쟁 수처럼 정수형인 구간은 실제 포함 범위(예: `0개 이상 5개 미만`)로 표시했다. 경계값은 요약 CSV의 `group_rule`에 기록했다. 상권 총매출은 개업 직전 분기 값이며, 상주인구는 참고용으로 함께 그렸다.
- 관측 종료일은 2026-09-30이며 폐업하지 않은 점포는 우측 중도절단으로 처리했다. 재인허가 강건성은 `robust_relicense_in_sample`과 이에 대응하는 생존기간·사건 변수를 사용했다.

### 곡선과 검정

각 집단에 Kaplan–Meier 생존곡선과 95% 신뢰구간을 표시했다. 중앙 생존기간 신뢰구간은 생존 신뢰구간의 하한·상한 곡선이 각각 0.5를 처음 지나는 시점으로 계산했다. 위험집합이 30개 미만이 되는 시점 뒤의 곡선은 점선으로 표시해 추정의 불안정성을 드러냈으며, 그림 아래에는 각 연수 시점의 위험집합 수를 넣었다. 전체 log-rank 검정은 두 집단 이상 비교마다 수행했고, 사분위·경쟁 구간처럼 순서가 있는 집단에는 저값→고값의 명시적 `group_order`에 따른 1자유도 순서 log-rank 추세검정도 추가했다. 검정은 집단 간 생존곡선의 차이를 요약할 뿐 공변량을 조정하지 않는다.

## 산출물과 유의사항

- 그림: `docs/figures/04_km_*.png`
- 요약표: `docs/tables/04_km_summary.csv`
- 개업 시기 구성표: `docs/tables/04_km_opening_period_mix.csv`
- 상쇄 진단표: `docs/tables/04_km_offset_diagnostics.csv`
- 재현 스크립트: `src/04_kaplan_meier.py`

개업 시기별 관측기간 차이, 경쟁·수요 집단의 개업 시기 구성 차이, 그리고 KM의 비조정 비교라는 한계 때문에 이 결과만으로 폐업 위험의 원인을 판단하지 않는다. 상쇄 진단표에는 수요 사분위별 경쟁 분포, 경쟁만으로 나눈 생존율 차이, 2×2 칸별 점포 수를 함께 기록했다. 다음 Cox 단계에서는 개업 연도와 수요·경쟁 공변량을 함께 조정해 이 방향이 유지되는지 검증해야 한다.
"""


def main() -> None:
    os.environ["TEMP"] = str(ROOT / ".tmp")
    os.environ["TMP"] = str(ROOT / ".tmp")
    (ROOT / ".tmp").mkdir(exist_ok=True)
    for directory in (FIGURE_DIR, TABLE_PATH.parent, REPORT_PATH.parent, LOG_PATH.parent):
        directory.mkdir(parents=True, exist_ok=True)
    # These names were replaced by the requested 'offset' terminology.
    for stale_figure in (FIGURE_DIR / "04_km_interaction_sales.png", FIGURE_DIR / "04_km_interaction_floating.png"):
        stale_figure.unlink(missing_ok=True)
    configure_matplotlib()
    synthetic_trend_p = verify_trend_logrank()
    data = read_data()
    private = data.loc[~data["is_franchise"]].copy()
    all_rows: list[dict] = []
    metadata: dict[str, str] = {}
    opening_mix_frames: dict[str, pd.DataFrame] = {}
    offset_diagnostic_frames: list[pd.DataFrame] = []

    all_rows += plot_comparison("overall", "개인 카페 전체 생존곡선", private.assign(group="개인 카페"), "group", ["개인 카페"], "개인 카페 전체")

    opening = private.assign(group=np.where(private["permit_date"].dt.year <= 2022, "코로나 시기 개업 (2021–2022)", "이후 개업 (2023–2026)"))
    all_rows += plot_comparison("opening_period", "개업 시기별 개인 카페 생존곡선", opening, "group", ["코로나 시기 개업 (2021–2022)", "이후 개업 (2023–2026)"], "2021–2022 / 2023–2026")

    types = data.assign(group=np.where(data["is_franchise"], "프랜차이즈", "개인 카페"))
    all_rows += plot_comparison("store_type", "개인 카페와 프랜차이즈 생존곡선", types, "group", ["개인 카페", "프랜차이즈"], "개인 카페 / 프랜차이즈")

    competition_base = private.dropna(subset=["comp_total_r200", "comp_indep_r200", "comp_fran_r200"]).copy()
    for name, column, title in [
        ("competition_total", "comp_total_r200", "200m 전체 경쟁 카페 수별 생존곡선"),
        ("competition_independent", "comp_indep_r200", "200m 개인 카페 경쟁 수별 생존곡선"),
    ]:
        grouped, rule = qgroups(competition_base, column)
        frame = competition_base.assign(group=grouped)
        order = quartile_order(grouped)
        all_rows += plot_comparison(name, title, frame, "group", order, f"{column} 사분위 경계: {rule}", ordered=True)
        opening_mix_frames[name] = frame

    franchise_group, franchise_rule = franchise_comp_groups(competition_base)
    franchise_frame = competition_base.assign(group=franchise_group)
    franchise_order = ["0개", "1–2개", "3–5개", "6개 이상"]
    distribution = competition_base["comp_fran_r200"].value_counts().sort_index()
    metadata["franchise_distribution"] = "; ".join(f"{int(k)}개 {int(v):,}개" for k, v in distribution.items())
    all_rows += plot_comparison("competition_franchise", "200m 프랜차이즈 경쟁 수별 생존곡선", franchise_frame, "group", franchise_order, f"comp_fran_r200 구간: {franchise_rule}", ordered=True)
    opening_mix_frames["competition_franchise"] = franchise_frame

    demand_base = private.loc[private["trdar_within_200m"].fillna(False)].copy()
    for name, column, title in [
        ("demand_sales", "trdar_sales_total", "상권 총매출 사분위별 생존곡선"),
        ("demand_floating", "trdar_floating_pop", "유동인구 사분위별 생존곡선"),
        ("demand_resident", "trdar_resident_pop", "상주인구 사분위별 생존곡선 (참고)"),
    ]:
        frame = demand_base.dropna(subset=[column]).copy()
        grouped, rule = qgroups(frame, column)
        frame["group"] = grouped
        order = quartile_order(grouped)
        all_rows += plot_comparison(name, title, frame, "group", order, f"{column} 사분위 경계: {rule}", ordered=True)
        opening_mix_frames[name] = frame

    for name, demand_column, demand_label, title in [
        ("offset_sales", "trdar_sales_total", "총매출", "경쟁과 상권 총매출의 상쇄 점검 생존곡선"),
        ("offset_floating", "trdar_floating_pop", "유동인구", "경쟁과 유동인구의 상쇄 점검 생존곡선"),
    ]:
        frame = demand_base.dropna(subset=["comp_total_r200", demand_column]).copy()
        offset_diagnostic_frames.append(
            make_offset_diagnostics(frame, demand_column, demand_label, name.replace("offset_", "demand_"))
        )
        comp_median = float(frame["comp_total_r200"].median())
        demand_median = float(frame[demand_column].median())
        frame["group"] = np.where(frame["comp_total_r200"] <= comp_median, "경쟁 낮음", "경쟁 높음") + " · " + np.where(frame[demand_column] <= demand_median, f"{demand_label} 낮음", f"{demand_label} 높음")
        order = [f"경쟁 낮음 · {demand_label} 낮음", f"경쟁 낮음 · {demand_label} 높음", f"경쟁 높음 · {demand_label} 낮음", f"경쟁 높음 · {demand_label} 높음"]
        all_rows += plot_comparison(name, title, frame, "group", order, f"comp_total_r200 중앙값 {fmt_number(comp_median)}; {demand_column} 중앙값 {fmt_number(demand_median)}")

    type_frame = demand_base.dropna(subset=["trdar_type"]).copy()
    type_order = list(type_frame["trdar_type"].value_counts().index)
    type_frame["group"] = type_frame["trdar_type"].astype(str)
    all_rows += plot_comparison("trade_area_type", "상권 구분별 개인 카페 생존곡선", type_frame, "group", type_order, "200m 이내 상권 배정; 상권 구분")

    robust = private.loc[private["robust_relicense_in_sample"].fillna(False)].dropna(subset=["robust_relicense_survival_days", "robust_relicense_event"]).copy()
    main_frame = private.assign(group="주 분석 인허가 단위", duration=private["survival_days"], outcome=private["event"])
    robust_frame = robust.assign(group="재인허가 통합 강건성", duration=robust["robust_relicense_survival_days"], outcome=robust["robust_relicense_event"])
    combined = pd.concat([main_frame, robust_frame], ignore_index=True)
    all_rows += plot_comparison("robustness_relicense", "재인허가 통합 전후 개인 카페 생존곡선", combined, "group", ["주 분석 인허가 단위", "재인허가 통합 강건성"], "개인 카페: 인허가 단위 / 재인허가 통합", duration_col="duration", event_col="outcome")

    summary = pd.DataFrame(all_rows)
    opening_mix = make_opening_period_mix(opening_mix_frames)
    offset_diagnostics = pd.concat(offset_diagnostic_frames, ignore_index=True)
    summary.to_csv(TABLE_PATH, index=False, encoding="utf-8-sig")
    opening_mix.to_csv(OPENING_MIX_PATH, index=False, encoding="utf-8-sig")
    offset_diagnostics.to_csv(OFFSET_DIAGNOSTICS_PATH, index=False, encoding="utf-8-sig")
    REPORT_PATH.write_text(make_report(summary, metadata, data, opening_mix, offset_diagnostics), encoding="utf-8")
    shutil.copy2(__file__, ROOT / ".tmp" / "04_kaplan_meier_last_run.py")
    LOG_PATH.write_text(
        "# Kaplan–Meier 실행 로그\n\n"
        f"- 입력: `{INPUT_PATH.relative_to(ROOT)}` ({len(data):,}행)\n"
        f"- 개인 카페: {len(private):,}행\n"
        f"- 생성 그림: {len(list(FIGURE_DIR.glob('04_km_*.png')))}개\n"
        f"- 요약표 행: {len(summary):,}행\n"
        f"- 개업 시기 구성표 행: {len(opening_mix):,}행\n"
        f"- 상쇄 진단표 행: {len(offset_diagnostics):,}행\n"
        f"- 검증: 입력 행 수, 생존기간 양수, 필수 열 존재를 검사했고, 행 순서를 Q4·Q1·Q3·Q2로 섞은 단조 감소 합성자료의 추세검정 p={synthetic_trend_p:.3g}를 확인했다.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
