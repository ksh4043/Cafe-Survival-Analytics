"""Create reproducible report figures and insert them into pre-Cox reports.

Run from the project root:
    .venv\\Scripts\\python.exe src\\report_figures.py

This script reads processed outputs and tables only.  It deliberately never
writes the 05/ or 06/ reports because those stages may be edited concurrently.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "cafes_analysis.parquet"
AREA = ROOT / "data" / "raw" / "area" / "서울시 상권분석서비스(영역-상권).shp"
FLOW = ROOT / "docs" / "tables" / "05_cox_sample_flow.csv"
FIGURES = ROOT / "docs" / "figures"
REPORTS = ROOT / "docs" / "reports"
LOG = ROOT / "docs" / "logs" / "report_figures_execution.md"
README = FIGURES / "README.md"

DPI = 200
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#555555"
LIGHT_GRAY = "#D9D9D9"
FIGURE_FILES = [
    "02_sample_flow.png", "02_openings_closures_by_quarter.png",
    "03_closure_vs_density_map.png", "03_competition_distribution.png",
    "03_competition_vs_demand.png",
]


def setup_style() -> None:
    """Use the requested Korean typeface when Windows supplies it."""
    mpl.rcParams.update({
        "font.family": "Malgun Gothic",
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"필수 입력 없음: {path.relative_to(ROOT)}")


def read_data() -> pd.DataFrame:
    require(DATA)
    data = pd.read_parquet(DATA).copy()
    for col in ("permit_date", "closure_date"):
        data[col] = pd.to_datetime(data[col], errors="coerce")
    required = {"permit_date", "closure_date", "event", "is_franchise", "brand", "coordinate_x", "coordinate_y",
                "comp_indep_r100", "comp_fran_r100", "comp_indep_r200", "comp_fran_r200", "comp_indep_r500",
                "comp_fran_r500", "trdar_nearest_dist_m", "trdar_sales_total", "trdar_within_200m"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"분석 파일 필수 열 없음: {missing}")
    return data


def quarter_label(series: pd.Series) -> pd.Series:
    return series.dt.to_period("Q").astype(str).str.replace("Q", "Q", regex=False)


def make_sample_flow(data: pd.DataFrame, flow: pd.DataFrame) -> None:
    final = flow.loc[flow["step"].eq("최종 공통 분석 표본")]
    if final.empty:
        raise ValueError("05_cox_sample_flow.csv에 최종 공통 분석 표본 행이 없습니다.")
    final_n, final_events = int(final.iloc[0]["n_after"]), int(final.iloc[0]["events_after"])
    steps = [
        ("원자료", 148302), ("커피숍 선택", 40665), ("유효 인허가", 40469),
        ("한시적·0일 이하·비카페 제외", 39602), ("개업 기간 분석 표본", len(data)),
        ("개인 카페\n(분석 대상)", int((~data["is_franchise"].fillna(False)).sum())),
        ("Cox 공통 표본", final_n),
    ]
    exclusions = [
        "커피숍 외 업태", "폐업일 결측·관측 종료 뒤", "한시적·0일 이하·비카페",
        "개업 기간 밖", "프랜차이즈\n(경쟁 집합 유지)", "200m 밖·공변량 결측",
    ]
    positions = np.arange(len(steps), dtype=float) * 1.48
    width = 1.02
    fig, ax = plt.subplots(figsize=(16, 4.7))
    ax.set_xlim(-.68, positions[-1] + .68); ax.set_ylim(.13, .92); ax.axis("off")
    for i, ((title, n), x) in enumerate(zip(steps, positions)):
        color = BLUE if title in {"개인 카페", "Cox 공통 표본"} else "#E9EEF2"
        if "개인 카페" in title or title == "Cox 공통 표본":
            color = BLUE
        patch = FancyBboxPatch((x - width / 2, .34), width, .27, boxstyle="round,pad=0.02,rounding_size=0.03",
                               linewidth=1.1, edgecolor=GRAY, facecolor=color)
        ax.add_patch(patch)
        text_color = "white" if color == BLUE else "black"
        ax.text(x, .535, title, ha="center", va="center", weight="bold", color=text_color, fontsize=9.2)
        ax.text(x, .405, f"{n:,}건", ha="center", va="center", color=text_color, fontsize=11)
        if i:
            loss = steps[i - 1][1] - n
            left, right = positions[i - 1] + width / 2 + .04, x - width / 2 - .04
            ax.annotate("", xy=(right, .475), xytext=(left, .475),
                        arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 1.2})
            ax.text((left + right) / 2, .69, f"-{loss:,}\n{exclusions[i - 1]}", ha="center", va="center", color=GRAY, fontsize=7.7)
    ax.set_title("분석 표본 흐름", loc="left", weight="bold")
    ax.text(positions[-1] / 2, .18, f"Cox 공통 표본은 200m 이내 상권 배정 개인 카페의 완전 사례이며, 폐업 사건은 {final_events:,}건이다.", ha="center", fontsize=9.5)
    save(fig, "02_sample_flow.png")


def make_open_close(data: pd.DataFrame) -> None:
    data = data.copy()
    data["유형"] = np.where(data["is_franchise"].fillna(False), "프랜차이즈", "개인 카페")
    openings = data.groupby([quarter_label(data["permit_date"]), "유형"]).size().unstack(fill_value=0)
    closures = data.loc[data["event"].eq(1) & data["closure_date"].notna()].groupby([quarter_label(data.loc[data["event"].eq(1) & data["closure_date"].notna(), "closure_date"]), "유형"]).size().unstack(fill_value=0)
    quarters = pd.period_range(data["permit_date"].min().to_period("Q"), pd.Timestamp("2026-09-30").to_period("Q"), freq="Q").astype(str)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"hspace": .12})
    x = np.arange(len(quarters)); width = .38
    for ax, frame, title, ylabel in [(axes[0], openings, "분기별 개업", "개업 점포 수"), (axes[1], closures, "분기별 폐업", "폐업 점포 수")]:
        for offset, label, color in [(-width / 2, "개인 카페", BLUE), (width / 2, "프랜차이즈", ORANGE)]:
            values = frame.reindex(quarters, fill_value=0).get(label, pd.Series(0, index=quarters))
            ax.bar(x + offset, values, width, label=label, color=color)
        ax.set_title(title, loc="left", weight="bold"); ax.set_ylabel(ylabel); ax.grid(axis="y", color="#E5E5E5", linewidth=.8)
    start = list(quarters).index("2021Q2"); end = list(quarters).index("2026Q1"); observed = list(quarters).index("2026Q3")
    for ax in axes:
        ax.axvline(start - .5, color=GRAY, lw=1); ax.axvline(end + .5, color=PURPLE, lw=1.2, ls="--")
        ax.axvline(observed + .5, color=GRAY, lw=1.2, ls=":")
    axes[0].text(start + .08, axes[0].get_ylim()[1] * .90, "개업 기간\n시작", ha="left", va="top", color=GRAY, fontsize=9)
    axes[0].text(end + .65, axes[0].get_ylim()[1] * .90, "개업 기간\n종료", ha="left", va="top", color=PURPLE, fontsize=9)
    axes[1].text(observed - .35, axes[1].get_ylim()[1] * .90, "관측 종료 분기\n(2026Q3)", ha="right", va="top", color=GRAY, fontsize=9)
    axes[1].set_xticks(x[::2], quarters[::2], rotation=45, ha="right")
    fig.suptitle("개업 기간과 관측 종료 시점", x=.08, ha="left", weight="bold", fontsize=17)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.55, .975))
    fig.subplots_adjust(top=.88)
    save(fig, "02_openings_closures_by_quarter.png")


def projected_points(data: pd.DataFrame) -> gpd.GeoDataFrame:
    points = data.dropna(subset=["coordinate_x", "coordinate_y"]).copy()
    return gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points["coordinate_x"], points["coordinate_y"]), crs="EPSG:5174").to_crs("EPSG:5181")


def map_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    require(AREA)
    areas = gpd.read_file(AREA).to_crs("EPSG:5181")
    if areas.empty: raise ValueError("상권 SHP가 비어 있습니다.")
    return areas, areas.dissolve()


def make_closure_vs_density_map(points: gpd.GeoDataFrame, areas: gpd.GeoDataFrame, outline: gpd.GeoDataFrame) -> None:
    """Compare spatial concentration and crude closure rates on identical hexagons."""
    private = points.loc[~points["is_franchise"].fillna(False)]
    x, y = private.geometry.x.to_numpy(), private.geometry.y.to_numpy()
    extent = (x.min(), x.max(), y.min(), y.max())
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.4), constrained_layout=True)
    for ax in axes:
        outline.plot(ax=ax, facecolor="#F7F7F7", edgecolor=GRAY, linewidth=.7, zorder=0)
        ax.set_aspect("equal"); ax.set_axis_off()
    density = axes[0].hexbin(x, y, gridsize=54, extent=extent, mincnt=1, cmap="YlOrRd", bins="log", linewidths=0, zorder=1)
    areas.boundary.plot(ax=axes[0], color="white", linewidth=.10, alpha=.22, zorder=2)
    density_cb = fig.colorbar(density, ax=axes[0], shrink=.68, pad=.01)
    density_cb.set_label("격자 내 개인 카페 수 (로그 척도)")
    axes[0].set_title("격자 내 개인 카페 수", loc="left", weight="bold")

    # The gray base shows every populated hexagon; the colored layer is drawn
    # only where at least ten shops make the crude proportion minimally stable.
    axes[1].hexbin(x, y, C=np.ones_like(x), reduce_C_function=np.sum, gridsize=54, extent=extent,
                   mincnt=1, cmap=mpl.colors.ListedColormap([LIGHT_GRAY]), linewidths=.18, edgecolors="white", zorder=1)
    closure_rate = axes[1].hexbin(x, y, C=private["event"].to_numpy(dtype=float), reduce_C_function=np.mean,
                                  gridsize=54, extent=extent, mincnt=10, cmap="YlOrRd", vmin=0, vmax=1,
                                  linewidths=.18, edgecolors="white", zorder=2)
    areas.boundary.plot(ax=axes[1], color="white", linewidth=.10, alpha=.22, zorder=3)
    rate_cb = fig.colorbar(closure_rate, ax=axes[1], shrink=.68, pad=.01)
    rate_cb.set_label("격자 내 개인 카페 폐업 비율")
    axes[1].set_title("격자 내 개인 카페 폐업 비율", loc="left", weight="bold")
    axes[1].text(.02, .02, "점포 10개 미만 격자: 회색", transform=axes[1].transAxes, fontsize=9, color=GRAY,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 2})
    fig.suptitle("개인 카페의 공간적 밀집과 단순 폐업 비율", x=.07, ha="left", weight="bold", fontsize=17)
    fig.text(.5, .035, "점포별 관측 기간이 달라 단순 폐업 비율은 생존분석 결과가 아니며, 지역 분포를 설명하는 참고값이다.", ha="center", fontsize=10, color=GRAY)
    save(fig, "03_closure_vs_density_map.png")


def make_competition_distribution(data: pd.DataFrame) -> None:
    subset = data.loc[~data["is_franchise"].fillna(False)]
    radii = [100, 200, 500]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for ax, radius in zip(axes, radii):
        indep = subset[f"comp_indep_r{radius}"].dropna(); fran = subset[f"comp_fran_r{radius}"].dropna()
        bins = np.arange(0, max(indep.max(), fran.max()) + 2) - .5
        ax.hist(indep, bins=bins, color=BLUE, alpha=.70, label="개인 카페 경쟁")
        ax.hist(fran, bins=bins, color=ORANGE, alpha=.62, label="프랜차이즈 경쟁")
        ax.axvline(indep.median(), color=BLUE, lw=1.4, ls="--"); ax.axvline(fran.median(), color=ORANGE, lw=1.4, ls="--")
        ax.set_title(f"반경 {radius}m", weight="bold"); ax.set_xlabel("개업 당시 경쟁 카페 수"); ax.grid(axis="y", color="#E5E5E5", linewidth=.8)
        ax.set_xlim(-.5, min(max(indep.max(), fran.max()) + 1, 80))
    axes[0].set_ylabel("개인 카페 점포 수"); axes[0].legend(frameon=False)
    fig.suptitle("개인 카페의 반경별 경쟁 분포", x=.08, ha="left", weight="bold", fontsize=17)
    fig.subplots_adjust(bottom=.18, top=.84)
    fig.text(.5, .065, "점선은 각 경쟁 유형의 중앙값이며, 500m 분포는 긴 오른쪽 꼬리를 보존하되 가독성을 위해 80개에서 절단", ha="center", fontsize=9, color=GRAY)
    save(fig, "03_competition_distribution.png")


def make_competition_demand(data: pd.DataFrame) -> None:
    base = data.loc[(~data["is_franchise"].fillna(False)) & data["trdar_within_200m"].fillna(False)]
    panels = [
        ("trdar_sales_total", "상권 총매출", "개업 직전 분기 상권 총매출 log(1+원)"),
        ("trdar_floating_pop", "유동인구", "개업 직전 분기 상권 유동인구 log(1+명)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)
    for ax, (column, title, ylabel) in zip(axes, panels):
        private = base[["comp_indep_r200", column]].dropna()
        x = private["comp_indep_r200"].to_numpy()
        y = np.log1p(private[column].to_numpy())
        corr = np.corrcoef(x, y)[0, 1]
        slope, intercept = np.polyfit(x, y, 1)
        hb = ax.hexbin(x, y, gridsize=42, mincnt=1, cmap="Blues", bins="log", linewidths=0)
        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, slope * grid + intercept, color=ORANGE, lw=2)
        cb = fig.colorbar(hb, ax=ax, shrink=.82, pad=.02); cb.set_label("점포 수 (로그 척도)")
        ax.set_title(f"개인 카페 경쟁과 {title}", loc="left", weight="bold")
        ax.set_xlabel("200m 이내 개인 카페 경쟁 수"); ax.set_ylabel(ylabel)
        ax.text(.03, .94, f"Pearson r = {corr:.3f}\nn = {len(private):,}", transform=ax.transAxes, va="top", fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": .9, "pad": 3})
    fig.text(.5, .003, "경쟁과 수요가 함께 몰리는 기초 상관이며 인과 효과가 아니다.", ha="center", fontsize=10, color=GRAY)
    save(fig, "03_competition_vs_demand.png")


def caption(name: str, text: str) -> str:
    return f"![{text}](../figures/{name})\n\n{text}\n"


def remove_figure_block(path: Path, filename: str) -> bool:
    """Remove a generated Markdown figure plus its one-sentence caption."""
    content = path.read_text(encoding="utf-8")
    pattern = rf"\n*!\[[^\]]*\]\(\.\./figures/{re.escape(filename)}\)\n\n[^\n]*\n?"
    updated, count = re.subn(pattern, "\n", content)
    if count:
        path.write_text(updated, encoding="utf-8")
    return bool(count)


def insert_after_heading(path: Path, heading: str, block: str) -> bool:
    content = path.read_text(encoding="utf-8")
    if f"../figures/{block.split('../figures/')[1].split(')')[0]}" in content:
        return False
    match = re.search(rf"(^##+ {re.escape(heading)}.*?$)", content, flags=re.MULTILINE)
    if not match:
        return False
    position = content.find("\n##", match.end())
    if position < 0: position = len(content)
    content = content[:position].rstrip() + "\n\n" + block.rstrip() + "\n" + content[position:]
    path.write_text(content, encoding="utf-8")
    return True


def append_before_methods(path: Path, block: str) -> bool:
    content = path.read_text(encoding="utf-8")
    key = block.split("../figures/")[1].split(")")[0]
    if f"../figures/{key}" in content: return False
    match = re.search(r"^## 방법과 검증.*?$", content, flags=re.MULTILINE)
    if not match: return False
    content = content[:match.start()].rstrip() + "\n\n" + block.rstrip() + "\n\n" + content[match.start():]
    path.write_text(content, encoding="utf-8")
    return True


def insert_reports() -> list[str]:
    updates: list[str] = []
    p02 = REPORTS / "02_preprocess_report.md"
    remove_figure_block(p02, "02_franchise_brands.png")
    for heading, filename, text in [
        ("처리 단계와 행 수", "02_sample_flow.png", "원자료에서 Cox 공통 분석 표본까지의 제외 단계와 잔여 표본 수를 보여 준다."),
        ("연도별 개업·폐업 (최종 분석 표본)", "02_openings_closures_by_quarter.png", "개업 기간 경계와 관측 종료 시점 주변의 분기별 개업·폐업 규모를 개인 카페와 프랜차이즈로 나누어 보여 준다."),
    ]:
        if insert_after_heading(p02, heading, caption(filename, text)): updates.append(f"02_preprocess_report.md: {filename}")
    p03sp = REPORTS / "03_spatial_join_report.md"
    remove_figure_block(p03sp, "03_cafe_map.png")
    remove_figure_block(p03sp, "03_competition_density_map.png")
    if append_before_methods(p03sp, caption("03_closure_vs_density_map.png", "같은 육각 격자에서 개인 카페 수와 단순 폐업 비율의 지역 분포를 비교한다.")):
        updates.append("03_spatial_join_report.md: 03_closure_vs_density_map.png")
    p03tr = REPORTS / "03_trade_area_variables_report.md"
    remove_figure_block(p03tr, "03_trade_area_distance.png")
    remove_figure_block(p03tr, "03_competition_vs_demand.png")
    if append_before_methods(p03tr, caption("03_competition_vs_demand.png", "200m 개인 카페 경쟁과 상권 총매출·유동인구의 기초 상관을 비교한다.")):
        updates.append("03_trade_area_variables_report.md: 03_competition_vs_demand.png")
    p03ro = REPORTS / "03_robustness_and_correlation_report.md"
    remove_figure_block(p03ro, "03_covariate_correlation.png")
    if append_before_methods(p03ro, caption("03_competition_distribution.png", "개인 카페의 100m·200m·500m 반경별 개인·프랜차이즈 경쟁 수 분포를 보여 준다.")): updates.append("03_robustness_and_correlation_report.md: 03_competition_distribution.png")
    p04 = REPORTS / "04_kaplan_meier_report.md"
    content = p04.read_text(encoding="utf-8")
    km_files = sorted(FIGURES.glob("04_km_*.png"))
    if not km_files:
        updates.append("04_kaplan_meier_report.md: 04_km 그림 없음(삽입 건너뜀)")
    else:
        # Replace a previous generated block so reruns keep figures immediately
        # after their matching result paragraphs, rather than accumulating copies.
        content = re.sub(
            r"\n*!\[[^\]]*\]\(\.\./figures/04_km_[^)]+\.png\)\n\n[^\n]*\n",
            "", content, flags=re.DOTALL,
        )
        km_blocks = {
            "overall": caption("04_km_overall.png", "개인 카페 전체의 Kaplan–Meier 생존곡선을 보여 준다."),
            "opening": caption("04_km_opening_period.png", "코로나 시기와 이후 개업 개인 카페의 Kaplan–Meier 생존곡선을 보여 준다."),
            "type": caption("04_km_store_type.png", "개인 카페와 프랜차이즈의 Kaplan–Meier 생존곡선을 보여 준다."),
            "competition": "\n\n".join([caption("04_km_competition_total.png", "200m 전체 경쟁 카페 수 사분위별 생존곡선을 보여 준다."), caption("04_km_competition_independent.png", "200m 개인 카페 경쟁 수 사분위별 생존곡선을 보여 준다."), caption("04_km_competition_franchise.png", "200m 프랜차이즈 경쟁 수 구간별 생존곡선을 보여 준다.")]),
            "demand": "\n\n".join([caption("04_km_demand_sales.png", "개업 직전 분기 상권 총매출 사분위별 생존곡선을 보여 준다."), caption("04_km_demand_floating.png", "유동인구 사분위별 생존곡선을 보여 준다."), caption("04_km_demand_resident.png", "상주인구 사분위별 생존곡선을 보여 준다.")]),
            "offset": "\n\n".join([caption("04_km_offset_sales.png", "경쟁과 상권 총매출의 2×2 집단별 생존곡선을 보여 준다."), caption("04_km_offset_floating.png", "경쟁과 유동인구의 2×2 집단별 생존곡선을 보여 준다.")]),
            "trade": caption("04_km_trade_area_type.png", "상권 구분별 개인 카페 생존곡선을 보여 준다."),
            "robust": caption("04_km_robustness_relicense.png", "인허가 단위와 재인허가 통합 강건성 표본의 생존곡선을 비교한다."),
        }
        anchors = [
            ("overall", "추정되지 않은 것이다."), ("opening", "3년 차이는 불안정할 수 있다."),
            ("type", "주 분석의 인과적 비교가 아니다."), ("competition", "경쟁이 많을수록 폐업 위험이 높다는 가설과 같은 방향이다."),
            ("demand", "높은 수요가 생존을 높인다는 가설과 일관된 단변량 방향을 보이지 않았다."),
            ("offset", "상쇄의 크기나 원인을 확정하지 않는다."), ("trade", "해당 구분 자체의 효과로 해석하지 않는다."),
            ("robust", "주 분석의 인허가 단위 정의를 바꾸지 않고 민감도만 점검한다."),
        ]
        for key, anchor in anchors:
            pos = content.find(anchor)
            if pos < 0:
                raise ValueError(f"04 보고서의 그림 삽입 문단을 찾지 못했습니다: {anchor}")
            pos += len(anchor)
            content = content[:pos] + "\n\n" + km_blocks[key].rstrip() + content[pos:]
        p04.write_text(content, encoding="utf-8")
        updates.append("04_kaplan_meier_report.md: 13개 그림을 결과 문단 아래에 삽입")
    return updates


def make_readme() -> None:
    descriptions = {
        "02_sample_flow.png": ("전처리·Cox 표본 흐름", "본문"), "02_openings_closures_by_quarter.png": ("분기별 개업·폐업", "본문/발표"),
        "03_closure_vs_density_map.png": ("개인 카페 수와 단순 폐업 비율 지도", "본문/발표"),
        "03_competition_distribution.png": ("반경별 경쟁 분포", "부록(반경 강건성 설명용)"),
        "03_competition_vs_demand.png": ("경쟁과 총매출·유동인구의 기초 상관", "본문/발표"),
    }
    rows = ["# 그림 목록", "", "| 파일 | 내용 | 보고서에서의 용도 |", "| --- | --- | --- |"]
    for name in FIGURE_FILES:
        desc, use = descriptions[name]; rows.append(f"| `{name}` | {desc} | {use} |")
    for prefix, label in [("04_", "Kaplan–Meier 분석 그림"), ("05_", "Cox 분석 그림")]:
        for path in sorted(FIGURES.glob(f"{prefix}*.png")):
            rows.append(f"| `{path.name}` | {label} | 본문/발표 |")
    README.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    os.environ["TEMP"] = str(ROOT / ".tmp"); os.environ["TMP"] = str(ROOT / ".tmp")
    (ROOT / ".tmp").mkdir(exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True); LOG.parent.mkdir(parents=True, exist_ok=True)
    setup_style()
    data = read_data(); require(FLOW); flow = pd.read_csv(FLOW, encoding="utf-8-sig")
    points = projected_points(data); areas, outline = map_layers()
    makers = [
        lambda: make_sample_flow(data, flow), lambda: make_open_close(data),
        lambda: make_closure_vs_density_map(points, areas, outline), lambda: make_competition_distribution(data),
        lambda: make_competition_demand(data),
    ]
    for maker in makers: maker()
    obsolete = ["02_franchise_brands.png", "03_cafe_map.png", "03_competition_density_map.png", "03_trade_area_distance.png", "03_covariate_correlation.png"]
    for name in obsolete:
        (FIGURES / name).unlink(missing_ok=True)
    report_updates = insert_reports(); make_readme()
    missing = [name for name in FIGURE_FILES if not (FIGURES / name).exists()]
    if missing: raise RuntimeError(f"그림 생성 실패: {missing}")
    LOG.write_text(
        "# 보고서 그림 실행 로그\n\n"
        f"- 입력: `data/processed/cafes_analysis.parquet` ({len(data):,}행)\n"
        f"- Cox 표본 흐름 입력: `docs/tables/05_cox_sample_flow.csv`\n"
        f"- 지도 좌표계: 인허가 EPSG:5174 → 표시 EPSG:5181; 상권 경계 EPSG:5181\n"
        f"- 생성 그림: {len(FIGURE_FILES)}개 (200dpi PNG)\n"
        f"- 그림 참조 보강: " + ("; ".join(report_updates) if report_updates else "이미 모두 삽입됨") + "\n"
        "- 검증: 필수 열·입력 파일·생성 파일 존재를 검사했다. 05·06 단계 보고서는 수정하지 않았다.\n",
        encoding="utf-8",
    )
    print(f"Created {len(FIGURE_FILES)} figures; updated {len(report_updates)} report sections.")


if __name__ == "__main__":
    main()
