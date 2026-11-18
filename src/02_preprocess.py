"""Prepare the Seoul coffee-shop licence cohort and franchise review files.

Run from the project root with ``.venv\\Scripts\\python.exe src\\02_preprocess.py``.
The script never changes the raw source file.  It writes all derived data and
reports deterministically; franchise-review sampling uses a fixed random seed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data/raw/refreshment_establishment/식품_휴게음식점_서울특별시.csv"
BRAND_PATH = PROJECT_ROOT / "src/franchise_brands.csv"
NONCAFE_PATTERN_PATH = PROJECT_ROOT / "src/noncafe_patterns.csv"
ALL_PROCESSED_PATH = PROJECT_ROOT / "data/processed/cafes_licenses_all.parquet"
CLEAN_PROCESSED_PATH = PROJECT_ROOT / "data/processed/cafes_clean.parquet"
LEGACY_PROCESSED_PATH = PROJECT_ROOT / "data/processed/cafes_licenses.parquet"
REPORT_PATH = PROJECT_ROOT / "docs/reports/02_preprocess_report.md"
REVIEW_PATH = PROJECT_ROOT / "docs/review/02_franchise_review.csv"
UNMATCHED_PATH = PROJECT_ROOT / "docs/review/02_unmatched_top.csv"
EXCLUDED_ROWS_PATH = PROJECT_ROOT / "docs/review/02_excluded_rows.csv"
BRAND_VARIANTS_PATH = PROJECT_ROOT / "docs/review/02_brand_variants.csv"
NARROWED_BRANDS_PATH = PROJECT_ROOT / "docs/review/02_narrowed_brands.csv"
CHAIN_CANDIDATES_PATH = PROJECT_ROOT / "docs/review/02_chain_candidates.csv"
CHAIN_CLASSIFICATION_PATH = PROJECT_ROOT / "docs/review/02_chain_classification.csv"

EXPECTED_SHA256 = "44a5c4cbe64b0dd7d9cbbfa28c0532f003910644ddf22b2cbf4bf24751c16f3b"
OBSERVATION_END = pd.Timestamp("2026-09-30")
STUDY_START = pd.Timestamp("2021-04-01")
STUDY_END = pd.Timestamp("2026-03-31")
RANDOM_SEED = 20261118

# Review decisions for the chain-candidate list. The output is
# intentionally an audit record rather than an inference rule: a candidate
# omitted from these sets remains ``불확실`` instead of being silently added.
CAFE_CHAIN_CANDIDATES = frozenset({
    "우지커피", "바나프레소", "백억커피", "커피사피엔스", "달리는 커피", "카페인 중독", "만랩커피",
    "커스텀커피", "포트 캔 커피", "벤티프레소", "블루포트", "플러스 82", "날쌘카페", "커피온리",
    "백미당", "설빙", "아마스빈", "빌리엔젤", "스무디킹", "마노핀 익스프레스", "이공커피",
    "하이오커피", "브루다 커피", "커피코트", "타래퀸", "24시 무인카페 만월경", "유캔두잇",
    "카페네스카페", "(주)하우", "요커 (yo.kr)", "카페홈즈", "국민우유집", "다시스", "마노핀",
    "메머드 익스프레스", "아마스빈 버블티",
    "오가다", "읍천리382", "트리플에이커피", "파란만잔", "홍콩다방", "디 초콜릿커피",
    "마리웨일 마카롱", "모노치즈", "아이갓 에브리씽", "카페 요아정", "카페일리터", "카페희다",
    "달다곰이", "발도스커피", "슈퍼커피", "영커피", "와드커피", "주커피", "카멜커피",
    "커피인류", "커피쿡", "프롬하츠커피", "고망고", "더커피", "로스터리 락온", "사과당",
    "쉬즈베이글", "에센스 커피", "차백도", "(주)학산 테라로사", "(주)학산 테라로사커피",
    "고품격 커피공장", "다시, 온 카페", "더착한커피", "데니스커피", "델트렌", "모리커피",
    "벌크커피", "베러 댄 와플", "봉명동내커피", "빙동댕", "스위트밀", "아메리칸 트레일러",
    "잠바주스", "쥬스식스", "치키차카초코", "카페 만월경", "카페 일마지오", "카페미뇽",
    "커피기업", "커피식스 쥬스식스", "킷사텐", "텍사스파파 크레페", "템포커피", "파머스빈",
    "페이브커피", "퐁치 커피익스프레스", "고디바", "곤트란 쉐리에", "나이스카페인클럽",
    "던킨도너츠", "떼루와", "라떼야 커피", "마이쥬스", "모리셔스브라운", "베브릿지", "시나본",
    "아이캔커피", "에슬로우", "오와케이크", "와이블랙", "제주스", "주식회사 빅바이트 컴퍼니 잠바주스",
    "차지", "카페 글렌", "카페 차이", "카페오아시아", "크로플덕", "타르타르", "테라로사",
    "테라커피", "팀 홀튼", "파운드커피", "퍼스트 커피랩", "푸르파파", "피에스타7 커피",
    "행복한 베이커리&카페", "히포커피", "(주)밀탑", "그라찌에", "글로리아진스 커피", "까사밀",
    "꽈페", "더 베이크", "더설", "던킨", "떠기다방", "띠아모", "라스베이글", "롤링핀", "리사르커피",
    "마실다실", "망원동티라미수", "메가후르츠", "몬도커피츄러스", "밀크힐", "반달커피", "브알라",
    "브이카페", "빙달", "스탠브루", "스트릿1988", "스트릿츄러스", "에이바우트커피", "오가페",
    "오슬로우", "오우야 에스프레소 바", "오페뜨", "와요커피", "우주라이크커피", "위치커피", "주스탐",
    "쥬시", "진심커피", "착한커피", "청자다방", "카페 두다트", "카페 띠아모", "카페 잇", "카페리빈",
    "카페보스", "카페스윗", "커피나인", "커피디지", "커피브라더", "커피식구", "커피식스", "컵오브커피",
    "크레이저커피", "크로플덕 오리아가씨", "타래커피", "텀브커피", "테이큰커피", "트러스트 커피",
    "펠어커피", "펠어커피초코", "헤이키커피", "희망카페", "히코코", "79파운야드", "가배차", "나리꼬모",
    "나우커피", "네스카페", "라떼킹", "랑데자뷰", "레인커피", "로티보이", "롯데웰푸드(주)파스퇴르밀크바",
    "마치마치", "만월경", "메타킹", "밀탑", "버터앤쉘터", "보난자커피", "블랙라인커피", "빨라쪼",
    "생활커피", "슈퍼보바", "스택빈", "아방베이커리", "아토커피", "알리바바커피앤도넛", "에그마카슈",
    "에이미스", "엑스엑스커피메이커", "엔젤리너스", "옐로우캔", "올더타임", "옵션스페셜티커피", "워너커피",
    "원유로 스페셜티", "위메프카페", "일리카페", "조앤더주스", "쥬스팩토리", "지유명차", "차차커피코",
    "천씨씨커피", "청춘커피", "카페 디아떼", "카페,진정성", "카페그랑", "카페뉴욕", "카페보니또",
    "카페안젤로", "카페차이 롯데백화점", "칸틴", "캔카페", "커스텀 커피 (KUSTOM COFFEE)", "커피홀",
    "코피발리", "콘레드커피", "콜롬비아 퍼펙토", "크래킹커피", "크로플각", "토다 베이글", "토즈커피하우스",
    "티라레미수", "패기파이", "페이브 베이커리", "푸가", "홈스테드 커피", "휘드카페",
})
NONCAFE_CHAIN_CANDIDATES = frozenset({
    "세븐일레븐", "씨유", "GS25", "이마트24", "지에스25", "지에스(GS)25", "미니스톱", "훼미리마트",
    "(주)비지에프리테일", "위드미", "놀숲", "짐박스피트니스", "샐러디", "커피랑 도서관", "랭스터디카페",
    "위펀", "써브웨이", "이삭토스트", "죠 샌드위치", "에그드랍", "아이키친", "뚜레쥬르", "홍루이젠",
    "자연드림", "히어로보드게임카페", "탭플레이", "(주)아워홈", "(주)케이씨 코퍼레이션", "세라젬 웰카페",
    "카페드림", "레스티오", "쿱카페 경희대", "함께그린카페",
})
NOT_CHAIN_CANDIDATES = frozenset({"(주)마체베트", "카페 아이엔지", "카페 아이엠티", "카페 서울숲"})

SEOUL_BOUNDS = {"lon_min": 126.76, "lon_max": 127.19, "lat_min": 37.42, "lat_max": 37.71}

# Counts in the report produced before this follow-up.  The source data and
# pre-existing validity rules are unchanged, so these are the comparable
# all-coffee-shop matching counts before the dictionary was expanded.
PREVIOUS_BRAND_COUNTS = {
    "스타벅스": 751, "이디야": 716, "매머드커피": 441, "컴포즈커피": 431,
    "투썸플레이스": 395, "빽다방": 303, "커피빈": 264, "공차": 259,
    "할리스": 254, "커피베이": 225, "더벤티": 182, "엔제리너스": 180,
    "탐앤탐스": 175, "요거프레소": 166, "카페베네": 152, "파스쿠찌": 132,
    "텐퍼센트커피": 109, "커피에반하다": 108, "커피나무": 105, "더카페": 100,
    "달콤": 95, "커피마마": 80, "셀렉토커피": 74, "폴바셋": 65,
    "아티제": 42, "팔공티": 42, "카페게이트": 38, "메가MGC커피": 34,
    "감성커피": 29, "드롭탑": 25, "밀크티": 24, "블루샥": 19,
    "하삼동커피": 19, "더리터": 18, "오설록": 15, "카페봄봄": 10,
    "커피스미스": 7, "카페051": 5, "카페마마스": 4, "커피명가": 2,
}

# The approved initial-dictionary result is the baseline for this follow-up.
PREVIOUS_COHORT_COUNTS = {
    "인허가(운영자) 수": 11_593,
    "개인 카페": 9_219,
    "프랜차이즈": 2_374,
    "사건": 4_271,
}

# Chain-addition output, immediately before the narrowly scoped dictionary patch.
PREVIOUS_CHAIN_ADDITION_COHORT_COUNTS = {
    "인허가(운영자) 수": 11_515,
    "개인 카페": 8_054,
    "프랜차이즈": 3_461,
    "사건": 4_253,
    "중도절단": 7_262,
}

# These are the broad rules before the approved narrowing. They are kept
# here only to create an auditable retained/released-name review file.
PREVIOUS_NARROW_PATTERNS = {
    "달콤": re.compile(r"달콤|dalkomm"),
    "더카페": re.compile(r"더카페|thecafe"),
    "커피나무": re.compile(r"커피나무|coffeetree"),
    "더커피": re.compile(r"더커피|thecoffee"),
}

# These deliberately broad normalized strings are for inspection only.  They
# never classify an outlet as a franchise without a dictionary-pattern match.
BROAD_SEARCH_TERMS = {
    "스타벅스": ("스타벅", "starbuck"), "이디야": ("이디", "ediya"),
    "메가MGC커피": ("메가", "mgc", "엠지씨"), "컴포즈커피": ("컴포즈", "compose"),
    "빽다방": ("빽", "paik"), "투썸플레이스": ("투썸", "twosome"),
    "할리스": ("할리스", "hollys"), "커피빈": ("커피빈", "coffeebean"),
    "파스쿠찌": ("파스쿠", "pascucci"), "요거프레소": ("요거", "yoger"),
    "탐앤탐스": ("탐앤", "tom"), "엔제리너스": ("엔제리", "angel"),
    "폴바셋": ("폴바", "paulbas"), "더벤티": ("더벤", "thevent"),
    "매머드커피": ("매머드", "mammoth"), "커피베이": ("커피베", "coffeebay"),
    "카페베네": ("베네", "caffebene"), "드롭탑": ("드롭", "droptop"),
    "공차": ("공차", "gongcha"), "달콤": ("달콤", "dalkomm"),
    "커피마마": ("커피마마", "coffeemama"), "텐퍼센트커피": ("텐퍼", "10percent"),
    "더리터": ("더리터", "theliter"), "감성커피": ("감성", "gamsung"),
    "카페봄봄": ("봄봄", "bombom"), "하삼동커피": ("하삼", "hasamdong"),
    "블루샥": ("블루샥", "blueshark"), "카페게이트": ("게이트", "cafegate"),
    "커피에반하다": ("반하다", "vanada"), "카페051": ("051", "cafe051"),
    "셀렉토커피": ("셀렉", "selecto"), "아티제": ("아티", "artisee"),
    "블루보틀": ("블루보", "bluebottle"), "커피스미스": ("스미스", "coffeesmith"),
    "커피나무": ("커피나무", "coffeetree"), "커피명가": ("커피명가", "coffeemyungga"),
    "더카페": ("더카페", "thecafe"), "카페마마스": ("마마스", "cafemamas"),
    "오설록": ("오설", "osulloc"), "팔공티": ("팔공", "palgong"),
    "밀크티": ("타이거", "흑화", "heychatea", "tigersugar"),
    "토프레소": ("토프", "topresso"), "더치앤빈": ("더치", "dutch"),
    "빈스빈스": ("빈스", "beansbeans"), "커피니": ("커피니", "coffeenie"),
    "카페7그램": ("7그램", "7gram", "7g"), "쥬씨": ("쥬씨", "juicy"),
    "디저트39": ("디저트", "dessert"),
    "해머스미스커피": ("해머스미스", "hammersmith"),
    "과일에반하다 프루타": ("과일에반하다", "프루타", "푸르타", "fruta"),
    "바나타이거": ("바나타이거", "banatiger"),
    "빈스앤베리즈": ("빈스앤베리즈", "beansberries"),
    "바빈스커피": ("바빈스", "bavins"),
    "벨라빈스": ("벨라빈스", "밸라빈스", "bellabeans"),
    "블루빈스커피": ("블루빈스", "bluebeans"),
    "빈스바네스": ("빈스바네스", "beansbarnes"),
    "배스킨라빈스": ("배스킨", "baskin"),
    "요거트월드": ("요거트월드", "yogurtworld"),
    "요거트퍼플": ("요거트퍼플", "yogurtpurple"),
    "요거트아이스크림의 정석": ("요거트아이스크림", "yogurticecream"),
    "요거트맨": ("요거트맨", "yogurtman"),
    "요고 프로즌 요거트": ("요고", "yogofrozen"),
    "아오스요거트": ("아오스", "aosyogurt"),
    "요미스라이스×요거트": ("요미스", "yomisrice"),
}


@dataclass(frozen=True)
class CoordinateResult:
    epsg: int
    longitude: pd.Series
    latitude: pd.Series
    in_seoul_box: int
    median_longitude: float
    median_latitude: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_name(value: object) -> str:
    """Normalize spacing, case, punctuation, and Unicode for brand matching."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    return re.sub(r"[\s\-_'`.,·()/\\&]+", "", text)


def study_period_mask(permit_date: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Return the inclusive licence-date study window; dates remain configurable."""
    return permit_date.between(start, end, inclusive="both")


def transform_coordinates(x: pd.Series, y: pd.Series, epsg: int) -> CoordinateResult:
    valid = x.notna() & y.notna()
    longitude = pd.Series(np.nan, index=x.index, dtype="float64")
    latitude = pd.Series(np.nan, index=x.index, dtype="float64")
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon_values, lat_values = transformer.transform(x.loc[valid].to_numpy(), y.loc[valid].to_numpy())
    longitude.loc[valid] = lon_values
    latitude.loc[valid] = lat_values
    in_box = (
        longitude.between(SEOUL_BOUNDS["lon_min"], SEOUL_BOUNDS["lon_max"])
        & latitude.between(SEOUL_BOUNDS["lat_min"], SEOUL_BOUNDS["lat_max"])
    )
    return CoordinateResult(
        epsg=epsg,
        longitude=longitude,
        latitude=latitude,
        in_seoul_box=int(in_box.sum()),
        median_longitude=float(longitude.median()),
        median_latitude=float(latitude.median()),
    )


def choose_coordinate_system(
    results: dict[int, CoordinateResult], reference_summary: pd.DataFrame
) -> int:
    """Choose the candidate supported by matched known-address reference points.

    A Seoul bounding-box check detects gross errors but cannot distinguish these
    close Korean projected CRSs.  The median error for known building addresses
    is used first, falling back to the bounding-box count only if no references
    occur in the source data.
    """
    reference_error = reference_summary.dropna(subset=["기준점 중앙거리(m)"]).groupby("EPSG")["기준점 중앙거리(m)"].mean()
    if not reference_error.empty:
        return int(reference_error.idxmin())
    best_count = max(result.in_seoul_box for result in results.values())
    tied = [epsg for epsg, result in results.items() if result.in_seoul_box == best_count]
    return min(tied)


def load_brand_dictionary(path: Path) -> pd.DataFrame:
    brands = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
    required = {"브랜드명", "매칭 패턴(정규식)", "비고", "source"}
    if set(brands.columns) != required:
        raise ValueError(f"Brand dictionary columns must be {sorted(required)}; found {brands.columns.tolist()}")
    allowed_sources = {"initial", "chain_candidates", "chain_candidates_pattern_patch", "pattern_patch"}
    if not set(brands["source"]).issubset(allowed_sources):
        raise ValueError("Brand dictionary has an invalid source value")
    brands["compiled_pattern"] = brands["매칭 패턴(정규식)"].map(re.compile)
    return brands


def load_noncafe_patterns(path: Path) -> pd.DataFrame:
    """Load approved non-cafe exclusion patterns on normalized business names."""
    patterns = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
    required = {"업종명", "매칭 패턴(정규식)", "비고"}
    if set(patterns.columns) != required:
        raise ValueError(f"Non-cafe pattern columns must be {sorted(required)}; found {patterns.columns.tolist()}")
    patterns["compiled_pattern"] = patterns["매칭 패턴(정규식)"].map(re.compile)
    return patterns


def classify_noncafes(names: pd.Series, patterns: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Flag only approved, clearly non-cafe chains or institutional operations."""
    normalized = names.map(normalize_name)
    excluded = pd.Series(False, index=names.index, dtype=bool)
    reason = pd.Series(pd.NA, index=names.index, dtype="string")
    for record in patterns.itertuples(index=False):
        matches = normalized.str.contains(record.compiled_pattern, na=False)
        newly_matched = matches & ~excluded
        excluded.loc[newly_matched] = True
        reason.loc[newly_matched] = record.업종명
    return excluded, reason


def make_chain_classification(candidates: pd.DataFrame) -> pd.DataFrame:
    """Create the review record for every chain candidate without auto-guessing."""
    columns = ["핵심 상호", "지점 수", "분류", "근거", "반영 여부"]
    rows: list[dict[str, object]] = []
    for name, branches, *_ in candidates.itertuples(index=False, name=None):
        branches = int(branches)
        if name in CAFE_CHAIN_CANDIDATES:
            classification = "카페 체인"
            evidence = "지점형 상호가 반복되며 커피·음료·디저트 판매 체인으로 판단"
            applied = "프랜차이즈 사전 반영" if branches >= 5 else "분류만(지점 수 5개 미만)"
        elif name in NONCAFE_CHAIN_CANDIDATES:
            classification = "카페가 아닌 업종"
            evidence = "편의점·학습/오락·운동·식사 중심 또는 기업·기관 운영으로 일반 카페 영업과 구분"
            applied = "비카페 제외 패턴 반영"
        elif name in NOT_CHAIN_CANDIDATES:
            classification = "체인 아님"
            evidence = "운영 법인·기관의 복수 사업장 표기로 보이며 독립 카페 체인으로 보기 어려움"
            applied = "미반영"
        else:
            classification = "불확실"
            evidence = "상호와 지점명 예시만으로 커피·음료·디저트 체인 여부를 확정할 수 없음"
            applied = "미반영"
        rows.append({"핵심 상호": name, "지점 수": branches, "분류": classification, "근거": evidence, "반영 여부": applied})
    return pd.DataFrame(rows, columns=columns).sort_values(["지점 수", "핵심 상호"], ascending=[False, True], kind="stable")


def brand_match_mask(names: pd.Series, normalized: pd.Series, record: object) -> pd.Series:
    """Apply a dictionary pattern, preserving word boundaries for ``달콤``.

    Most dictionary rules deliberately operate on punctuation/space-stripped
    names.  For 달콤, stripping these separators turns unrelated names such as
    ``알콩달콤 커피`` into ``알콩달콤커피``.  Its approved rule therefore uses
    NFKC-normalized display text: standalone 달콤커피 (not an embedded word),
    or a leading 달콤 followed by a separated branch label ending in 점.
    """
    if record.브랜드명 != "달콤":
        return normalized.str.contains(record.compiled_pattern, na=False)
    display = names.fillna("").astype(str).map(lambda value: unicodedata.normalize("NFKC", value).lower().strip())
    dalkomm_coffee = display.str.contains(r"(?:^|[\s(])달콤\s*커피", regex=True, na=False)
    dalkomm_branch = display.str.contains(r"^달콤(?:\s+|[.(].*)[^\s]*점$", regex=True, na=False)
    dalkomm_n_branch = display.str.contains(r"^달콤[.\s]*엔.*점$", regex=True, na=False)
    return dalkomm_coffee | dalkomm_branch | dalkomm_n_branch


def classify_franchises(names: pd.Series, brands: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    normalized = names.map(normalize_name)
    brand = pd.Series(pd.NA, index=names.index, dtype="string")
    for record in brands.itertuples(index=False):
        matches = brand_match_mask(names, normalized, record)
        brand.loc[matches & brand.isna()] = record.브랜드명
    return brand.notna().astype(bool), brand


def make_brand_variant_review(names: pd.Series, brands: pd.DataFrame) -> pd.DataFrame:
    """List broad-search candidates which do not match their brand pattern.

    No candidate in this review is automatically added to the brand dictionary.
    This keeps a deliberately broad diagnostic search from changing the study
    population through a false-positive match.
    """
    normalized = names.map(normalize_name)
    display_names = names.fillna("").astype(str).str.strip()
    generic_terms = {"감성", "더치", "dutch", "드롭", "디저트", "dessert"}
    rows: list[dict[str, object]] = []
    for record in brands.itertuples(index=False):
        pattern_matches = brand_match_mask(names, normalized, record)
        for term in BROAD_SEARCH_TERMS.get(record.브랜드명, ()):
            query = normalize_name(term)
            candidates = normalized.str.contains(re.escape(query), regex=True, na=False) & ~pattern_matches
            counts = display_names.loc[candidates & display_names.ne("")].value_counts()
            for business_name, count in counts.items():
                if query in generic_terms:
                    judgement = "개인 카페"
                    basis = f"검색어 '{term}'은(는) 일반 어휘이며 브랜드 고유 표기가 없어 개인 카페로 판단"
                else:
                    judgement = "불확실"
                    basis = "넓은 검색어에는 일치하지만 현재 브랜드 정규식에는 일치하지 않아 상호만으로 가맹 여부를 확정할 수 없음"
                rows.append(
                    {
                        "브랜드명": record.브랜드명,
                        "검색어": term,
                        "사업장명": business_name,
                        "건수": int(count),
                        "Codex 판단": judgement,
                        "근거": basis,
                    }
                )
    columns = ["브랜드명", "검색어", "사업장명", "건수", "Codex 판단", "근거"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["브랜드명", "건수", "검색어", "사업장명"], ascending=[True, False, True, True], kind="stable"
    )


def make_narrowed_brand_review(names: pd.Series, brands: pd.DataFrame) -> pd.DataFrame:
    """Compare the approved narrow rules with their previous broad patterns."""
    normalized = names.map(normalize_name)
    display_names = names.fillna("").astype(str).str.strip()
    rows: list[dict[str, object]] = []
    for brand_name, previous_pattern in PREVIOUS_NARROW_PATTERNS.items():
        current_record = next(record for record in brands.itertuples(index=False) if record.브랜드명 == brand_name)
        previous_matches = normalized.str.contains(previous_pattern, na=False)
        current_matches = brand_match_mask(names, normalized, current_record)
        compared = previous_matches | current_matches
        counts = display_names.loc[compared & display_names.ne("")].value_counts()
        for business_name, count in counts.items():
            name_mask = display_names.eq(business_name)
            retained = bool((current_matches & name_mask).any())
            rows.append(
                {
                    "브랜드명": brand_name,
                    "결과": "매칭 유지" if retained else "매칭 해제",
                    "사업장명": business_name,
                    "건수": int(count),
                }
            )
    columns = ["브랜드명", "결과", "사업장명", "건수"]
    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows, columns=columns)
    result["결과 정렬"] = result["결과"].map({"매칭 유지": 0, "매칭 해제": 1})
    return result.sort_values(
        ["브랜드명", "결과 정렬", "건수", "사업장명"],
        ascending=[True, True, False, True],
        kind="stable",
    ).drop(columns="결과 정렬")


def make_chain_candidates(names: pd.Series, top_names: pd.Series) -> pd.DataFrame:
    """Find unmatched names sharing a spaced prefix and at least three branches.

    A candidate requires a final whitespace-separated ``…점`` token.  This
    intentionally conservative rule avoids inferring a chain from arbitrary
    Korean substrings; candidates remain a review list, never a classification
    input.  The top-100 unmatched list is unioned with all unmatched names to
    make both requested inspection sources explicit.
    """
    observed_names = pd.concat([names, top_names], ignore_index=True).fillna("").astype(str).str.strip()
    observed_names = observed_names.loc[observed_names.ne("")].drop_duplicates()
    rows: list[dict[str, str]] = []
    for business_name in observed_names:
        match = re.match(r"^(?P<core>.+?)\s+(?P<branch>\S+점)$", business_name)
        if match is None:
            continue
        core = match.group("core").strip()
        branch = match.group("branch").strip()
        if not core or not branch:
            continue
        rows.append(
            {
                "핵심 상호 키": normalize_name(core),
                "핵심 상호": core,
                "지점명 키": normalize_name(branch),
                "사업장명": business_name,
            }
        )
    columns = ["핵심 상호", "지점 수", "예시 상호 1", "예시 상호 2", "예시 상호 3"]
    if not rows:
        return pd.DataFrame(columns=columns)
    branches = pd.DataFrame(rows)
    candidates: list[dict[str, object]] = []
    for _, group in branches.groupby("핵심 상호 키", sort=True):
        distinct_branches = group.drop_duplicates("지점명 키")
        branch_count = len(distinct_branches)
        if branch_count < 3:
            continue
        core = group.sort_values("핵심 상호", kind="stable").iloc[0]["핵심 상호"]
        examples = distinct_branches.sort_values("사업장명", kind="stable")["사업장명"].head(3).tolist()
        candidates.append(
            {
                "핵심 상호": core,
                "지점 수": branch_count,
                "예시 상호 1": examples[0],
                "예시 상호 2": examples[1],
                "예시 상호 3": examples[2],
            }
        )
    if not candidates:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(candidates, columns=columns).sort_values(
        ["지점 수", "핵심 상호"], ascending=[False, True], kind="stable"
    )


def make_bene_ambiguous_review(names: pd.Series) -> pd.DataFrame:
    """List unclassified ``베네 …점``-style names without assigning a brand."""
    normalized = names.map(normalize_name)
    display_names = names.fillna("").astype(str).str.strip()
    candidate = (
        normalized.str.contains(r"(?:^|주)베네(?!치아)[^점]+점$", regex=True, na=False)
        & ~normalized.str.contains(r"(?:카페|까페)베네|caffebene", regex=True, na=False)
    )
    return (
        display_names.loc[candidate & display_names.ne("")]
        .value_counts()
        .rename_axis("사업장명")
        .reset_index(name="건수")
        .sort_values(["건수", "사업장명"], ascending=[False, True], kind="stable")
    )


def format_int(value: int | float) -> str:
    return f"{int(value):,}"


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(해당 없음)"
    columns = [str(column) for column in frame.columns]

    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    rows.extend("| " + " | ".join(cell(value) for value in record) + " |" for record in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def reference_address_diagnostics(frame: pd.DataFrame, coordinate_results: dict[int, CoordinateResult]) -> pd.DataFrame:
    """Compare matched cafe addresses to three known Seoul reference points.

    This is an address-based sanity check, not a substitute for an official CRS
    field.  Multiple business units can share an address, so median error is
    reported rather than assuming every licence sits at a building centroid.
    """
    reference_points = (
        ("서울시청", "세종대로 110", 126.9779692, 37.566535),
        ("서울역", "한강대로 405", 126.97075, 37.55468),
    )
    address = frame["도로명주소"].fillna("").astype(str).str.strip()
    rows: list[dict[str, object]] = []
    for point_name, address_text, point_lon, point_lat in reference_points:
        target = address.str.contains(address_text, regex=False)
        for epsg, result in coordinate_results.items():
            lon = result.longitude.loc[target]
            lat = result.latitude.loc[target]
            valid = lon.notna() & lat.notna()
            # Equirectangular approximation is adequate for a building-level check.
            dx = (lon.loc[valid] - point_lon) * 88_200
            dy = (lat.loc[valid] - point_lat) * 111_000
            distance = np.sqrt(dx**2 + dy**2)
            rows.append(
                {
                    "기준점": point_name,
                    "주소 일치 문자열": address_text,
                    "EPSG": epsg,
                    "해당 카페 행": int(target.sum()),
                    "좌표 보유": int(valid.sum()),
                    "기준점 중앙거리(m)": round(float(np.median(distance)), 1) if len(distance) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write Parquet using the installed pandas engine, failing transparently otherwise."""
    try:
        frame.to_parquet(path, index=False)
    except ImportError as error:
        raise RuntimeError(
            "Parquet writer is unavailable in the prepared environment. "
            "Install neither packages nor write a substitute: add pyarrow or fastparquet to the managed environment."
        ) from error


def main() -> None:
    for directory in (ALL_PROCESSED_PATH.parent, REPORT_PATH.parent, REVIEW_PATH.parent):
        directory.mkdir(parents=True, exist_ok=True)

    observed_hash = sha256(RAW_PATH)
    if observed_hash != EXPECTED_SHA256:
        raise RuntimeError(f"Raw-file SHA256 mismatch: expected {EXPECTED_SHA256}, got {observed_hash}")

    raw = pd.read_csv(RAW_PATH, encoding="cp949", dtype=str, keep_default_na=True)
    raw_rows = len(raw)
    cafe = raw.loc[raw["업태구분명"].str.strip().eq("커피숍")].copy()
    coffee_rows = len(cafe)

    cafe["permit_date"] = pd.to_datetime(cafe["인허가일자"].str.strip(), format="%Y-%m-%d", errors="coerce")
    cafe["closure_date"] = pd.to_datetime(cafe["폐업일자"].str.strip(), format="%Y-%m-%d", errors="coerce")
    cafe["operation_status"] = cafe["영업상태명"].fillna("").str.strip()
    closure_status = cafe["operation_status"].eq("폐업")
    closure_status_missing_date = closure_status & cafe["closure_date"].isna()
    permit_parse_failure = cafe["permit_date"].isna()
    closure_date_parse_failure = cafe["폐업일자"].notna() & cafe["closure_date"].isna()

    # A licence cannot enter a cohort ending before it opened.
    permit_after_observation = cafe["permit_date"].gt(OBSERVATION_END)
    base_exclusions = closure_status_missing_date | permit_after_observation | permit_parse_failure
    all_cafes = cafe.loc[~base_exclusions].copy()

    all_cafes["event"] = all_cafes["closure_date"].le(OBSERVATION_END).astype("int8")
    all_cafes["analysis_end_date"] = all_cafes["closure_date"].where(all_cafes["event"].eq(1), OBSERVATION_END)
    all_cafes["survival_days"] = (all_cafes["analysis_end_date"] - all_cafes["permit_date"]).dt.days.astype("Int64")
    all_cafes["in_study_period"] = study_period_mask(all_cafes["permit_date"], STUDY_START, STUDY_END)
    all_cafes["excluded_temporary_permit"] = all_cafes["사업장명"].fillna("").str.contains("한시적", regex=False)
    all_cafes["excluded_nonpositive_survival"] = all_cafes["survival_days"].le(0).fillna(False)

    all_cafes["coordinate_x"] = pd.to_numeric(all_cafes["좌표정보(X)"].astype("string").str.strip(), errors="coerce")
    all_cafes["coordinate_y"] = pd.to_numeric(all_cafes["좌표정보(Y)"].astype("string").str.strip(), errors="coerce")
    coordinate_results = {
        epsg: transform_coordinates(all_cafes["coordinate_x"], all_cafes["coordinate_y"], epsg)
        for epsg in (5174, 2097, 5181)
    }
    reference_summary = reference_address_diagnostics(all_cafes, coordinate_results)
    selected_epsg = choose_coordinate_system(coordinate_results, reference_summary)
    all_cafes["coordinate_crs_epsg"] = selected_epsg
    all_cafes["longitude"] = coordinate_results[selected_epsg].longitude
    all_cafes["latitude"] = coordinate_results[selected_epsg].latitude
    all_cafes["has_coordinates"] = all_cafes["coordinate_x"].notna() & all_cafes["coordinate_y"].notna()

    brands = load_brand_dictionary(BRAND_PATH)
    noncafe_patterns = load_noncafe_patterns(NONCAFE_PATTERN_PATH)

    # Recreate the review population with the dictionary before chain addition,
    # before adding chains or excluding non-cafes.  This makes the 294-row
    # classification record reproducible even after the live candidate list is
    # refreshed below.
    brands_before_chain_addition = brands.loc[
        ~brands["source"].isin({"chain_candidates", "chain_candidates_pattern_patch"})
    ].copy()
    mammoth_index = brands_before_chain_addition["브랜드명"].eq("매머드커피")
    brands_before_chain_addition.loc[mammoth_index, "compiled_pattern"] = re.compile(
        r"매머드(?:커피|익스프레스)?|mammoth(?:coffee|express)?"
    )
    franchise_before_chain_addition, _ = classify_franchises(all_cafes["사업장명"], brands_before_chain_addition)
    clean_before_chain_addition = all_cafes.loc[
        ~(all_cafes["excluded_temporary_permit"] | all_cafes["excluded_nonpositive_survival"])
    ]
    unmatched_before_chain_addition = clean_before_chain_addition.loc[
        ~franchise_before_chain_addition.loc[clean_before_chain_addition.index], "사업장명"
    ]
    unmatched_top_before_chain_addition = (
        unmatched_before_chain_addition.fillna("").astype(str).str.strip().replace("", pd.NA).dropna()
        .value_counts().rename_axis("사업장명").reset_index(name="건수").head(100)
    )
    baseline_chain_candidates = make_chain_candidates(
        unmatched_before_chain_addition, unmatched_top_before_chain_addition["사업장명"]
    )

    all_cafes["is_franchise"], all_cafes["brand"] = classify_franchises(all_cafes["사업장명"], brands)
    all_cafes["excluded_noncafe"], all_cafes["noncafe_category"] = classify_noncafes(
        all_cafes["사업장명"], noncafe_patterns
    )

    new_exclusions = (
        all_cafes["excluded_temporary_permit"]
        | all_cafes["excluded_nonpositive_survival"]
        | all_cafes["excluded_noncafe"]
    )
    excluded_rows = all_cafes.loc[
        new_exclusions,
        ["관리번호", "사업장명", "인허가일자", "폐업일자", "excluded_temporary_permit", "excluded_nonpositive_survival", "excluded_noncafe"],
    ].copy()
    exclusion_labels = pd.DataFrame({
        "한시적 인허가": excluded_rows["excluded_temporary_permit"],
        "생존기간 0일 이하": excluded_rows["excluded_nonpositive_survival"],
        "카페가 아닌 업종": excluded_rows["excluded_noncafe"],
    })
    excluded_rows["제외 사유"] = exclusion_labels.apply(
        lambda row: "; ".join(label for label, matched in row.items() if matched), axis=1
    )
    excluded_rows = excluded_rows[["관리번호", "사업장명", "인허가일자", "폐업일자", "제외 사유"]].sort_values(
        ["제외 사유", "관리번호"], kind="stable"
    )
    excluded_rows.to_csv(EXCLUDED_ROWS_PATH, index=False, encoding="utf-8-sig")
    clean = all_cafes.loc[~new_exclusions].copy()

    # Review samples are deterministic but selected independently within each brand.
    franchise_rows = clean.loc[
        clean["is_franchise"], ["brand", "사업장명", "관리번호", "인허가일자", "도로명주소"]
    ]
    review_parts = [
        group.sample(n=min(len(group), 10), random_state=RANDOM_SEED)
        for _, group in franchise_rows.groupby("brand", sort=True)
    ]
    review = pd.concat(review_parts, ignore_index=True).sort_values(
        ["brand", "사업장명", "관리번호"], kind="stable"
    )
    review.to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")

    unmatched_top = (
        clean.loc[~clean["is_franchise"], "사업장명"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .rename_axis("사업장명")
        .reset_index(name="건수")
        .head(100)
    )
    unmatched_top.to_csv(UNMATCHED_PATH, index=False, encoding="utf-8-sig")
    brand_variants = make_brand_variant_review(clean["사업장명"], brands)
    brand_variants.to_csv(BRAND_VARIANTS_PATH, index=False, encoding="utf-8-sig")
    narrowed_brand_review = make_narrowed_brand_review(clean["사업장명"], brands)
    narrowed_brand_review.to_csv(NARROWED_BRANDS_PATH, index=False, encoding="utf-8-sig")
    unmatched_names = clean.loc[~clean["is_franchise"], "사업장명"]
    chain_candidates = make_chain_candidates(unmatched_names, unmatched_top["사업장명"])
    chain_candidates.to_csv(CHAIN_CANDIDATES_PATH, index=False, encoding="utf-8-sig")
    chain_classification = make_chain_classification(baseline_chain_candidates)
    chain_classification.to_csv(CHAIN_CLASSIFICATION_PATH, index=False, encoding="utf-8-sig")
    bene_ambiguous = make_bene_ambiguous_review(clean["사업장명"])

    parquet_engine_available = any(
        importlib.util.find_spec(package) is not None for package in ("pyarrow", "fastparquet")
    )
    parquet_engine_status = "사용 가능" if parquet_engine_available else "사용 불가"
    parquet_limitation = (
        ""
        if parquet_engine_available
        else "\n5. `pyarrow`와 `fastparquet`가 모두 설치되어 있지 않아 Parquet 파일을 만들 수 없었다. "
        "지시에 따라 패키지를 설치하거나 CSV 대체본을 만들지 않았다. 관리된 환경에 두 패키지 중 하나를 추가한 뒤 같은 명령을 다시 실행해야 한다."
    )

    cohort = clean.loc[clean["in_study_period"]].copy()
    cohort["permit_year"] = cohort["permit_date"].dt.year
    annual = (
        cohort.groupby(["permit_year", "is_franchise"], dropna=False)
        .agg(개업_수=("관리번호", "size"), 폐업_수=("event", "sum"))
        .reset_index()
    )
    annual["구분"] = np.where(annual["is_franchise"], "프랜차이즈", "개인 카페")
    annual = annual.rename(columns={"permit_year": "개업 연도"})[["개업 연도", "구분", "개업_수", "폐업_수"]]
    annual = annual.rename(columns={"개업_수": "개업 수", "폐업_수": "폐업 수"})

    brand_counts = (
        clean.loc[clean["is_franchise"], "brand"]
        .value_counts()
        .rename_axis("브랜드")
        .reset_index(name="매칭 수")
        .sort_values(["매칭 수", "브랜드"], ascending=[False, True], kind="stable")
    )
    coordinate_summary = pd.DataFrame(
        [
            {
                "EPSG": result.epsg,
                "서울 근사 경계 내": result.in_seoul_box,
                "중앙 경도": round(result.median_longitude, 6),
                "중앙 위도": round(result.median_latitude, 6),
            }
            for result in coordinate_results.values()
        ]
    )

    brand_counts_all = (
        all_cafes.loc[all_cafes["is_franchise"], "brand"]
        .value_counts()
        .rename_axis("브랜드")
        .reset_index(name="현재 매칭 수")
    )
    brand_change = pd.DataFrame({"브랜드": sorted(set(PREVIOUS_BRAND_COUNTS) | set(brand_counts_all["브랜드"]))})
    brand_change["이전 매칭 수"] = brand_change["브랜드"].map(PREVIOUS_BRAND_COUNTS).fillna(0).astype(int)
    brand_change = brand_change.merge(brand_counts_all, on="브랜드", how="left")
    brand_change["현재 매칭 수"] = brand_change["현재 매칭 수"].fillna(0).astype(int)
    brand_change["증감"] = brand_change["현재 매칭 수"] - brand_change["이전 매칭 수"]
    brand_change = brand_change.sort_values(["증감", "현재 매칭 수", "브랜드"], ascending=[False, False, True], kind="stable")
    all_cohort_before_new_exclusions = all_cafes.loc[all_cafes["in_study_period"]]
    cohort_comparison = pd.DataFrame(
        [
            {"지표": "인허가(운영자) 수", "체인 추가 후": PREVIOUS_CHAIN_ADDITION_COHORT_COUNTS["인허가(운영자) 수"], "현재 결과": len(cohort)},
            {"지표": "개인 카페", "체인 추가 후": PREVIOUS_CHAIN_ADDITION_COHORT_COUNTS["개인 카페"], "현재 결과": int((~cohort["is_franchise"]).sum())},
            {"지표": "프랜차이즈", "체인 추가 후": PREVIOUS_CHAIN_ADDITION_COHORT_COUNTS["프랜차이즈"], "현재 결과": int(cohort["is_franchise"].sum())},
            {"지표": "사건", "체인 추가 후": PREVIOUS_CHAIN_ADDITION_COHORT_COUNTS["사건"], "현재 결과": int(cohort["event"].sum())},
            {"지표": "중도절단", "체인 추가 후": PREVIOUS_CHAIN_ADDITION_COHORT_COUNTS["중도절단"], "현재 결과": int((1 - cohort["event"]).sum())},
        ]
    )
    cohort_comparison["증감"] = cohort_comparison["현재 결과"] - cohort_comparison["체인 추가 후"]
    mega_cohort_mask = all_cohort_before_new_exclusions["사업장명"].map(normalize_name).str.contains("메가", regex=False)
    mega_cohort_total = int(mega_cohort_mask.sum())
    mega_cohort_matched = int(
        (mega_cohort_mask & all_cohort_before_new_exclusions["brand"].eq("메가MGC커피")).sum()
    )
    mega_unmatched = (
        all_cohort_before_new_exclusions.loc[
            mega_cohort_mask & all_cohort_before_new_exclusions["brand"].fillna("").ne("메가MGC커피"),
            "사업장명",
        ]
        .fillna("").astype(str).str.strip().replace("", pd.NA).dropna().value_counts()
        .rename_axis("사업장명").reset_index(name="건수")
    )
    temporary_all = int(all_cafes["excluded_temporary_permit"].sum())
    nonpositive_all = int(all_cafes["excluded_nonpositive_survival"].sum())
    noncafe_all = int(all_cafes["excluded_noncafe"].sum())
    temporary_cohort = int(all_cohort_before_new_exclusions["excluded_temporary_permit"].sum())
    nonpositive_cohort = int(all_cohort_before_new_exclusions["excluded_nonpositive_survival"].sum())
    noncafe_cohort = int(all_cohort_before_new_exclusions["excluded_noncafe"].sum())
    exclusion_overlap_all = int(
        (all_cafes[["excluded_temporary_permit", "excluded_nonpositive_survival", "excluded_noncafe"]].sum(axis=1) > 1).sum()
    )
    exclusion_overlap_cohort = int(
        (all_cohort_before_new_exclusions[["excluded_temporary_permit", "excluded_nonpositive_survival", "excluded_noncafe"]].sum(axis=1) > 1).sum()
    )
    added_brands = brands.loc[brands["source"].ne("initial"), "브랜드명"]
    added_brand_counts = (
        all_cafes.loc[all_cafes["brand"].isin(added_brands)]
        .groupby("brand", dropna=True).size().rename("전체 기간 매칭 수")
        .to_frame().join(
            all_cafes.loc[all_cafes["in_study_period"] & all_cafes["brand"].isin(added_brands)]
            .groupby("brand", dropna=True).size().rename("개업 기간 표본 매칭 수"), how="outer"
        ).fillna(0).astype(int).rename_axis("브랜드").reset_index()
        .sort_values(["전체 기간 매칭 수", "브랜드"], ascending=[False, True], kind="stable")
    )
    uncertain_chain_candidates = chain_classification.loc[
        chain_classification["분류"].eq("불확실"), ["핵심 상호", "지점 수", "근거"]
    ]
    coordinate_missing = int((~clean["has_coordinates"]).sum())
    narrowed_summary = (
        narrowed_brand_review.groupby(["브랜드명", "결과"], sort=True)["건수"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=["매칭 유지", "매칭 해제"], fill_value=0)
        .reset_index()
    )
    report = f"""# 02 인허가 전처리 보고서

실행일: 2026-11-18  
입력: `data/raw/refreshment_establishment/식품_휴게음식점_서울특별시.csv` (CP949)  
입력 SHA256: `{observed_hash}` (지정값과 일치)  
관측 종료일: {OBSERVATION_END.date()}  
개업 기간: {STUDY_START.date()}~{STUDY_END.date()} (양 끝 포함)  
프랜차이즈 검수 표본 시드: {RANDOM_SEED}

## 처리 단계와 행 수

| 단계 | 행 수 | 제외 수 | 제외 사유 |
| --- | ---: | ---: | --- |
| 원본 CP949 읽기 | {format_int(raw_rows)} | 0 | - |
| 업태구분명 `커피숍` 선택 | {format_int(coffee_rows)} | {format_int(raw_rows - coffee_rows)} | 커피숍 외 업태 |
| 기존 날짜 유효성 검사 후 전체 보존본 | {format_int(len(all_cafes))} | {format_int(int(base_exclusions.sum()))} | 폐업 상태인데 폐업일 없음 {format_int(int(closure_status_missing_date.sum()))}; 인허가일이 관측 종료일 뒤 {format_int(int(permit_after_observation.sum()))}; 인허가일 파싱 실패 {format_int(int(permit_parse_failure.sum()))} |
| `cafes_clean.parquet` 신규 제외 적용 | {format_int(len(clean))} | {format_int(int(new_exclusions.sum()))} | 한시적 인허가 {format_int(temporary_all)}; 생존기간 0일 이하 {format_int(nonpositive_all)}; 카페가 아닌 업종 {format_int(noncafe_all)}; 복수 사유 중복 {format_int(exclusion_overlap_all)} |
| 최종 개업 기간 분석 표본 | {format_int(len(cohort))} | {format_int(len(clean) - len(cohort))} | 기간 밖 행은 `cafes_clean.parquet`에 유지 |

폐업일 비결측인데 파싱에 실패한 행은 {format_int(int(closure_date_parse_failure.sum()))}건이다. 개업 기간 표본(신규 제외 전)의 한시적 인허가는 {format_int(temporary_cohort)}건, 생존기간 0일 이하는 {format_int(nonpositive_cohort)}건, 카페가 아닌 업종은 {format_int(noncafe_cohort)}건이다. 세 제외 기준에서 복수 사유가 겹치는 행은 {format_int(exclusion_overlap_cohort)}건이다.

`cafes_licenses_all.parquet`은 기존 유효성 검사를 통과한 커피숍 행을 모두 보존한다. `cafes_clean.parquet`은 한시적 인허가, 생존기간 0일 이하, 카페가 아닌 업종 행을 제외한 이후 단계의 입력이며, 이 행들은 분석 표본과 향후 경쟁 점포 수 계산 모두에서 제외한다. 카페가 아닌 업종은 편의점, 만화·스터디·보드게임카페, 헬스장, 식사 중심 체인, 기업·기관 운영처럼 일반 카페 영업으로 보기 어려운 상호의 승인된 패턴으로 한정했다. 제외 목록은 `docs/review/02_excluded_rows.csv`에 남겼다.

## 개업 기간 표본의 사건과 표본 구성

| 지표 | 전체 | 개인 카페 | 프랜차이즈 |
| --- | ---: | ---: | ---: |
| 인허가(운영자) 수 | {format_int(len(cohort))} | {format_int(int((~cohort['is_franchise']).sum()))} | {format_int(int(cohort['is_franchise'].sum()))} |
| 사건(폐업일 ≤ 관측 종료일) | {format_int(int(cohort['event'].sum()))} | {format_int(int(cohort.loc[~cohort['is_franchise'], 'event'].sum()))} | {format_int(int(cohort.loc[cohort['is_franchise'], 'event'].sum()))} |
| 중도절단 | {format_int(int((1 - cohort['event']).sum()))} | {format_int(int((1 - cohort.loc[~cohort['is_franchise'], 'event']).sum()))} | {format_int(int((1 - cohort.loc[cohort['is_franchise'], 'event']).sum()))} |

아래 표는 체인 추가 후 결과(개인 카페 8,054곳, 프랜차이즈 3,461곳, 사건 4,253건)와 현재 사전 보완 후 결과를 비교한다.

{markdown_table(cohort_comparison)}

분석 대상은 `is_franchise == False`인 개인 카페다. 프랜차이즈를 포함한 `cafes_clean.parquet`은 이후 경쟁 수 계산에 사용한다.

## 연도별 개업·폐업 (최종 분석 표본)

{markdown_table(annual)}

## 좌표 처리와 좌표계 판단

원문 X/Y의 앞뒤 공백을 제거한 후 숫자로 변환했다. `cafes_clean.parquet` 중 X 또는 Y가 없거나 숫자가 아닌 행은 {format_int(coordinate_missing)}건이며, 이 행은 제외하지 않고 `has_coordinates=False`, 경위도 결측으로 표시했다. 서울 근사 경계는 경도 126.76~127.19, 위도 37.42~37.71이다.

{markdown_table(coordinate_summary)}

세 후보의 서울 근사 경계 내 행 수는 같아 경계 검사만으로는 판별할 수 없었다. 알려진 기준점 주소의 중앙거리 평균이 가장 작은 **EPSG:{selected_epsg}**를 선택해 `longitude`·`latitude`를 저장했다. 이는 공식 좌표계 필드의 확인을 대체하지 않는다.

{markdown_table(reference_summary)}

**좌표계 확인 필요:** EPSG:{selected_epsg}가 세 기준점 검증에서 가장 가까웠으나, 데이터 제공처의 CRS 명세를 확보해 최종 확인해야 한다. 공간 결합 전에는 이 공식 확인을 마친다.

## 프랜차이즈 매칭과 표기 변형 검토

브랜드 사전 `src/franchise_brands.csv`은 {format_int(len(brands))}개 브랜드다. 검수 후보 중 지점이 5개 이상인 커피·음료·디저트 체인만 추가하고, 띄어쓰기·영문·법인명·오타 변형을 함께 반영했다. 더커피는 상호 시작·지점명 조건으로 축소하고, 엔젤리너스 및 네스카페 시작 표기를 보완했다. 사업장명은 NFKC 정규화 후 공백·일부 문장부호를 제거해 매칭한다. 지점 수 3~4개 또는 상호만으로 확정하지 못한 후보는 분류만 하고 사전에 넣지 않았다.

### 체인 추가·사전 보완 브랜드의 매칭 수

전체 기간은 `cafes_licenses_all.parquet`, 개업 기간 표본은 신규 제외 적용 전 유효 커피숍을 기준으로 하며, 비카페 제외 대상은 별도 패턴으로 제거한다.

{markdown_table(added_brand_counts)}

### 최종 정제본의 브랜드별 매칭 수

{markdown_table(brand_counts)}

### 사전 보강 전후 비교

아래 비교는 신규 제외 기준을 적용하지 않은 `cafes_licenses_all.parquet` 기준이다. 이전 실행의 동일 원자료·기존 사전 결과와 비교하므로 사전 변화만 읽을 수 있다.

{markdown_table(brand_change)}

### 오탐이 많은 브랜드의 승인된 축소 규칙

달콤은 `달콤커피` 또는 `달콤 …점`, 더카페는 상호 시작의 `더카페`/`더 카페`/`The Cafe` 뒤 지점명·`더카페엔 …점`·이랜드 표기, 커피나무는 상호 시작의 `커피나무 …점`만 매칭했다. 더커피는 상호 시작의 `더커피`/`더 커피`/`The Coffee` 뒤 지점명(…점) 및 `롯데프라자더커피`만 매칭한다. 단독 브랜드명 및 일반어 포함 상호는 개인 카페로 남겼다. 아래는 최종 정제본에서 이전 넓은 규칙 대비 유지·해제 건수이며, 개별 상호 목록은 `docs/review/02_narrowed_brands.csv`에 있다.

{markdown_table(narrowed_summary)}

개업 기간 표본(신규 제외 전)에서 `메가`가 포함된 행은 {format_int(mega_cohort_total)}건으로 사전 확인값 589건과 일치한다. 이 중 메가MGC커피 매칭은 기존 34건에서 {format_int(mega_cohort_matched)}건으로 늘었다. 매칭되지 않은 {format_int(int(mega_unmatched['건수'].sum()))}건({format_int(len(mega_unmatched))}개 상호)은 개인 카페·다른 브랜드 입점·놓친 표기를 구분하기 위해 빈도순으로 남겼다.

{markdown_table(mega_unmatched.head(50))}

`docs/review/02_brand_variants.csv`에는 모든 브랜드별 넓은 검색어 일치·사전 미매칭 상호를 남겼다. 이 파일의 {format_int(len(brand_variants))}행은 넓은 검색이므로 자동 반영하지 않았으며, 상호만으로 확정할 수 없어 `불확실`로 표시했다. 놓친 표기로 확정되는 항목은 사용자 검수 후 사전에 추가해야 한다. 커피니는 일반 단어 일부 오탐 여부를 이 파일과 브랜드 검수 표본에서 확인하도록 비고에 남겼다.

검수용 파일:

- `docs/review/02_franchise_review.csv`: 브랜드별 무작위 최대 10건(시드 고정)
- `docs/review/02_unmatched_top.csv`: 최종 정제본 미매칭 상호 빈도 상위 100개
- `docs/review/02_brand_variants.csv`: 넓은 검색어 일치·사전 미매칭 상호
- `docs/review/02_narrowed_brands.csv`: 달콤·더카페·커피나무·더커피의 이전 규칙 대비 매칭 유지·해제 상호
- `docs/review/02_chain_candidates.csv`: 체인 추가 뒤 남은 미매칭 지점형 체인 후보
- `docs/review/02_chain_classification.csv`: 체인 후보 294개의 분류·근거·반영 여부

## 산출물

- `data/processed/cafes_licenses_all.parquet`: 기존 유효성 검사를 통과한 전체 커피숍 보존본
- `data/processed/cafes_clean.parquet`: 신규 제외 기준을 적용한 이후 단계용 전체 커피숍
- `docs/reports/02_preprocess_report.md`: 이 보고서
- `docs/review/02_excluded_rows.csv` 및 위 검수 CSV: 제외·프랜차이즈 판별 검수용

기존 `data/processed/cafes_licenses.parquet`은 새 파일명으로 대체되어 삭제했다. Parquet 엔진은 {parquet_engine_status}이다.

## 결정 필요·한계

1. 프랜차이즈 매칭은 현재 상호명 기반 사전 분류다. 동명 개인점·상호 변경·신규/누락 체인이 있을 수 있다. 체인 후보 294개의 분류는 `docs/review/02_chain_classification.csv`에 남겼고, 이 중 상호만으로 확정할 수 없는 {format_int(len(uncertain_chain_candidates))}개는 사전에 넣지 않았다. 체인 추가 뒤에도 공백으로 구분된 마지막 `…점` 지점명이 3개 이상인 핵심 상호 {format_int(len(chain_candidates))}개가 남아 `docs/review/02_chain_candidates.csv`에 있다.

{markdown_table(uncertain_chain_candidates)}
2. **결정 필요 — 베네 OO점:** 카페베네 및 까페베네 표기 외에, `베네` 뒤 지점명으로 보이는 상호 {format_int(int(bene_ambiguous['건수'].sum()))}건({format_int(len(bene_ambiguous))}개 상호)을 찾았다. 상호만으로 카페베네의 변형인지 별도 체인인지 확정할 수 없어 사전에 넣지 않았다.

{markdown_table(bene_ambiguous)}

3. `이디커피`, `커피반하다`, `봄봄`·`커피봄봄`·`봄봄커피숍`, `마마스` 계열, `커피베리`는 승인 범위에 따라 사전에 추가하지 않았다.
4. 원자료에는 운영자 변경(양수도)을 식별하는 전용 필드가 없다. 승인된 정의에 따라 인허가(운영자) 단위로 유지했다.
5. 관측 종료일 이후 폐업일은 중도절단으로 처리했다. 원자료의 최신성이 2026-09-30까지 완전하다는 별도 근거는 이 작업에서 검증하지 않았다.
6. 좌표계는 공식 명세로 확정되지 않았으므로, 위의 임시 EPSG:{selected_epsg} 경위도는 향후 공간 결합 전에 재검증해야 한다.{parquet_limitation}
"""
    write_parquet(all_cafes, ALL_PROCESSED_PATH)
    write_parquet(clean, CLEAN_PROCESSED_PATH)
    if LEGACY_PROCESSED_PATH.exists():
        LEGACY_PROCESSED_PATH.unlink()
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Wrote {ALL_PROCESSED_PATH}")
    print(f"Wrote {CLEAN_PROCESSED_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {REVIEW_PATH}")
    print(f"Wrote {UNMATCHED_PATH}")
    print(f"Wrote {EXCLUDED_ROWS_PATH}")
    print(f"Wrote {BRAND_VARIANTS_PATH}")
    print(f"Wrote {NARROWED_BRANDS_PATH}")
    print(f"Wrote {CHAIN_CANDIDATES_PATH}")
    print(f"Study cohort: {len(cohort)}; events: {int(cohort['event'].sum())}; censored: {int((1 - cohort['event']).sum())}")


if __name__ == "__main__":
    main()
