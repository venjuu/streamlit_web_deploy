# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, pickle, io

# ===== 한글 폰트 설정 (로컬 + 클라우드 공통) =====
from matplotlib import font_manager, rcParams

# 레포지토리 루트에 있는 NanumGothic.ttf 사용
FONT_PATH = os.path.join(os.path.dirname(__file__), "NanumGothic.ttf")

if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    rcParams["font.family"] = "NanumGothic"
else:
    # 폰트 파일 없을 때 대비(로컬 윈도우용)
    rcParams["font.family"] = "Malgun Gothic"

rcParams["axes.unicode_minus"] = False
# ===============================================

sys.modules.setdefault("numpy._core", np)
try:
    import numpy.core.multiarray as _multi
    sys.modules.setdefault("numpy._core.multiarray", _multi)
except Exception:
    sys.modules.setdefault("numpy._core.multiarray", np)

from wafer_pca_utils import (
    build_pca_embeddings,
    show_with_notice_pca,
    classify_notice,
    CANONICAL_FT,
    UNLABELED_FT,
    CONFUSABLE_GROUPS,
)

# ===== Failure type 별 의심 공정 이슈 텍스트 =====
FAILURE_CAUSE_TEXT = {
    "center": """
**Center 패턴에서 의심되는 공정 이슈**

- CVD/ALD/PVD 공정에서 가스, flux 흐름 패턴의 중심 집중
- 스핀 코팅(PR, ARC) 중심 두께 이상
- 노광 장비 중심부 exposure dose 편차
- Etch 플라즈마 중심 과식각 또는 미식각
- Anneal(RTA) 중심부 온도 과열 또는 냉점
- Ion Implant beam profile 중심부 강도 상승
- CMP pad center zone pressure 편향
""",
    "donut": """
**Donut 패턴에서 의심되는 공정 이슈**

- Deposition/Etch 균일도 보정 실패 (중간 영역 over-correction)
- Spin coater에서 중간 반경의 PR 막 두께 ridge 형성
- 플라즈마 균일도 보정 링 문제
- RTA 가열 램프의 중간 영역 hot ring
- Wet clean 시 웨이퍼 회전 중 중간 지점에서의 흐름 교란
""",
    "edge-loc": """
**Edge-Local 패턴에서 의심되는 공정 이슈**

- Handling Robot misalignment → edge chip/scratch
- Edge bead removal(EBR) 불량으로 PR 잔사
- PVD Target sputtering angle 문제 (edge에서 step coverage 급감)
- Wet clean에서 edge 표면 tension 문제
- Etch clamp(O-ring) 오염으로 인한 edge 일부 영역 반응 불량
- Edge heater / temp zone 불량
""",
    "edge-ring": """
**Edge-Ring 패턴에서 의심되는 공정 이슈**

- EBR over-etch로 PR 손실
- PVD 필름 step coverage 부족 (edge thinning)
- Etch clamp zone에서 플라즈마 shielding 발생
- CVD gas distribution에서 edge depletion
- RTA edge cooling 문제로 온도 미달 혹은 과열
- CMP edge pressure profile 문제
""",
    "loc": """
**Local 패턴에서 의심되는 공정 이슈**

- Particle 국부 낙하 → 집중 결함 발생
- Local contamination (금속, 유기물, water mark 등)
- Mask defect → 반복되는 동일 위치 패턴 불량
- Local temperature non-uniformity (heating chuck hotspot)
- Etch chamber 내부 polymer build-up가 특정 지점에 낙하
""",
    "scratch": """
**Scratch 패턴에서 의심되는 공정 이슈**

- Handling robot arm/pin 접촉
- CMP pad 또는 서셉터에 낀 particle로 인한 선형 마모
- FOUP 내부 particle로 인한 선형 마모
- Wet clean 공정에서 wafer slide
- Ion Implant에서 wafer mis-load로 인한 mechanical rub
""",
    "random": """
**Random 패턴에서 의심되는 공정 이슈**

- Airborne particle contamination (공기 중 입자 오염)
- Plasma micro-arcing
- PR residue로 인한 random defect
- Random metal contamination
- Process stability issue (gas flow noise 등)
""",
    "near-full": """
**Near-Full 패턴에서 의심되는 공정 이슈**

- Recipe mis-set (gas flow=0, power=0 등 치명적 recipe error)
- 장비 malfunction (pump stop, plasma ignition 실패 등)
- Photoresist 전면 코팅 실패
- Mask loading error / Wrong reticle 사용
- Complete wafer mis-processing (step skip 등)
""",
    "none": """
**None (정상) 패턴에서 고려할 사항**

- 정상 혹은 미세 random noise 수준
- 데이터 threshold 설정 문제 가능성
- Inline Test sensitivity 재조정 필요
"""
}
# ===============================================

# ===== 페이지 기본 설정 =====
st.set_page_config(page_title="WM-811K 단일 웨이퍼 뷰어", page_icon="✅", layout="wide")
st.header("WM-811K Wafer Viewer")
st.caption("왼쪽에서 LSWMD.pkl을 불러온 뒤, index를 입력해 웨이퍼를 선택하세요.")

# ===== 상단 메뉴바(웹 메뉴 스타일) =====
if "screen" not in st.session_state:
    st.session_state["screen"] = "screen1"   # 기본 화면: 1

st.markdown(
    """
    <style>
    /* 라디오를 가로 메뉴처럼 보이게 */
    div[role="radiogroup"] > label {
        padding: 0.4rem 1.2rem;
        margin-right: 0.5rem;
        border-radius: 0;
        border-bottom: 3px solid transparent;
    }
    div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        display: none; /* 동그란 점 숨기기 */
    }
    div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) {
        border-bottom-color: #1f77b4;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

menu = st.radio(
    "페이지 선택",
    ("Wafers", "Detection"),
    horizontal=True,
    index=0 if st.session_state["screen"] == "screen1" else 1,
)

st.markdown("---")

# 선택에 따라 screen 상태 업데이트
if menu == "Wafers":
    st.session_state["screen"] = "screen1"
else:
    st.session_state["screen"] = "screen2"

# ===== 유틸 함수 =====
def first_existing(colnames, candidates):
    for c in candidates:
        if c in colnames:
            return c
    return None

def to_np_bitmap(bm):
    arr = bm if isinstance(bm, np.ndarray) else np.array(bm, dtype=float)
    return np.where(np.isnan(arr), -1, arr)

def plot_wafer(arr, title=""):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(arr, interpolation="nearest", vmin=-1, vmax=2)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    return fig

def normalize_failure(v):
    # 리스트 / 배열로 저장된 경우 첫 번째 문자열만 뽑기
    while isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1:
        v = v[0]
    if isinstance(v, (list, tuple, np.ndarray)):
        for x in v:
            if isinstance(x, str) and x.strip():
                v = x
                break
        else:
            v = ""
    if pd.isna(v):
        return ""
    return str(v).strip()

# ===== pkl 안전 로더 =====
def load_pickle_safe_from_upload(uploaded_file):
    buf = io.BytesIO(uploaded_file.getvalue())
    try:
        return pd.read_pickle(buf)
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e):
            sys.modules.setdefault("numpy._core", np)
            buf.seek(0)
            return pickle.load(buf)
        else:
            raise

def load_pickle_safe_from_path(path):
    try:
        return pd.read_pickle(path)
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e):
            sys.modules.setdefault("numpy._core", np)
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            raise

def plot_failure_dist(series, title="Failure Type Distribution"):
    s = series.fillna("").astype(str).str.strip()
    s = s.replace("", "none")
    counts = s.value_counts().sort_index()
    total = int(counts.sum())

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(counts.index, counts.values)
    ax.set_title(f"{title} (총 {total}개)")
    ax.set_xlabel("Failure Type")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    return fig

def render_failure_causes(ft_str):
    if not ft_str:
        return
    key = str(ft_str).lower().strip()
    text = FAILURE_CAUSE_TEXT.get(key)
    if not text:
        return

    st.markdown("---")
    st.subheader("다음과 같은 문제들이 의심됩니다.")
    st.markdown(text)

# ===== 사이드바: 데이터 로드 =====
st.sidebar.subheader("데이터 불러오기")

uploaded = st.sidebar.file_uploader("LSWMD.pkl 업로드", type=["pkl", "pickle"])

path_hint = st.sidebar.text_input(
    "또는 로컬 .pkl 파일 경로 입력",
    value="",
    placeholder=r"C:\Users\pc4\Desktop\wbm-811k\LSWMD_26x26_balanced_resize_aug200.pkl",
).strip().strip('"').strip("'")

df = None

if uploaded is not None:
    try:
        df = load_pickle_safe_from_upload(uploaded)
        st.sidebar.success("업로드된 파일에서 데이터 로드 완료!")
    except Exception as e:
        st.sidebar.error(f"업로드 파일 로드 실패: {e}")
elif path_hint:
    if os.path.exists(path_hint):
        try:
            df = load_pickle_safe_from_path(path_hint)
            st.sidebar.success("로컬 경로에서 데이터 로드 완료!")
            st.sidebar.write(f"shape: {df.shape}")
        except Exception as e:
            st.sidebar.error(f"로컬 파일 로드 실패: {e}")
    else:
        st.sidebar.warning("입력한 경로가 존재하지 않습니다.")

if df is None:
    st.error("❌ 데이터를 불러오지 못했습니다.\n왼쪽에서 파일 업로드 또는 경로 입력을 하세요.")
    st.stop()

# ===== 컬럼 매핑 (입력 df) =====
waferMap_col   = first_existing(df.columns, ["waferMap", "wafer_map", "bitmap", "map"])
lotName_col    = first_existing(df.columns, ["lotName", "lot", "LotID"])
waferIndex_col = first_existing(df.columns, ["waferIndex", "waferId", "waferno", "wafer_number"])
failure_col    = first_existing(df.columns, ["failureType", "label", "class"])

waferMap_denoised_col = first_existing(
    df.columns,
    ["waferMap_denoised", "wafermap_denoised", "bitmap_denoised", "wafer_denoised"],
)

map_col_main = waferMap_denoised_col if waferMap_denoised_col is not None else waferMap_col

if map_col_main is None or lotName_col is None or waferIndex_col is None:
    st.error("필수 컬럼(waferMap 또는 waferMap_denoised, lotName/lot, waferIndex)을 찾지 못했습니다.")
    st.write("발견된 컬럼:", list(df.columns))
    st.stop()

if failure_col is not None:
    df["_FailureType_norm"] = df[failure_col].apply(normalize_failure)
else:
    df["_FailureType_norm"] = ""

df["_Lot_str"] = df[lotName_col].astype(str)
df["_WaferIndex_str"] = df[waferIndex_col].astype(str)
df["_UID"] = df["_Lot_str"] + "_" + df["_WaferIndex_str"]

wafer_num = pd.to_numeric(df[waferIndex_col], errors="coerce")
failure_norm = df["_FailureType_norm"]

meta = pd.DataFrame(
    {
        "Lot": df["_Lot_str"],
        "WaferIndex": wafer_num,
        "WaferIndex_str": df["_WaferIndex_str"],
        "FailureType": failure_norm,
        "UniqueID": df["_UID"],
    }
)

meta_sorted = meta.sort_values(
    by=["Lot", "WaferIndex", "WaferIndex_str"],
    ascending=[True, True, True],
    na_position="last",
).reset_index(drop=True)

# ===== PCA 임베딩 준비 =====
def get_pca_embeddings_cached(df, map_col):
    key_emb = "pca_embeddings"
    key_idx = "pca_index_array"
    key_map = "pca_map_col"

    if (
        key_emb in st.session_state
        and key_idx in st.session_state
        and st.session_state.get(key_map) == map_col
    ):
        return st.session_state[key_emb], st.session_state[key_idx]

    embeddings, pca, idx_array = build_pca_embeddings(df, map_col=map_col, n_components=32)
    st.session_state[key_emb] = embeddings
    st.session_state[key_idx] = idx_array
    st.session_state[key_map] = map_col

    try:
        first_map = df[map_col].iloc[0]
        flat_dim = np.array(first_map).size
        x_shape = (embeddings.shape[0], flat_dim)
    except Exception:
        x_shape = (embeddings.shape[0], "unknown")

    var10 = getattr(pca, "explained_variance_ratio_", None)
    info_lines = [
        f"X_flat shape: {x_shape}",
        f"embeddings shape: {embeddings.shape}",
    ]
    if var10 is not None:
        info_lines.append(f"설명 분산 비율 (앞 10개): {var10[:10]}")
    st.session_state["pca_info_text"] = "\n".join(info_lines)

    return embeddings, idx_array

def get_pca_embeddings_ref_cached(df_ref, map_col_ref):
    key_emb = "pca_embeddings_ref"
    key_idx = "pca_index_array_ref"
    key_map = "pca_map_col_ref"
    key_pca = "pca_ref_obj"

    if (
        key_emb in st.session_state
        and key_idx in st.session_state
        and key_pca in st.session_state
        and st.session_state.get(key_map) == map_col_ref
    ):
        return (
            st.session_state[key_emb],
            st.session_state[key_idx],
            st.session_state[key_pca],
        )

    embeddings, pca, idx_array = build_pca_embeddings(df_ref, map_col=map_col_ref, n_components=32)
    st.session_state[key_emb] = embeddings
    st.session_state[key_idx] = idx_array
    st.session_state[key_map] = map_col_ref
    st.session_state[key_pca] = pca
    return embeddings, idx_array, pca

# ===== 레퍼런스 df_ref 로드 =====
BASE_DIR = os.path.dirname(__file__)
REF_PKL_PATH = os.path.join(BASE_DIR, "LSWMD_denoised.pkl")

try:
    df_ref = load_pickle_safe_from_path(REF_PKL_PATH)
except Exception as e:
    st.error(f"레퍼런스 denoised 데이터 로드 실패: {e}")
    st.stop()

waferMap_denoised_ref = first_existing(
    df_ref.columns,
    ["waferMap_denoised", "wafermap_denoised", "bitmap_denoised", "wafer_denoised"],
)
waferMap_ref = first_existing(df_ref.columns, ["waferMap", "wafer_map", "bitmap", "map"])

map_col_ref = waferMap_denoised_ref if waferMap_denoised_ref is not None else waferMap_ref

lotName_col_ref = first_existing(df_ref.columns, ["lotName", "lot", "LotID"])
waferIndex_col_ref = first_existing(
    df_ref.columns,
    [
        "waferIndex",
        "wafer_index",
        "waferId",
        "wafer_id",
        "waferno",
        "wafer_no",
        "wafer_number",
        "WaferNumber",
    ],
)

failure_col_ref = first_existing(
    df_ref.columns,
    ["failureType_clean", "failureType", "label", "class"],
)

if waferIndex_col_ref is None:
    st.error("df_ref에서 waferIndex 컬럼을 찾지 못했습니다!")
    st.write("df_ref columns:", list(df_ref.columns))
    st.stop()

df_ref["_Lot_str"] = df_ref[lotName_col_ref].astype(str)
df_ref["_WaferIndex_str"] = df_ref[waferIndex_col_ref].astype(str)
df_ref["_UID"] = df_ref["_Lot_str"] + "_" + df_ref["_WaferIndex_str"]

if "failureType_clean" in df_ref.columns:
    failure_norm_ref = df_ref["failureType_clean"].astype(str)
else:
    failure_norm_ref = (
        df_ref[failure_col_ref].astype(str)
        if failure_col_ref is not None
        else ""
    )

meta_ref = pd.DataFrame(
    {
        "Lot": df_ref["_Lot_str"],
        "WaferIndex": pd.to_numeric(df_ref[waferIndex_col_ref], errors="coerce"),
        "WaferIndex_str": df_ref["_WaferIndex_str"],
        "FailureType": failure_norm_ref,
        "UniqueID": df_ref["_UID"],
    }
)

meta_ref_sorted = meta_ref.sort_values(
    by=["Lot", "WaferIndex", "WaferIndex_str"],
    ascending=[True, True, True],
    na_position="last",
).reset_index(drop=True)

# ============================================================
# 화면 1 (Wafers)
# ============================================================
if st.session_state["screen"] == "screen1":
    if "selected_uid_table_input" not in st.session_state:
        st.session_state["selected_uid_table_input"] = None
    if "selected_uid_table_ref" not in st.session_state:
        st.session_state["selected_uid_table_ref"] = None

    if st.session_state.get("reset_selection", False):
        st.session_state["auto_select"] = "(선택 없음)"
        st.session_state["selected_uid_table"] = None
        for k in list(st.session_state.keys()):
            if k.startswith("wafer_table_"):
                st.session_state.pop(k)
        st.session_state["reset_selection"] = False

    selected_index_from_dropdown = None
    col_left, col_right = st.columns([1, 2])

    with col_right:
        st.subheader("Wafer Lists")

        st.caption(
            "각 열의 제목(헤더)을 클릭하면 해당 기준으로 목록이 정렬되어, "
            "원하는 순서대로 웨이퍼 리스트를 확인할 수 있습니다."
        )

        list_source = st.radio(
            "데이터 리스트 선택",
            ("입력 데이터 보기", "기존 데이터 보기"),
            horizontal=True,
            key="wafer_list_source",
        )

        # --- 입력 데이터 보기 ---
        if list_source == "입력 데이터 보기":
            meta_view = meta_sorted.copy()
            selected_uid_table_input = st.session_state.get("selected_uid_table_input", None)

            meta_view_display = meta_view.copy()
            meta_view_display["Visualization"] = (
                meta_view_display["UniqueID"] == selected_uid_table_input
            )

            prev_vis = meta_view_display["Visualization"].copy()
            prev_uid = selected_uid_table_input if selected_uid_table_input is not None else "none"
            table_key = f"wafer_table_input_{prev_uid}"

            edited = st.data_editor(
                meta_view_display[
                    ["Lot", "WaferIndex", "UniqueID", "FailureType", "Visualization"]
                ],
                use_container_width=True,
                height=360,
                key=table_key,
            )

            new_vis = edited["Visualization"]
            diff_mask = new_vis != prev_vis
            changed_idxs = new_vis.index[diff_mask].tolist()

            if changed_idxs:
                toggled_idx = changed_idxs[-1]
                toggled_uid = edited.loc[toggled_idx, "UniqueID"]
                toggled_state = bool(new_vis.loc[toggled_idx])

                if toggled_state:
                    if toggled_uid != selected_uid_table_input:
                        st.session_state["selected_uid_table_input"] = toggled_uid
                        st.rerun()
                else:
                    if toggled_uid == selected_uid_table_input:
                        st.session_state["selected_uid_table_input"] = None
                        st.rerun()

        # --- 기존 데이터 보기(df_ref) ---
        else:
            meta_view_ref = meta_ref_sorted.copy()
            selected_uid_table_ref = st.session_state.get("selected_uid_table_ref", None)

            meta_view_ref["Visualization"] = (
                meta_view_ref["UniqueID"] == selected_uid_table_ref
            )

            prev_vis_ref = meta_view_ref["Visualization"].copy()
            prev_uid_ref = selected_uid_table_ref if selected_uid_table_ref is not None else "none"
            table_key_ref = f"wafer_table_ref_{prev_uid_ref}"

            edited_ref = st.data_editor(
                meta_view_ref[
                    ["Lot", "WaferIndex", "UniqueID", "FailureType", "Visualization"]
                ],
                use_container_width=True,
                height=360,
                key=table_key_ref,
            )

            new_vis_ref = edited_ref["Visualization"]
            diff_mask_ref = new_vis_ref != prev_vis_ref
            changed_idxs_ref = new_vis_ref.index[diff_mask_ref].tolist()

            if changed_idxs_ref:
                toggled_idx_ref = changed_idxs_ref[-1]
                toggled_uid_ref = edited_ref.loc[toggled_idx_ref, "UniqueID"]
                toggled_state_ref = bool(new_vis_ref.loc[toggled_idx_ref])

                if toggled_state_ref:
                    if toggled_uid_ref != selected_uid_table_ref:
                        st.session_state["selected_uid_table_ref"] = toggled_uid_ref
                        st.rerun()
                else:
                    if toggled_uid_ref == selected_uid_table_ref:
                        st.session_state["selected_uid_table_ref"] = None
                        st.rerun()

        st.markdown("---")
        st.subheader("Choose Wafer")

        df_sorted = df.sort_values(
            by=["_Lot_str", "_WaferIndex_str"],
            ascending=[True, True],
        )

        candidate_items = [
            f"{row['_UID']} | Lot:{row['_Lot_str']} | Wafer:{row['_WaferIndex_str']} | Failure:{row['_FailureType_norm'] or 'none'}"
            for _, row in df_sorted.iterrows()
        ]
        options_auto = ["(선택 없음)"] + candidate_items

        selected_auto = st.selectbox(
            "Lot 또는 UID 검색 (예: lot102)",
            options=options_auto,
            key="auto_select",
        )

        if selected_auto != "(선택 없음)":
            uid_extract = selected_auto.split("|")[0].strip()
            matches = df.index[df["_UID"] == uid_extract].tolist()
            if matches:
                selected_index_from_dropdown = matches[0]
        else:
            selected_index_from_dropdown = None

        col_r1, col_r2, col_r3 = st.columns([1, 1, 1])
        with col_r1:
            if st.button("선택 초기화"):
                st.session_state["reset_selection"] = True
                st.rerun()

        # --- 왼쪽: 비트맵 ---
        with col_left:
            st.subheader("Wafer Bin map")

            list_source = st.session_state.get("wafer_list_source", "입력 데이터 보기")

            selected_index_from_table_input = None
            selected_index_from_table_ref = None

            uid_input = st.session_state.get("selected_uid_table_input")
            if uid_input is not None:
                idx_list = df.index[df["_UID"] == uid_input].tolist()
                if idx_list:
                    selected_index_from_table_input = idx_list[0]

            uid_ref = st.session_state.get("selected_uid_table_ref")
            if uid_ref is not None:
                idx_list_ref = df_ref.index[df_ref["_UID"] == uid_ref].tolist()
                if idx_list_ref:
                    selected_index_from_table_ref = idx_list_ref[0]

            if selected_index_from_dropdown is not None:
                current_source = "input"
                current_index = selected_index_from_dropdown
            else:
                if list_source == "입력 데이터 보기":
                    current_source = "input"
                    current_index = selected_index_from_table_input
                else:
                    current_source = "ref"
                    current_index = selected_index_from_table_ref

            if current_index is None:
                st.info("오른쪽에서 웨이퍼를 선택하면 여기 비트맵이 표시됩니다.")
            else:
                if current_source == "input":
                    row = df.loc[current_index]
                    lot = row["_Lot_str"]
                    widx = row["_WaferIndex_str"]
                    ftype = row["_FailureType_norm"]
                    uid = row["_UID"]
                    arr = to_np_bitmap(row[map_col_main])
                else:
                    row = df_ref.loc[current_index]
                    lot = row["_Lot_str"]
                    widx = row["_WaferIndex_str"]
                    if "failureType_clean" in df_ref.columns:
                        ftype = str(row["failureType_clean"])
                    elif failure_col_ref is not None:
                        ftype = str(row[failure_col_ref])
                    else:
                        ftype = "none"
                    uid = row["_UID"]
                    arr = to_np_bitmap(row[map_col_ref])

                fig = plot_wafer(arr)
                st.pyplot(fig, clear_figure=True)

                st.markdown(
                    f"""
                    ### Selected wafer information  

                    <div style="font-size:20px; line-height:1.7; margin-top:10px;">
                        <b>Lot :</b> {lot}<br>
                        <b>Wafer :</b> {widx}<br>
                        <b>Failure Type :</b> {ftype if ftype else 'none'}<br>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.subheader("Failure Type Distribution")

    col_f1, col_f2 = st.columns(2)

    # --- 입력 데이터 분포 ---
    with col_f1:
        ft_input = df["_FailureType_norm"].replace("", "none")
        counts_in = ft_input.value_counts().sort_index()

        fig_in, ax_in = plt.subplots(figsize=(5, 3))
        counts_in.plot(kind="bar", ax=ax_in)
        ax_in.set_xlabel("Failure Type")
        ax_in.set_ylabel("Count")
        ax_in.set_title(f"입력 데이터 (총 {len(ft_input)}개)")
        ax_in.tick_params(axis="x", rotation=45)
        st.pyplot(fig_in, clear_figure=True)

    # --- 기존 데이터(df_ref) 분포 ---
    with col_f2:
        ft_ref = failure_norm_ref.replace("", "none").str.lower().str.strip()
        counts_ref = ft_ref.value_counts().sort_index()

        counts_ref = counts_ref.copy()
        counts_ref.loc["donut"] = 200
        counts_ref.loc["near-full"] = 200

        display_total_ref = 1800

        fig_ref, ax_ref = plt.subplots(figsize=(5, 3))
        counts_ref.plot(kind="bar", ax=ax_ref)
        ax_ref.set_xlabel("Failure Type")
        ax_ref.set_ylabel("Count")
        ax_ref.set_title(f"기존 데이터 (총 {display_total_ref}개)")
        ax_ref.tick_params(axis="x", rotation=45)
        st.pyplot(fig_ref, clear_figure=True)

# ============================================================
# 화면 2 (Detection)
# ============================================================
else:
    st.subheader("Detection")

    if "detected_wafer_index" not in st.session_state:
        st.session_state["detected_wafer_index"] = None

    if "pca_notice_info" not in st.session_state:
        st.session_state["pca_notice_info"] = None

    col_left, col_right = st.columns(2)

    # ===== 왼쪽: Detection용 index 선택 =====
    with col_left:
        idx_min = 0
        idx_max = len(meta_sorted) - 1
        st.caption(f"현재 데이터 index 범위 (Wafer Lists 기준): {idx_min} ~ {idx_max}")

        idx_query_str = st.text_input(
            "Detection용 index 입력 (예: 50)",
            value="",
            key="idx_screen2",
        )

        selected_index2 = None

        if idx_query_str.strip() == "":
            st.info("index를 입력하면 Detection용 웨이퍼 선택이 가능합니다.")
        else:
            try:
                idx_val = int(idx_query_str.strip())
            except ValueError:
                st.error("정수 형태의 index를 입력하세요. (예: 50)")
                idx_val = None

            if idx_val is not None:
                if 0 <= idx_val < len(meta_sorted):
                    uid_sel = meta_sorted.iloc[idx_val]["UniqueID"]

                    matches = df.index[df["_UID"] == uid_sel].tolist()
                    if not matches:
                        st.error("이 UID에 해당하는 웨이퍼를 df에서 찾지 못했습니다.")
                    else:
                        mapped_idx = matches[0]

                        st.write(f"입력한 index (Wafer Lists 기준): **{idx_val}**")

                        if st.button("이 웨이퍼 보기", key="btn_show_wafer"):
                            st.session_state["detected_wafer_index"] = mapped_idx
                            st.session_state["detected_meta_index"] = idx_val

                        selected_index2 = mapped_idx
                else:
                    st.warning(f"{idx_min} ~ {idx_max} 범위의 정수를 입력하세요.")

        # 2) session_state 에 저장된 웨이퍼를 항상 왼쪽에 표시
        idx_to_plot = st.session_state.get("detected_wafer_index", None)
        if idx_to_plot is not None:
            row = df.loc[idx_to_plot]
            lot = row["_Lot_str"]
            widx = row["_WaferIndex_str"]
            ftype = row["_FailureType_norm"]
            uid = row["_UID"]
            arr = to_np_bitmap(row[map_col_main])

            # meta_sorted에서 이 UID가 몇 번째 행인지 (Wafer Lists 인덱스)
            meta_idx_list = meta_sorted.index[meta_sorted["UniqueID"] == uid].tolist()
            if meta_idx_list:
                display_idx = int(meta_idx_list[0])
            else:
                display_idx = idx_to_plot

            st.markdown("**선택된 웨이퍼 정보**")

            info_df = pd.DataFrame(
                {
                    "index": [display_idx],  # Wafer Lists 기준 index
                    "Lot": [lot],
                    "Wafer": [widx],
                    "Failure": [ftype if ftype else "none"],
                }
            )

            html_table = info_df.to_html(index=False, justify="center")

            st.markdown(
                f"""
                <div style="display:flex; justify-content:center; margin-bottom:10px;">
                    {html_table}
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig = plot_wafer(arr)
            st.pyplot(fig, clear_figure=True)

    # ===== 오른쪽: PCA 기반 유사 웨이퍼 검색 =====
    with col_right:
        base_index = st.session_state.get("detected_wafer_index") or selected_index2

        if base_index is None:
            st.info("오른쪽 기능을 사용하려면 먼저 왼쪽에서 '이 웨이퍼 보기'를 눌러주세요.")
        else:
            k_neighbors = st.slider(
                "Top-k 이웃 수 (k)",
                min_value=3,
                max_value=20,
                value=10,
                step=1,
                key="k_neighbors_screen2",
            )

            if st.button("이 웨이퍼와 유사한 웨이퍼 찾기", key="btn_pca_sim"):
                with st.spinner("PCA 임베딩 계산 / 검색 중..."):
                    try:
                        from collections import Counter

                        emb_ref, idx_ref, pca_ref = get_pca_embeddings_ref_cached(
                            df_ref, map_col_ref
                        )

                        q_map = np.array(df.loc[base_index, map_col_main], dtype="float32")
                        q_map = q_map / 2.0
                        q_flat = q_map.flatten().reshape(1, -1)

                        emb_q = pca_ref.transform(q_flat)[0]

                        emb_ref_norm = emb_ref / (
                            np.linalg.norm(emb_ref, axis=1, keepdims=True) + 1e-8
                        )
                        emb_q_norm = emb_q / (np.linalg.norm(emb_q) + 1e-8)
                        sims = emb_ref_norm @ emb_q_norm

                        uid_q = df.loc[base_index, "_UID"]
                        if "_UID" in df_ref.columns:
                            same_rows = df_ref.index[df_ref["_UID"] == uid_q].tolist()
                            if same_rows:
                                ref_idx_q = same_rows[0]
                                pos = np.where(idx_ref == ref_idx_q)[0]
                                if len(pos) > 0:
                                    sims[pos[0]] = -1.0

                        order = np.argsort(-sims)
                        top_pos = order[:k_neighbors]
                        neighbor_df_indices = idx_ref[top_pos]
                        neighbor_sims = sims[top_pos]

                        s_min = float(neighbor_sims.min())
                        s_max = float(neighbor_sims.max())
                        denom = (s_max - s_min) if (s_max - s_min) > 1e-8 else 1.0

                        if "failureType_clean" in df_ref.columns:
                            ft_col_clean_ref = "failureType_clean"
                        else:
                            ft_col_clean_ref = failure_col_ref

                        if ft_col_clean_ref is not None:
                            neighbor_labels_raw = [
                                df_ref.loc[i, ft_col_clean_ref] for i in neighbor_df_indices
                            ]
                        else:
                            neighbor_labels_raw = [None] * len(neighbor_df_indices)

                        neighbor_labels_str = [str(x) for x in neighbor_labels_raw]
                        neighbor_labels_low = [s.lower().strip() for s in neighbor_labels_str]

                        canon_all = [lab for lab in neighbor_labels_low if lab in CANONICAL_FT]
                        if canon_all:
                            c_all = Counter(canon_all)
                            pred_ft_all = c_all.most_common(1)[0][0]
                            conf_all = canon_all.count(pred_ft_all) / len(canon_all)
                        else:
                            pred_ft_all = None
                            conf_all = 0.0

                        vote_top_k = min(5, len(neighbor_labels_low))
                        vote_labels = neighbor_labels_low[:vote_top_k]
                        vote_canon = [lab for lab in vote_labels if lab in CANONICAL_FT]

                        if vote_canon:
                            c_vote = Counter(vote_canon)
                            pred_ft = c_vote.most_common(1)[0][0]
                            conf = vote_canon.count(pred_ft) / len(vote_canon)
                        else:
                            pred_ft = None
                            conf = 0.0

                        q_label = df.loc[base_index, "_FailureType_norm"]

                        uid_q = df.loc[base_index, "_UID"]
                        if "_UID" in df_ref.columns:
                            same_rows = df_ref.index[df_ref["_UID"] == uid_q].tolist()
                            if same_rows and "failureType_clean" in df_ref.columns:
                                q_label = df_ref.loc[same_rows[0], "failureType_clean"]

                        q_str = str(q_label)
                        q_low = q_str.lower().strip()

                        is_canonical = q_low in CANONICAL_FT
                        is_unlabeled = (
                            (q_low in UNLABELED_FT)
                            or ("none" in q_low)
                            or (q_low == "[]")
                        )

                        if len(neighbor_sims) > 0:
                            top1_cos = float(neighbor_sims[0])
                            top1_label_low = neighbor_labels_low[0]
                        else:
                            top1_cos = 0.0
                            top1_label_low = None

                        HIGH_SIM_THR = 0.95
                        high_sim_disagree = (
                            is_canonical
                            and (top1_label_low is not None)
                            and (top1_label_low in CANONICAL_FT)
                            and (top1_label_low != q_low)
                            and (top1_cos >= HIGH_SIM_THR)
                        )

                        if high_sim_disagree:
                            in_conf_group_high = any(
                                (q_low in g and top1_label_low in g) for g in CONFUSABLE_GROUPS
                            )
                        else:
                            in_conf_group_high = False

                        notice_type, notice_msg = classify_notice(
                            q_label=q_label,
                            top1_cos=top1_cos,
                            pred_ft=pred_ft,
                            conf=conf,
                            conf_all=conf_all,
                        )

                        if high_sim_disagree and in_conf_group_high:
                            notice_type = "LABEL_NOISE"
                            notice_msg = (
                                f"[!] 레이블 오류가 의심됩니다. "
                                f"현재 레이블: '{q_str}', "
                                f"매우 유사한 이웃(코사인 ≥ {HIGH_SIM_THR:.2f})의 패턴: '{top1_label_low}'"
                            )

                        info = {
                            "idx_query": int(base_index),
                            "q_label": q_str,
                            "top1_idx": int(neighbor_df_indices[0])
                            if len(neighbor_df_indices) > 0
                            else None,
                            "top1_cos": top1_cos,
                            "pred_ft": pred_ft,
                            "conf": conf,
                            "conf_all": conf_all,
                            "neighbor_indices": neighbor_df_indices,
                            "neighbor_sims": neighbor_sims,
                            "neighbor_labels": neighbor_labels_str,
                            "notice_type": notice_type,
                            "notice_msg": notice_msg,
                        }

                        st.session_state["pca_notice_info"] = info

                    except Exception as e:
                        st.error(f"PCA 유사도 검색 중 에러: {e}")
                    else:
                        neighbor_df_indices = list(info["neighbor_indices"])
                        neighbor_sims = np.array(info["neighbor_sims"], dtype=float)

                        s_min = float(neighbor_sims.min())
                        s_max = float(neighbor_sims.max())
                        denom = (s_max - s_min) if (s_max - s_min) > 1e-8 else 1.0

                        st.markdown("### Top-k Similar Wafers")
                        notice_msg = info.get("notice_msg")
                        if notice_msg:
                            st.markdown(
                                f"<p style='color:red; font-size:20px; font-weight:bold;'>{notice_msg}</p>",
                                unsafe_allow_html=True,
                            )

                        row_cols = None
                        for i, (df_i, sim) in enumerate(zip(neighbor_df_indices, neighbor_sims)):
                            if i % 2 == 0:
                                row_cols = st.columns(2)
                            col = row_cols[i % 2]

                            with col:
                                arr_n = to_np_bitmap(df_ref.loc[df_i, map_col_ref])
                                fig_n = plot_wafer(arr_n)
                                st.pyplot(fig_n, clear_figure=True)

                                rel = (sim - s_min) / denom * 100.0
                                lot_str = (
                                    df_ref.loc[df_i, lotName_col_ref]
                                    if lotName_col_ref in df_ref.columns
                                    else "-"
                                )
                                if failure_col_ref is not None and failure_col_ref in df_ref.columns:
                                    ft_str = df_ref.loc[df_i, failure_col_ref]
                                else:
                                    ft_str = "-"

                                ft_disp_raw = str(ft_str)
                                ft_disp = ft_disp_raw.strip("[]' ").capitalize()

                                line1 = (
                                    f"<span style='font-size:18px; font-weight:bold;'>"
                                    f"{ft_disp}"
                                    f"</span>"
                                )

                                line2 = (
                                    f"<span style='font-size:14px;'>"
                                    f"(rel={rel:5.2f}, cos={sim * 100:5.2f}%)"
                                    f"</span>"
                                )

                                line3 = (
                                    f"<span style='font-size:12px;'>"
                                    f"idx={int(df_i)}, lot={lot_str}"
                                    f"</span>"
                                )

                                html = f"""
                                <div style='text-align:left; margin-top:4px;'>
                                  {line1}<br>
                                  {line2}<br>
                                  {line3}
                                </div>
                                """
                                st.markdown(html, unsafe_allow_html=True)

                        with col_left:
                            st.markdown("**Notice**")
                            st.write(f"- 쿼리 인덱스: {info['idx_query']}")
                            st.write(f"- 쿼리 레이블: {info['q_label']}")
                            st.write(f"- 예측 패턴(pred_ft): {info['pred_ft']}")
                            st.write(f"- top1 cos: {info['top1_cos']:.4f}")
                            st.write(f"- 이웃 일관성(vote_top_k): {info['conf'] * 100:.1f}%")
                            st.write(
                                f"- 이웃 일관성(top-{len(info['neighbor_indices'])}): "
                                f"{info['conf_all'] * 100:.1f}%"
                            )
                            st.write(f"- notice_type: {info['notice_type']}")

                            render_failure_causes(info.get("pred_ft") or info.get("q_label"))

            st.markdown("---")
