import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== 페이지 기본 설정 =====
st.set_page_config(page_icon="✅", page_title="WM-811K: 좌/우 비교 뷰어", layout="wide")
st.header("웨이퍼맵 신규 패턴 탐지 — 좌/우 비교")
st.subheader("목록에서 웨이퍼를 선택해 좌(A) / 우(B)로 배치하고 나란히 비교하세요.")

# ===== 유틸 =====
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
    ax.set_xticks([]); ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    return fig

def normalize_failure(v):
    while isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1:
        v = v[0]
    if isinstance(v, (list, tuple, np.ndarray)):
        for x in v:
            if isinstance(x, str) and x.strip():
                v = x; break
        else:
            v = ""
    if pd.isna(v): return ""
    return str(v).strip()

def render_one_side(title_prefix, uid_key, meta_sorted, df, waferMap_col, lotName_col, waferIndex_col):
    """한쪽(A 또는 B) 선택 UI + 렌더링. 선택 UID는 session_state[uid_key]에 저장."""
    st.markdown(f"### {title_prefix}")
    tab1, tab2, tab3 = st.tabs(["Lot/Index로 선택", "Unique ID로 선택", "Failure Type로 선택 (none/빈값 제외)"])

    if uid_key not in st.session_state:
        st.session_state[uid_key] = None

    def choose_by_uid(uid):
        if uid: st.session_state[uid_key] = uid

    # Lot/Index
    with tab1:
        lots = meta_sorted["Lot"].dropna().unique().tolist()
        lot_sel = st.selectbox("Lot 선택", options=lots, index=0 if lots else None, key=f"{uid_key}_lot")
        sub = meta_sorted[meta_sorted["Lot"] == lot_sel]

        idx_options = []
        for _, r in sub.iterrows():
            idx_disp = r["WaferIndex"] if pd.notna(r["WaferIndex"]) else r["WaferIndex_str"]
            idx_disp = str(int(idx_disp)) if isinstance(idx_disp, (int, float)) and pd.notna(idx_disp) else str(idx_disp)
            idx_options.append(f"{idx_disp} ({r['FailureType']})")

        def idx_key(s):
            tok = s.split()[0]
            return (0, int(tok)) if tok.isdigit() else (1, tok)
        idx_options = sorted(set(idx_options), key=idx_key)

        idx_sel = st.selectbox("WaferIndex 선택", options=idx_options if idx_options else [], key=f"{uid_key}_idx")
        if st.button("이 웨이퍼 보기", key=f"{uid_key}_show_by_lotidx"):
            idx_token = idx_sel.split()[0] if idx_sel else None
            cand = sub.copy()
            cand["_idx_num"] = pd.to_numeric(cand["WaferIndex"], errors="coerce")
            chosen_row = None
            if idx_token is not None:
                maybe_num = pd.to_numeric(pd.Series([idx_token]), errors="coerce").iloc[0]
                if pd.notna(maybe_num):
                    hit = cand[cand["_idx_num"] == float(maybe_num)]
                    if len(hit) >= 1: chosen_row = hit.iloc[0]
                if chosen_row is None:
                    hit = cand[cand["WaferIndex_str"].astype(str) == str(idx_token)]
                    if len(hit) >= 1: chosen_row = hit.iloc[0]
            if chosen_row is not None:
                choose_by_uid(chosen_row["UniqueID"])
            else:
                st.warning("선택한 Lot/Index에 해당하는 웨이퍼를 찾지 못했습니다.")

    # UID
    with tab2:
        uid_options = meta_sorted["UniqueID"].tolist()
        uid_sel = st.selectbox("Unique ID 선택", options=uid_options if uid_options else [], key=f"{uid_key}_uid")
        if st.button("이 웨이퍼 보기 (UID)", key=f"{uid_key}_show_by_uid"):
            choose_by_uid(uid_sel)

    # FailureType
    with tab3:
        ft_all = meta_sorted["FailureType"].fillna("").astype(str).str.strip()
        ft_list = sorted({ft for ft in ft_all if ft and ft.lower() != "none"})
        ft_sel = st.selectbox("Failure Type 선택 (none/빈값 제외)", options=ft_list if ft_list else [], key=f"{uid_key}_ft")
        sub_ft = meta_sorted[meta_sorted["FailureType"].str.strip() == ft_sel]
        uid_list = sub_ft["UniqueID"].tolist()
        uid_sel_ft = st.selectbox("해당 Failure Type의 Unique ID 선택", options=uid_list if uid_list else [], key=f"{uid_key}_uid_ft")
        if st.button("이 웨이퍼 보기 (FailureType)", key=f"{uid_key}_show_by_ft"):
            choose_by_uid(uid_sel_ft)

    # 렌더링
    if st.session_state.get(uid_key):
        mhit = meta_sorted[meta_sorted["UniqueID"] == st.session_state[uid_key]]
        if len(mhit) == 1:
            meta_row = mhit.iloc[0]
            lot_v = meta_row["Lot"]
            idx_num = meta_row["WaferIndex"]
            idx_str = meta_row["WaferIndex_str"]
            ftype = meta_row["FailureType"]
            main_cause = meta_row["Main cause of defect"]
            title_idx = idx_str if pd.isna(idx_num) else int(idx_num)

            cand = df.copy()
            if lotName_col:
                cand = cand[cand[lotName_col].astype(str) == str(lot_v)]
            hit = None
            if waferIndex_col:
                cand_num = cand.copy()
                cand_num["_idx_num"] = pd.to_numeric(cand_num[waferIndex_col], errors="coerce")
                if pd.notna(idx_num):
                    hit = cand_num[cand_num["_idx_num"] == float(idx_num)]
                if (hit is None) or (len(hit) == 0):
                    cand["_idx_str"] = cand[waferIndex_col].astype(str)
                    hit = cand[cand["_idx_str"] == str(idx_str)]

            if (hit is None) or (len(hit) == 0):
                st.error("원본 데이터에서 선택한 웨이퍼를 찾지 못했습니다. (Lot/Index 매칭 실패)")
                return None, None, None, None
            else:
                row = hit.iloc[0]
                arr = to_np_bitmap(row[waferMap_col])

                st.markdown(f"**선택된 웨이퍼:** Lot:{lot_v} | Wafer:{title_idx} | Failure:{ftype} | UID:{meta_row['UniqueID']}")
                fig = plot_wafer(arr, title="")
                st.pyplot(fig, clear_figure=True)

                info_df = pd.DataFrame([{
                    "Lot": lot_v,
                    "WaferIndex": title_idx,
                    "UniqueID": meta_row["UniqueID"],
                    "FailureType": ftype,
                    "Main cause of defect": main_cause
                }])
                st.table(info_df)
                return meta_row, arr, ftype, (lot_v, title_idx)
        else:
            st.warning("선택한 UID에 해당하는 행을 찾지 못했습니다. 다시 선택해 주세요.")
    else:
        st.info("위 탭에서 선택 후 **이 웨이퍼 보기** 버튼을 눌러주세요.")
    return None, None, None, None

# ===== 사이드바: 데이터 로드 =====
st.sidebar.markdown("### 데이터 불러오기")
uploaded = st.sidebar.file_uploader("LSWMD.pkl 업로드", type=["pkl","pickle"])
path_hint = st.sidebar.text_input("또는 로컬 경로 입력", value="", placeholder=r"C:\Users\pc4\Desktop\wafer_dataset\LSWMD.pkl").strip().strip('"').strip("'")

# ===== 데이터 로드 =====
df = None; load_err = None
try:
    if uploaded is not None: df = pd.read_pickle(uploaded)
    elif path_hint: df = pd.read_pickle(path_hint)
except Exception as e:
    load_err = str(e)

if load_err: st.error(f"불러오기 실패: {load_err}")
if df is None:
    st.info("왼쪽에서 파일 업로드(200MB 한도) 또는 로컬 경로를 입력하세요."); st.stop()

# ===== 컬럼 매핑 =====
waferMap_col   = first_existing(df.columns, ["waferMap","wafer_map","bitmap","map"])
lotName_col    = first_existing(df.columns, ["lotName","lot","LotID"])
waferIndex_col = first_existing(df.columns, ["waferIndex","waferId","waferno","wafer_number"])
failure_col    = first_existing(df.columns, ["failureType","label","class"])
if waferMap_col is None:
    st.error("웨이퍼 비트맵 컬럼(예: 'waferMap')을 찾지 못했습니다."); st.write("발견된 컬럼:", list(df.columns)); st.stop()

# ===== 메타테이블 =====
lot_str   = df[lotName_col].astype(str) if lotName_col else pd.Series([""]*len(df))
wafer_num = pd.to_numeric(df[waferIndex_col], errors="coerce") if waferIndex_col else pd.Series([np.nan]*len(df))
wafer_str = df[waferIndex_col].astype(str) if waferIndex_col else pd.Series([""]*len(df))
failure_norm = df[failure_col].apply(normalize_failure) if failure_col else pd.Series([""]*len(df))

meta = pd.DataFrame({
    "Lot": lot_str, "WaferIndex": wafer_num, "WaferIndex_str": wafer_str, "FailureType": failure_norm
})
meta["UniqueID"] = meta["Lot"].astype(str) + "_" + meta["WaferIndex_str"].astype(str)
if "Main cause of defect" not in meta.columns: meta["Main cause of defect"] = ""

# 기본 정렬
meta_sorted = meta.sort_values(by=["Lot","WaferIndex","WaferIndex_str"], ascending=[True,True,True],
                               na_position="last").reset_index(drop=True)

# ===== 목록(표) & 정렬/필터 =====
st.markdown("---")
st.subheader("목록(표) & 정렬/필터 옵션")
sort_mode = st.radio("정렬 기준 선택", ["Index 순 (Lot→WaferIndex)", "특정 FailureType만 보기"], index=0, horizontal=True)

ft_all = meta_sorted["FailureType"].fillna("").astype(str).str.strip()
ft_candidates = sorted({ft for ft in ft_all if ft and ft.lower() != "none"})

if sort_mode == "특정 FailureType만 보기":
    ft_sel_for_table = st.selectbox("Failure Type 선택 (none/빈값 제외)", options=ft_candidates if ft_candidates else [])
    if ft_sel_for_table:
        meta_view = meta_sorted[meta_sorted["FailureType"].str.strip() == ft_sel_for_table] \
            .sort_values(by=["Lot","WaferIndex","WaferIndex_str"], ascending=[True,True,True]).reset_index(drop=True)
    else:
        meta_view = meta_sorted.copy()
else:
    meta_view = meta_sorted.copy()

st.markdown(f"**표시 행 수:** {len(meta_view)}")
st.dataframe(meta_view[["Lot","WaferIndex","UniqueID","FailureType","Main cause of defect"]],
             use_container_width=True, height=360)

# ===== 좌(A) | 우(B) 비교 영역 =====
st.markdown("---")
colA, colB = st.columns(2, gap="large")
with colA:
    metaA, arrA, fA, idA = render_one_side("입력 웨이퍼 선택", "uid_A",
                                           meta_sorted, df, waferMap_col, lotName_col, waferIndex_col)
with colB:
    metaB, arrB, fB, idB = render_one_side("비교 웨이퍼 선택", "uid_B",
                                           meta_sorted, df, waferMap_col, lotName_col, waferIndex_col)

# ===== 간단 비교표 =====
if (arrA is not None) and (arrB is not None):
    st.markdown("---")
    st.subheader("비교 요약")
    compare = pd.DataFrame([
        {"측면": "A", "Lot": idA[0], "WaferIndex": idA[1], "FailureType": fA,
         "해상도(h×w)": f"{arrA.shape[0]}×{arrA.shape[1]}"},
        {"측면": "B", "Lot": idB[0], "WaferIndex": idB[1], "FailureType": fB,
         "해상도(h×w)": f"{arrB.shape[0]}×{arrB.shape[1]}"},
    ])
    st.table(compare)
else:
    st.info("좌와우 모두 웨이퍼를 선택하면 비교표가 나타납니다.")
