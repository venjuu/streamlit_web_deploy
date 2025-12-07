# wafer_pca_utils.py
import numpy as np
from numpy.linalg import norm
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# ============================
# 0. 글로벌 상수 및 유틸 함수
# ============================

# Canonical(정상 라벨) / Unlabeled 정의 (소문자 기준)
CANONICAL_FT = {
    'center', 'donut', 'edge-loc', 'edge-ring', 'loc', 'scratch', 'near-full'
}
UNLABELED_FT = {'random'}

# 자주 헷갈리는 패턴 그룹들
CONFUSABLE_GROUPS = [
    {'edge-loc', 'edge-ring'},
    # {'loc', 'edge-loc'},
    # {'loc', 'donut'},
]


def most_common(lst):
    """리스트에서 최빈값 반환, 비어 있으면 None."""
    cnt = Counter(lst)
    if not cnt:
        return None
    return cnt.most_common(1)[0][0]


# ============================
# 1. PCA 임베딩 생성
# ============================

def build_pca_embeddings(df, map_col='waferMap_denoised', n_components=32):
    """
    웨이퍼맵(26x26)을 flatten해서 PCA 임베딩 생성.

    Parameters
    ----------
    df : pandas.DataFrame
        웨이퍼 데이터 (예: data_denoised, df_demo 등)
    map_col : str
        웨이퍼맵 컬럼 이름 (각 셀에 26x26 배열)
    n_components : int
        PCA 차원 수

    Returns
    -------
    embeddings : np.ndarray, shape (N, n_components)
    pca : sklearn.decomposition.PCA
        학습된 PCA 객체
    idx_array : np.ndarray, shape (N,)
        임베딩 각 행에 대응되는 DataFrame 인덱스
    """
    if map_col not in df.columns:
        raise ValueError(f"'{map_col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

    # 웨이퍼맵을 flatten 해서 (N, 26*26) 벡터로 만들기
    imgs_flat = []
    for wm in df[map_col]:
        arr = np.array(wm, dtype='float32')      # (26, 26)
        arr = arr / 2.0                          # 값 0,1,2 → 0~1 스케일
        imgs_flat.append(arr.flatten())          # (676,)

    X_flat = np.stack(imgs_flat, axis=0)         # (N, 676)
    print("X_flat shape:", X_flat.shape)

    # PCA 학습
    pca = PCA(n_components=n_components, random_state=42)
    embeddings = pca.fit_transform(X_flat)       # (N, n_components)

    idx_array = df.index.to_numpy()
    print("embeddings shape:", embeddings.shape)
    print("설명 분산 비율 (앞 10개):", pca.explained_variance_ratio_[:10])

    return embeddings, pca, idx_array


# ============================
# 2. PCA 기반 유사 웨이퍼 시각화
# ============================

def show_similar_wafermaps_pca(
    df,
    idx_query,
    embeddings,
    df_index_array,
    k=10,
    map_col='waferMap_denoised',
    lot_col='lotName',
    ft_cols=('failureType_clean', 'failureType')
):
    """
    PCA 임베딩 기반 유사 웨이퍼 검색 & 시각화

    Parameters
    ----------
    df : pandas.DataFrame
    idx_query : int
        쿼리 웨이퍼 df 인덱스
    embeddings : np.ndarray, shape (N, d)
        PCA 임베딩 (build_pca_embeddings 결과)
    df_index_array : np.ndarray, shape (N,)
        embeddings 각 행에 대응되는 df 인덱스
    k : int
        top-k 개수
    map_col : str
        웨이퍼맵 컬럼명
    lot_col : str
        lotName 컬럼명
    ft_cols : tuple
        Failure Type 후보 컬럼명들

    Returns
    -------
    fig : matplotlib.figure.Figure
        시각화 Figure (PyCharm: plt.show(), Streamlit: st.pyplot(fig))
    neighbor_df_indices : np.ndarray
        top-k 이웃의 df 인덱스 배열
    neighbor_sims : np.ndarray
        top-k 이웃의 코사인 유사도 배열
    """

    if map_col not in df.columns:
        raise ValueError(f"'{map_col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

    # Failure Type 컬럼 자동 탐색
    ft_col = None
    for col in ft_cols:
        if col in df.columns:
            ft_col = col
            break

    if ft_col is None:
        print("⚠ Failure Type 컬럼이 없어 FT 정보 없이 진행합니다.")

    # 쿼리 위치 찾기
    try:
        pos_q = np.where(df_index_array == idx_query)[0][0]
    except IndexError:
        raise ValueError(f"idx_query={idx_query} 가 embeddings에 대응되지 않습니다.")

    emb_q = embeddings[pos_q]                  # (d,)

    # 코사인 유사도 계산
    num = embeddings @ emb_q                   # (N,)
    denom = (norm(embeddings, axis=1) * norm(emb_q) + 1e-8)
    sims = num / denom                         # -1 ~ 1

    # 자기 자신 제외
    sims_self = sims.copy()
    sims_self[pos_q] = -1.0

    # top-k 인덱스 (embeddings 배열 기준 위치)
    order = np.argsort(-sims_self)
    top_positions = order[:k]

    neighbor_df_indices = df_index_array[top_positions]
    neighbor_sims = sims_self[top_positions]

    # 상대 점수 (0~100 스케일)
    s_min = neighbor_sims.min()
    s_max = neighbor_sims.max()
    rel_scores = (neighbor_sims - s_min) / (s_max - s_min + 1e-8) * 100

    # ---- Figure 생성 (plt.show()는 여기서 호출하지 않음) ----
    n_cols = k + 1
    n_rows = 2
    fig = plt.figure(figsize=(2.5 * (n_cols // 2 + 1), 4 * n_rows))

    # (0) QUERY
    plt.subplot(n_rows, n_cols // 2 + 1, 1)
    query_map = np.array(df.loc[idx_query, map_col])
    plt.imshow(query_map, cmap='gray')

    title = f"QUERY\nidx={idx_query}"
    if lot_col in df.columns:
        title += f"\nlot={df.loc[idx_query, lot_col]}"
    if ft_col:
        title += f"\nFT={df.loc[idx_query, ft_col]}"
    plt.title(title, fontsize=10)
    plt.axis('off')

    # (1~k) neighbors
    for j, (df_i, sim, rscore) in enumerate(zip(neighbor_df_indices, neighbor_sims, rel_scores), start=2):
        plt.subplot(n_rows, n_cols // 2 + 1, j)
        cand_map = np.array(df.loc[df_i, map_col])
        plt.imshow(cand_map, cmap='gray')

        # raw cosine (−1~1)과 상대 점수(0~100)를 둘 다 보여줌
        raw_pct = sim * 100

        if ft_col:
            ft = df.loc[df_i, ft_col]
        else:
            ft = "-"

        if lot_col in df.columns:
            lot = df.loc[df_i, lot_col]
            t = f"idx={df_i}\nlot={lot}\nFT={ft}\nrel={rscore:4.1f} / cos={raw_pct:4.1f}%"
        else:
            t = f"idx={df_i}\nFT={ft}\nrel={rscore:4.1f} / cos={raw_pct:4.1f}%"

        plt.title(t, fontsize=8)
        plt.axis('off')

    plt.tight_layout()

    # ---- 텍스트 요약 ----
    print("=== Top-k Similar Wafers (PCA embedding) ===")
    for rank, (df_i, sim, rscore) in enumerate(zip(neighbor_df_indices, neighbor_sims, rel_scores), start=1):
        lot = df.loc[df_i, lot_col] if lot_col in df.columns else "-"
        ft  = df.loc[df_i, ft_col] if ft_col else "-"
        print(f"{rank:2d}. idx={df_i:4d}, lot={lot:8}, FT={ft:12}, "
              f"rel={rscore:6.2f}, cos={sim*100:6.2f}%")

    return fig, neighbor_df_indices, neighbor_sims


# ============================
# 3. Notice 판단 로직
# ============================

def classify_notice(
    q_label,             # 쿼리의 failureType_clean 값 (문자열)
    top1_cos,            # top-1 코사인 유사도
    pred_ft,             # 이웃 다수결 예측 FT (문자열 또는 None, 소문자/원문 상관없음)
    conf,                # 이웃 중 pred_ft 비율 (0~1, vote_top_k 기준)
    conf_all=None,       # top-k 전체 기준 일관성 (0~1) - 없으면 None
    cos_strong=0.88,
    cos_ok=0.85,
    conf_strong=0.7,
    conf_ok=0.6
):
    """
    쿼리 웨이퍼에 대해 TYPICAL / LABEL_NOISE / NEW_PATTERN / UNCERTAIN / UNLABELED_TO_KNOWN
    등으로 notice를 생성.
    """
    q_str = str(q_label)
    q_low = q_str.lower().strip()

    is_canonical = q_low in CANONICAL_FT
    is_unlabeled = (
        (q_low in UNLABELED_FT) or
        ('none' in q_low) or
        (q_low == '[]')
    )

    # ===== ① 레이블 있는 데이터, Typical한 입력 =====
    if is_canonical and (pred_ft is not None):
        pred_low = pred_ft.lower()
        same_label = (pred_low == q_low)

        # (A) 기본 강한 기준
        strong_cond = (top1_cos >= cos_strong) and (conf >= conf_strong)

        # (B) 보너스 기준: top-k 전체가 한 목소리일 때
        if conf_all is not None:
            bonus_cond = (conf_all >= 0.9) and (top1_cos >= 0.75)
        else:
            bonus_cond = False

        if same_label and (strong_cond or bonus_cond):
            msg = f"이 웨이퍼맵은 '{q_str}' 패턴으로 판단됩니다."
            return "TYPICAL", msg

    # ===== ② Random / unlabeled / none → 기존 FT에 잘 매핑되는 경우 =====
    if is_unlabeled and (pred_ft is not None):
        if (top1_cos >= cos_ok) and (conf >= conf_ok):
            msg = f"이 웨이퍼맵은 '{pred_ft}' 패턴일 가능성이 있습니다."
            return "UNLABELED_TO_KNOWN", msg

    # ===== ③ 레이블 있는 데이터, 라벨노이즈 의심 =====
    if is_canonical and (pred_ft is not None):
        pred_low = pred_ft.lower()

        # (1) 기본(빡센) 조건
        base_cond = (pred_low != q_low) and (top1_cos >= cos_ok) and (conf >= conf_strong)

        # (2) 혼동 그룹(Edge-Loc vs Edge-Ring 등)에 대한 완화 조건
        in_conf_group = any(
            (q_low in g and pred_low in g) for g in CONFUSABLE_GROUPS
        )
        relaxed_cond = (
            in_conf_group and
            (pred_low != q_low) and
            (top1_cos >= 0.90) and   # 유사도는 매우 높고
            (conf >= 0.5)            # 이웃 절반 이상이 pred_ft 쪽이면
        )

        if base_cond or relaxed_cond:
            msg = (
                f"[!] 레이블 오류가 의심됩니다. "
                f"현재 레이블: '{q_str}', 이웃 패턴: '{pred_ft}' "
                f"(이웃 중 약 {conf*100:.1f}%가 '{pred_ft}')"
            )
            return "LABEL_NOISE", msg

    # ===== ④ Random / none / unlabeled + 기존 FT로도 설명 안 되는 경우 =====
    if is_unlabeled:
        if (top1_cos < 0.80) or (conf < 0.5) or (pred_ft is None):
            msg = "[!] 신규 패턴일 가능성이 있습니다."
            return "NEW_PATTERN", msg

    # ===== 그 외 애매한 케이스 =====
    msg = "패턴 판단이 애매합니다. 이웃 웨이퍼들을 함께 검토해 주세요."
    return "UNCERTAIN", msg


# ============================
# 4. PCA 기반 분석 함수
# ============================

def analyze_wafer_pca(
    df,
    idx_query,
    embeddings,
    df_index_array,
    k_neighbors=10,
    vote_top_k=5,
    ft_col_clean='failureType_clean'
):
    """
    - idx_query: 쿼리 웨이퍼 df 인덱스
    - k_neighbors: 이웃 몇 개까지 볼지 (유사도 순)
    - vote_top_k: 다수결에 사용할 상위 이웃 개수

    Returns
    -------
    info : dict
        분석 결과 및 notice 정보
    """
    # 쿼리 위치 찾기
    try:
        pos_q = np.where(df_index_array == idx_query)[0][0]
    except IndexError:
        raise ValueError(f"idx_query={idx_query} 가 embeddings에 대응되지 않습니다.")

    emb_q = embeddings[pos_q]
    emb_norm = embeddings / (norm(embeddings, axis=1, keepdims=True) + 1e-8)
    emb_q_n = emb_q / (norm(emb_q) + 1e-8)

    sims = emb_norm @ emb_q_n        # 코사인 유사도
    sims[pos_q] = -1.0               # 자기 자신 제외

    order = np.argsort(-sims)
    top_pos = order[:k_neighbors]
    neighbor_df_indices = df_index_array[top_pos]
    neighbor_sims = sims[top_pos]

    # 이웃 레이블 수집 (clean 기준)
    if ft_col_clean in df.columns:
        neighbor_labels_raw = [df.loc[i, ft_col_clean] for i in neighbor_df_indices]
    else:
        neighbor_labels_raw = [None] * len(neighbor_df_indices)

    neighbor_labels_str = [str(x) for x in neighbor_labels_raw]
    neighbor_labels_low = [x.lower().strip() for x in neighbor_labels_str]

    # ---- (1) top-k 전체 기준 canonical 일관성 (conf_all) ----
    canon_all = [lab for lab in neighbor_labels_low if lab in CANONICAL_FT]
    if canon_all:
        pred_ft_all = most_common(canon_all)
        conf_all = canon_all.count(pred_ft_all) / len(canon_all)
    else:
        pred_ft_all = None
        conf_all = 0.0

    # ---- (2) vote_top_k 기반 다수결(pred_ft, conf) ----
    k_vote = min(vote_top_k, len(neighbor_labels_low))
    vote_labels = neighbor_labels_low[:k_vote]

    vote_canon = [lab for lab in vote_labels if lab in CANONICAL_FT]
    if vote_canon:
        pred_ft_low = most_common(vote_canon)
        pred_ft = pred_ft_low  # (소문자)
        conf = vote_canon.count(pred_ft_low) / len(vote_canon)
    else:
        pred_ft = None
        conf = 0.0

    # ---- 쿼리 레이블 ----
    if ft_col_clean in df.columns:
        q_label = df.loc[idx_query, ft_col_clean]
    else:
        q_label = "UNKNOWN"

    q_str = str(q_label)
    q_low = q_str.lower().strip()

    is_canonical = q_low in CANONICAL_FT
    is_unlabeled = (
        (q_low in UNLABELED_FT) or
        ('none' in q_low) or
        (q_low == '[]')
    )

    # ---- (3) "최고 유사 이웃이 다른 FT" 체크 ----
    HIGH_SIM_THR = 0.95  # 거의 동일하다고 보는 기준

    # top-1 이웃 정보
    if len(neighbor_sims) > 0:
        top1_cos = float(neighbor_sims[0])
        top1_label_low = neighbor_labels_low[0]
    else:
        top1_cos = 0.0
        top1_label_low = None

    # top-1 이웃이 canonical + 쿼리와 FT가 다름
    high_sim_disagree = (
        is_canonical and
        (top1_label_low is not None) and
        (top1_label_low in CANONICAL_FT) and
        (top1_label_low != q_low) and
        (top1_cos >= HIGH_SIM_THR)
    )

    # top-1 라벨이 q_label과 같은 혼동 그룹에 속하는지?
    if high_sim_disagree:
        in_conf_group_high = any(
            (q_low in g and top1_label_low in g) for g in CONFUSABLE_GROUPS
        )
    else:
        in_conf_group_high = False

    # ---- (4) 기본 notice 평가 ----
    notice_type, notice_msg = classify_notice(
        q_label=q_label,
        top1_cos=top1_cos,
        pred_ft=pred_ft,
        conf=conf,
        conf_all=conf_all
    )

    # ---- (5) "거의 완전히 같은데 FT 다른 이웃"이 있으면 라벨노이즈로 override ----
    if high_sim_disagree and in_conf_group_high:
        # 예: q_label = edge-ring, top1_label_low = edge-loc, cos ~ 1.0
        notice_type = "LABEL_NOISE"
        notice_msg = (
            f"[!] 레이블 오류가 의심됩니다. "
            f"현재 레이블: '{q_str}', "
            f"매우 유사한 이웃(코사인 ≥ {HIGH_SIM_THR:.2f})의 패턴: '{top1_label_low}'"
        )

    # ---- 요약 정보 반환 ----
    info = {
        'idx_query': idx_query,
        'q_label': q_str,
        'top1_idx': int(neighbor_df_indices[0]) if len(neighbor_df_indices) > 0 else None,
        'top1_cos': top1_cos,
        'pred_ft': pred_ft,
        'conf': conf,
        'conf_all': conf_all,
        'neighbor_indices': neighbor_df_indices,
        'neighbor_sims': neighbor_sims,
        'neighbor_labels': neighbor_labels_str,
        'notice_type': notice_type,
        'notice_msg': notice_msg
    }

    return info


# ============================
# 5. 분석 + 시각화 통합 함수
# ============================

def show_with_notice_pca(
    df,
    idx_query,
    embeddings,
    df_index_array,
    k=10,
    map_col='waferMap_denoised',
    lot_col='lotName'
):
    """
    - analyze_wafer_pca로 notice 생성
    - show_similar_wafermaps_pca로 그림 생성
    둘을 한 번에 수행하고 info, fig를 반환.
    """
    # 1) 분석 + notice 생성
    info = analyze_wafer_pca(
        df=df,
        idx_query=idx_query,
        embeddings=embeddings,
        df_index_array=df_index_array,
        k_neighbors=k
    )

    print("=== 분석 요약 ===")
    print(f"- idx_query      : {info['idx_query']}")
    print(f"- 쿼리 레이블    : {info['q_label']}")
    print(f"- top1 이웃 idx  : {info['top1_idx']}")
    print(f"- top1 cos       : {info['top1_cos']:.4f}")
    print(f"- pred_ft        : {info['pred_ft']}")
    print(f"- 이웃 일관성(vote_top_k) : {info['conf']*100:.1f}%")
    print(f"- 이웃 일관성(top-{k})   : {info['conf_all']*100:.1f}%")
    print(f"- notice_type    : {info['notice_type']}")
    print(f"- notice_msg     : {info['notice_msg']}")
    print()

    # 2) PCA 유사도 그림 생성
    fig, _, _ = show_similar_wafermaps_pca(
        df=df,
        idx_query=idx_query,
        embeddings=embeddings,
        df_index_array=df_index_array,
        k=k,
        map_col=map_col,
        lot_col=lot_col
    )

    # 3) 마지막에 메시지 한 번 더 출력
    print(">>> NOTICE:", info['notice_msg'])

    return info, fig
