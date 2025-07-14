# dust_fractal_analyzer.py
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage import exposure
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor

# ------------------------------------------------------------------ Streamlit
st.set_page_config(page_title="Dust Fractal Analyzer", layout="wide")
st.title("🧹 フラクタル埃解析 — 4 指標×4 段階評価")

# ---------------------------------------------------- 1. 画像処理ユーティリティ
def resize_if_large(img: np.ndarray, max_side: int = 1024) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(h, w) / max_side
    return img if scale <= 1.0 else cv2.resize(img, (int(w/scale), int(h/scale)),
                                               interpolation=cv2.INTER_AREA)

def equalize_hist_uniformity(gray: np.ndarray) -> float:
    """ヒストグラム均一度 (1 − 分散/最大分散)。0=不均一, 1=最均一"""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist /= hist.sum()
    variance = ((np.arange(256) - hist.mean()) ** 2 * hist).sum()
    max_var = (255 ** 2) / 4
    return 1.0 - variance / max_var

def preprocess(img_bgr: np.ndarray, block: int, C: int, k: int) -> Tuple[np.ndarray, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bin_adp = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, block, C)
    kernel = np.ones((k, k), np.uint8)
    clean = cv2.morphologyEx(bin_adp, cv2.MORPH_OPEN, kernel, iterations=2)

    _, _, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    mean_size = areas.mean() if areas.size else 0.0
    return clean, mean_size

# ----------------------------------------------- 2. フラクタル & 量ユーティリティ
def box_count(img: np.ndarray, size: int) -> int:
    S = np.add.reduceat(np.add.reduceat(img,
                 np.arange(0, img.shape[0], size), axis=0),
                 np.arange(0, img.shape[1], size), axis=1)
    return np.count_nonzero(S)

def fractal_dimension(mask: np.ndarray):
    min_exp = 1
    max_exp = int(np.log2(min(mask.shape))) - 1
    sizes = 2 ** np.arange(min_exp, max_exp)
    counts = [box_count(mask, s) for s in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0], sizes, counts

def occupancy_rate(mask: np.ndarray) -> float:
    return np.count_nonzero(mask) / mask.size * 100

# ---------------------------------------------------- 3. 総合評価ロジック
LABELS = ["とても汚い", "やや汚い", "やや綺麗", "とても綺麗"]
COLORS = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

def score_to_label(score: float):
    if score < 25:  return LABELS[0], COLORS[0]
    if score < 50:  return LABELS[1], COLORS[1]
    if score < 75:  return LABELS[2], COLORS[2]
    return LABELS[3], COLORS[3]

def compute_scores(img_bgr: np.ndarray, block: int, C: int, k: int):
    clean_mask, mean_size = preprocess(img_bgr, block, C, k)

    with ThreadPoolExecutor() as ex:
        fd_future  = ex.submit(fractal_dimension, clean_mask)
        occ_future = ex.submit(occupancy_rate, clean_mask)
        hist_future= ex.submit(equalize_hist_uniformity,
                               cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))

    fractal_dim, sizes, counts = fd_future.result()
    occ = occ_future.result()
    hist_uni = hist_future.result()

    # 0–100 スケール
    occ_s  = np.clip(occ, 0, 20) * 5
    size_s = np.clip(mean_size, 0, 200)/200*100
    fd_s   = (np.clip(fractal_dim, 1.0, 1.6)-1.0)/0.6*100
    hist_s = hist_uni*100

    total_score = np.mean([occ_s, size_s, fd_s, hist_s])
    label, color = score_to_label(total_score)

    return dict(mask=clean_mask, sizes=sizes, counts=counts,
                occ=occ, mean_size=mean_size, fd=fractal_dim,
                hist_uni=hist_uni, score=total_score,
                label=label, color=color)

# ---------------------------------------------------- 4. Streamlit UI
st.sidebar.header("⚙️ 前処理パラメータ")
block = st.sidebar.slider("適応二値化ブロック", 5, 51, 11, 2)
C_val = st.sidebar.slider("定数 C", 0, 10, 2)
ksz   = st.sidebar.slider("カーネルサイズ", 1, 7, 3)

uploaded = st.sidebar.file_uploader("画像を選択", ["jpg", "png", "jpeg", "bmp"])

if uploaded:
    img_color = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8),
                             cv2.IMREAD_COLOR)
    img_color = resize_if_large(img_color)

    result = compute_scores(img_color, block, C_val, ksz)

    # ① 元画像 & マスク横並び
    col_img, col_mask = st.columns(2)
    with col_img:
        st.subheader("📸 元画像 (縮小表示)")
        st.image(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB),
                 channels="RGB", use_column_width=True)
    with col_mask:
        st.subheader("🧹 二値マスク (縮小表示)")
        st.image(result["mask"], clamp=True, use_column_width=True)

    # ② サマリー
    st.metric("総合スコア", f"{result['score']:.1f} / 100")
    st.markdown(f"<h3 style='color:{result['color']};'>{result['label']}</h3>",
                unsafe_allow_html=True)

    cols = st.columns(4)
    cols[0].metric("空間占有率 %", f"{result['occ']:.2f}")
    cols[1].metric("平均粒径 px", f"{result['mean_size']:.1f}")
    cols[2].metric("フラクタル次元", f"{result['fd']:.3f}")
    cols[3].metric("ヒスト均一度", f"{result['hist_uni']:.2f}")

    # ③ 詳細グラフ
    with st.expander("▼ 詳細グラフ"):
        indic_vals  = [result['occ'],
                       result['mean_size'],
                       result['fd']*100,
                       result['hist_uni']*100]
        indic_names = ["占有率", "平均粒径", "FD×100", "ヒスト均一度×100"]

        fig = go.Figure(go.Scatterpolar(
            r=indic_vals + [indic_vals[0]],
            theta=indic_names + [indic_names[0]],
            fill='toself'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        fig2, ax2 = plt.subplots()
        ax2.plot(np.log(result['sizes']), np.log(result['counts']), 'o-')
        ax2.set_xlabel('log(Box Size)')
        ax2.set_ylabel('log(Count)')
        ax2.set_title(f"Fractal Dimension: {result['fd']:.3f}")
        st.pyplot(fig2)
else:
    st.info("サイドバーから画像をアップロードしてください。")
