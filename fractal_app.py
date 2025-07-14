import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

st.set_page_config(page_title="フラクタル埃解析", layout="wide")
st.title("📐 フラクタル次元による埃解析 Web アプリ")

# -----------------------------------------------------------------------------
# 画像処理ユーティリティ
# -----------------------------------------------------------------------------

def clahe_luminance(img_bgr: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """Lab 色空間の L チャンネルに CLAHE を適用し輝度のダイナミクスを拡張する。"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    L = clahe.apply(L)
    lab_clahe = cv2.merge([L, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def gamma_correct(img_bgr: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """シンプルなガンマ補正 (γ<1: 暗部持ち上げ, γ>1: 明部抑え)"""
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img_bgr, lut)


def preprocess(
    img_bgr: np.ndarray,
    block: int = 11,
    C: int = 2,
    kernel_size: int = 3,
    apply_clahe: bool = True,
    gamma: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """前処理: CLAHE→ガンマ→適応二値化→形態学開閉→Canny でエッジ抽出。

    Returns
    -------
    edges : np.ndarray
        ボックスカウント入力用のエッジ画像 (0/255)
    clean : np.ndarray
        採用前のクリーンな二値マスク (0/255)
    sat_mask : np.ndarray
        白飛び飽和領域マスク (True=飽和)
    """
    img = img_bgr.copy()

    # 1) CLAHE (オプション)
    if apply_clahe:
        img = clahe_luminance(img)

    # 2) ガンマ補正
    if abs(gamma - 1.0) > 1e-3:
        img = gamma_correct(img, gamma)

    # 飽和マスク (灰度 250 以上)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sat_mask = gray > 250

    # 3) 適応二値化 (輝度成分のみ)
    bin_adp = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, C
    )

    # 4) 形態学的開閉で微小ノイズ除去 & 小粒連結
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    clean = cv2.morphologyEx(bin_adp, cv2.MORPH_OPEN, kernel, iterations=2)

    # 5) Canny エッジ抽出
    edges = cv2.Canny(clean, 50, 150)

    # 6) 飽和領域は解析対象外に
    edges[sat_mask] = 0
    clean[sat_mask] = 0

    return edges, clean, sat_mask


# -----------------------------------------------------------------------------
# フラクタル解析ユーティリティ
# -----------------------------------------------------------------------------

def box_count(img: np.ndarray, size: int) -> int:
    """与えられたボックスサイズで非ゼロ領域を数える。"""
    S = np.add.reduceat(np.add.reduceat(img, np.arange(0, img.shape[0], size), axis=0),
                        np.arange(0, img.shape[1], size), axis=1)
    return np.count_nonzero(S)


def evaluate_cleanliness(rate: float) -> str:
    """空間占有率→簡易清潔度ラベル"""
    if rate >= 10:
        return "汚い"
    elif rate >= 1:
        return "やや汚い"
    return "綺麗"


def analyze_image(
    image_bytes: bytes,
    block: int,
    C: int,
    kernel_size: int,
    apply_clahe: bool,
    gamma: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[str], Optional[float], Optional[np.ndarray]]:
    """画像バイト列を解析し結果を返す。失敗時はすべて None"""
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_color is None:
        return (None,) * 6

    edges, clean, sat_mask = preprocess(img_color, block, C, kernel_size, apply_clahe, gamma)

    occupancy = np.count_nonzero(clean) / clean.size * 100
    cleanliness = evaluate_cleanliness(occupancy)

    # ボックスカウント (エッジのみ)
    min_exp = 1  # 2px
    max_exp = int(np.log2(min(edges.shape))) - 1
    if max_exp <= min_exp:
        return (None,) * 6
    sizes = 2 ** np.arange(min_exp, max_exp)
    counts = [box_count(edges, s) for s in sizes]

    # 線形回帰で勾配→フラクタル次元
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    fractal_dim = -coeffs[0]

    return img_color, edges, occupancy, cleanliness, fractal_dim, sat_mask


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.sidebar.header("⚙️ 前処理パラメータ")
block = st.sidebar.slider("適応二値化ブロックサイズ", 5, 51, 11, 2)
C_val = st.sidebar.slider("適応二値化定数 C", 0, 10, 2)
kernel_sz = st.sidebar.slider("形態学カーネルサイズ", 1, 7, 3)
apply_clahe = st.sidebar.checkbox("CLAHE (輝度強調)", True)
gamma_val = st.sidebar.slider("ガンマ補正 γ", 0.4, 1.6, 0.8, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📤 画像アップロード")
uploaded_file = st.sidebar.file_uploader("解析する画像を選択", type=["png", "jpg", "jpeg", "bmp"])

# メイン表示領域
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    (
        img_color,
        edges_img,
        occupancy,
        cleanliness,
        fractal_dim,
        sat_mask,
    ) = analyze_image(
        image_bytes,
        block,
        C_val,
        kernel_sz,
        apply_clahe,
        gamma_val,
    )

    if img_color is None:
        st.error("画像の解析に失敗しました。対応フォーマットか確認してください。")
        st.stop()

    col1, col2 = st.columns(2)

    # 画像表示
    with col1:
        st.subheader("📸 元画像 (カラー)")
        st.image(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB), channels="RGB")

    with col2:
        st.subheader("🧹 抽出エッジ")
        st.image(edges_img, clamp=True)

    # 飽和率チェック
    sat_rate = np.count_nonzero(sat_mask) / sat_mask.size * 100
    if sat_rate > 5:
        st.warning(f"白飛び領域が {sat_rate:.1f}% あります。露出を下げて再撮影すると精度向上します。")

    # ボックスカウントグラフ
    st.subheader("📈 ボックスカウントグラフ")
    min_exp = 1
    max_exp = int(np.log2(min(edges_img.shape))) - 1
    sizes = 2 ** np.arange(min_exp, max_exp)
    counts = [box_count(edges_img, s) for s in sizes]

    fig, ax = plt.subplots()
    ax.plot(np.log(sizes), np.log(counts), "o-", label="Box Counting")
    ax.set_xlabel("log(Box Size)")
    ax.set_ylabel("log(Count)")
    ax.set_title(f"Fractal Dimension: {fractal_dim:.4f}")
    ax.legend()
    st.pyplot(fig)

    # テキスト結果
    st.markdown(
        f"**空間占有率:** {occupancy:.2f}%  ｜  **清潔度評価:** {cleanliness}  ｜  **フラクタル次元:** {fractal_dim:.4f}"
    )
else:
    st.info("サイドバーから解析したい画像をアップロードしてください。")
