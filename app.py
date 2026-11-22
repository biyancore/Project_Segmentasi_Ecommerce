"""
Dashboard Segmentasi Pelanggan E-Commerce untuk Manajer CRM.

Model   : K-Means (k = 3) -> model_kmeans.pkl
Scaler  : StandardScaler (fit pada data latih) -> scaler.pkl
Fitur   : Tenure, WarehouseToHome, HourSpendOnApp, OrderCount, CashbackAmount,
          Total_Orders_Engagement (= OrderCount + CouponUsed)

Pemetaan nama segmen TIDAK di-hardcode per nomor klaster, melainkan diturunkan
dari centroid aktual model, sehingga selalu konsisten dengan model_kmeans.pkl
(termasuk setelah retraining yang mengubah urutan nomor klaster).
"""

import io
import re
import warnings
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

# =============================================================================
# KONFIGURASI & KONSTANTA
# =============================================================================
st.set_page_config(
    page_title="Segmentasi Pelanggan E-Commerce | CRM Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_kmeans.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
DATA_PATH = BASE_DIR / "E_Commerce_Dataset.csv"

# Urutan WAJIB sama dengan urutan saat scaler & model dilatih di notebook
FEATURES = ["Tenure", "WarehouseToHome", "HourSpendOnApp", "OrderCount",
            "CashbackAmount", "Total_Orders_Engagement"]
# Kolom yang diisi pengguna (Total_Orders_Engagement dihitung otomatis)
INPUT_COLS = ["Tenure", "WarehouseToHome", "HourSpendOnApp", "OrderCount",
              "CouponUsed", "CashbackAmount"]
# Kolom yang diimputasi median pada notebook (logika cleaning tim)
IMPUTE_COLS = ["Tenure", "WarehouseToHome", "HourSpendOnApp", "OrderAmountHikeFromlastYear",
               "CouponUsed", "OrderCount", "DaySinceLastOrder"]
# Median data mentah (dipakai bila dataset tidak tersedia saat prediksi massal)
FALLBACK_MEDIANS = {"Tenure": 9.0, "WarehouseToHome": 14.0, "HourSpendOnApp": 3.0,
                    "OrderCount": 2.0, "CouponUsed": 1.0, "CashbackAmount": 163.0}

FEATURE_LABELS = {
    "Tenure": "Tenure (bulan)",
    "WarehouseToHome": "Jarak Gudang (km)",
    "HourSpendOnApp": "Jam di Aplikasi",
    "OrderCount": "Jumlah Pesanan",
    "CouponUsed": "Kupon Dipakai",
    "CashbackAmount": "Cashback",
    "Total_Orders_Engagement": "Total Engagement",
    "DaySinceLastOrder": "Hari Sejak Order Terakhir",
}

# Hasil evaluasi dari notebook (Bagian 5.4 & 7.3) — ditampilkan sebagai referensi cepat
NOTEBOOK_EVAL = {
    "kmeans_sil_train": 0.2821, "kmeans_sil_test": 0.2799, "ward_sil_train": 0.3018,
    "ari": 0.5476, "kmeans_sizes": [2862, 643, 999], "ward_sizes": [3342, 756, 406],
}

# Persona segmen. Kunci persona dipetakan ke nomor klaster secara dinamis dari centroid.
PERSONAS = {
    "low": {
        "name": "Pelanggan Baru Beraktivitas Rendah",
        "short": "Pelanggan Baru",
        "icon": "🌱",
        "color": "#D97706",
        "color2": "#F59E0B",
        "theme": "Aktivasi & Retensi Awal",
        "summary": "Masa keanggotaan paling singkat, frekuensi belanja dan cashback paling rendah, "
                   "serta risiko churn paling tinggi.",
        "traits": [
            "Tenure paling rendah — mayoritas baru bergabung",
            "Hanya 1–2 pesanan per bulan, pemakaian kupon rendah",
            "Cashback terendah di antara ketiga segmen",
            "Churn rate tertinggi dan penyumbang terbesar pelanggan yang churn",
        ],
        "actions": [
            "Onboarding journey 90 hari: welcome series, panduan aplikasi, rekomendasi produk awal",
            "Kupon bersyarat untuk pembelian ke-2 dan ke-3 (gratis ongkir / potongan minimum belanja)",
            "Ubah cashback flat menjadi cashback berbasis aksi (hanya jika terjadi repeat order)",
            "Cross-sell aksesori, garansi, dan cicilan untuk kategori elektronik",
            "Dorong migrasi ke aplikasi seluler dengan bonus transaksi pertama di aplikasi",
        ],
        "kpi": "Repeat-order rate 90 hari, churn rate segmen, aktivasi aplikasi",
    },
    "active": {
        "name": "Pembeli Aktif Pemburu Promo",
        "short": "Pembeli Aktif",
        "icon": "🔥",
        "color": "#DC2626",
        "color2": "#F97316",
        "theme": "Optimasi Nilai & Efisiensi Kupon",
        "summary": "Frekuensi pesanan dan penggunaan kupon tertinggi; sangat responsif terhadap promo "
                   "dengan pola belanja borongan lalu jeda.",
        "traits": [
            "Jumlah pesanan dan Total Engagement jauh di atas segmen lain",
            "Penggunaan kupon paling tinggi — sensitif terhadap promo",
            "Jeda sejak pesanan terakhir paling panjang (pola borongan)",
            "Cashback menengah-tinggi",
        ],
        "actions": [
            "Kupon personal bertingkat (tiered) dengan syarat nilai keranjang minimum",
            "Program poin/membership untuk mengurangi ketergantungan pada kupon massal",
            "Reminder repurchase otomatis berbasis riwayat ketika jeda belanja mulai panjang",
            "Bundling & cross-sell Fashion, Grocery, dan Laptop & Accessory untuk menaikkan nilai order",
        ],
        "kpi": "Average order value, biaya kupon per order, frekuensi repeat order",
    },
    "loyal": {
        "name": "Pelanggan Setia Bernilai Tinggi",
        "short": "Pelanggan Setia",
        "icon": "💎",
        "color": "#4F46E5",
        "color2": "#7C3AED",
        "theme": "Apresiasi & Advokasi",
        "summary": "Masa keanggotaan terpanjang dan cashback tertinggi dengan risiko churn paling rendah.",
        "traits": [
            "Tenure paling panjang — pelanggan lama dan setia",
            "Cashback tertinggi di antara ketiga segmen",
            "Frekuensi pesanan sedang namun stabil",
            "Churn rate paling rendah",
        ],
        "actions": [
            "Tier VIP dengan benefit non-diskon: akses awal flash sale, layanan prioritas, gratis ongkir eksklusif",
            "Uji A/B penurunan cashback bertahap untuk menjaga efisiensi anggaran tanpa menaikkan churn",
            "Program referral & ulasan — jadikan pelanggan setia sebagai advokat merek",
            "Langganan / auto-reorder untuk kebutuhan rutin (Grocery) via push notification aplikasi",
        ],
        "kpi": "Retention rate, jumlah referral, rasio cashback terhadap nilai transaksi",
    },
}


# =============================================================================
# STYLE
# =============================================================================
st.markdown(
    """
    <style>
    .block-container {padding-top: 3.4rem; padding-bottom: 3rem; max-width: 1320px;}
    h1, h2, h3, h4 {letter-spacing: -0.01em;}
    .hero {background: linear-gradient(120deg, #1E1B4B 0%, #4338CA 55%, #7C3AED 100%);
           color: #fff; padding: 1.5rem 1.9rem; border-radius: 18px; margin-bottom: 1.1rem;
           box-shadow: 0 12px 32px rgba(67, 56, 202, .25);}
    .hero h1 {color: #fff; font-size: 1.75rem; margin: 0 0 .35rem 0; font-weight: 750; padding: 0;}
    .hero p {color: #E0E7FF; margin: 0; font-size: .98rem; max-width: 860px;}
    .chip {display: inline-block; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.3);
           padding: .18rem .7rem; border-radius: 999px; font-size: .76rem; margin: .75rem .35rem 0 0; color: #fff;}
    div[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px;
           padding: 14px 18px; box-shadow: 0 1px 3px rgba(16,24,40,.06);}
    div[data-testid="stMetricLabel"] p {font-size: .78rem; color: #6B7280; font-weight: 650;
           text-transform: uppercase; letter-spacing: .04em;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem; font-weight: 750; color: #111827;}
    .section-title {font-size: 1.12rem; font-weight: 750; color: #111827; margin: 1.3rem 0 .1rem 0;}
    .section-sub {color: #6B7280; font-size: .88rem; margin-bottom: .55rem;}
    .badge {display: inline-block; padding: .2rem .7rem; border-radius: 999px; font-size: .75rem;
            font-weight: 700; color: #fff; letter-spacing: .02em;}
    .persona-card {border-radius: 16px; padding: 1.05rem 1.2rem 1.15rem 1.2rem; background: #FFFFFF;
            border: 1px solid #E5E7EB; border-top: 5px solid var(--accent); height: 100%;
            box-shadow: 0 1px 3px rgba(16,24,40,.06);}
    .persona-card h3 {margin: .55rem 0 .25rem 0; font-size: 1.06rem; color: #111827; padding: 0;}
    .persona-card .desc {color: #4B5563; font-size: .86rem; margin-bottom: .55rem; min-height: 3.6em;}
    .stat-row {display: flex; justify-content: space-between; font-size: .86rem; padding: .3rem 0;
            border-bottom: 1px dashed #E5E7EB; color: #374151;}
    .stat-row b {color: #111827;}
    .strategy-card {border-radius: 16px; padding: 1.1rem 1.25rem; background: #FFFFFF;
            border: 1px solid #E5E7EB; border-left: 6px solid var(--accent); margin-bottom: .9rem;
            box-shadow: 0 1px 3px rgba(16,24,40,.05);}
    .strategy-card h4 {margin: 0 0 .15rem 0; font-size: 1.02rem; color: #111827; padding: 0;}
    .strategy-card .theme {color: var(--accent); font-weight: 700; font-size: .84rem; margin-bottom: .45rem;}
    .strategy-card ol, .strategy-card ul {margin: .2rem 0 .35rem 0; padding-left: 1.15rem; font-size: .88rem; color: #374151;}
    .strategy-card .kpi {font-size: .8rem; color: #6B7280; background: #F9FAFB; padding: .4rem .6rem; border-radius: 8px;}
    .result-card {border-radius: 18px; padding: 1.35rem 1.6rem; color: #fff;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
            box-shadow: 0 12px 28px rgba(17,24,39,.18);}
    .result-card .label {font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; opacity: .85;}
    .result-card h2 {color: #fff; margin: .15rem 0 .35rem 0; font-size: 1.55rem; padding: 0;}
    .result-card p {margin: 0; opacity: .95; font-size: .93rem;}
    .result-card .pill {display: inline-block; background: rgba(255,255,255,.2); border-radius: 999px;
            padding: .15rem .65rem; font-size: .78rem; margin: .7rem .35rem 0 0;}
    .info-box {background: #EEF2FF; border: 1px solid #C7D2FE; color: #3730A3; border-radius: 12px;
            padding: .7rem .9rem; font-size: .86rem;}
    .stTabs [data-baseweb="tab-list"] {gap: .35rem;}
    .stTabs [data-baseweb="tab"] {background: #F3F4F6; border-radius: 10px 10px 0 0;
            padding: .55rem 1.05rem; font-weight: 650;}
    .stTabs [aria-selected="true"] {background: #EEF2FF; color: #4338CA;}
    section[data-testid="stSidebar"] {background: #F8FAFC; border-right: 1px solid #E5E7EB;}
    .sidebar-brand {font-weight: 800; font-size: 1.05rem; color: #1E1B4B;}
    .sidebar-muted {color: #6B7280; font-size: .8rem;}
    .footer {text-align: center; color: #9CA3AF; font-size: .78rem; margin-top: 2.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# UTILITAS
# =============================================================================
def fmt(x, dec=1, sign=False):
    """Format angka gaya Indonesia: 1.234,5 (sign=True menambahkan tanda +)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    s = f"{x:+,.{dec}f}" if sign else f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def section(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def badge(persona):
    return (f'<span class="badge" style="background:{persona["color"]}">'
            f'{persona["icon"]} {persona["short"]}</span>')


def _norm_col(name):
    """Normalisasi nama kolom: huruf kecil, tanpa spasi/underscore/simbol."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# =============================================================================
# MEMUAT ARTEFAK & DATA
# =============================================================================
@st.cache_resource(show_spinner="Memuat model dan scaler...")
def load_artifacts():
    """Memuat model & scaler, sekaligus memvalidasi kesesuaian keduanya."""
    missing = [p.name for p in (MODEL_PATH, SCALER_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"File artefak tidak ditemukan: {', '.join(missing)}. Letakkan file tersebut "
            f"satu folder dengan app.py (jalankan Bagian 8 notebook untuk membuatnya)."
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    load_warnings = sorted({str(w.message).split(".")[0] for w in caught
                            if "InconsistentVersion" in type(w.message).__name__})

    if not hasattr(model, "cluster_centers_") or not hasattr(model, "predict"):
        raise TypeError("model_kmeans.pkl bukan model K-Means yang sudah dilatih.")
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "transform"):
        raise TypeError("scaler.pkl bukan StandardScaler yang sudah dilatih.")
    if model.n_clusters != 3:
        raise ValueError(f"Aplikasi dirancang untuk 3 klaster, model berisi {model.n_clusters} klaster.")
    for name, obj in (("scaler", scaler), ("model", model)):
        names = getattr(obj, "feature_names_in_", None)
        if names is not None and list(names) != FEATURES:
            raise ValueError(f"Urutan fitur {name} tidak sesuai. Diharapkan {FEATURES}, ditemukan {list(names)}.")
        if obj.n_features_in_ != len(FEATURES):
            raise ValueError(f"{name} mengharapkan {obj.n_features_in_} fitur, aplikasi memakai {len(FEATURES)}.")
    return model, scaler, load_warnings


def scale(df_features):
    """Transformasi dengan scaler, hasil tetap DataFrame bernama kolom (sesuai cara model dilatih)."""
    return pd.DataFrame(scaler.transform(df_features[FEATURES]), columns=FEATURES, index=df_features.index)


def predict(df_features):
    """Mengembalikan (label klaster, jarak ke setiap centroid)."""
    xs = scale(df_features)
    return model.predict(xs), model.transform(xs)


def build_persona_map(centroids):
    """Petakan nomor klaster -> persona berdasarkan centroid aktual (skala asli).
    - Engagement (OrderCount + CouponUsed) tertinggi  -> Pembeli Aktif
    - Dari sisanya, Tenure tertinggi                  -> Pelanggan Setia
    - Sisanya                                         -> Pelanggan Baru Beraktivitas Rendah
    """
    active = int(centroids["Total_Orders_Engagement"].idxmax())
    rest = centroids.drop(index=active)
    loyal = int(rest["Tenure"].idxmax())
    low = int(rest.drop(index=loyal).index[0])
    return {low: "low", active: "active", loyal: "loyal"}


@st.cache_data(show_spinner="Menyiapkan dataset...")
def load_dataset(path_str, _mtime):
    """Membaca dataset dan menerapkan cleaning yang sama dengan notebook. Mengembalikan (df, pesan_error)."""
    path = Path(path_str)
    if not path.exists():
        return None, f"File `{path.name}` tidak ditemukan."
    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # termasuk file rusak / sisa merge conflict
        return None, f"`{path.name}` tidak dapat dibaca ({type(exc).__name__}). Periksa format file."
    raw.columns = raw.columns.str.strip()
    needed = set(IMPUTE_COLS + ["CashbackAmount", "Churn", "Complain", "CityTier", "PreferedOrderCat",
                                "PreferredPaymentMode", "CustomerID"])
    if not needed.issubset(raw.columns):
        return None, f"Kolom dataset tidak lengkap. Hilang: {sorted(needed - set(raw.columns))}"

    df = raw.copy()
    df[IMPUTE_COLS] = df[IMPUTE_COLS].fillna(raw[IMPUTE_COLS].median())       # imputasi median
    df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace(           # penyeragaman CC/COD
        {"CC": "Credit Card", "COD": "Cash on Delivery"})
    df["Total_Orders_Engagement"] = df["OrderCount"] + df["CouponUsed"]       # feature engineering
    return df, None


# --- Muat model (wajib) -------------------------------------------------------
try:
    model, scaler, LOAD_WARNINGS = load_artifacts()
except Exception as exc:
    st.error(f"**Model tidak dapat dimuat.** {exc}")
    st.info("Pastikan `model_kmeans.pkl` dan `scaler.pkl` berasal dari pelatihan yang sama "
            "(Bagian 8 notebook) dan berada di folder yang sama dengan `app.py`.")
    st.stop()

CENTROIDS = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=FEATURES)
CENTROIDS.index.name = "Klaster"
CENTROIDS_Z = pd.DataFrame(model.cluster_centers_, columns=FEATURES)
PERSONA_MAP = build_persona_map(CENTROIDS)
CLUSTER_IDS = sorted(PERSONA_MAP)


def persona_of(cluster_id):
    return PERSONAS[PERSONA_MAP[int(cluster_id)]]


def segment_label(cluster_id):
    p = persona_of(cluster_id)
    return f"Klaster {cluster_id} · {p['short']}"


SEGMENT_DOMAIN = [segment_label(c) for c in CLUSTER_IDS]
SEGMENT_RANGE = [persona_of(c)["color"] for c in CLUSTER_IDS]
SEGMENT_SCALE = alt.Scale(domain=SEGMENT_DOMAIN, range=SEGMENT_RANGE)

# --- Muat dataset (opsional: dashboard butuh, prediksi tetap jalan tanpanya) ----
_mtime = DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0
DATA, DATA_ERROR = load_dataset(str(DATA_PATH), _mtime)
if DATA is not None:
    DATA = DATA.copy()
    DATA["Klaster"], _ = predict(DATA)
    DATA["Segmen"] = DATA["Klaster"].map(segment_label)
    MEDIANS = {c: float(DATA[c].median()) for c in INPUT_COLS}
    RANGES = {c: (float(DATA[c].min()), float(DATA[c].max())) for c in INPUT_COLS}
else:
    MEDIANS = FALLBACK_MEDIANS
    RANGES = {}


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🛍️ CRM Segmentation Suite</div>'
                '<div class="sidebar-muted">Segmentasi pelanggan berbasis perilaku belanja & aktivitas digital</div>',
                unsafe_allow_html=True)
    st.divider()

    st.markdown("**🧠 Informasi Model**")
    st.markdown(
        f"""
        - **Algoritma:** K-Means (k = {model.n_clusters})
        - **Preprocessing:** StandardScaler
        - **Data latih:** {fmt(scaler.n_samples_seen_, 0)} pelanggan (split 80:20)
        - **Fitur:** {len(FEATURES)} fitur perilaku
        - **scikit-learn runtime:** {sklearn.__version__}
        """
    )
    for c in CLUSTER_IDS:
        st.markdown(badge(persona_of(c)) + f" &nbsp;<span class='sidebar-muted'>Klaster {c}</span>",
                    unsafe_allow_html=True)
    if LOAD_WARNINGS:
        st.warning("Versi scikit-learn saat ini berbeda dari versi saat model disimpan. "
                   "Samakan versi di `requirements.txt` agar hasil terjamin konsisten.", icon="⚠️")

    st.divider()
    st.markdown("**🎛️ Filter Dashboard** <span class='sidebar-muted'>(Tab Executive Dashboard)</span>",
                unsafe_allow_html=True)
    if DATA is not None:
        tier_opts = sorted(DATA["CityTier"].unique().tolist())
        sel_tier = st.multiselect("City Tier", tier_opts, default=tier_opts,
                                  format_func=lambda t: f"Tier {t}")
        cat_opts = DATA["PreferedOrderCat"].value_counts().index.tolist()
        sel_cat = st.multiselect("Kategori Pesanan Favorit", cat_opts, default=cat_opts)
        sel_seg = st.multiselect("Segmen", CLUSTER_IDS, default=CLUSTER_IDS, format_func=segment_label)
    else:
        st.caption("Filter nonaktif karena dataset belum tersedia.")
        sel_tier, sel_cat, sel_seg = [], [], []

    st.divider()
    with st.expander("📖 Panduan Singkat", expanded=False):
        st.markdown(
            """
            1. **Executive Dashboard** — pantau KPI dan distribusi pelanggan; gunakan filter di atas.
            2. **Profil Segmen & Strategi** — bandingkan karakteristik tiap klaster dan baca rekomendasi CRM.
            3. **Prediksi Segmen** — prediksi satu pelanggan lewat formulir, atau banyak pelanggan sekaligus
               dengan mengunggah CSV, lalu unduh hasilnya.

            *Total Engagement* = `OrderCount` + `CouponUsed` dihitung otomatis.
            """
        )
    st.markdown("<div class='sidebar-muted'>Data: Kaggle — E-commerce Customer Churn Analysis "
                "and Prediction (Ankit Verma), CC BY-NC-SA 4.0.</div>", unsafe_allow_html=True)


# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    """
    <div class="hero">
      <h1>🛍️ Dashboard Segmentasi Pelanggan E-Commerce</h1>
      <p>Segmentasi pelanggan berdasarkan perilaku belanja dan aktivitas digital menggunakan K-Means Clustering
      untuk mendukung strategi personalisasi pemasaran, efisiensi anggaran cashback/kupon, dan program retensi.</p>
      <span class="chip">CRISP-DM</span><span class="chip">K-Means · k = 3</span>
      <span class="chip">StandardScaler</span><span class="chip">6 fitur perilaku</span>
      <span class="chip">Stakeholder: Manajer CRM</span>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_dash, tab_profile, tab_predict = st.tabs(
    ["📊 Executive Dashboard & EDA", "🧩 Profil Segmen & Strategi CRM", "🎯 Prediksi Segmen"]
)


# =============================================================================
# TAB 1 — EXECUTIVE DASHBOARD & EDA
# =============================================================================
with tab_dash:
    if DATA is None:
        st.warning(f"Dashboard membutuhkan dataset. {DATA_ERROR} "
                   f"Fitur prediksi pada tab **Prediksi Segmen** tetap dapat digunakan.", icon="📂")
    else:
        view = DATA[DATA["CityTier"].isin(sel_tier) & DATA["PreferedOrderCat"].isin(sel_cat)
                    & DATA["Klaster"].isin(sel_seg)]
        is_filtered = len(view) != len(DATA)

        if view.empty:
            st.warning("Tidak ada pelanggan yang sesuai dengan kombinasi filter. Longgarkan filter di sidebar.")
        else:
            if is_filtered:
                st.caption(f"Menampilkan **{fmt(len(view), 0)}** dari {fmt(len(DATA), 0)} pelanggan "
                           f"sesuai filter. Delta dibandingkan seluruh pelanggan.")

            def delta(col, dec=1):
                if not is_filtered:
                    return None
                return f"{fmt(view[col].mean() - DATA[col].mean(), dec, sign=True)} vs total"

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Pelanggan", fmt(len(view), 0),
                      delta=f"{fmt(len(view) / len(DATA) * 100)}% dari total" if is_filtered else None,
                      delta_color="off", delta_arrow="off")
            k2.metric("Rata-rata Tenure", f"{fmt(view['Tenure'].mean())} bln", delta=delta("Tenure"))
            k3.metric("Rata-rata Order", f"{fmt(view['OrderCount'].mean(), 2)}", delta=delta("OrderCount", 2))
            k4.metric("Rata-rata Cashback", fmt(view["CashbackAmount"].mean()), delta=delta("CashbackAmount"))

            k5, k6, k7, k8 = st.columns(4)
            k5.metric("Churn Rate", f"{fmt(view['Churn'].mean() * 100)}%",
                      delta=(f"{fmt((view['Churn'].mean() - DATA['Churn'].mean()) * 100, sign=True)} pp"
                             if is_filtered else None), delta_color="inverse")
            k6.metric("Complain Rate", f"{fmt(view['Complain'].mean() * 100)}%",
                      delta=(f"{fmt((view['Complain'].mean() - DATA['Complain'].mean()) * 100, sign=True)} pp"
                             if is_filtered else None), delta_color="inverse")
            k7.metric("Rata-rata Kupon", fmt(view["CouponUsed"].mean(), 2), delta=delta("CouponUsed", 2))
            k8.metric("Jam di Aplikasi", f"{fmt(view['HourSpendOnApp'].mean(), 2)} jam",
                      delta=delta("HourSpendOnApp", 2))

            # --- Komposisi segmen & risiko churn -----------------------------------
            c1, c2 = st.columns([1, 1.25])
            seg_stats = (view.groupby("Segmen")
                         .agg(Pelanggan=("CustomerID", "count"), Churn=("Churn", "mean"),
                              Complain=("Complain", "mean"))
                         .reset_index())
            seg_stats["Proporsi"] = seg_stats["Pelanggan"] / seg_stats["Pelanggan"].sum()
            with c1:
                section("Komposisi Segmen", "Proporsi pelanggan pada setiap segmen hasil K-Means")
                donut = alt.Chart(seg_stats).mark_arc(innerRadius=70, cornerRadius=4).encode(
                    theta=alt.Theta("Pelanggan:Q"),
                    color=alt.Color("Segmen:N", scale=SEGMENT_SCALE, legend=alt.Legend(orient="bottom", columns=1, title=None)),
                    tooltip=["Segmen", alt.Tooltip("Pelanggan:Q", format=","),
                             alt.Tooltip("Proporsi:Q", format=".1%")],
                ).properties(height=320)
                st.altair_chart(donut, width="stretch")
            with c2:
                section("Risiko Churn & Komplain per Segmen", "Churn tidak dipakai saat pelatihan — "
                        "berfungsi sebagai validasi bisnis segmen")
                risk = seg_stats.melt(id_vars="Segmen", value_vars=["Churn", "Complain"],
                                      var_name="Indikator", value_name="Rate")
                risk["Indikator"] = risk["Indikator"].map({"Churn": "Churn Rate", "Complain": "Complain Rate"})
                risk_chart = alt.Chart(risk).mark_bar(cornerRadiusEnd=4).encode(
                    y=alt.Y("Segmen:N", title=None, sort=SEGMENT_DOMAIN, axis=alt.Axis(labelLimit=220)),
                    x=alt.X("Rate:Q", title="Persentase pelanggan", axis=alt.Axis(format="%")),
                    yOffset="Indikator:N",
                    color=alt.Color("Indikator:N", scale=alt.Scale(range=["#EF4444", "#94A3B8"]),
                                    legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["Segmen", "Indikator", alt.Tooltip("Rate:Q", format=".1%")],
                ).properties(height=320)
                st.altair_chart(risk_chart, width="stretch")

            # --- Distribusi fitur numerik --------------------------------------------
            section("Distribusi Fitur Perilaku", "Pilih fitur untuk melihat sebarannya per segmen")
            num_options = ["Tenure", "OrderCount", "CouponUsed", "CashbackAmount", "HourSpendOnApp",
                           "WarehouseToHome", "DaySinceLastOrder", "Total_Orders_Engagement"]
            f1, f2 = st.columns([3, 1])
            with f2:
                feat = st.selectbox("Fitur", num_options, index=3, format_func=lambda c: FEATURE_LABELS.get(c, c))
                stacked = st.toggle("Pisahkan per segmen", value=True)
                st.markdown(
                    f"""<div class="info-box">Median: <b>{fmt(view[feat].median())}</b><br>
                    Rata-rata: <b>{fmt(view[feat].mean())}</b><br>
                    Min–Maks: <b>{fmt(view[feat].min())} – {fmt(view[feat].max())}</b></div>""",
                    unsafe_allow_html=True)
            with f1:
                base = alt.Chart(view[[feat, "Segmen"]]).mark_bar(opacity=.9).encode(
                    x=alt.X(f"{feat}:Q", bin=alt.Bin(maxbins=30), title=FEATURE_LABELS.get(feat, feat)),
                    y=alt.Y("count():Q", title="Jumlah pelanggan"),
                    tooltip=[alt.Tooltip("count():Q", title="Pelanggan")],
                )
                hist = base.encode(color=alt.Color("Segmen:N", scale=SEGMENT_SCALE,
                                                   legend=alt.Legend(orient="top", title=None))) \
                    if stacked else base.encode(color=alt.value("#6366F1"))
                st.altair_chart(hist.properties(height=330), width="stretch")

            # --- Distribusi kategorikal & scatter -------------------------------------
            c3, c4 = st.columns(2)
            with c3:
                section("Kategori Pesanan Favorit", "Komposisi segmen pada setiap kategori")
                cat_chart = alt.Chart(view[["PreferedOrderCat", "Segmen"]]).mark_bar(cornerRadiusEnd=3).encode(
                    y=alt.Y("PreferedOrderCat:N", sort="-x", title=None),
                    x=alt.X("count():Q", title="Jumlah pelanggan"),
                    color=alt.Color("Segmen:N", scale=SEGMENT_SCALE, legend=None),
                    tooltip=["PreferedOrderCat", "Segmen", alt.Tooltip("count():Q", title="Pelanggan")],
                ).properties(height=300)
                st.altair_chart(cat_chart, width="stretch")
            with c4:
                section("Tenure vs Cashback", "Sampel maksimal 1.500 pelanggan agar ringan dimuat")
                sample = view.sample(min(1500, len(view)), random_state=42)
                scatter = alt.Chart(sample).mark_circle(size=38, opacity=.6).encode(
                    x=alt.X("Tenure:Q", title="Tenure (bulan)"),
                    y=alt.Y("CashbackAmount:Q", title="Cashback", scale=alt.Scale(zero=False)),
                    color=alt.Color("Segmen:N", scale=SEGMENT_SCALE, legend=None),
                    tooltip=["CustomerID", "Segmen", "Tenure", "CashbackAmount", "OrderCount"],
                ).properties(height=300).interactive()
                st.altair_chart(scatter, width="stretch")

            c5, c6 = st.columns(2)
            with c5:
                section("Metode Pembayaran", "Setelah penyeragaman label CC/COD")
                pay = (view["PreferredPaymentMode"].value_counts().rename_axis("Metode")
                       .reset_index(name="Pelanggan"))
                st.bar_chart(pay, x="Metode", y="Pelanggan", horizontal=True, color="#6366F1", height=260, sort="-Pelanggan")
            with c6:
                section("Perangkat Login", "Kanal akses yang paling sering digunakan")
                dev = (view["PreferredLoginDevice"].value_counts().rename_axis("Perangkat")
                       .reset_index(name="Pelanggan"))
                st.bar_chart(dev, x="Perangkat", y="Pelanggan", horizontal=True, color="#0EA5E9", height=260, sort="-Pelanggan")

            with st.expander("🔎 Lihat & unduh data pelanggan (sesuai filter)"):
                show_cols = ["CustomerID", "Segmen", "Klaster"] + INPUT_COLS + [
                    "Total_Orders_Engagement", "CityTier", "PreferedOrderCat", "PreferredPaymentMode",
                    "Complain", "Churn"]
                st.dataframe(view[show_cols], hide_index=True, height=320)
                st.download_button("⬇️ Unduh CSV", view[show_cols].to_csv(index=False).encode("utf-8-sig"),
                                   file_name="pelanggan_tersegmentasi.csv", mime="text/csv")


# =============================================================================
# TAB 2 — PROFIL SEGMEN & STRATEGI CRM
# =============================================================================
@st.cache_data(show_spinner="Menghitung evaluasi model (K-Means vs Hierarchical)...")
def live_evaluation(df_features):
    """Reproduksi split notebook lalu hitung Silhouette K-Means vs Ward secara live."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.model_selection import train_test_split

    x_train, x_test = train_test_split(df_features[FEATURES], test_size=0.2, random_state=42)
    if len(x_train) != int(scaler.n_samples_seen_) or not np.allclose(x_train.mean().values, scaler.mean_):
        return None
    xs_train, xs_test = scale(x_train), scale(x_test)
    km_train = model.predict(xs_train)
    ward = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(xs_train)
    return {
        "kmeans_sil_train": silhouette_score(xs_train, km_train),
        "kmeans_sil_test": silhouette_score(xs_test, model.predict(xs_test)),
        "ward_sil_train": silhouette_score(xs_train, ward),
        "ari": adjusted_rand_score(km_train, ward),
        "kmeans_sizes": np.bincount(km_train).tolist(),
        "ward_sizes": np.bincount(ward).tolist(),
    }


with tab_profile:
    # --- Kartu persona ----------------------------------------------------------
    section("Persona Segmen Pelanggan", "Nama segmen diturunkan otomatis dari centroid model_kmeans.pkl")
    if DATA is not None:
        prof = DATA.groupby("Klaster").agg(
            n=("CustomerID", "count"), churn=("Churn", "mean"), tenure=("Tenure", "mean"),
            orders=("OrderCount", "mean"), cashback=("CashbackAmount", "mean"))
        prof["share"] = prof["n"] / prof["n"].sum()
    cols = st.columns(3)
    for col, cid in zip(cols, CLUSTER_IDS):
        p = persona_of(cid)
        if DATA is not None:
            r = prof.loc[cid]
            stats = (f'<div class="stat-row"><span>Pelanggan</span><b>{fmt(r.n, 0)} ({fmt(r.share * 100)}%)</b></div>'
                     f'<div class="stat-row"><span>Rata-rata tenure</span><b>{fmt(r.tenure)} bln</b></div>'
                     f'<div class="stat-row"><span>Rata-rata order</span><b>{fmt(r.orders, 2)}</b></div>'
                     f'<div class="stat-row"><span>Rata-rata cashback</span><b>{fmt(r.cashback)}</b></div>'
                     f'<div class="stat-row"><span>Churn rate</span><b>{fmt(r.churn * 100)}%</b></div>')
        else:
            c_ = CENTROIDS.loc[cid]
            stats = (f'<div class="stat-row"><span>Centroid tenure</span><b>{fmt(c_.Tenure)} bln</b></div>'
                     f'<div class="stat-row"><span>Centroid order</span><b>{fmt(c_.OrderCount, 2)}</b></div>'
                     f'<div class="stat-row"><span>Centroid cashback</span><b>{fmt(c_.CashbackAmount)}</b></div>')
        col.markdown(
            f"""<div class="persona-card" style="--accent:{p['color']}">
                {badge(p)} <span class="sidebar-muted">Klaster {cid}</span>
                <h3>{p['icon']} {p['name']}</h3>
                <div class="desc">{p['summary']}</div>{stats}</div>""",
            unsafe_allow_html=True)

    # --- Tabel perbandingan ----------------------------------------------------
    section("Perbandingan Karakteristik Rata-rata per Klaster",
            "🟩 nilai tertinggi &nbsp;·&nbsp; 🟥 nilai terendah pada setiap kolom")
    if DATA is not None:
        table = DATA.groupby("Klaster").agg(
            **{"Jumlah Pelanggan": ("CustomerID", "count"),
               "Tenure": ("Tenure", "mean"),
               "WarehouseToHome": ("WarehouseToHome", "mean"),
               "HourSpendOnApp": ("HourSpendOnApp", "mean"),
               "OrderCount": ("OrderCount", "mean"),
               "CouponUsed": ("CouponUsed", "mean"),
               "Total_Orders_Engagement": ("Total_Orders_Engagement", "mean"),
               "CashbackAmount": ("CashbackAmount", "mean"),
               "DaySinceLastOrder": ("DaySinceLastOrder", "mean"),
               "Churn Rate (%)": ("Churn", lambda s: s.mean() * 100),
               "Complain Rate (%)": ("Complain", lambda s: s.mean() * 100),
               "SatisfactionScore": ("SatisfactionScore", "mean")})
        source_note = "Rata-rata aktual seluruh pelanggan dataset yang dilabeli oleh model."
    else:
        table = CENTROIDS.copy()
        source_note = "Dataset tidak tersedia — menampilkan centroid model (skala asli)."
    table.index = [segment_label(i) for i in table.index]
    styled = (table.style
              .highlight_max(axis=0, props="background-color:#DCFCE7; color:#166534; font-weight:700;")
              .highlight_min(axis=0, props="background-color:#FEE2E2; color:#991B1B; font-weight:700;")
              .format(precision=2, thousands=".", decimal=","))
    if "Jumlah Pelanggan" in table.columns:
        styled = styled.format(precision=0, thousands=".", decimal=",", subset=["Jumlah Pelanggan"])
    st.dataframe(styled, width="stretch")
    st.caption(source_note)

    c1, c2 = st.columns([1.3, 1])
    with c1:
        section("Profil Centroid (z-score)", "0 = rata-rata pelanggan data latih; ±1 = satu standar deviasi")
        cz = CENTROIDS_Z.copy()
        cz["Segmen"] = [segment_label(i) for i in cz.index]
        cz_long = cz.melt(id_vars="Segmen", var_name="Fitur", value_name="z")
        cz_long["Fitur"] = cz_long["Fitur"].map(FEATURE_LABELS)
        zchart = alt.Chart(cz_long).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("Fitur:N", title=None, sort=[FEATURE_LABELS[f] for f in FEATURES], axis=alt.Axis(labelLimit=200)),
            x=alt.X("z:Q", title="z-score centroid"),
            yOffset=alt.YOffset("Segmen:N", sort=SEGMENT_DOMAIN),
            color=alt.Color("Segmen:N", scale=SEGMENT_SCALE,
                            legend=alt.Legend(orient="bottom", title=None, columns=1, labelLimit=260)),
            tooltip=["Segmen", "Fitur", alt.Tooltip("z:Q", format=".2f")],
        ).properties(height=380)
        st.altair_chart(zchart, width="stretch")
    with c2:
        section("Evaluasi Model", "K-Means (model final) vs Hierarchical Clustering (Ward)")
        use_live = st.toggle("Hitung ulang secara live dari dataset", value=False,
                             disabled=DATA is None,
                             help="Mereproduksi split 80:20 (random_state=42) seperti notebook lalu "
                                  "menghitung Silhouette & Adjusted Rand Index. Memerlukan beberapa detik.")
        ev = NOTEBOOK_EVAL
        if use_live:
            live = live_evaluation(DATA[FEATURES])
            if live is None:
                st.warning("Data latih hasil reproduksi tidak cocok dengan scaler.pkl — menampilkan hasil notebook.")
            else:
                ev = live
        m1, m2 = st.columns(2)
        m1.metric("Silhouette K-Means", fmt(ev["kmeans_sil_train"], 4), border=True)
        m2.metric("Silhouette Ward", fmt(ev["ward_sil_train"], 4), border=True)
        m3, m4 = st.columns(2)
        m3.metric("Silhouette K-Means (uji)", fmt(ev["kmeans_sil_test"], 4), border=True)
        m4.metric("Adjusted Rand Index", fmt(ev["ari"], 4), border=True)
        eval_df = pd.DataFrame({
            "Aspek": ["Ukuran klaster (data latih)", "Prediksi pelanggan baru", "Stabilitas data uji", "Status"],
            "K-Means": [" / ".join(fmt(v, 0) for v in ev["kmeans_sizes"]), "Ya — predict()",
                        f"Silhouette {fmt(ev['kmeans_sil_test'], 4)}", "✅ Model final"],
            "Hierarchical (Ward)": [" / ".join(fmt(v, 0) for v in ev["ward_sizes"]), "Tidak — harus dilatih ulang",
                                    "Tidak dapat diuji", "Pembanding"],
        })
        st.dataframe(eval_df, hide_index=True, width="stretch")
        st.caption(("Sumber: perhitungan live. " if ev is not NOTEBOOK_EVAL else "Sumber: notebook (Bagian 5.4 & 7.3). ")
                   + "Ward sedikit lebih tinggi pada Silhouette, tetapi K-Means dipilih karena dapat memprediksi "
                     "pelanggan baru, segmennya lebih seimbang, dan stabil di data uji.")

    # --- Strategi CRM -----------------------------------------------------------
    section("Actionable Marketing Strategy", "Rekomendasi program CRM untuk setiap segmen")
    s_cols = st.columns(3)
    for col, cid in zip(s_cols, CLUSTER_IDS):
        p = persona_of(cid)
        actions = "".join(f"<li>{a}</li>" for a in p["actions"])
        col.markdown(
            f"""<div class="strategy-card" style="--accent:{p['color']}">
                <h4>{p['icon']} Klaster {cid} — {p['short']}</h4>
                <div class="theme">🎯 {p['theme']}</div>
                <ol>{actions}</ol>
                <div class="kpi">📈 <b>KPI:</b> {p['kpi']}</div></div>""",
            unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">🤝 <b>Program lintas segmen:</b> pelanggan yang pernah komplain memiliki churn '
        'sekitar 3× lebih tinggi daripada yang tidak komplain, dan complain rate hampir sama di semua segmen. '
        'Terapkan SLA penanganan komplain cepat untuk seluruh segmen.</div>', unsafe_allow_html=True)

    with st.expander("🔬 Verifikasi konsistensi model (untuk tim data)"):
        st.markdown("Centroid aktual `model_kmeans.pkl` dalam skala asli (hasil `scaler.inverse_transform`). "
                    "Pemetaan persona: *engagement* tertinggi → Pembeli Aktif; dari sisanya *tenure* tertinggi → "
                    "Pelanggan Setia; sisanya → Pelanggan Baru.")
        ver = CENTROIDS.round(2).copy()
        ver.insert(0, "Persona", [persona_of(i)["name"] for i in ver.index])
        st.dataframe(ver, width="stretch")
        self_check = model.predict(pd.DataFrame(model.cluster_centers_, columns=FEATURES))
        st.markdown(
            f"- Urutan fitur scaler & model: `{FEATURES}` ✅\n"
            f"- Centroid diprediksi ke klasternya sendiri: {'✅' if (self_check == np.arange(3)).all() else '❌'}\n"
            f"- Scaler dilatih pada {fmt(scaler.n_samples_seen_, 0)} baris (data latih)")


# =============================================================================
# TAB 3 — PREDIKSI SEGMEN
# =============================================================================
def comparison_chart(row_values, cid):
    """Grafik batang z-score: profil pelanggan vs rata-rata (centroid) klaster hasil prediksi."""
    z_input = scale(pd.DataFrame([row_values], columns=FEATURES)).iloc[0]
    z_cent = CENTROIDS_Z.loc[cid]
    p = persona_of(cid)
    df_cmp = pd.DataFrame({
        "Fitur": [FEATURE_LABELS[f] for f in FEATURES] * 2,
        "Sumber": ["Pelanggan ini"] * len(FEATURES) + [f"Rata-rata Klaster {cid}"] * len(FEATURES),
        "z": list(z_input.values) + list(z_cent.values),
    })
    return alt.Chart(df_cmp).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("Fitur:N", title=None, sort=[FEATURE_LABELS[f] for f in FEATURES]),
        x=alt.X("z:Q", title="z-score (0 = rata-rata pelanggan)"),
        yOffset="Sumber:N",
        color=alt.Color("Sumber:N", scale=alt.Scale(range=["#111827", p["color"]]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Sumber", "Fitur", alt.Tooltip("z:Q", format=".2f")],
    ).properties(height=330)


def read_uploaded_csv(uploaded):
    raw = uploaded.getvalue()
    if not raw.strip():
        raise ValueError("File CSV kosong.")
    last_err = None
    for enc in ("utf-8-sig", "latin-1"):
        try:  # sep=None -> deteksi otomatis koma / titik koma (CSV Excel Indonesia)
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError as exc:
            last_err = exc
        except Exception as exc:
            raise ValueError(f"Format CSV tidak dapat dibaca ({type(exc).__name__}: {exc}).") from exc
    raise ValueError(f"Encoding file tidak didukung ({last_err}).")


def predict_batch(df_in):
    """Validasi, imputasi, dan prediksi massal. Mengembalikan (hasil, ringkasan)."""
    lookup = {_norm_col(c): c for c in INPUT_COLS + ["Total_Orders_Engagement"]}
    rename, used = {}, set()
    for col in df_in.columns:
        target = lookup.get(_norm_col(col))
        if target and target not in used:
            rename[col] = target
            used.add(target)
    df = df_in.rename(columns=rename)
    missing = [c for c in INPUT_COLS if c not in df.columns]
    if missing:
        raise KeyError(missing)

    num = df[INPUT_COLS].apply(pd.to_numeric, errors="coerce")
    non_numeric = int((num.isna() & df[INPUT_COLS].notna()).sum().sum())
    empty_cells = int(df[INPUT_COLS].isna().sum().sum())
    all_empty = num.isna().all(axis=1)          # baris tanpa satu pun nilai valid
    rows_imputed = num.isna().any(axis=1) & ~all_empty
    negative = (num < 0).any(axis=1)
    num = num.fillna(MEDIANS)  # imputasi median, konsisten dengan cleaning notebook
    num["Total_Orders_Engagement"] = num["OrderCount"] + num["CouponUsed"]

    engagement_mismatch = 0
    if "Total_Orders_Engagement" in df.columns:
        given = pd.to_numeric(df["Total_Orders_Engagement"], errors="coerce")
        engagement_mismatch = int((given.notna() & (given != num["Total_Orders_Engagement"])).sum())

    out = df_in.copy()
    out["Total_Orders_Engagement_Hitung"] = num["Total_Orders_Engagement"]
    out["Klaster"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["Segmen"] = "Tidak diprediksi"
    out["Strategi_CRM"] = "-"
    out["Status_Data"] = np.where(rows_imputed, "Diimputasi median", "Valid")
    out.loc[negative, "Status_Data"] = "Ditolak: nilai negatif"
    out.loc[all_empty, "Status_Data"] = "Ditolak: baris kosong"

    valid = ~negative & ~all_empty
    if valid.any():
        labels, _ = predict(num.loc[valid, FEATURES])
        out.loc[valid, "Klaster"] = labels
        out.loc[valid, "Segmen"] = [persona_of(c)["name"] for c in labels]
        out.loc[valid, "Strategi_CRM"] = [persona_of(c)["theme"] for c in labels]
    summary = {"total": len(out), "predicted": int(valid.sum()), "rejected": int((~valid).sum()),
               "imputed_rows": int((rows_imputed & valid).sum()), "non_numeric": non_numeric,
               "empty_cells": empty_cells, "mismatch": engagement_mismatch, "renamed": rename,
               "features": num}
    return out, summary


with tab_predict:
    mode = st.radio("Mode prediksi", ["👤 Prediksi Tunggal", "📁 Prediksi Massal (Batch CSV)"],
                    horizontal=True, label_visibility="collapsed")

    # ------------------------------------------------------------------ TUNGGAL
    if mode.startswith("👤"):
        section("Input Profil Pelanggan Baru", "Total Engagement (OrderCount + CouponUsed) dihitung otomatis")
        with st.form("single_form", border=True):
            col1, col2 = st.columns(2)
            with col1:
                tenure = st.number_input("Masa Keanggotaan / Tenure (Bulan)", min_value=0.0, max_value=100.0, value=12.0)
                warehouse_to_home = st.number_input("Jarak Gudang ke Rumah (KM)", min_value=0.0, max_value=200.0, value=15.0)
                hour_spend = st.number_input("Waktu Akses Aplikasi (Jam)", min_value=0.0, max_value=24.0, value=3.0)
            with col2:
                order_count = st.number_input("Total Pesanan Bulan Lalu (OrderCount)", min_value=0, max_value=100, value=5)
                coupon_used = st.number_input("Jumlah Kupon Digunakan Bulan Lalu (CouponUsed)", min_value=0, max_value=100, value=1)
                cashback_amount = st.number_input("Rata-rata Cashback (Mata Uang)", min_value=0.0, max_value=500.0, value=150.0)
            submitted = st.form_submit_button("🚀 Prediksi Segmen Pelanggan", type="primary", width="stretch")

        if submitted:
            # Feature engineering otomatis + susun sesuai urutan fitur saat scaling
            total_orders_engagement = order_count + coupon_used
            values = [tenure, warehouse_to_home, hour_spend, order_count, cashback_amount, total_orders_engagement]
            input_df = pd.DataFrame([values], columns=FEATURES)
            labels, dists = predict(input_df)
            st.session_state["single_result"] = {
                "values": values, "coupon": coupon_used, "cluster": int(labels[0]), "dists": dists[0].tolist()}

        res = st.session_state.get("single_result")
        if res is None:
            st.info("Isi profil pelanggan lalu tekan **Prediksi Segmen Pelanggan** untuk melihat hasilnya.", icon="👆")
        else:
            cid, values, dists = res["cluster"], res["values"], np.array(res["dists"])
            p = persona_of(cid)
            sorted_d = np.sort(dists)
            clarity = (1 - sorted_d[0] / sorted_d[1]) * 100 if sorted_d[1] > 0 else 100.0
            st.markdown(
                f"""<div class="result-card" style="--accent:{p['color']}; --accent2:{p['color2']}">
                    <div class="label">Hasil Prediksi Segmen</div>
                    <h2>{p['icon']} Klaster {cid} — {p['name']}</h2>
                    <p>{p['summary']}</p>
                    <span class="pill">🎯 Fokus CRM: {p['theme']}</span>
                    <span class="pill">Total Engagement: {fmt(values[5], 0)}</span>
                    <span class="pill">Kejelasan segmen: {fmt(clarity, 0)}%</span>
                </div>""",
                unsafe_allow_html=True)

            out_of_range = [FEATURE_LABELS[c] for c, v in zip(
                ["Tenure", "WarehouseToHome", "HourSpendOnApp", "OrderCount", "CouponUsed", "CashbackAmount"],
                [values[0], values[1], values[2], values[3], res["coupon"], values[4]])
                if c in RANGES and not (RANGES[c][0] <= v <= RANGES[c][1])]
            if out_of_range:
                st.warning(f"Nilai {', '.join(out_of_range)} berada di luar rentang data historis. "
                           f"Prediksi tetap dibuat, namun interpretasikan dengan hati-hati.", icon="⚠️")
            if clarity < 15:
                st.info("Pelanggan ini berada dekat batas antar-segmen (jarak ke dua centroid terdekat hampir sama). "
                        "Pertimbangkan juga strategi segmen terdekat kedua.", icon="ℹ️")

            r1, r2 = st.columns([1, 1.25])
            with r1:
                section("Karakteristik Segmen")
                st.markdown("\n".join(f"- {t}" for t in p["traits"]))
                section("Rekomendasi Aksi CRM")
                st.markdown("\n".join(f"{i}. {a}" for i, a in enumerate(p["actions"], 1)))
                st.caption(f"📈 KPI: {p['kpi']}")
            with r2:
                section("Profil Pelanggan vs Rata-rata Klaster",
                        "Dibandingkan dalam z-score agar fitur berbeda satuan dapat disandingkan")
                st.altair_chart(comparison_chart(values, cid), width="stretch")
                cmp_table = pd.DataFrame({
                    "Fitur": [FEATURE_LABELS[f] for f in FEATURES],
                    "Pelanggan ini": values,
                    f"Rata-rata Klaster {cid}": CENTROIDS.loc[cid].values,
                })
                cmp_table["Selisih"] = cmp_table["Pelanggan ini"] - cmp_table[f"Rata-rata Klaster {cid}"]
                st.dataframe(cmp_table.style.format(precision=2, decimal=",", thousands="."),
                             hide_index=True, width="stretch")

            dist_df = pd.DataFrame({"Segmen": [segment_label(c) for c in range(len(dists))],
                                    "Jarak": dists})
            section("Jarak ke Setiap Centroid", "Semakin kecil jarak, semakin mirip dengan segmen tersebut")
            dist_chart = alt.Chart(dist_df).mark_bar(cornerRadiusEnd=4).encode(
                y=alt.Y("Segmen:N", title=None, sort=SEGMENT_DOMAIN, axis=alt.Axis(labelLimit=220)),
                x=alt.X("Jarak:Q", title="Jarak Euclidean (ruang z-score)"),
                color=alt.Color("Segmen:N", scale=SEGMENT_SCALE, legend=None),
                tooltip=["Segmen", alt.Tooltip("Jarak:Q", format=".3f")],
            ).properties(height=170)
            st.altair_chart(dist_chart, width="stretch")

    # -------------------------------------------------------------------- BATCH
    else:
        section("Prediksi Massal dari File CSV",
                "Unggah daftar pelanggan baru untuk diprediksi sekaligus menggunakan scaler.pkl & model_kmeans.pkl")
        g1, g2 = st.columns([1.6, 1])
        with g1:
            st.markdown(
                """
                **Kolom wajib** (nama tidak peka huruf besar/kecil, spasi, atau underscore):
                `Tenure`, `WarehouseToHome`, `HourSpendOnApp`, `OrderCount`, `CouponUsed`, `CashbackAmount`.
                Kolom lain (misalnya `CustomerID`) akan dipertahankan pada hasil.

                - Pemisah koma `,` maupun titik koma `;` didukung.
                - Nilai kosong/non-angka diisi **median** data historis (sesuai cleaning notebook).
                - Baris dengan nilai negatif **ditolak** dan tidak diprediksi.
                - `Total_Orders_Engagement` selalu dihitung ulang dari `OrderCount + CouponUsed`.
                """
            )
        with g2:
            template = pd.DataFrame({
                "CustomerID": ["CUST-001", "CUST-002", "CUST-003"],
                "Tenure": [2, 12, 25], "WarehouseToHome": [12, 15, 9], "HourSpendOnApp": [3, 3, 4],
                "OrderCount": [1, 9, 2], "CouponUsed": [1, 5, 1], "CashbackAmount": [140, 200, 260]})
            st.download_button("📄 Unduh Template CSV", template.to_csv(index=False).encode("utf-8-sig"),
                               file_name="template_prediksi_segmen.csv", mime="text/csv", width="stretch")
            use_sample = st.button("🧪 Coba dengan 300 sampel dataset", width="stretch",
                                   disabled=DATA is None)
            if use_sample:
                st.session_state["batch_source"] = "sample"

        uploaded = st.file_uploader("Unggah file CSV", type=["csv"], accept_multiple_files=False)
        if uploaded is not None:
            st.session_state["batch_source"] = "upload"

        source = st.session_state.get("batch_source")
        batch_df, source_name = None, None
        if source == "upload" and uploaded is not None:
            try:
                batch_df = read_uploaded_csv(uploaded)
                source_name = uploaded.name
            except ValueError as exc:
                st.error(f"❌ {exc}")
        elif source == "sample" and DATA is not None:
            batch_df = DATA.sample(300, random_state=7)[["CustomerID"] + INPUT_COLS].reset_index(drop=True)
            source_name = "sampel dataset"

        if batch_df is not None:
            if batch_df.empty:
                st.error("❌ File tidak berisi baris data.")
            elif len(batch_df) > 200_000:
                st.error("❌ Maksimal 200.000 baris per unggahan. Pecah file menjadi beberapa bagian.")
            else:
                try:
                    result, info = predict_batch(batch_df)
                except KeyError as exc:
                    missing_cols = exc.args[0]
                    st.error(f"❌ Format kolom tidak sesuai. Kolom wajib yang tidak ditemukan: "
                             f"**{', '.join(missing_cols)}**.")
                    st.caption(f"Kolom pada file: {', '.join(map(str, batch_df.columns))}. "
                               f"Gunakan template CSV sebagai acuan.")
                else:
                    st.success(f"✅ {fmt(info['predicted'], 0)} dari {fmt(info['total'], 0)} pelanggan "
                               f"berhasil diprediksi ({source_name}).")
                    notes = []
                    if info["imputed_rows"]:
                        notes.append(f"{fmt(info['imputed_rows'], 0)} baris memiliki nilai kosong/non-angka "
                                     f"({fmt(info['empty_cells'] + info['non_numeric'], 0)} sel) dan diisi median.")
                    if info["rejected"]:
                        notes.append(f"{fmt(info['rejected'], 0)} baris ditolak (nilai negatif atau baris kosong) "
                                     f"— lihat kolom Status_Data.")
                    if info["mismatch"]:
                        notes.append(f"{fmt(info['mismatch'], 0)} nilai Total_Orders_Engagement pada file berbeda "
                                     f"dari OrderCount + CouponUsed; aplikasi memakai hasil hitung ulang.")
                    renamed = {k: v for k, v in info["renamed"].items() if k != v}
                    if renamed:
                        notes.append("Kolom disesuaikan otomatis: " +
                                     ", ".join(f"`{k}` → `{v}`" for k, v in renamed.items()) + ".")
                    for n in notes:
                        st.warning(n, icon="⚠️")

                    ok = result[result["Klaster"].notna()]
                    if not ok.empty:
                        counts = ok["Klaster"].astype(int).value_counts()
                        mcols = st.columns(len(CLUSTER_IDS) + 1)
                        mcols[0].metric("Total Diprediksi", fmt(len(ok), 0))
                        for mc, cid in zip(mcols[1:], CLUSTER_IDS):
                            n = int(counts.get(cid, 0))
                            mc.metric(f"{persona_of(cid)['icon']} {persona_of(cid)['short']}", fmt(n, 0),
                                      delta=f"{fmt(n / len(ok) * 100)}%", delta_color="off", delta_arrow="off")

                        b1, b2 = st.columns([1, 1.3])
                        with b1:
                            section("Distribusi Segmen Hasil Prediksi")
                            dist = pd.DataFrame({"Segmen": [segment_label(c) for c in CLUSTER_IDS],
                                                 "Pelanggan": [int(counts.get(c, 0)) for c in CLUSTER_IDS]})
                            bar = alt.Chart(dist).mark_bar(cornerRadiusEnd=4).encode(
                                y=alt.Y("Segmen:N", title=None, sort=SEGMENT_DOMAIN, axis=alt.Axis(labelLimit=220)),
                                x=alt.X("Pelanggan:Q", title="Jumlah pelanggan"),
                                color=alt.Color("Segmen:N", scale=SEGMENT_SCALE, legend=None),
                                tooltip=["Segmen", "Pelanggan"],
                            ).properties(height=210)
                            st.altair_chart(bar, width="stretch")
                        with b2:
                            section("Rata-rata Fitur per Segmen (Hasil Unggahan)")
                            feats = info["features"].loc[ok.index].assign(Klaster=ok["Klaster"].astype(int))
                            agg = feats.groupby("Klaster")[INPUT_COLS + ["Total_Orders_Engagement"]].mean()
                            agg.index = [segment_label(i) for i in agg.index]
                            st.dataframe(agg.style.format(precision=2, decimal=",", thousands="."),
                                         width="stretch")

                    section("Hasil Prediksi", "Pratinjau 200 baris pertama — unduh file untuk hasil lengkap")
                    st.dataframe(result.head(200), hide_index=True, width="stretch", height=320)
                    st.download_button(
                        "⬇️ Unduh Hasil Prediksi (CSV)", result.to_csv(index=False).encode("utf-8-sig"),
                        file_name="hasil_prediksi_segmen.csv", mime="text/csv", type="primary")

st.markdown(
    "<div class='footer'>Praktikum Data Science · Informatika UPN \"Veteran\" Yogyakarta · "
    "Model: K-Means (model_kmeans.pkl) + StandardScaler (scaler.pkl)</div>",
    unsafe_allow_html=True)
