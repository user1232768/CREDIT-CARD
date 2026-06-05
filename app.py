"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        Fraud Detection Platform                              ║
║         Built with Streamlit · RandomForest Model · Plotly Charts            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Pages:
  1. 🔐 Login           – Secure access gate
  2. 🏠 Home            – Welcome dashboard & KPIs
  3. 🔍 Prediction      – Single transaction fraud predictor
  4. 📊 Data Explorer   – Dataset visualisations
  5. 📈 Model Insights  – Feature importances & performance
  6. ℹ️  About           – App & model information
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import hashlib
from pathlib import Path
from typing import Optional

def hex_to_rgba(hex_color, alpha=0.2):
    """Convert #rrggbb to rgba() string — Plotly-safe transparency."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="InsureShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths — adjust if running locally
MODEL_PATH = Path("model.pkl")
DATA_PATH  = Path("creditcard.csv")

# Demo credentials (hash passwords in production!)
USERS = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
    "analyst": hashlib.sha256("analyst@2025".encode()).hexdigest(),
    "demo": hashlib.sha256("demo".encode()).hexdigest(),
}

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "authenticated": False,
        "username": "",
        "dark_mode": False,
        "page": "🏠 Home",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# ──────────────────────────────────────────────────────────────────────────────
# THEME HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_theme():
    """Return a dict of colour tokens based on current dark/light mode."""
    if st.session_state.dark_mode:
        return {
            "bg":         "#0f1117",
            "card":       "#1c1f2e",
            "border":     "#2d3147",
            "text":       "#e8eaf0",
            "subtext":    "#9199b0",
            "accent":     "#4f8ef7",
            "accent2":    "#7c5cbf",
            "success":    "#2ecc71",
            "danger":     "#e74c3c",
            "warn":       "#f39c12",
            "chart_bg":   "rgba(28,31,46,0.9)",
            "grid":       "#2d3147",
            "plotly_template": "plotly_dark",
        }
    return {
        "bg":         "#f5f7fb",
        "card":       "#ffffff",
        "border":     "#dde3f0",
        "text":       "#1a1d2e",
        "subtext":    "#6b7494",
        "accent":     "#3b6ff0",
        "accent2":    "#6c3fc5",
        "success":    "#27ae60",
        "danger":     "#c0392b",
        "warn":       "#d68910",
        "chart_bg":   "rgba(255,255,255,0.95)",
        "grid":       "#eaecf3",
        "plotly_template": "plotly_white",
    }

def inject_css(t):
    """Inject global CSS using current theme tokens."""
    st.markdown(f"""
    <style>
    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {t['bg']} !important;
        color: {t['text']} !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}
    [data-testid="stSidebar"] {{
        background-color: {t['card']} !important;
        border-right: 1.5px solid {t['border']};
    }}
    /* ── Metric cards ── */
    .kpi-card {{
        background: {t['card']};
        border: 1px solid {t['border']};
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}
    .kpi-card h1 {{ font-size: 2rem; margin: 0; color: {t['accent']}; }}
    .kpi-card p  {{ margin: 0.2rem 0 0; color: {t['subtext']}; font-size: 0.85rem; }}
    /* ── Section headers ── */
    .section-title {{
        font-size: 1.3rem;
        font-weight: 700;
        color: {t['text']};
        border-left: 4px solid {t['accent']};
        padding-left: 0.7rem;
        margin: 1.5rem 0 0.8rem;
    }}
    /* ── Alert boxes ── */
    .alert-fraud {{
        background: rgba(231,76,60,0.12);
        border: 1.5px solid {t['danger']};
        border-radius: 10px;
        padding: 1rem 1.4rem;
        color: {t['danger']};
        font-weight: 600;
    }}
    .alert-safe {{
        background: rgba(46,204,113,0.12);
        border: 1.5px solid {t['success']};
        border-radius: 10px;
        padding: 1rem 1.4rem;
        color: {t['success']};
        font-weight: 600;
    }}
    /* ── Login card ── */
    .login-card {{
        max-width: 440px;
        margin: 4rem auto;
        background: {t['card']};
        border: 1px solid {t['border']};
        border-radius: 18px;
        padding: 2.5rem 2.8rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.10);
    }}
    /* ── Streamlit overrides ── */
    .stButton > button {{
        border-radius: 8px !important;
        font-weight: 600 !important;
    }}
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {{
        background: {t['bg']} !important;
        color: {t['text']} !important;
        border-color: {t['border']} !important;
        border-radius: 8px !important;
    }}
    div[data-testid="stMetricValue"] {{ color: {t['accent']} !important; }}
    .stDataFrame {{ border-radius: 10px; overflow: hidden; }}
    footer {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADERS  (cached for performance)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

# Feature columns the model was trained on (Amount and Time are normalized at fit time)
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["normAmount", "normTime"]

# Raw input columns shown to the user (plain units)
RAW_INPUT_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]


def normalize_input(df_raw: pd.DataFrame, ref_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Convert raw Amount / Time → normAmount / normTime using StandardScaler
    fitted on the reference dataset.  If no reference is passed the values
    are standardised using dataset-wide statistics loaded from disk.
    """
    df = df_raw.copy()
    if ref_df is None:
        ref_df = load_data()

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(ref_df[["Amount", "Time"]])

    scaled = scaler.transform(df[["Amount", "Time"]])
    df["normAmount"] = scaled[:, 0]
    df["normTime"]   = scaled[:, 1]

    # Drop the raw columns — model never saw them
    df = df.drop(columns=["Amount", "Time"], errors="ignore")

    # Return in the exact column order the model expects
    return df[FEATURE_COLS]

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar(t):
    with st.sidebar:
        # Logo / branding
        st.markdown(f"""
        <div style="text-align:center; padding: 1rem 0 0.5rem;">
            <span style="font-size:2.8rem;">🛡️</span>
            <h2 style="margin:0.2rem 0 0; color:{t['accent']}; font-size:1.3rem;">InsureShield AI</h2>
            <p style="color:{t['subtext']}; font-size:0.78rem; margin:0;">Fraud & Risk Analytics</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<hr style='border-color:{t['border']}; margin:0.8rem 0;'>", unsafe_allow_html=True)

        # Theme toggle
        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.markdown(f"<span style='color:{t['subtext']}; font-size:0.85rem;'>🌙 Dark Mode</span>", unsafe_allow_html=True)
        with col_b:
            dark = st.toggle("", value=st.session_state.dark_mode, key="theme_toggle", label_visibility="collapsed")
            if dark != st.session_state.dark_mode:
                st.session_state.dark_mode = dark
                st.rerun()

        st.markdown(f"<hr style='border-color:{t['border']}; margin:0.6rem 0 1rem;'>", unsafe_allow_html=True)

        # Navigation
        pages = ["🏠 Home", "🔍 Prediction", "📊 Data Explorer", "📈 Model Insights", "ℹ️ About"]
        for p in pages:
            active = st.session_state.page == p
            btn_style = f"background:{t['accent']}22; color:{t['accent']}; border:1px solid {t['accent']}44;" if active else ""
            if st.button(p, use_container_width=True, key=f"nav_{p}"):
                st.session_state.page = p
                st.rerun()

        st.markdown(f"<hr style='border-color:{t['border']}; margin:1rem 0 0.5rem;'>", unsafe_allow_html=True)

        # Logged-in user + logout
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
            <span style="font-size:1.4rem;">👤</span>
            <span style="color:{t['subtext']}; font-size:0.82rem;">Signed in as <b style='color:{t['text']}'>{st.session_state.username}</b></span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.session_state.page = "🏠 Home"
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: LOGIN
# ──────────────────────────────────────────────────────────────────────────────

def page_login(t):
    inject_css(t)

    # Centered hero
    st.markdown(f"""
    <div style="text-align:center; margin-top:3rem; margin-bottom:1.5rem;">
        <span style="font-size:4rem;">🛡️</span>
        <h1 style="margin:0.3rem 0 0; color:{t['accent']}; font-size:2rem;">InsureShield AI</h1>
        <p style="color:{t['subtext']};">Life Insurance Fraud Detection Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Theme toggle on login page too
    cols = st.columns([3, 1, 3])
    with cols[1]:
        dark = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()

    # Login form
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown(f"""
        <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:18px;
                    padding:2rem 2.2rem; box-shadow:0 8px 32px rgba(0,0,0,0.10);">
            <h3 style="margin-top:0; color:{t['text']}; text-align:center;">Sign In</h3>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔐 Login", use_container_width=True, type="primary"):
            hashed = hashlib.sha256(password.encode()).hexdigest()
            if username in USERS and USERS[username] == hashed:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success(f"Welcome back, **{username}**! Redirecting…")
                st.rerun()
            else:
                st.error("❌ Invalid username or password. Please try again.")

        st.markdown(f"""
        <p style="text-align:center; color:{t['subtext']}; font-size:0.78rem; margin-top:1rem;">
            Demo credentials — user: <b>demo</b> · password: <b>demo</b>
        </p>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ──────────────────────────────────────────────────────────────────────────────

def page_home(t):
    df  = load_data()

    st.markdown(f"<h1 style='color:{t['text']}'>🏠 Dashboard Overview</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{t['subtext']}'>Welcome, <b>{st.session_state.username}</b>. Here's your platform snapshot.</p>", unsafe_allow_html=True)

    # ── KPI row ──
    total        = len(df)
    fraud_count  = int(df["Class"].sum())
    legit_count  = total - fraud_count
    fraud_rate   = fraud_count / total * 100
    avg_amount   = df["Amount"].mean()
    fraud_amount = df.loc[df["Class"] == 1, "Amount"].mean()

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (c1, "📋 Total Transactions", f"{total:,}",       "Records in dataset"),
        (c2, "✅ Legitimate",          f"{legit_count:,}", "Non-fraudulent"),
        (c3, "🚨 Fraud Cases",         f"{fraud_count:,}", f"{fraud_rate:.3f}% of all"),
        (c4, "💳 Avg. Amount",         f"${avg_amount:.2f}", f"Fraud avg: ${fraud_amount:.2f}"),
    ]
    for col, label, value, sub in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <p style="color:{t['subtext']}; font-size:0.8rem; margin:0 0 0.3rem;">{label}</p>
                <h1 style="color:{t['accent']}; font-size:1.8rem; margin:0;">{value}</h1>
                <p style="color:{t['subtext']}; font-size:0.78rem; margin:0.2rem 0 0;">{sub}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f"<div class='section-title'>Transaction Class Distribution</div>", unsafe_allow_html=True)
        pie_df = pd.DataFrame({
            "Label": ["Legitimate", "Fraud"],
            "Count": [legit_count, fraud_count],
        })
        fig_pie = px.pie(
            pie_df, names="Label", values="Count",
            color="Label",
            color_discrete_map={"Legitimate": t["success"], "Fraud": t["danger"]},
            hole=0.55,
            template=t["plotly_template"],
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown(f"<div class='section-title'>Transaction Amount Distribution</div>", unsafe_allow_html=True)
        sample = df.sample(min(5000, len(df)), random_state=42)
        fig_hist = px.histogram(
            sample, x="Amount", color="Class",
            nbins=60, barmode="overlay",
            color_discrete_map={0: t["accent"], 1: t["danger"]},
            labels={"Class": "Type", "Amount": "Amount (USD)"},
            template=t["plotly_template"],
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
            legend_title_text="",
        )
        fig_hist.for_each_trace(lambda tr: tr.update(name="Legitimate" if tr.name == "0" else "Fraud"))
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Fraud over time ──
    st.markdown(f"<div class='section-title'>Fraud Events Over Time</div>", unsafe_allow_html=True)
    time_df = df.copy()
    time_df["hour"] = (time_df["Time"] // 3600).astype(int)
    hourly = time_df.groupby(["hour", "Class"]).size().reset_index(name="count")
    fraud_hourly = hourly[hourly["Class"] == 1]
    fig_line = px.bar(
        fraud_hourly, x="hour", y="count",
        labels={"hour": "Hour of Day", "count": "Fraud Count"},
        color_discrete_sequence=[t["danger"]],
        template=t["plotly_template"],
    )
    fig_line.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        height=260,
    )
    st.plotly_chart(fig_line, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICTION
# ──────────────────────────────────────────────────────────────────────────────

def page_prediction(t):
    model = load_model()
    df    = load_data()

    st.markdown(f"<h1 style='color:{t['text']}'>🔍 Fraud Prediction</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{t['subtext']}'>Enter transaction features manually, pick a sample row, or upload a CSV for batch scoring.</p>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["⌨️ Manual Entry", "📂 Batch Upload"])

    # ── Manual Entry ──
    with tab1:
        col_ctrl, _ = st.columns([2, 3])
        with col_ctrl:
            sample_type = st.selectbox("🎲 Pre-fill with sample", ["— Enter manually —", "Random LEGIT sample", "Random FRAUD sample"])

        # Build default values using raw columns (Amount & Time, not normalized)
        raw_defaults = {col: 0.0 for col in RAW_INPUT_COLS}
        if sample_type == "Random LEGIT sample":
            row = df[df["Class"] == 0].sample(1).iloc[0]
            raw_defaults = {col: float(row[col]) for col in RAW_INPUT_COLS}
        elif sample_type == "Random FRAUD sample":
            row = df[df["Class"] == 1].sample(1).iloc[0]
            raw_defaults = {col: float(row[col]) for col in RAW_INPUT_COLS}

        st.markdown(f"<div class='section-title'>PCA Features (V1 – V28)</div>", unsafe_allow_html=True)

        # V1-V28 in a 4-column grid
        v_cols = st.columns(4)
        v_vals = {}
        for i, feat in enumerate([f"V{j}" for j in range(1, 29)]):
            with v_cols[i % 4]:
                v_vals[feat] = st.number_input(feat, value=raw_defaults[feat], format="%.4f", key=f"feat_{feat}")

        st.markdown(f"<div class='section-title'>Transaction Details</div>", unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            amt  = st.number_input("💵 Amount (USD)", min_value=0.0, value=raw_defaults["Amount"], format="%.2f")
        with cb:
            time = st.number_input("⏱️ Time (seconds)", min_value=0.0, value=raw_defaults["Time"], format="%.0f")

        if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
            # Build raw DataFrame then normalize Amount & Time → normAmount / normTime
            raw_data   = pd.DataFrame([[v_vals[f"V{i}"] for i in range(1, 29)] + [amt, time]],
                                      columns=RAW_INPUT_COLS)
            input_data = normalize_input(raw_data)
            prob  = model.predict_proba(input_data)[0]
            pred  = model.predict(input_data)[0]
            fraud_prob = prob[1] * 100

            st.markdown("<br>", unsafe_allow_html=True)
            c_res, c_gauge = st.columns([1, 1])

            with c_res:
                if pred == 1:
                    st.markdown(f"""
                    <div class="alert-fraud">
                        🚨 FRAUD DETECTED<br>
                        <span style="font-size:0.9rem; font-weight:400;">
                        This transaction shows strong indicators of fraudulent activity.
                        Confidence: <b>{fraud_prob:.1f}%</b>
                        </span>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-safe">
                        ✅ TRANSACTION LEGITIMATE<br>
                        <span style="font-size:0.9rem; font-weight:400;">
                        No fraud indicators detected.
                        Fraud probability: <b>{fraud_prob:.2f}%</b>
                        </span>
                    </div>""", unsafe_allow_html=True)

            with c_gauge:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fraud_prob,
                    title={"text": "Fraud Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": t["danger"] if pred == 1 else t["success"]},
                        "steps": [
                            {"range": [0, 30],  "color": hex_to_rgba(t["success"], 0.2)},
                            {"range": [30, 70], "color": hex_to_rgba(t["warn"], 0.2)},
                            {"range": [70, 100],"color": hex_to_rgba(t["danger"], 0.2)},
                        ],
                        "threshold": {"line": {"color": t["danger"], "width": 3}, "value": 50},
                    },
                    number={"suffix": "%", "font": {"color": t["text"]}},
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color=t["text"],
                    height=260,
                    margin=dict(t=40, b=10, l=20, r=20),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Batch Upload ──
    with tab2:
        st.info("Upload a CSV with the same columns as the training data (V1–V28, Amount, Time). The model will score every row.")
        uploaded = st.file_uploader("📤 Upload CSV", type=["csv"])
        if uploaded:
            batch_df = pd.read_csv(uploaded)
            missing  = [c for c in FEATURE_COLS if c not in batch_df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                # Normalize raw Amount/Time columns if present, else use normAmount/normTime directly
                if "Amount" in batch_df.columns and "Time" in batch_df.columns:
                    score_df = normalize_input(batch_df[RAW_INPUT_COLS])
                else:
                    score_df = batch_df[FEATURE_COLS]
                probs  = model.predict_proba(score_df)[:, 1]
                preds  = model.predict(score_df)
                batch_df["Fraud_Probability"] = (probs * 100).round(2)
                batch_df["Prediction"]        = ["🚨 Fraud" if p == 1 else "✅ Legit" for p in preds]

                fraud_n = int(preds.sum())
                st.success(f"Scored **{len(batch_df):,}** rows — **{fraud_n}** flagged as fraud ({fraud_n/len(batch_df)*100:.2f}%).")
                st.dataframe(
                    batch_df[["Amount", "Time", "Fraud_Probability", "Prediction"]].head(200),
                    use_container_width=True,
                )
                csv_out = batch_df.to_csv(index=False).encode()
                st.download_button("⬇️ Download Results", csv_out, "scored_transactions.csv", "text/csv")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: DATA EXPLORER
# ──────────────────────────────────────────────────────────────────────────────

def page_data_explorer(t):
    df = load_data()

    st.markdown(f"<h1 style='color:{t['text']}'>📊 Data Explorer</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{t['subtext']}'>Interactively explore the underlying transaction dataset.</p>", unsafe_allow_html=True)

    # ── Filters ──
    with st.expander("🔧 Filters", expanded=True):
        fa, fb, fc = st.columns(3)
        with fa:
            class_filter = st.multiselect("Transaction Type", [0, 1], default=[0, 1],
                                          format_func=lambda x: "✅ Legit" if x == 0 else "🚨 Fraud")
        with fb:
            amt_min, amt_max = float(df["Amount"].min()), float(df["Amount"].max())
            amt_range = st.slider("Amount Range ($)", amt_min, amt_max, (amt_min, min(amt_max, 1000.0)))
        with fc:
            sample_n = st.slider("Sample size (rows)", 500, 10000, 3000, step=500)

    filtered = df[df["Class"].isin(class_filter) & df["Amount"].between(*amt_range)]
    sample   = filtered.sample(min(sample_n, len(filtered)), random_state=1)

    st.markdown(f"<div class='section-title'>Raw Data Preview ({len(filtered):,} rows matched)</div>", unsafe_allow_html=True)
    st.dataframe(sample.head(50), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Correlation heatmap ──
    st.markdown(f"<div class='section-title'>Feature Correlation Heatmap</div>", unsafe_allow_html=True)
    corr_cols = [f"V{i}" for i in range(1, 15)] + ["Amount", "Class"]
    corr = sample[corr_cols].corr()
    fig_heat = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r",
        template=t["plotly_template"],
    )
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        height=500,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Scatter ──
    st.markdown(f"<div class='section-title'>Feature Scatter Explorer</div>", unsafe_allow_html=True)
    num_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    sc_a, sc_b = st.columns(2)
    with sc_a:
        x_feat = st.selectbox("X-axis feature", num_cols, index=0)
    with sc_b:
        y_feat = st.selectbox("Y-axis feature", num_cols, index=1)

    fig_sc = px.scatter(
        sample, x=x_feat, y=y_feat, color="Class",
        color_discrete_map={0: t["accent"], 1: t["danger"]},
        opacity=0.55,
        template=t["plotly_template"],
        labels={"Class": "Type"},
    )
    fig_sc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        height=380,
    )
    fig_sc.for_each_trace(lambda tr: tr.update(name="Legitimate" if tr.name == "0" else "Fraud"))
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Box plots ──
    st.markdown(f"<div class='section-title'>Feature Distribution by Class</div>", unsafe_allow_html=True)
    box_feat = st.selectbox("Select feature", [f"V{i}" for i in range(1, 29)] + ["Amount"])
    fig_box = px.box(
        sample, x="Class", y=box_feat,
        color="Class",
        color_discrete_map={0: t["accent"], 1: t["danger"]},
        template=t["plotly_template"],
        labels={"Class": "Type"},
    )
    fig_box.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        height=340,
    )
    fig_box.for_each_trace(lambda tr: tr.update(name="Legitimate" if tr.name == "0" else "Fraud"))
    st.plotly_chart(fig_box, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL INSIGHTS
# ──────────────────────────────────────────────────────────────────────────────

def page_model_insights(t):
    model = load_model()
    df    = load_data()

    st.markdown(f"<h1 style='color:{t['text']}'>📈 Model Insights</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{t['subtext']}'>Understand what drives the model's predictions.</p>", unsafe_allow_html=True)

    # ── Feature importances ──
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    top_n = 15

    st.markdown(f"<div class='section-title'>Top {top_n} Feature Importances</div>", unsafe_allow_html=True)
    fig_imp = px.bar(
        x=importances.values[:top_n],
        y=importances.index[:top_n],
        orientation="h",
        color=importances.values[:top_n],
        color_continuous_scale=[[0, t["accent2"]], [1, t["accent"]]],
        template=t["plotly_template"],
        labels={"x": "Importance", "y": "Feature"},
    )
    fig_imp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        height=400,
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── Model performance metrics ──
    st.markdown(f"<div class='section-title'>Model Performance (Training Set Estimate)</div>", unsafe_allow_html=True)

    # Quick eval on a stratified sample (avoid full 284K for speed)
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (confusion_matrix, classification_report,
                                 roc_auc_score, roc_curve, precision_recall_curve)

    sample = df.sample(min(20000, len(df)), random_state=42)
    # CSV has raw Amount/Time — normalize to normAmount/normTime before scoring
    X = normalize_input(sample[RAW_INPUT_COLS], ref_df=df)
    y = sample["Class"].reset_index(drop=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    report  = classification_report(y_te, y_pred, output_dict=True)
    auc     = roc_auc_score(y_te, y_proba)

    # Metric chips
    m1, m2, m3, m4 = st.columns(4)
    chips = [
        (m1, "ROC-AUC",   f"{auc:.4f}"),
        (m2, "Precision", f"{report['1']['precision']:.4f}"),
        (m3, "Recall",    f"{report['1']['recall']:.4f}"),
        (m4, "F1-Score",  f"{report['1']['f1-score']:.4f}"),
    ]
    for col, label, val in chips:
        with col:
            st.metric(label, val)

    # ── Confusion matrix ──
    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown(f"<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
        cm = confusion_matrix(y_te, y_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted Legit", "Predicted Fraud"],
            y=["Actual Legit",    "Actual Fraud"],
            color_continuous_scale=[[0, t["card"]], [1, t["accent"]]],
            template=t["plotly_template"],
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10),
            height=340,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_roc:
        st.markdown(f"<div class='section-title'>ROC Curve</div>", unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_te, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"AUC = {auc:.4f}",
            line=dict(color=t["accent"], width=2.5),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random",
            line=dict(color=t["subtext"], width=1.5, dash="dash"),
        ))
        fig_roc.update_layout(
            template=t["plotly_template"],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            margin=dict(t=10, b=10, l=10, r=10),
            height=340,
            legend=dict(x=0.55, y=0.1),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── Precision-Recall ──
    st.markdown(f"<div class='section-title'>Precision–Recall Curve</div>", unsafe_allow_html=True)
    prec_arr, rec_arr, _ = precision_recall_curve(y_te, y_proba)
    fig_pr = go.Figure(go.Scatter(
        x=rec_arr, y=prec_arr, mode="lines",
        line=dict(color=t["accent2"], width=2.5),
        fill="tozeroy", fillcolor=hex_to_rgba(t["accent2"], 0.13),
    ))
    fig_pr.update_layout(
        template=t["plotly_template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Recall",
        yaxis_title="Precision",
        margin=dict(t=10, b=10, l=10, r=10),
        height=300,
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    # ── Model parameters ──
    with st.expander("🔧 Model Hyperparameters"):
        params = model.get_params()
        param_df = pd.DataFrame(params.items(), columns=["Parameter", "Value"])
        st.dataframe(param_df, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ──────────────────────────────────────────────────────────────────────────────

def page_about(t):
    st.markdown(f"<h1 style='color:{t['text']}'>ℹ️ About InsureShield AI</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(f"""
        <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:14px; padding:1.8rem 2rem; margin-bottom:1rem;">
            <h3 style="margin-top:0; color:{t['accent']};">🛡️ Platform Overview</h3>
            <p style="color:{t['text']}; line-height:1.7;">
                <b>InsureShield AI</b> is a professional analytics platform built for life insurance
                fraud detection and risk assessment. It combines machine learning with interactive
                visualisations to help analysts identify suspicious transactions in real time.
            </p>
            <hr style="border-color:{t['border']};">
            <h4 style="color:{t['text']};">🤖 Model</h4>
            <ul style="color:{t['subtext']}; line-height:1.9;">
                <li>Algorithm: <b style="color:{t['text']}">Random Forest Classifier</b></li>
                <li>Estimators: <b style="color:{t['text']}">100 trees</b></li>
                <li>Min samples split: <b style="color:{t['text']}">5</b></li>
                <li>Random state: <b style="color:{t['text']}">42</b></li>
                <li>Training data: <b style="color:{t['text']}">284,807 transactions</b></li>
                <li>Fraud prevalence: <b style="color:{t['text']}">0.172% (492 cases)</b></li>
            </ul>
            <hr style="border-color:{t['border']};">
            <h4 style="color:{t['text']};">📦 Tech Stack</h4>
            <ul style="color:{t['subtext']}; line-height:1.9;">
                <li>Frontend: <b style="color:{t['text']}">Streamlit</b></li>
                <li>ML: <b style="color:{t['text']}">scikit-learn, XGBoost</b></li>
                <li>Visualisation: <b style="color:{t['text']}">Plotly</b></li>
                <li>Data: <b style="color:{t['text']}">Pandas, NumPy</b></li>
                <li>Serialisation: <b style="color:{t['text']}">Joblib</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:14px; padding:1.8rem 2rem; margin-bottom:1rem;">
            <h3 style="margin-top:0; color:{t['accent']};">📋 Pages</h3>
            <ul style="color:{t['subtext']}; line-height:2.1; list-style:none; padding:0;">
                <li>🏠 <b style="color:{t['text']}">Home</b> — KPI dashboard</li>
                <li>🔍 <b style="color:{t['text']}">Prediction</b> — Real-time scoring</li>
                <li>📊 <b style="color:{t['text']}">Data Explorer</b> — Visual EDA</li>
                <li>📈 <b style="color:{t['text']}">Model Insights</b> — Performance & SHAP</li>
                <li>ℹ️ <b style="color:{t['text']}">About</b> — This page</li>
            </ul>
        </div>
        <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:14px; padding:1.8rem 2rem;">
            <h3 style="margin-top:0; color:{t['accent']};">🔑 Demo Credentials</h3>
            <table style="width:100%; color:{t['text']}; border-collapse:collapse;">
                <tr style="border-bottom:1px solid {t['border']};">
                    <th style="padding:0.4rem 0; color:{t['subtext']}; text-align:left;">User</th>
                    <th style="padding:0.4rem 0; color:{t['subtext']}; text-align:left;">Password</th>
                </tr>
                <tr><td style="padding:0.4rem 0;">admin</td><td>admin123</td></tr>
                <tr><td style="padding:0.4rem 0;">analyst</td><td>analyst@2025</td></tr>
                <tr><td style="padding:0.4rem 0;">demo</td><td>demo</td></tr>
            </table>
            <p style="color:{t['subtext']}; font-size:0.78rem; margin-top:0.8rem;">
                ⚠️ For production use, replace with a secure auth provider.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN ROUTER
# ──────────────────────────────────────────────────────────────────────────────

def main():
    t = get_theme()
    inject_css(t)

    if not st.session_state.authenticated:
        page_login(t)
        return

    render_sidebar(t)

    page = st.session_state.page
    if   page == "🏠 Home":            page_home(t)
    elif page == "🔍 Prediction":      page_prediction(t)
    elif page == "📊 Data Explorer":   page_data_explorer(t)
    elif page == "📈 Model Insights":  page_model_insights(t)
    elif page == "ℹ️ About":           page_about(t)

if __name__ == "__main__":
    main()
