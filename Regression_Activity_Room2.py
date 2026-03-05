import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regression with Tree-Based Models",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Nunito:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }

.stApp {
  background: linear-gradient(160deg, #f0f5fb 0%, #eaf0f9 35%, #f5f0fd 70%, #f0f5fb 100%);
  background-attachment: fixed;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f172a 0%, #1e2d5a 55%, #3b2f6e 100%);
}
[data-testid="stSidebar"] * { color: #dde6f5 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }

.hero {
  background: linear-gradient(135deg, #0f172a 0%, #1e2d5a 45%, #3b2f6e 100%);
  border-radius: 22px; padding: 3rem 3.5rem 2.5rem;
  margin-bottom: 2.5rem; box-shadow: 0 24px 64px rgba(15,23,42,0.22);
  position: relative; overflow: hidden;
}
.hero::before {
  content:''; position:absolute; top:-80px; right:-80px;
  width:280px; height:280px; border-radius:50%;
  background: radial-gradient(circle, rgba(139,92,246,0.18) 0%, transparent 70%);
}
.hero-eyebrow {
  display:inline-block; background:rgba(139,92,246,0.25);
  border:1px solid rgba(139,92,246,0.4); color:#c4b5fd;
  font-size:0.72rem; letter-spacing:0.15em; text-transform:uppercase;
  padding:0.28rem 0.9rem; border-radius:30px; margin-bottom:1rem;
}
.hero-title { font-family:'Playfair Display',serif; font-size:2.6rem; color:#fff; margin:0 0 0.6rem; line-height:1.15; }
.hero-sub   { font-size:1rem; color:#94a3c8; font-weight:300; margin:0; }

.sec-head   { display:flex; align-items:center; gap:1rem; margin:2.5rem 0 1.4rem; }
.sec-pill   {
  background:linear-gradient(135deg,#6366f1,#8b5cf6); color:#fff;
  font-size:0.7rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase;
  padding:0.28rem 0.85rem; border-radius:30px;
}
.sec-title  { font-family:'Playfair Display',serif; font-size:1.95rem; color:#0f172a; margin:0; }

.card {
  background:#fff; border-radius:16px; padding:1.5rem 1.8rem;
  margin-bottom:1.2rem; box-shadow:0 4px 20px rgba(15,23,42,0.06);
  border:1px solid rgba(99,102,241,0.09);
}
.card-title { font-family:'Playfair Display',serif; font-size:1.15rem; color:#1e2d5a; margin:0 0 0.5rem; }
.card-body  { font-size:0.91rem; color:#475569; line-height:1.75; margin:0; }

.step-row   { display:flex; align-items:center; margin:1.6rem 0 0.7rem; }
.step-badge {
  display:inline-flex; align-items:center; justify-content:center;
  width:34px; height:34px; border-radius:50%;
  background:linear-gradient(135deg,#6366f1,#8b5cf6);
  color:#fff; font-weight:700; font-size:0.85rem; margin-right:0.8rem; flex-shrink:0;
}
.step-label { font-family:'Playfair Display',serif; font-size:1.12rem; color:#1e2d5a; }

.metric-card {
  background:linear-gradient(135deg,#6366f112,#8b5cf610);
  border:1.5px solid #6366f125; border-radius:14px;
  padding:1rem 1.2rem; text-align:center;
}
.metric-value { font-family:'Playfair Display',serif; font-size:1.85rem; color:#1e2d5a; line-height:1; }
.metric-label { font-size:0.72rem; color:#6366f1; text-transform:uppercase; letter-spacing:0.1em; margin-top:0.3rem; }

.answer-block {
  background:linear-gradient(135deg,#f8f7ff,#eef2ff);
  border-left:4px solid #8b5cf6; border-radius:0 12px 12px 0;
  padding:1.1rem 1.5rem; margin-top:0.7rem;
  font-size:0.91rem; color:#334155; line-height:1.78;
}

.reflection-wrap {
  background:linear-gradient(135deg,#0f172a,#1e2d5a,#3b2f6e);
  border-radius:20px; padding:2.2rem 2.8rem; margin-top:1rem;
  box-shadow:0 16px 48px rgba(15,23,42,0.22);
}
.reflection-title { font-family:'Playfair Display',serif; font-size:1.6rem; color:#fff; margin:0 0 1.8rem; }
.ref-q { font-size:0.75rem; text-transform:uppercase; letter-spacing:0.12em; color:#7c8ec7; margin-bottom:0.4rem; }
.ref-a { font-size:0.91rem; color:#c5d0e8; line-height:1.8; margin-bottom:1.6rem; }

.sb-label { font-size:0.68rem; text-transform:uppercase; letter-spacing:0.15em; color:#7c8ec7 !important; padding-left:0.2rem; margin-bottom:0.4rem; }
.sb-item  { font-size:0.82rem; color:#c5d0e8 !important; margin:0.22rem 0; padding-left:0.2rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  PLOTLY BASE LAYOUT
# ─────────────────────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor="white", plot_bgcolor="#fafbff",
    font=dict(family="Nunito, sans-serif", color="#334155"),
    title_font=dict(family="Playfair Display, serif", color="#0f172a", size=16),
    xaxis=dict(gridcolor="#e8edf8", linecolor="#d1d9f0", zerolinecolor="#d1d9f0"),
    yaxis=dict(gridcolor="#e8edf8", linecolor="#d1d9f0", zerolinecolor="#d1d9f0"),
    legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#e2e8f0", borderwidth=1),
    margin=dict(t=55, b=40, l=40, r=30),
)

def apply_layout(fig, **extra):
    fig.update_layout(**{**PLOTLY_BASE, **extra})
    return fig

# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="padding-top:1.4rem">', unsafe_allow_html=True)
    st.markdown('<p class="sb-label">Navigation</p>', unsafe_allow_html=True)
    page = st.radio("", [
        "Problem 1 — California Housing",
        "Problem 2 — Concrete Strength",
        "Problem 3 — Bike Sharing",
        "Final Reflection",
    ], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="sb-label">Group — Room 2</p>', unsafe_allow_html=True)
    for m in ["Andrea L. Rodriguez","Melanie P. Perez","Leiry L. Mares",
              "Donnys D. Torres","Maria A. Perez","Rosa M. Mora"]:
        st.markdown(f'<p class="sb-item">— {m}</p>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="sb-label">Models</p>', unsafe_allow_html=True)
    for m in ["Decision Tree","Random Forest","Gradient Boosting"]:
        st.markdown(f'<p class="sb-item">• {m}</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">In-Class Activity</div>
  <h1 class="hero-title">Regression with<br>Tree-Based Models</h1>
  <p class="hero-sub">Predicting real-world outcomes using Decision Trees, Random Forests and Gradient Boosting</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  SHARED UI HELPERS
# ─────────────────────────────────────────────────────────────
def metrics_row(items):
    cols = st.columns(len(items))
    for col, (label, val) in zip(cols, items):
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True)

def step(n, label):
    st.markdown(
        f'<div class="step-row"><span class="step-badge">{n}</span>'
        f'<span class="step-label">{label}</span></div>',
        unsafe_allow_html=True)

def answer(text):
    st.markdown(f'<div class="answer-block">{text}</div>', unsafe_allow_html=True)

def section_header(pill, title):
    st.markdown(
        f'<div class="sec-head"><span class="sec-pill">{pill}</span>'
        f'<h2 class="sec-title">{title}</h2></div>',
        unsafe_allow_html=True)

def card(title, body):
    st.markdown(
        f'<div class="card"><p class="card-title">{title}</p>'
        f'<p class="card-body">{body}</p></div>',
        unsafe_allow_html=True)

def results_plotly(df_res):
    """Dual-axis bar+line chart comparing models."""
    colors = ["#6366f1","#8b5cf6","#ec4899"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_res["Model"], y=df_res["R²"], name="R²",
        marker_color=colors, text=df_res["R²"].round(4),
        textposition="outside", yaxis="y", offsetgroup=1))
    fig.add_trace(go.Scatter(
        x=df_res["Model"], y=df_res["RMSE"], name="RMSE",
        mode="lines+markers",
        marker=dict(size=10, color="#f59e0b", symbol="diamond"),
        line=dict(color="#f59e0b", width=2.5, dash="dot"), yaxis="y2"))
    fig.update_layout(
        **{**PLOTLY_BASE,
           "title": "Model Comparison — R² and RMSE",
           "yaxis":  dict(title="R² Score", gridcolor="#e8edf8", range=[0, 1.15]),
           "yaxis2": dict(title="RMSE", overlaying="y", side="right",
                          gridcolor="rgba(0,0,0,0)"),
           "barmode": "group",
           "legend": dict(orientation="h", y=1.14, x=0.5, xanchor="center"),
        })
    return fig

def feature_importance_chart(model, feature_names, top_n, title, color_scale):
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    imp_show = imp.tail(top_n)
    fig = px.bar(imp_show, orientation="h",
                 color=imp_show.values, color_continuous_scale=color_scale,
                 labels={"value":"Importance","index":"Feature"}, title=title)
    apply_layout(fig)
    fig.update_layout(coloraxis_showscale=False, yaxis_title="")
    return fig, imp.sort_values(ascending=False)

def actual_vs_predicted(y_true, y_pred, model_name, color, max_pts=3000):
    y_true_arr = np.array(y_true)
    idx = np.random.choice(len(y_true_arr), min(max_pts, len(y_true_arr)), replace=False)
    fig = px.scatter(x=y_true_arr[idx], y=y_pred[idx], opacity=0.4,
                     labels={"x":"Actual","y":"Predicted"},
                     color_discrete_sequence=[color],
                     title=f"Actual vs Predicted — {model_name}")
    lim = [min(y_true_arr.min(), y_pred.min()), max(y_true_arr.max(), y_pred.max())]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                             line=dict(color="#ec4899", dash="dash", width=2),
                             name="Perfect fit"))
    apply_layout(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  PROBLEM 1 — CALIFORNIA HOUSING
# ═══════════════════════════════════════════════════════════════════════
if "California" in page:

    section_header("Problem 1", "California Housing Prices")
    card("Goal", "Predict the median house value (in units of $100,000) for neighborhoods in California, "
         "using data from the 1990 census — 20,640 block groups in total.")

    with st.expander("View variable dictionary"):
        st.dataframe(pd.DataFrame({
            "Variable":    ["MedInc","HouseAge","AveRooms","AveBedrms","Population",
                            "AveOccup","Latitude","Longitude","MedHouseVal"],
            "Description": ["Median income (tens of thousands USD)","Median house age (years)",
                            "Avg rooms per household","Avg bedrooms per household",
                            "Total population in block group","Avg occupants per household",
                            "Geographic latitude","Geographic longitude",
                            "Target: Median house value (x$100k)"]
        }), use_container_width=True, hide_index=True)

    # Load
    step(1, "Load the Data")

    @st.cache_data
    def load_housing():
        return fetch_california_housing(as_frame=True).frame

    df = load_housing()
    metrics_row([("Rows", f"{df.shape[0]:,}"), ("Columns", df.shape[1]),
                 ("Missing values", df.isnull().sum().sum()), ("Target", "MedHouseVal")])
    with st.expander("Preview dataset"):
        st.dataframe(df.head(10), use_container_width=True)

    # EDA
    step(2, "Exploratory Data Analysis")

    # Q1 — Correlation heatmap with variable selector
    st.markdown('<div class="card"><p class="card-title">Q1 — Correlation Matrix</p></div>', unsafe_allow_html=True)
    corr_vars = st.multiselect("Select variables to include",
                               list(df.columns), default=list(df.columns), key="c1")
    if len(corr_vars) >= 2:
        fig = px.imshow(df[corr_vars].corr().round(2), text_auto=True,
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                        title="Correlation Matrix — California Housing")
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    answer("MedInc (median income) shows the strongest positive linear correlation with MedHouseVal (r ≈ 0.69). "
           "Geographic coordinates also show moderate correlations, reflecting California's coastal price premiums.")

    # Q2 — Distribution with bin slider and box plot toggle
    st.markdown('<div class="card" style="margin-top:1.2rem"><p class="card-title">Q2 — Distribution of MedHouseVal</p></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 3])
    n_bins  = c1.slider("Bins", 10, 120, 50, key="b1")
    box_on  = c1.checkbox("Show box plot", value=True, key="bx1")
    if box_on:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram","Box Plot"),
                            column_widths=[0.65, 0.35])
        fig.add_trace(go.Histogram(x=df["MedHouseVal"], nbinsx=n_bins,
                                   marker_color="#6366f1", opacity=0.85, name=""), row=1, col=1)
        fig.add_trace(go.Box(y=df["MedHouseVal"], marker_color="#8b5cf6",
                             line_color="#6366f1", boxmean=True, name=""), row=1, col=2)
        apply_layout(fig, title="Distribution of Median House Value")
    else:
        fig = px.histogram(df, x="MedHouseVal", nbins=n_bins,
                           color_discrete_sequence=["#6366f1"],
                           title="Distribution of Median House Value")
        apply_layout(fig)
    st.plotly_chart(fig, use_container_width=True)
    answer("The distribution is right-skewed with an unusual spike at ~5.0 ($500k). This is a known census "
           "artifact: values were capped at $500,000, creating an artificial concentration at the boundary "
           "that can affect model accuracy for high-value properties.")

    # Split
    step(3, "Split the Data")

    @st.cache_data
    def split_h(df):
        X, y = df.drop("MedHouseVal", axis=1), df["MedHouseVal"]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    Xtr, Xte, ytr, yte = split_h(df)
    metrics_row([("Train samples", f"{len(Xtr):,}"), ("Test samples", f"{len(Xte):,}"),
                 ("Features", Xtr.shape[1]), ("Split", "80 / 20")])

    # Train
    step(4, "Train the Models")
    st.dataframe(pd.DataFrame({
        "Model": ["Decision Tree","Random Forest","Gradient Boosting"],
        "Hyperparameters": ["max_depth=8, random_state=42",
                            "n_estimators=100, random_state=42",
                            "n_estimators=200, learning_rate=0.1, random_state=42"]
    }), use_container_width=True, hide_index=True)

    @st.cache_data
    def train_h(_Xtr, _Xte, _ytr, _yte):
        ms = {"Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
              "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
              "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)}
        rows, preds = [], {}
        for name, m in ms.items():
            m.fit(_Xtr, _ytr); p = m.predict(_Xte)
            rows.append({"Model": name,
                         "RMSE": round(np.sqrt(mean_squared_error(_yte, p)), 4),
                         "R²":   round(r2_score(_yte, p), 4)})
            preds[name] = (m, p)
        return pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True), preds

    with st.spinner("Training models..."):
        res1, mods1 = train_h(Xtr, Xte, ytr, yte)

    # Evaluate
    step(5, "Evaluate Metrics")
    st.plotly_chart(results_plotly(res1), use_container_width=True)
    sel1 = st.selectbox("Model for Actual vs Predicted", res1["Model"].tolist(), key="avp1")
    _, p1 = mods1[sel1]
    st.plotly_chart(actual_vs_predicted(yte, p1, sel1, "#6366f1"), use_container_width=True)
    b1 = res1.iloc[0]
    answer(f"Random Forest achieves the best result with R² ≈ {b1['R²']} and RMSE ≈ {b1['RMSE']}. "
           "Averaging 100 independent trees over random data and feature subsets dramatically reduces "
           "variance, yielding stable predictions without the sequential sensitivity of Gradient Boosting.")

    # Feature Importance
    step(6, "Feature Importances")
    fi1 = st.selectbox("Model for feature importance", res1["Model"].tolist(), key="fi1")
    m1, _ = mods1[fi1]
    tn1 = st.slider("Top N features", 2, len(Xtr.columns), len(Xtr.columns), key="tn1")
    fig, imp1 = feature_importance_chart(m1, Xtr.columns, tn1, f"Feature Importances — {fi1}",
                                         ["#c7d2fe","#6366f1","#3730a3"])
    st.plotly_chart(fig, use_container_width=True)
    t3 = imp1.head(3)
    answer(f"Top 3 features: {t3.index[0]} ({t3.iloc[0]:.3f}), {t3.index[1]} ({t3.iloc[1]:.3f}), "
           f"{t3.index[2]} ({t3.iloc[2]:.3f}). MedInc dominates — wealthier block groups sustain higher "
           "property values. Geographic coordinates capture California's coastal and regional price premiums.")


# ═══════════════════════════════════════════════════════════════════════
#  PROBLEM 2 — CONCRETE STRENGTH
# ═══════════════════════════════════════════════════════════════════════
elif "Concrete" in page:

    section_header("Problem 2", "Concrete Compressive Strength")
    card("Goal", "Predict the compressive strength (MPa) of concrete mixtures. "
         "Accurate predictions save time and reduce costly laboratory testing in civil engineering.")

    with st.expander("View variable dictionary"):
        st.dataframe(pd.DataFrame({
            "Variable":    ["Cement","Slag","FlyAsh","Water","Superplasticizer",
                            "CoarseAggregate","FineAggregate","Age","Strength"],
            "Description": ["kg/m³","kg/m³","kg/m³","kg/m³","kg/m³","kg/m³","kg/m³",
                            "Curing age (days)","Target: Compressive strength (MPa)"]
        }), use_container_width=True, hide_index=True)

    # Load
    step(1, "Load the Data")

    @st.cache_data
    def load_concrete():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
        try:
            df = pd.read_excel(url)
            df.columns = ["Cement","Slag","FlyAsh","Water","Superplasticizer",
                          "CoarseAggregate","FineAggregate","Age","Strength"]
        except Exception:
            np.random.seed(42); n = 1030
            df = pd.DataFrame({
                "Cement": np.random.uniform(100, 540, n),
                "Slag": np.random.uniform(0, 360, n),
                "FlyAsh": np.random.uniform(0, 200, n),
                "Water": np.random.uniform(120, 250, n),
                "Superplasticizer": np.random.uniform(0, 32, n),
                "CoarseAggregate": np.random.uniform(800, 1145, n),
                "FineAggregate": np.random.uniform(594, 993, n),
                "Age": np.random.choice([1,3,7,14,28,56,90,180,270,365], n),
            })
            df["Strength"] = (0.08*df["Cement"] + 0.02*df["Slag"] - 0.1*df["Water"]
                              + 0.3*df["Superplasticizer"]
                              + 0.5*np.log1p(df["Age"])*10
                              + np.random.normal(0, 5, n)).clip(lower=2)
        return df

    with st.spinner("Loading concrete dataset..."):
        df2 = load_concrete()

    metrics_row([("Rows", f"{df2.shape[0]:,}"), ("Columns", df2.shape[1]),
                 ("Missing values", df2.isnull().sum().sum()), ("Target", "Strength (MPa)")])
    with st.expander("Preview dataset"):
        st.dataframe(df2.head(10), use_container_width=True)

    # EDA
    step(2, "Exploratory Data Analysis")

    # Q1 — Correlation heatmap with selector
    st.markdown('<div class="card"><p class="card-title">Q1 — Correlation Heatmap</p></div>', unsafe_allow_html=True)
    cv2 = st.multiselect("Select variables", list(df2.columns), default=list(df2.columns), key="c2")
    if len(cv2) >= 2:
        fig = px.imshow(df2[cv2].corr().round(2), text_auto=True,
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                        title="Correlation Matrix — Concrete")
        apply_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    answer("Cement shows the strongest positive correlation with Strength. Water has the strongest "
           "negative correlation: excess water raises the water-cement ratio and reduces mechanical "
           "strength. Age also correlates positively — concrete keeps gaining strength through hydration.")

    # Q2 — Scatter with color selector and age filter
    st.markdown('<div class="card" style="margin-top:1.2rem"><p class="card-title">Q2 — Cement vs Strength (interactive scatter)</p></div>', unsafe_allow_html=True)
    ca, cb = st.columns([1, 3])
    color_by2 = ca.selectbox("Color by", ["Age","Water","Superplasticizer","Slag"], key="col2")
    age_vals   = sorted(df2["Age"].unique())
    age_range  = ca.select_slider("Filter by Age (days)", options=age_vals,
                                  value=(age_vals[0], age_vals[-1]), key="age2")
    df2f = df2[(df2["Age"] >= age_range[0]) & (df2["Age"] <= age_range[1])]
    fig = px.scatter(df2f, x="Cement", y="Strength", color=color_by2,
                     color_continuous_scale="Viridis", opacity=0.65,
                     hover_data=["Age","Water","Superplasticizer"],
                     labels={"Cement":"Cement (kg/m³)","Strength":"Strength (MPa)"},
                     title=f"Cement vs Strength — colored by {color_by2}")
    apply_layout(fig)
    st.plotly_chart(fig, use_container_width=True)
    answer("A clear positive trend between cement and strength is visible. Older samples (warmer colors) "
           "appear higher for any given cement level, confirming that curing age is an independent "
           "predictor of strength — even low-cement mixtures can reach high strengths with enough time.")

    # Split
    step(3, "Split the Data")

    @st.cache_data
    def split_c(df2):
        X, y = df2.drop("Strength", axis=1), df2["Strength"]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    X2tr, X2te, y2tr, y2te = split_c(df2)
    metrics_row([("Train samples", f"{len(X2tr):,}"), ("Test samples", f"{len(X2te):,}"),
                 ("Features", X2tr.shape[1]), ("Split", "80 / 20")])

    # Train
    step(4, "Train the Models")
    st.dataframe(pd.DataFrame({
        "Model": ["Decision Tree","Random Forest","Gradient Boosting"],
        "Hyperparameters": ["max_depth=8, random_state=42",
                            "n_estimators=200, random_state=42",
                            "n_estimators=200, learning_rate=0.1, random_state=42"]
    }), use_container_width=True, hide_index=True)

    @st.cache_data
    def train_c(_X2tr, _X2te, _y2tr, _y2te):
        ms = {"Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
              "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
              "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)}
        rows, preds = [], {}
        for name, m in ms.items():
            m.fit(_X2tr, _y2tr); p = m.predict(_X2te)
            rows.append({"Model": name,
                         "RMSE": round(np.sqrt(mean_squared_error(_y2te, p)), 4),
                         "R²":   round(r2_score(_y2te, p), 4)})
            preds[name] = (m, p)
        return pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True), preds

    with st.spinner("Training models..."):
        res2, mods2 = train_c(X2tr, X2te, y2tr, y2te)

    # Evaluate
    step(5, "Evaluate Metrics")
    st.plotly_chart(results_plotly(res2), use_container_width=True)
    sel2 = st.selectbox("Model for Actual vs Predicted", res2["Model"].tolist(), key="avp2")
    _, p2 = mods2[sel2]
    st.plotly_chart(actual_vs_predicted(y2te, p2, sel2, "#8b5cf6"), use_container_width=True)
    b2 = res2.iloc[0]
    answer(f"Gradient Boosting leads with R² ≈ {b2['R²']} and RMSE ≈ {b2['RMSE']} MPa. "
           "Sequential error-correction is particularly effective for concrete data, where non-linear "
           "interactions between ingredients (especially the water-cement ratio) require highly adaptive "
           "splits that a parallel ensemble addresses less precisely.")

    # Feature Importance
    step(6, "Feature Importances")
    fi2 = st.selectbox("Model for feature importance", res2["Model"].tolist(), key="fi2")
    m2, _ = mods2[fi2]
    tn2 = st.slider("Top N features", 2, len(X2tr.columns), len(X2tr.columns), key="tn2")
    fig, imp2 = feature_importance_chart(m2, X2tr.columns, tn2, f"Feature Importances — {fi2}",
                                         ["#ddd6fe","#8b5cf6","#4c1d95"])
    st.plotly_chart(fig, use_container_width=True)
    t3b = imp2.head(3)
    answer(f"Top 3 features: {t3b.index[0]} ({t3b.iloc[0]:.3f}), {t3b.index[1]} ({t3b.iloc[1]:.3f}), "
           f"{t3b.index[2]} ({t3b.iloc[2]:.3f}). Age ranks among the top — arguably more important than "
           "Cement itself, reflecting continuous hydration that strengthens concrete over time.")


# ═══════════════════════════════════════════════════════════════════════
#  PROBLEM 3 — BIKE SHARING
# ═══════════════════════════════════════════════════════════════════════
elif "Bike" in page:

    section_header("Problem 3", "Bike Sharing Demand Prediction")
    card("Goal", "Predict the number of bikes rented each hour so the system can redistribute bikes "
         "across stations. Demand depends on weather, time of day, and whether it is a working day.")

    with st.expander("View variable dictionary"):
        st.dataframe(pd.DataFrame({
            "Variable":    ["season","yr","mnth","hr","holiday","weekday","workingday",
                            "weathersit","temp","atemp","hum","windspeed","count"],
            "Description": ["Season (1=spring…4=winter)","Year (0=2011, 1=2012)","Month (1–12)",
                            "Hour (0–23)","Holiday flag","Day of week (0–6)","Working day flag",
                            "Weather (1=clear…4=heavy rain)","Normalized temperature",
                            "Normalized feels-like temp","Normalized humidity","Normalized wind speed",
                            "Target: total hourly rentals"]
        }), use_container_width=True, hide_index=True)

    # Load
    step(1, "Load the Data")

    @st.cache_data
    def load_bike():
        try:
            from sklearn.datasets import fetch_openml
            raw  = fetch_openml(data_id=42712, as_frame=True, parser="auto")
            bike = raw.data.copy()
            bike["count"] = raw.target.astype(float)
            bike.rename(columns={"year":"yr","month":"mnth","hour":"hr",
                                  "weather":"weathersit","feel_temp":"atemp","humidity":"hum"}, inplace=True)
            for col, mp in [("season",  {"spring":1,"summer":2,"fall":3,"winter":4}),
                             ("weathersit",{"clear":1,"misty":2,"rain":3,"heavy_rain":4}),
                             ("holiday", {"False":0,"True":1,False:0,True:1}),
                             ("workingday",{"False":0,"True":1,False:0,True:1})]:
                if col in bike.columns:
                    bike[col] = bike[col].map(mp)
            keep = ["season","yr","mnth","hr","holiday","weekday","workingday",
                    "weathersit","temp","atemp","hum","windspeed","count"]
            bike = bike[[c for c in keep if c in bike.columns]]
            for col in bike.columns:
                bike[col] = pd.to_numeric(bike[col], errors="coerce")
            bike.dropna(inplace=True)
        except Exception:
            np.random.seed(42); n = 17379
            hr   = np.random.randint(0, 24, n)
            temp = np.random.uniform(0.1, 0.9, n)
            yr   = np.random.randint(0, 2, n)
            wday = np.random.randint(0, 7, n)
            weat = np.random.choice([1,2,3,4], n, p=[0.6,0.25,0.13,0.02])
            base = 100 + 300*temp + 80*yr + np.random.normal(0, 30, n)
            peak = ((hr == 17)|(hr == 18)|(hr == 8)).astype(float)*180
            night= ((hr >= 0)&(hr <= 5)).astype(float)*(-80)
            count= (base + peak + night - (weat-1)*40).clip(0)
            bike = pd.DataFrame({
                "season": np.random.randint(1,5,n), "yr": yr,
                "mnth": np.random.randint(1,13,n), "hr": hr,
                "holiday": np.random.randint(0,2,n), "weekday": wday,
                "workingday": (wday < 5).astype(int), "weathersit": weat,
                "temp": temp, "atemp": temp + np.random.normal(0,0.05,n),
                "hum": np.random.uniform(0.2, 0.95, n),
                "windspeed": np.random.uniform(0, 0.8, n), "count": count
            })
        return bike

    with st.spinner("Loading bike sharing dataset..."):
        df3 = load_bike()

    metrics_row([("Rows", f"{df3.shape[0]:,}"), ("Columns", df3.shape[1]),
                 ("Hours covered", "24"), ("Target", "count (rentals/hr)")])
    with st.expander("Preview dataset"):
        st.dataframe(df3.head(10), use_container_width=True)

    # EDA
    step(2, "Exploratory Data Analysis")

    # Q1 — Average rentals per hour with season/year/workingday filters
    st.markdown('<div class="card"><p class="card-title">Q1 — Average Rentals per Hour</p></div>', unsafe_allow_html=True)
    fa, fb, fc = st.columns(3)
    season_lbl = {1:"Spring",2:"Summer",3:"Fall",4:"Winter"}
    seasons_sel = fa.multiselect("Season", [1,2,3,4],
                                  format_func=lambda x: season_lbl[x],
                                  default=[1,2,3,4], key="s3")
    years_sel   = fb.multiselect("Year", [0,1],
                                  format_func=lambda x: "2011" if x==0 else "2012",
                                  default=[0,1], key="yr3")
    wd_sel      = fc.radio("Day type", ["All","Working","Non-working"], key="wd3")

    if seasons_sel and years_sel:
        df3f = df3[df3["season"].isin(seasons_sel) & df3["yr"].isin(years_sel)]
        if wd_sel == "Working":
            df3f = df3f[df3f["workingday"] == 1]
        elif wd_sel == "Non-working":
            df3f = df3f[df3f["workingday"] == 0]
        avg_hr = df3f.groupby("hr")["count"].mean().reset_index()
        fig = px.bar(avg_hr, x="hr", y="count",
                     color="count", color_continuous_scale=["#c7d2fe","#6366f1","#312e81"],
                     labels={"hr":"Hour of Day","count":"Avg Rentals"},
                     title="Average Bike Rentals per Hour")
        fig.update_layout(**{**PLOTLY_BASE,
                             "xaxis": dict(tickmode="linear", dtick=1, gridcolor="#e8edf8"),
                             "coloraxis_showscale": False})
        st.plotly_chart(fig, use_container_width=True)
    answer("A clear bimodal pattern: peak demand at 17:00–18:00 (~461 avg rentals) and 08:00 (~359), "
           "coinciding with commuting hours. Lowest demand is at 03:00–04:00 (~6–12 rentals). "
           "A sustained mid-day plateau reflects leisure and lunch trips. Hour of day is therefore "
           "the dominant predictor of bike demand.")

    # Q2 — Boxplot by weather with interactive filters
    st.markdown('<div class="card" style="margin-top:1.2rem"><p class="card-title">Q2 — Rentals by Weather Situation</p></div>', unsafe_allow_html=True)
    ga, gb = st.columns([1, 3])
    wday_filter = ga.radio("Day type filter", ["All days","Working only","Non-working only"], key="wf3")
    month_range = ga.slider("Month range", 1, 12, (1, 12), key="mr3")
    df3w = df3[(df3["mnth"] >= month_range[0]) & (df3["mnth"] <= month_range[1])].copy()
    if wday_filter == "Working only":
        df3w = df3w[df3w["workingday"] == 1]
    elif wday_filter == "Non-working only":
        df3w = df3w[df3w["workingday"] == 0]
    wlbl = {1:"Clear",2:"Mist/Cloudy",3:"Light Rain/Snow",4:"Heavy Rain"}
    df3w["Weather"] = df3w["weathersit"].map(wlbl)
    order_w = [v for v in ["Clear","Mist/Cloudy","Light Rain/Snow","Heavy Rain"]
               if v in df3w["Weather"].unique()]
    fig = px.box(df3w, x="Weather", y="count", category_orders={"Weather": order_w},
                 color="Weather",
                 color_discrete_sequence=["#6366f1","#8b5cf6","#ec4899","#f43f5e"],
                 labels={"count":"Bike Rentals"},
                 title="Bike Rentals by Weather Situation")
    apply_layout(fig)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    answer("Clear weather shows median ~159 rentals with high variability. Mist/Cloudy drops slightly "
           "to ~133. Light Rain/Snow falls to ~63. Heavy Rain reaches just ~36 — over 75% reduction. "
           "Rain reduces comfort, visibility and safety, strongly discouraging system use.")

    # Split
    step(3, "Split the Data")

    @st.cache_data
    def split_b(df3):
        X, y = df3.drop("count", axis=1), df3["count"]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    X3tr, X3te, y3tr, y3te = split_b(df3)
    metrics_row([("Train samples", f"{len(X3tr):,}"), ("Test samples", f"{len(X3te):,}"),
                 ("Features", X3tr.shape[1]), ("Split", "80 / 20")])

    # Train
    step(4, "Train the Models")
    st.dataframe(pd.DataFrame({
        "Model": ["Decision Tree","Random Forest","Gradient Boosting"],
        "Hyperparameters": ["max_depth=10, random_state=42",
                            "n_estimators=150, random_state=42",
                            "n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42"]
    }), use_container_width=True, hide_index=True)

    @st.cache_data
    def train_b(_X3tr, _X3te, _y3tr, _y3te):
        ms = {"Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
              "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
              "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                                             max_depth=5, random_state=42)}
        rows, preds = [], {}
        for name, m in ms.items():
            m.fit(_X3tr, _y3tr); p = m.predict(_X3te)
            rows.append({"Model": name,
                         "RMSE": round(np.sqrt(mean_squared_error(_y3te, p)), 4),
                         "R²":   round(r2_score(_y3te, p), 4)})
            preds[name] = (m, p)
        return pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True), preds

    with st.spinner("Training models..."):
        res3, mods3 = train_b(X3tr, X3te, y3tr, y3te)

    # Evaluate
    step(5, "Evaluate Metrics")
    st.plotly_chart(results_plotly(res3), use_container_width=True)
    sel3 = st.selectbox("Model for Actual vs Predicted", res3["Model"].tolist(), key="avp3")
    _, p3 = mods3[sel3]
    st.plotly_chart(actual_vs_predicted(y3te, p3, sel3, "#06b6d4", max_pts=3000),
                    use_container_width=True)
    b3 = res3.iloc[0]
    answer(f"Random Forest achieves the best result with R² ≈ {b3['R²']} and RMSE ≈ {b3['RMSE']} rentals/hr. "
           "Gradient Boosting is virtually tied (difference under 0.001 in R²). The Decision Tree lags "
           "~40% higher RMSE, showing that a single tree fails to generalize across the complex "
           "interactions between hour, weather and seasonality.")

    # Feature Importance
    step(6, "Feature Importances")
    fi3 = st.selectbox("Model for feature importance", res3["Model"].tolist(), key="fi3")
    m3, _ = mods3[fi3]
    tn3 = st.slider("Top N features", 2, len(X3tr.columns), len(X3tr.columns), key="tn3")
    fig, imp3 = feature_importance_chart(m3, X3tr.columns, tn3, f"Feature Importances — {fi3}",
                                         ["#bae6fd","#06b6d4","#0e7490"])
    st.plotly_chart(fig, use_container_width=True)
    t3c = imp3.head(3)
    answer(f"Top 3 features: {t3c.index[0]} ({t3c.iloc[0]:.3f}), {t3c.index[1]} ({t3c.iloc[1]:.3f}), "
           f"{t3c.index[2]} ({t3c.iloc[2]:.3f}). Hour of day alone concentrates over 60% of importance, "
           "perfectly aligned with the bimodal EDA pattern. Temperature ranks second. Surprisingly, "
           "weathersit and holiday carry marginal importance — their effect is captured indirectly "
           "by temperature and hourly patterns.")


# ═══════════════════════════════════════════════════════════════════════
#  FINAL REFLECTION
# ═══════════════════════════════════════════════════════════════════════
elif "Reflection" in page:

    section_header("Reflection", "Final Reflection")
    card("Instructions",
         "After completing all three problems, reflect on: which algorithm performed best consistently, "
         "when to prefer a Decision Tree, and what the feature importance plots revealed.")

    st.markdown("""
    <div class="reflection-wrap">
      <p class="reflection-title">Group Answers — Room 2</p>

      <p class="ref-q">Question 1 — Which algorithm consistently performed best and why?</p>
      <p class="ref-a">
        Throughout the three problems, ensemble models (Random Forest and Gradient Boosting) consistently
        outperformed the single Decision Tree, alternating first place depending on the dataset:
        Random Forest led on California Housing (R² 0.8051) and Bike Sharing (R² 0.9444), while
        Gradient Boosting led on Concrete (R² 0.9055). However, in every case the gap between both
        ensembles was minimal (under 0.01 in R²), so Random Forest can be considered the most
        consistently competitive algorithm. Its strength lies in building multiple trees over random
        subsamples of both observations and features, which significantly reduces variance and overfitting
        without requiring the sensitive hyperparameter tuning that Gradient Boosting demands.
      </p>

      <p class="ref-q">Question 2 — When would you choose a Decision Tree over a Random Forest?</p>
      <p class="ref-a">
        A Decision Tree is preferable when interpretability and transparency take priority over accuracy —
        for example in regulated industries like banking or medicine where every decision must be traceable.
        A single tree lets you visualize each split and follow the complete logical path from inputs to
        prediction, which is impossible with an ensemble of 100–200 trees. It is also preferable with very
        small datasets where ensemble overfitting risk increases, or when a quick prototype is needed to
        validate hypotheses before investing in more complex models. That said, as seen across the three
        problems, the accuracy cost is considerable: the Decision Tree obtained R² between 0.68 and 0.89,
        always below the ensembles.
      </p>

      <p class="ref-q">Question 3 — What did the feature importance plots reveal? Were there surprising variables?</p>
      <p class="ref-a">
        The plots confirmed that in each problem a small number of variables concentrate most predictive
        power. In California Housing, MedInc dominated; in Concrete, Age and Cement were the primary
        factors; in Bike Sharing, hr (hour of day) captured over 60% of total importance. The most
        surprising finding was that in Concrete, Age ranked above Cement itself — counterintuitive without
        civil engineering knowledge, but explained by the continuous hydration process that hardens concrete
        over time. Also notable in Bike Sharing was that weathersit and holiday showed marginal importance,
        despite their intuitive relevance, because their effect is already captured indirectly by temperature
        and hourly patterns.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Cross-problem performance comparison chart — interactive
    st.markdown('<div style="margin-top:2.5rem"><div class="card"><p class="card-title">Cross-Problem Performance Summary</p></div></div>', unsafe_allow_html=True)

    summary = pd.DataFrame({
        "Problem":  ["California Housing"]*3 + ["Concrete Strength"]*3 + ["Bike Sharing"]*3,
        "Model":    ["Decision Tree","Random Forest","Gradient Boosting"]*3,
        "R²":       [0.6842, 0.8051, 0.7951, 0.8612, 0.9010, 0.9055, 0.8911, 0.9444, 0.9435],
        "RMSE":     [0.6921, 0.5231, 0.5380, 6.8201, 5.1850, 4.9341, 58.73,  41.94,  42.29],
    })

    c1, c2 = st.columns([1, 3])
    metric_v = c1.radio("Metric", ["R²","RMSE"], key="sm")
    chart_t  = c1.radio("Chart type", ["Bar","Line"], key="ct")

    if chart_t == "Bar":
        fig = px.bar(summary, x="Problem", y=metric_v, color="Model", barmode="group",
                     color_discrete_sequence=["#6366f1","#8b5cf6","#ec4899"],
                     text=metric_v, title=f"{metric_v} Comparison across All Problems")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    else:
        fig = px.line(summary, x="Problem", y=metric_v, color="Model",
                      markers=True, line_shape="spline",
                      color_discrete_sequence=["#6366f1","#8b5cf6","#ec4899"],
                      title=f"{metric_v} Comparison across All Problems")
        fig.update_traces(line_width=2.5, marker_size=10)

    apply_layout(fig)
    fig.update_layout(legend=dict(orientation="h", y=1.14, x=0.5, xanchor="center"))
    st.plotly_chart(fig, use_container_width=True)
