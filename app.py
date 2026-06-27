import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="WELON AGENCY Analytics",
    page_icon="📊",
    layout="wide"
)

np.random.seed(42)

@st.cache_data
def load_data():
    clients = pd.DataFrame([
        {"client_id": 1, "name": "Romantic",  "industry": "Retail / Flowers",     "city": "Almaty",  "contract_start": "2023-03-01"},
        {"client_id": 2, "name": "Biosfera",  "industry": "Healthcare / Pharmacy", "city": "Almaty",  "contract_start": "2023-01-15"},
        {"client_id": 3, "name": "Jetour",    "industry": "Automotive",            "city": "Almaty",  "contract_start": "2023-06-10"},
        {"client_id": 4, "name": "Kinetik",   "industry": "Healthcare / Sports",   "city": "Astana",  "contract_start": "2023-09-01"},
    ])
    clients["contract_start"] = pd.to_datetime(clients["contract_start"])
    campaign_types = ["SMM", "Targeted Ads", "SEO", "Content Marketing", "Email Marketing", "Influencer"]
    rows = []
    cid = 1
    for _, client in clients.iterrows():
        for i in range(np.random.randint(8, 16)):
            start = client["contract_start"] + timedelta(days=np.random.randint(0, 300))
            end   = start + timedelta(days=np.random.randint(14, 60))
            budget_ranges = {
                "Romantic": [30000, 50000, 80000, 100000],
                "Biosfera": [80000, 120000, 200000, 350000],
                "Kinetik":  [100000, 150000, 200000, 300000],
                "Jetour":   [200000, 350000, 500000, 800000],
            }
            budget = np.random.choice(budget_ranges[client["name"]])
            spend  = round(budget * np.random.uniform(0.75, 1.0))
            impressions = int(spend * np.random.uniform(0.4, 1.0))       # CPM ~1000-2500 KZT
            clicks      = int(impressions * np.random.uniform(0.01, 0.03)) # CTR ~1-3%
            conversions = int(clicks * np.random.uniform(0.02, 0.08))      # CVR ~2-8%
            revenue     = round(conversions * np.random.uniform(3000, 10000)) # realistic ROAS
            rows.append({
                "campaign_id": cid, "client_id": client["client_id"],
                "client_name": client["name"],
                "campaign_type": np.random.choice(campaign_types),
                "start_date": start, "end_date": end,
                "budget_kzt": budget, "spend_kzt": spend,
                "impressions": impressions, "clicks": clicks,
                "conversions": conversions, "revenue_kzt": revenue,
            })
            cid += 1
    df = pd.DataFrame(rows)
    df["CTR"] = (df["clicks"] / df["impressions"] * 100).round(2)
    df["CVR"] = (df["conversions"] / df["clicks"].replace(0,1) * 100).round(2)
    df["ROI"] = ((df["revenue_kzt"] - df["spend_kzt"]) / df["spend_kzt"] * 100).round(1)
    df["CPC"] = (df["spend_kzt"] / df["clicks"].replace(0,1)).round(0)
    return df, clients

df, clients = load_data()

# ── SIDEBAR ───────────────────────────────────────────────
# FIX: убрали via.placeholder.com (сервис недоступен -> битая картинка на защите)
st.sidebar.markdown("## 📊 WELON AGENCY")
st.sidebar.caption("Campaign Analytics")
st.sidebar.title("Navigation")
# FIX: непустой label + label_visibility вместо "" (иначе warning о доступности)
page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Overview",
        "🎯 RFM Segmentation",
        "🤖 ML Models",
        "📈 Campaign Forecast",
        "💰 Budget Optimizer",
    ],
    label_visibility="collapsed",
)

client_filter = st.sidebar.multiselect(
    "Filter by Client",
    options=df["client_name"].unique(),
    default=df["client_name"].unique()
)
df_filtered = df[df["client_name"].isin(client_filter)]

# FIX: защита от пустого фильтра — иначе метрики/графики падают на пустом df
if df_filtered.empty:
    st.warning("⚠️ Выберите хотя бы одного клиента в фильтре слева.")
    st.stop()

# ── PAGE 1: OVERVIEW ──────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 WELON AGENCY — Campaign Analytics Dashboard")
    st.markdown("**Data Analyst Intern:** Aitakhmetova Sabina · MCS-2401 · June–July 2026")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Campaigns", len(df_filtered))
    col2.metric("Total Spend", f"{df_filtered['spend_kzt'].sum()/1_000_000:.2f}M KZT")
    col3.metric("Avg CTR", f"{df_filtered['CTR'].mean():.2f}%")
    col4.metric("Total Conversions", f"{df_filtered['conversions'].sum():,}")

    st.subheader("Spend & ROI by Client")
    col1, col2 = st.columns(2)
    with col1:
        spend = df_filtered.groupby("client_name")["spend_kzt"].sum()/1_000_000
        fig = go.Figure(go.Bar(x=spend.index, y=spend.values,
            marker_color=["#E91E63","#4CAF50","#FF9800","#2196F3"]))
        fig.update_layout(title="Total Spend (M KZT)", height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        roi = df_filtered.groupby("client_name")["ROI"].mean()/1000
        fig = go.Figure(go.Bar(x=roi.index, y=roi.values,
            marker_color=["#E91E63","#4CAF50","#FF9800","#2196F3"]))
        fig.update_layout(title="Avg ROI (K%)", height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Campaign Type Performance")
    type_perf = df_filtered.groupby("campaign_type").agg(
        avg_ctr=("CTR","mean"), avg_roi=("ROI","mean"),
        total_conv=("conversions","sum"), count=("campaign_id","count")
    ).round(2).reset_index()
    st.dataframe(type_perf, use_container_width=True)

# ── PAGE 2: RFM ───────────────────────────────────────────
elif page == "🎯 RFM Segmentation":
    st.title("🎯 RFM Client Segmentation")
    st.divider()

    snap = df["end_date"].max() + timedelta(days=1)
    rfm = df.groupby("client_name").agg(
        Recency=("end_date", lambda x: (snap - x.max()).days),
        Frequency=("campaign_id","count"),
        Monetary=("spend_kzt","sum")
    ).reset_index()
    rfm["R"] = rfm["Recency"].rank(ascending=True).astype(int)
    rfm["F"] = rfm["Frequency"].rank(ascending=False).astype(int)
    rfm["M"] = rfm["Monetary"].rank(ascending=False).astype(int)
    rfm["Score"] = rfm["R"] + rfm["F"] + rfm["M"]
    def _seg(r):
        sc = r["Score"]
        if sc <= 5:   return "Champion"
        elif sc <= 6: return "Loyal"
        elif sc <= 7: return "Potential"
        else:         return "At Risk" if r["Recency"] >= 200 else "Potential"
    rfm["Segment"] = rfm.apply(_seg, axis=1)
    rfm["Monetary_M"] = (rfm["Monetary"]/1_000_000).round(2)

    seg_colors = {"Champion":"🏆","Loyal":"💚","Potential":"⚠️","At Risk":"🔴"}
    for _, row in rfm.iterrows():
        icon = seg_colors.get(row["Segment"],"")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Client", row["client_name"])
        col2.metric("Recency", f"{row['Recency']} days")
        col3.metric("Monetary", f"{row['Monetary_M']}M KZT")
        col4.metric("Segment", f"{icon} {row['Segment']}")
        st.divider()

# ── PAGE 3: ML MODELS ─────────────────────────────────────
elif page == "🤖 ML Models":
    st.title("🤖 Machine Learning Models")
    st.divider()

    snap = df["end_date"].max() + timedelta(days=1)
    rfm = df.groupby("client_name").agg(
        Recency=("end_date", lambda x: (snap - x.max()).days),
        Frequency=("campaign_id","count"),
        Monetary=("spend_kzt","sum")
    ).reset_index()
    rfm["R"] = rfm["Recency"].rank(ascending=True).astype(int)
    rfm["F"] = rfm["Frequency"].rank(ascending=False).astype(int)
    rfm["M"] = rfm["Monetary"].rank(ascending=False).astype(int)
    rfm["Score"] = rfm["R"] + rfm["F"] + rfm["M"]
    rfm["Churn"] = ((rfm["Score"] >= 8) & (rfm["Recency"] >= 200)).astype(int)
    rfm["Monetary_M"] = (rfm["Monetary"]/1_000_000).round(2)

    X = rfm[["Recency","Frequency","Monetary_M"]]
    y = rfm["Churn"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # FIX: если все клиенты в одном классе (никто/все в риске),
    # LogisticRegression падает -> показываем честное сообщение вместо краша
    if y.nunique() < 2:
        st.info("ℹ️ Все клиенты сейчас в одном классе риска — модель обучить нельзя. "
                "Это нормально для 4 клиентов; на реальной базе классы появятся.")
    else:
        model = LogisticRegression()
        model.fit(X_scaled, y)
        rfm["Churn_Prob"] = model.predict_proba(X_scaled)[:,1].round(2)
        rfm["Risk"] = rfm["Churn_Prob"].apply(lambda p: "🔴 High" if p>0.6 else ("🟡 Medium" if p>0.3 else "🟢 Low"))

        st.subheader("Churn Prediction (Logistic Regression)")
        st.caption("⚠️ Демонстрационная модель: обучена всего на 4 клиентах, "
                   "иллюстрирует логику, а не даёт статистически надёжный прогноз.")
        st.dataframe(rfm[["client_name","Recency","Frequency","Monetary_M","Churn_Prob","Risk"]], use_container_width=True)

    st.subheader("K-Means Clustering")
    # FIX: работаем с копией, чтобы не мутировать закэшированный df
    dfc = df.copy()
    features = ["spend_kzt","CTR","CVR","ROI","CPC"]
    X_c = StandardScaler().fit_transform(dfc[features])
    dfc["Cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_c)

    # FIX: номера кластеров у KMeans случайны -> ранжируем по реальному ROI
    # (худший -> лучший) и только потом вешаем подписи Low/Medium/High
    order = dfc.groupby("Cluster")["ROI"].mean().sort_values().index
    rank_map = {cl: r for r, cl in enumerate(order)}
    perf_labels = {0: "🔴 Low Performance", 1: "🟡 Medium Performance", 2: "🟢 High Performance"}
    dfc["Performance"] = dfc["Cluster"].map(rank_map).map(perf_labels)

    cluster_summary = dfc.groupby("Performance").agg(
        count=("campaign_id","count"),
        avg_ctr=("CTR","mean"),
        avg_roi=("ROI","mean"),
        avg_spend=("spend_kzt","mean")
    ).round(1).reindex(list(perf_labels.values()))
    st.dataframe(cluster_summary, use_container_width=True)

# ── PAGE 4: FORECAST ──────────────────────────────────────
elif page == "📈 Campaign Forecast":
    st.title("📈 Conversion Forecast")
    st.divider()

    st.subheader("Adjust Campaign Parameters")
    col1, col2 = st.columns(2)
    with col1:
        spend = st.slider("Budget (KZT)", 50000, 500000, 200000, 10000)
        ctr   = st.slider("Expected CTR (%)", 2.0, 10.0, 5.5, 0.1)
    with col2:
        impressions = st.slider("Expected Impressions", 100000, 5000000, 2000000, 100000)
        cpc = st.slider("Expected CPC (KZT)", 1.0, 10.0, 2.5, 0.1)

    X_reg = df[["spend_kzt","impressions","CTR","CPC"]].values
    y_reg = df["conversions"].values
    reg = LinearRegression()
    reg.fit(X_reg, y_reg)
    pred = int(reg.predict([[spend, impressions, ctr, cpc]])[0])

    st.metric("Predicted Conversions", f"{max(0, pred):,}")
    st.info(f"With a budget of {spend:,} KZT, CTR of {ctr}%, and {impressions:,} impressions — estimated conversions: **{max(0,pred):,}**")

# ── PAGE 5: BUDGET OPTIMIZER ──────────────────────────────
elif page == "💰 Budget Optimizer":
    st.title("💰 Budget Optimization")
    st.divider()

    total = st.slider("Total Budget (M KZT)", 1, 20, 10) * 1_000_000
    roi_by_client = df.groupby("client_name")["ROI"].mean() / 100
    roi_vals = roi_by_client.values

    # FIX: цель линейная -> SLSQP застревает в стартовой точке и НЕ оптимизирует.
    # Используем linprog (метод HiGHS) — корректный линейный оптимум.
    # Максимизируем sum(b*(1+roi)) == минимизируем -sum((1+roi)*b)
    from scipy.optimize import linprog
    c = -(1 + roi_vals)
    A_eq = [[1] * len(roi_vals)]
    b_eq = [total]
    bounds = [(total * 0.1, total * 0.5)] * len(roi_vals)  # каждый клиент: 10%..50%
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    alloc = result.x
    expected_revenue = float((alloc * (1 + roi_vals)).sum())

    opt = pd.DataFrame({
        "Client": roi_by_client.index,
        "Optimal Budget (KZT)": alloc.astype(int),
        "Budget Share (%)": (alloc/total*100).round(1),
        "Avg ROI (%)": (roi_vals*100).round(0)
    }).sort_values("Optimal Budget (KZT)", ascending=False)

    st.dataframe(opt, use_container_width=True)
    st.metric("Total Expected Revenue", f"{expected_revenue/1_000_000:.1f}M KZT")
    st.caption("Стратегия: концентрируем бюджет на клиентах с высшим ROI "
               "при ограничениях 10%–50% на каждого. Доход оценён по историческому ROI.")

    fig = go.Figure(go.Pie(
        labels=opt["Client"],
        values=opt["Optimal Budget (KZT)"],
        hole=0.4
    ))
    fig.update_layout(title="Optimal Budget Distribution", height=400)
    st.plotly_chart(fig, use_container_width=True)
