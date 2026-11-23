"""
Retail Sales - Interactive Dashboard (Streamlit)

Run locally:   streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parent / "data" / "cleaned_retail_data.csv"
BLUE, ORANGE, GRAY = "#1F4E79", "#E07B39", "#B8BFC9"
SEG_COLORS = {"Consumer": "#1F4E79", "Corporate": "#5B9BD5", "Home Office": "#A9BCD0"}

st.set_page_config(page_title="Retail Sales Dashboard", page_icon="📊", layout="wide")


@st.cache_data
def load_data():
    d = pd.read_csv(DATA, parse_dates=["Order_Date", "Month"])
    d["MonthLabel"] = d["Month"].dt.strftime("%Y-%m")
    return d


df = load_data()

# ----------------------------------------------------------------- sidebar: 4 filters
st.sidebar.header("Filters")
months = sorted(df["MonthLabel"].unique())
m_from, m_to = st.sidebar.select_slider("Month range", options=months, value=(months[0], months[-1]))
all_regions = sorted(df["Region"].unique())
all_cats = sorted(df["Product_Category"].unique())
all_segs = sorted(df["Customer_Segment"].unique())
regions = st.sidebar.multiselect("Region", all_regions, default=all_regions)
cats = st.sidebar.multiselect("Product category", all_cats, default=all_cats)
segs = st.sidebar.multiselect("Customer segment", all_segs, default=all_segs)

f = df[(df["MonthLabel"] >= m_from) & (df["MonthLabel"] <= m_to)
        & df["Region"].isin(regions) & df["Product_Category"].isin(cats) & df["Customer_Segment"].isin(segs)]

st.title("📊 Retail Sales Dashboard")
st.caption("Retail transactions 2023–2024 · change the filters on the left, every chart updates · hover over charts for details")

if f.empty:
    st.warning("No data for this combination of filters. Select at least one value in every filter.")
    st.stop()

# ----------------------------------------------------------------- KPIs
sales, profit = f["Sales"].sum(), f["Profit"].sum()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Sales", f"₹{sales / 1e6:,.2f}M")
k2.metric("Total Profit", f"₹{profit / 1e6:,.2f}M")
k3.metric("Profit Margin", f"{profit / sales:.1%}")
k4.metric("Average Order Value", f"₹{f['Sales'].mean():,.0f}")
k5.metric("Return Rate", f"{f['Returned_Flag'].mean():.1%}")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Sales overview", "Customers", "Relationships", "Returns & satisfaction"])

# ----------------------------------------------------------------- tab 1: overview
with tab1:
    monthly = f.groupby("Month", as_index=False)["Sales"].sum()
    fig = px.line(monthly, x="Month", y="Sales", markers=True, title="Monthly sales trend",
                    color_discrete_sequence=[BLUE], labels={"Sales": "Sales (₹)", "Month": "Order month"})
    st.plotly_chart(fig)

    c1, c2 = st.columns(2)
    reg = f.groupby("Region", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    weakest = reg.loc[reg["Sales"].idxmin(), "Region"]
    reg["Highlight"] = np.where(reg["Region"] == weakest, "Weakest region", "Other regions")
    fig = px.bar(reg, x="Region", y="Sales", color="Highlight", title="Total sales by region",
                    color_discrete_map={"Weakest region": ORANGE, "Other regions": BLUE}, labels={"Sales": "Sales (₹)"})
    c1.plotly_chart(fig)

    cat = f.groupby("Product_Category", as_index=False)["Sales"].sum().sort_values("Sales")
    fig = px.bar(cat, x="Sales", y="Product_Category", orientation="h", title="Total sales by product category",
                    color_discrete_sequence=[BLUE], labels={"Sales": "Sales (₹)", "Product_Category": "Category"})
    c2.plotly_chart(fig)

    piv = f.pivot_table(index="Product_Category", columns="Region", values="Sales", aggfunc="mean")
    fig = px.imshow(piv, text_auto=".0f", aspect="auto", color_continuous_scale="Blues",
                    title="Average sales per transaction: category × region",
                    labels={"color": "Avg sales (₹)"})
    st.plotly_chart(fig)

# ----------------------------------------------------------------- tab 2: customers
with tab2:
    c1, c2 = st.columns(2)
    fig = px.histogram(f.dropna(subset=["Age"]), x="Age", nbins=25, title="Distribution of customer age",
                        color_discrete_sequence=[BLUE], labels={"Age": "Age (years)"})
    fig.update_layout(yaxis_title="Transactions", bargap=0.05)
    c1.plotly_chart(fig)

    inc = f.dropna(subset=["Income"])
    fig = px.histogram(inc, x="Income", nbins=40, title="Distribution of annual income (right-skewed)",
                        color_discrete_sequence=[BLUE], labels={"Income": "Annual income (₹)"})
    fig.update_layout(yaxis_title="Transactions", bargap=0.05)
    c2.plotly_chart(fig)

    seg = f.groupby("Customer_Segment", as_index=False)["Sales"].sum()
    fig = px.pie(seg, names="Customer_Segment", values="Sales", hole=0.5, title="Share of sales by customer segment",
                    color="Customer_Segment", color_discrete_map=SEG_COLORS)
    st.plotly_chart(fig)

# ----------------------------------------------------------------- tab 3: relationships
with tab3:
    c1, c2 = st.columns(2)

    def scatter_with_trend(data, x, y, title, xlabel):
        d = data[[x, y]].dropna()
        sample = d.sample(min(len(d), 3000), random_state=1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sample[x], y=sample[y], mode="markers", name="Transactions",
                            marker=dict(color=BLUE, size=5, opacity=0.35)))
        if len(d) > 2:
            slope, intercept = np.polyfit(d[x], d[y], 1)
            xs = np.linspace(d[x].min(), d[x].max(), 50)
            r = d[x].corr(d[y])
            fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name=f"Trend line (r = {r:.2f})",
                            line=dict(color=ORANGE, width=3)))
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="Profit per transaction (₹)")
        return fig

    c1.plotly_chart(scatter_with_trend(f, "Sales", "Profit", "Sales vs profit", "Sales per transaction (₹)"))
    c2.plotly_chart(scatter_with_trend(f, "Discount", "Profit", "Discount vs profit", "Discount (share of list price)"))

    fig = px.box(f, x="Product_Category", y="Profit", title="Profit per transaction by category",
                            color_discrete_sequence=["#7FA7CF"], labels={"Product_Category": "Category", "Profit": "Profit (₹)"})
    st.plotly_chart(fig)

    corr = f[["Sales", "Profit", "Discount", "Quantity", "Age", "Income"]].corr()
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                    title="Correlation between key numeric variables", labels={"color": "Correlation"})
    st.plotly_chart(fig)

# ----------------------------------------------------------------- tab 4: returns & satisfaction
with tab4:
    c1, c2 = st.columns(2)
    ret = (f.groupby("Product_Category")["Returned_Flag"].mean() * 100).reset_index(name="Return rate (%)")
    ret["Highlight"] = np.where(ret["Return rate (%)"] == ret["Return rate (%)"].max(), "Highest", "Other")
    fig = px.bar(ret, x="Product_Category", y="Return rate (%)", color="Highlight", title="Return rate by category",
                    color_discrete_map={"Highest": ORANGE, "Other": BLUE}, labels={"Product_Category": "Category"})
    c1.plotly_chart(fig)

    sat = f.dropna(subset=["Satisfaction_Level"])
    ct = (pd.crosstab(sat["Delivery_Mode"], sat["Satisfaction_Level"], normalize="index") * 100).reset_index()
    ct = ct.melt(id_vars="Delivery_Mode", var_name="Satisfaction", value_name="Share (%)")
    fig = px.bar(ct, x="Share (%)", y="Delivery_Mode", color="Satisfaction", orientation="h",
                    title="Satisfaction by delivery mode",
                    category_orders={"Satisfaction": ["High", "Medium", "Low"]},
                    color_discrete_map={"High": BLUE, "Medium": GRAY, "Low": ORANGE}, labels={"Delivery_Mode": "Delivery mode"})
    c2.plotly_chart(fig)

with st.expander("Show filtered data table"):
    st.dataframe(f.drop(columns=["MonthLabel"]).head(500))
