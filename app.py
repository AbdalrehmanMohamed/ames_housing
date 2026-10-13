import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Estimator", page_icon="🏠")


@st.cache_resource
def load_bundle():
    return joblib.load("house_model.joblib")


b = load_bundle()
opt, rng, dflt = b["options"], b["ranges"], b["defaults"]

st.title("🏠 House Price Estimator")
st.caption("Based on the Ames, Iowa housing dataset. Prices are in US dollars.")

with st.sidebar:
    st.header("Model info")
    st.write("**Model:** Ridge Regression")
    st.write(f"**R² (log price):** {b['r2']:.3f}")
    st.write(f"**Average error (MAE):** ${b['mae']:,.0f}")

c1, c2 = st.columns(2)
with c1:
    qual = st.slider("Overall quality (1-10)", 1, 10, dflt["OverallQual"])
    area = st.number_input("Living area (sq ft)", rng["GrLivArea"][0], rng["GrLivArea"][1],
                           dflt["GrLivArea"], step=50)
    bsmt = st.number_input("Basement area (sq ft)", 0, rng["TotalBsmtSF"][1], dflt["TotalBsmtSF"], step=50)
    lot = st.number_input("Lot area (sq ft)", rng["LotArea"][0], rng["LotArea"][1], dflt["LotArea"], step=500)
    year = st.slider("Year built", rng["YearBuilt"][0], rng["YearBuilt"][1], dflt["YearBuilt"])
with c2:
    neigh = st.selectbox("Neighborhood", opt["Neighborhood"],
                         index=opt["Neighborhood"].index(dflt["Neighborhood"]))
    kitchen = st.selectbox("Kitchen quality", opt["KitchenQual"],
                           index=opt["KitchenQual"].index(dflt["KitchenQual"]),
                           help="Ex = Excellent, Gd = Good, TA = Typical/Average, Fa = Fair, Po = Poor")
    air = st.selectbox("Central air", opt["CentralAir"], index=opt["CentralAir"].index(dflt["CentralAir"]))
    garage = st.slider("Garage capacity (cars)", rng["GarageCars"][0], rng["GarageCars"][1], dflt["GarageCars"])
    baths = st.slider("Full bathrooms", rng["FullBath"][0], rng["FullBath"][1], dflt["FullBath"])

if st.button("Estimate price", type="primary"):
    row = pd.DataFrame([{
        "OverallQual": qual, "GrLivArea": area, "TotalBsmtSF": bsmt, "GarageCars": garage,
        "FullBath": baths, "YearBuilt": year, "LotArea": lot,
        "Neighborhood": neigh, "KitchenQual": kitchen, "CentralAir": air,
    }])[b["num"] + b["cat"]]
    price = float(np.expm1(b["model"].predict(row)[0]))
    st.success(f"Estimated price: **${price:,.0f}**")
    st.write(f"Typical range (± average error): ${max(price - b['mae'], 0):,.0f} – ${price + b['mae']:,.0f}")

st.caption("This is a machine-learning estimate for demonstration, not an official appraisal.")
