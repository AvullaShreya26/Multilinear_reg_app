import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Multilinear Regression",
    page_icon="📊",
    layout="centered"
)

# ---------------- LOAD CSS ----------------
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ---------------- TITLE ----------------
st.markdown("""
<div class="card title-card">
<h1>📊 Multilinear Regression App</h1>
<p>Predict <b>Disease Progression</b> using multiple health features</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df

df = load_data()

# ---------------- DATASET PREVIEW ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FEATURE SELECTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🎯 Feature Selection")

features = st.multiselect(
    "Select input features",
    options=df.columns[:-1],
    default=["bmi", "bp", "s5"]
)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREPARE DATA ----------------
X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- TRAIN MODEL ----------------
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------- METRICS ----------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(features) - 1)

# ---------------- VISUALIZATION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📈 Feature vs Target (BMI Example)")

fig, ax = plt.subplots()
ax.scatter(df["bmi"], df["target"], alpha=0.6)
ax.set_xlabel("BMI")
ax.set_ylabel("Disease Progression")
st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PERFORMANCE ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("⚡ Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("MSE", f"{mse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R² Score", f"{r2:.2f}")
c4.metric("Adjusted R²", f"{adj_r2:.2f}")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- COEFFICIENTS ----------------
st.markdown("""
<div class="card">
<h3>📌 Model Coefficients</h3>
""", unsafe_allow_html=True)

for f, c in zip(features, model.coef_):
    st.markdown(f"**{f}** : {c:.3f}")

st.markdown(f"<br><b>Intercept:</b> {model.intercept_:.3f}", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔮 Predict Disease Progression")

input_data = []
for f in features:
    val = st.slider(
        f"{f}",
        float(df[f].min()),
        float(df[f].max()),
        float(df[f].mean())
    )
    input_data.append(val)

input_scaled = scaler.transform([input_data])
prediction = model.predict(input_scaled)[0]

st.markdown(
    f"<div class='prediction-box'>Predicted Value : {prediction:.2f}</div>",
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
