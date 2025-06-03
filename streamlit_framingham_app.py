
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

st.title("Framingham Heart Disease Prediction")

# Load data
uploaded_file = st.file_uploader("Upload the Framingham dataset CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_clean = df.dropna()

    st.subheader("Data Overview")
    st.dataframe(df_clean.head())

    # Feature selection
    X = df_clean.drop(columns=["TenYearCHD"])
    y = df_clean["TenYearCHD"]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    st.subheader("Model Performance (Logistic Regression)")
    st.write(f"Accuracy: {acc:.3f}")
    st.write(f"ROC AUC: {auc:.3f}")

    # Correlation
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_clean.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Distribution plots
    st.subheader("Feature Distributions")
    feature = st.selectbox("Select a feature to explore by CHD Risk", df_clean.columns[:-1])
    fig2, ax2 = plt.subplots()
    sns.histplot(data=df_clean, x=feature, hue="TenYearCHD", multiple="stack", ax=ax2)
    st.pyplot(fig2)
