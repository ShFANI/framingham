
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score

st.set_page_config(page_title="Framingham CHD Prediction", layout="wide")
st.title("🫀 Framingham Heart Study - CHD Risk Prediction")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("framingham.csv")

df = load_data()

# Sidebar EDA options
st.sidebar.header("🔍 Exploratory Data Analysis")
eda_option = st.sidebar.selectbox("Choose analysis type", ["Data Overview", "Correlation Heatmap", "Distribution Plot"])

if eda_option == "Data Overview":
    st.subheader("Dataset Preview")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Missing Values:", df.isnull().sum())
elif eda_option == "Correlation Heatmap":
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(plt)
elif eda_option == "Distribution Plot":
    st.subheader("Distribution of Numerical Features")
    num_col = st.selectbox("Select column", df.select_dtypes(include="number").columns)
    fig, ax = plt.subplots()
    sns.histplot(df[num_col], kde=True, ax=ax)
    st.pyplot(fig)

# Preprocessing
df_clean = df.dropna()
X = df_clean.drop("TenYearCHD", axis=1)
y = df_clean["TenYearCHD"]
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model selection
st.sidebar.header("⚙️ Machine Learning Models")
model_option = st.sidebar.selectbox("Choose a model", ["Logistic Regression", "Random Forest", "Neural Network"])

if model_option == "Logistic Regression":
    model = LogisticRegression()
elif model_option == "Random Forest":
    model = RandomForestClassifier()
elif model_option == "Neural Network":
    model = MLPClassifier(max_iter=1000)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Output results
st.subheader(f"Results: {model_option}")
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))
st.write("ROC AUC Score:", roc_auc_score(y_test, y_proba))
