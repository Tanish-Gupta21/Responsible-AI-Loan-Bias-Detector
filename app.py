import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Problem Statement")
st.header("Loan Approval Bias Detection Dashboard")


@st.cache_data
def load_data():
    df = pd.read_csv("loan_data.csv")
    for col in df.columns:
        try:
            converted = pd.to_numeric(df[col])
            df[col] = converted
        except (ValueError, TypeError):
            pass  
    return df


df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Missing Values")
st.write(df.isnull().sum())

if "df" not in st.session_state:
    st.session_state.df = df.copy()

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Remove Missing Values"):
        st.session_state.df = st.session_state.df.dropna()
        st.success("Missing values removed.")

with col2:
    if st.button("Fill Missing Values"):
        st.session_state.df = st.session_state.df.ffill() 
        st.success("Missing values forward-filled.")

with col3:
    if st.button("Remove Duplicates"):
        st.session_state.df = st.session_state.df.drop_duplicates()
        st.success("Duplicates removed.")

df = st.session_state.df

st.subheader("Columns")
st.write(df.columns.tolist())

df = df.drop_duplicates()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    elif pd.api.types.is_numeric_dtype(df[col]):  
        df[col] = df[col].fillna(df[col].median())

st.write("Rows:", df.shape[0])
st.write("Columns:", df.shape[1])

st.subheader("Income vs Loan Approval")

income_col = next((c for c in df.columns if "income" in c.lower()), None)
target_col = next((c for c in df.columns if "loan_approved" in c.lower() or "approved" in c.lower()), None)

if income_col and target_col:
    fig, ax = plt.subplots()
    df.boxplot(column=income_col, by=target_col, ax=ax)
    plt.suptitle("")
    ax.set_title(f"{income_col} vs {target_col}")
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning(f"Could not find Income or Loan_Approved columns. Available: {df.columns.tolist()}")

st.header("Automatic Bias Detection")

if target_col is None:
    st.error("No 'Loan_Approved' column found. Check your CSV column names.")
    st.stop()

target = target_col
categorical_cols = df.select_dtypes(include="object").columns.tolist()
categorical_cols = [c for c in categorical_cols if c != target]

if not categorical_cols:
    st.warning("No categorical columns found for bias check.")
else:
    selected_col = st.selectbox("Select column for bias check", categorical_cols)
    bias = pd.crosstab(df[selected_col], df[target], normalize="index")

    fig, ax = plt.subplots()
    bias.plot(kind="bar", ax=ax)
    plt.title(f"{selected_col} vs {target}")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if 1 in bias.columns:
        approval_rates = bias[1]
    elif "Y" in bias.columns:
        approval_rates = bias["Y"]
    elif "Yes" in bias.columns:
        approval_rates = bias["Yes"]
    else:
        approval_rates = bias.iloc[:, -1]

    fairness = 1 - (approval_rates.max() - approval_rates.min())
    st.success(f"Fairness Score: {round(fairness * 100, 2)} %")

st.header("Numeric Data Visualization")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if not numeric_cols:
    st.warning("No numeric columns found.")
else:
    num_col = st.selectbox("Select numeric column", numeric_cols)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    df[num_col].dropna().hist(bins=25, ax=ax2)
    ax2.set_title(num_col)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

if st.checkbox("Show full dataset"):
    st.dataframe(df)
