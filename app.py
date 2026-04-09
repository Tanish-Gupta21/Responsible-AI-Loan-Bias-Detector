import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Problem Statement")
st.header("Loan Approval Bias Detection Dashboard")

def load_data():
    df = pd.read_csv("Loan Dataset.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())


st.subheader("Missing Values")
st.write(df.isnull().sum())


if st.button("Remove Missing Values"):
    df = df.dropna()
    st.write("Cleaned Data")
    st.write(df)


if st.button("Fill Missing Values"):
    df = df.fillna(method='ffill')
    st.write("Filled Data")
    st.write(df)

if st.button("Remove Duplicates"):
    df = df.drop_duplicates()
    st.write("Duplicates Removed")
    st.write(df)


st.subheader("Columns")
st.write(df.columns)



df = df.drop_duplicates()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())


st.write("Rows:", df.shape[0])
st.write("Columns:", df.shape[1])

st.subheader("Income vs Loan Approval")

fig, ax = plt.subplots()
df.boxplot(column='Income', by='Loan_Approved', ax=ax)
st.pyplot(fig)


st.header("Automatic Bias Detection")

target = "Loan_Approved"

categorical_cols = df.select_dtypes(include="object").columns.tolist()


categorical_cols = [c for c in categorical_cols if c != target]

selected_col = st.selectbox("Select column for bias check", categorical_cols)

bias = pd.crosstab(df[selected_col], df[target], normalize='index')

fig, ax = plt.subplots()
bias.plot(kind='bar', ax=ax)
plt.title(f"{selected_col} vs Loan Approval")
st.pyplot(fig)

if 1 in bias.columns:
    approval_rates = bias[1]
else:
    approval_rates = bias.iloc[:, -1]

fairness = 1 - (approval_rates.max() - approval_rates.min())
st.success(f"Fairness Score: {round(fairness*100,2)} %")


st.header("Numeric Data Visualization")

numeric_cols = df.select_dtypes(include=np.number).columns
num_col = st.selectbox("Select numeric column", numeric_cols)

fig2, ax2 = plt.subplots(figsize=(10,5))
df[num_col].dropna().hist(bins=25, ax=ax2)
plt.title(num_col)
st.pyplot(fig2)

if st.checkbox("Show full dataset"):
    st.dataframe(df)




