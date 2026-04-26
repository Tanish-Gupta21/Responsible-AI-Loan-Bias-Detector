import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Problem Statement")
st.header("Loan Approval Bias Detection Dashboard")

def load_data():
    df = pd.read_csv("loan_data.csv")
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
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    except:
        df[col] = df[col].fillna(df[col].mode()[0])




st.write("Rows:", df.shape[0])
st.write("Columns:", df.shape[1])

st.subheader("Income vs Loan Approval")
df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
df['Loan_Approved'] = pd.to_numeric(df['Loan_Approved'], errors='coerce')
df = df.dropna(subset=['Income', 'Loan_Approved'])


st.subheader("Income vs Loan Approval")

# Column exist check
if 'Income' in df.columns and 'Loan_Approved' in df.columns:

    # Convert to numeric
    df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
    df['Loan_Approved'] = pd.to_numeric(df['Loan_Approved'], errors='coerce')

    # Remove null values
    temp_df = df[['Income', 'Loan_Approved']].dropna()

    if not temp_df.empty:
        fig, ax = plt.subplots()
        temp_df.boxplot(column='Income', by='Loan_Approved', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough data to plot")

else:
    st.error("Required columns not found")



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




