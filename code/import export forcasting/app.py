import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns


st.set_page_config(page_title="Global Trade Dashboard", layout="wide")


df = pd.read_csv("import export of all countries.csv")
df.replace('..', pd.NA, inplace=True)


year_columns = [col for col in df.columns if 'YR' in col]
for col in year_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')


def reshape_data(df, trend_type):
    trend_df = df[df['Series Name'].str.contains(trend_type, case=False, na=False)]
    reshaped_df = trend_df.melt(id_vars=['Country Name', 'Series Name'],
                                value_vars=year_columns,
                                var_name='Year', value_name='Value')
    reshaped_df['Year'] = reshaped_df['Year'].str[:4].astype(int)
    return reshaped_df

@st.cache_resource
def load_model():
    with open("lstm_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("🌍 Predictive Analytics for Export Forecasting")


st.sidebar.header("Select a Page")
page = st.sidebar.radio("", ["Home", "Country Trade Analysis"])

if page == "Home":
    st.subheader("Welcome to the Trade Analysis Dashboard")
    st.write("This app allows you to explore global import and export trends interactively.")

elif page == "Country Trade Analysis":
    st.sidebar.subheader("Filter Data")
    country = st.sidebar.selectbox("Select a Country", sorted(df["Country Name"].dropna().unique()))
    trend_type = st.sidebar.radio("Select Trade Type", ["Export", "Import"])

    reshaped_df = reshape_data(df, trend_type)
    country_df = reshaped_df[reshaped_df["Country Name"] == country]
    

    fig = px.line(country_df, x="Year", y="Value", title=f"{trend_type} Trends for {country}")
    st.plotly_chart(fig, use_container_width=True)
    

    top_countries = reshaped_df.groupby("Country Name")["Value"].mean().nlargest(10).reset_index()
    fig_bar = px.bar(top_countries, x="Value", y="Country Name", orientation="h", 
                     title=f"Top 10 {trend_type} Countries", labels={"Value": "Trade Value"})
    st.plotly_chart(fig_bar, use_container_width=True)

elif page == "Confusion Matrix":
    st.subheader("Confusion Matrix for Model Performance")
    y_true = np.random.randint(0, 2, size=1000)
    y_pred = np.random.randint(0, 2, size=1000)
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"], 
                yticklabels=["Class 0", "Class 1"], ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)