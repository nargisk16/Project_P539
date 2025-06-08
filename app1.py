import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Page config
st.set_page_config(page_title="Customer Personality Analysis", layout="wide")
st.title("🧠 Customer Segmentation App")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_excel("marketing_campaign.xlsx")

data = load_data()

# Show data
st.subheader("📄 Raw Data")
st.write(data.head())

# Preprocessing
def preprocess(df):
    df = df.copy()
    df['Age'] = 2025 - df['Year_Birth']
    df['Family_Size'] = df['Kidhome'] + df['Teenhome']
    df['TotalSpent'] = df[['MntWines', 'MntFruits', 'MntMeatProducts',
                           'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
    selected_features = ['Age', 'Income', 'Recency', 'Family_Size', 'TotalSpent']
    df_model = df[selected_features].dropna()
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_model)
    return df_model, df_scaled, scaler, selected_features

df_model, df_scaled, scaler, selected_features = preprocess(data)

# Sidebar: Cluster count
st.sidebar.title("🔧 Model Settings")
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=4)

# Train KMeans
@st.cache_resource
def train_model(data_scaled, k):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(data_scaled)
    return model

model = train_model(df_scaled, n_clusters)
clusters = model.predict(df_scaled)
df_model['Cluster'] = clusters

# Show clustered data
st.subheader("📊 Customer Segments")
st.write(df_model.head())

# Cluster distribution
st.subheader("📈 Cluster Distribution")
st.bar_chart(df_model['Cluster'].value_counts().sort_index())

# Cluster profiles
st.subheader("📌 Cluster Profiles")
st.dataframe(df_model.groupby("Cluster").mean())

# PCA Visualization
st.subheader("🧬 Cluster Visualization (PCA)")
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df_scaled)
df_model['PCA1'] = pca_data[:, 0]
df_model['PCA2'] = pca_data[:, 1]

fig, ax = plt.subplots()
sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=df_model, palette="tab10", ax=ax)
plt.title("Customer Segments (PCA 2D View)")
st.pyplot(fig)

# --- Manual Prediction Section ---
st.markdown("---")
st.header("📝 Predict Cluster for New Customer (Manual Input)")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=50)

    with col2:
        income = st.number_input("Yearly Income", min_value=0, step=1000, value=50000)
        kidhome = st.number_input("Number of Kids", min_value=0, max_value=5, value=1)
        teenhome = st.number_input("Number of Teens", min_value=0, max_value=5, value=1)

    with col3:
        mnt_wines = st.number_input("Amount spent on Wine", min_value=0, value=200)
        mnt_meat = st.number_input("Amount spent on Meat", min_value=0, value=300)
        mnt_fruits = st.number_input("Amount spent on Fruits", min_value=0, value=100)
        mnt_fish = st.number_input("Amount spent on Fish", min_value=0, value=100)
        mnt_sweets = st.number_input("Amount spent on Sweets", min_value=0, value=50)
        mnt_gold = st.number_input("Amount spent on Gold", min_value=0, value=150)

    submitted = st.form_submit_button("Predict Cluster")

    if submitted:
        family_size = kidhome + teenhome
        total_spent = mnt_wines + mnt_meat + mnt_fruits + mnt_fish + mnt_sweets + mnt_gold

        # Create feature vector
        input_features = pd.DataFrame([[age, income, recency, family_size, total_spent]],
                                      columns=selected_features)
        input_scaled = scaler.transform(input_features)
        predicted_cluster = model.predict(input_scaled)[0]

        # Define cluster name mapping
        cluster_names = {
            0: "💎 High-Value Customers",
            1: "🛍️ Regular Buyers",
            2: "🧊 Low Engagement",
            3: "🎯 Deal Seekers",
            4: "🔍 New/Occasional Buyers",
            5: "📉 Price-Sensitive Shoppers",
            6: "🎁 Seasonal Buyers",
            7: "🚀 Fast Movers",
            8: "🧠 Explorers",
            9: "📦 Bulk Buyers"
        }

        # Get descriptive name
        cluster_label = cluster_names.get(predicted_cluster, f"Cluster {predicted_cluster}")
        st.success(f"🧩 This customer belongs to **{cluster_label}**.")
        st.write("📌 Feature Summary:")
        st.write(input_features)
