import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mt
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
from scipy import stats
import geopandas as gpd
import plotly.express as px
import folium as fl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture

# Title of the Streamlit App
st.title('Machine Learning Assignment-G6')

# File Uploader to load dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset loaded successfully!")
    st.dataframe(df.head())

    # Add missing data visualization
    st.subheader("Missing Data Visualization")
    msno.matrix(df)
    st.pyplot(plt)
    
    # Example assuming each row represents an individual
    df['Count'] = 1  # This line should only be executed if df is defined
    
    # Pivot the data
    marital_counts = df.pivot_table(index='Country', columns='Marital Status', values='Count', aggfunc='sum', fill_value=0).reset_index()
    
    # Rename columns for clarity
    marital_counts.columns = ['Country', 'Divorced', 'Married', 'Single', 'Widowed', 'Separated']
    
    # Preprocessing: Choose relevant columns and scale the data
    X = marital_counts[['Divorced', 'Married', 'Single', 'Widowed', 'Separated']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means clustering
    st.subheader("K-Means Clustering")
    n_clusters = st.slider("Select number of clusters (k)", 2, 10, value=4)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    marital_counts['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Show clustering results
    st.write("Clustering Results:")
    st.write(marital_counts[['Country', 'Cluster']].head())

    # Elbow method visualization
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans_test = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
        inertia.append(kmeans_test.inertia_)
    
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertia, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    st.pyplot(plt)

    # Silhouette score calculation
    silhouette_avg = silhouette_score(X_scaled, marital_counts['Cluster'])
    st.write(f'Silhouette Score for {n_clusters} clusters: {silhouette_avg}')
    
    # PCA for dimensionality reduction and visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = marital_counts['Cluster']
    
    st.subheader("PCA Visualization of Clusters")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='Set1')
    plt.title('K-Mean Clusters Visualization using PCA')
    st.pyplot(plt)

    # Geographical visualization of clusters using Plotly
    st.subheader("Geographical Visualization of Clusters")
    fig = px.choropleth(marital_counts, 
                        locations="Country", 
                        locationmode="country names",
                        color="Cluster", 
                        hover_name="Country", 
                        color_continuous_scale="Viridis")
    st.plotly_chart(fig)