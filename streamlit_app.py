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

from sklearn.cluster import AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture

# Additional clustering methods

# Hierarchical Clustering
st.subheader("Hierarchical Clustering")
n_clusters_hierarchical = st.slider("Select number of clusters for Hierarchical Clustering", 2, 10, value=4)
hierarchical = AgglomerativeClustering(n_clusters=n_clusters_hierarchical)
marital_counts['Hierarchical Cluster'] = hierarchical.fit_predict(X_scaled)
st.write(marital_counts[['Country', 'Hierarchical Cluster']].head())

# PCA Visualization for Hierarchical Clustering
st.subheader("PCA Visualization of Hierarchical Clusters")
pca_df['Hierarchical Cluster'] = marital_counts['Hierarchical Cluster']
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Hierarchical Cluster', data=pca_df, palette='Set2')
plt.title('Hierarchical Clustering Visualization using PCA')
st.pyplot(plt)

# Spectral Clustering
st.subheader("Spectral Clustering")
n_clusters_spectral = st.slider("Select number of clusters for Spectral Clustering", 2, 10, value=4)
spectral = SpectralClustering(n_clusters=n_clusters_spectral, random_state=42)
marital_counts['Spectral Cluster'] = spectral.fit_predict(X_scaled)
st.write(marital_counts[['Country', 'Spectral Cluster']].head())

# PCA Visualization for Spectral Clustering
st.subheader("PCA Visualization of Spectral Clusters")
pca_df['Spectral Cluster'] = marital_counts['Spectral Cluster']
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Spectral Cluster', data=pca_df, palette='Set3')
plt.title('Spectral Clustering Visualization using PCA')
st.pyplot(plt)

# GMM Clustering
st.subheader("GMM Clustering")
n_clusters_gmm = st.slider("Select number of clusters for GMM", 2, 10, value=4)
gmm = GaussianMixture(n_components=n_clusters_gmm, random_state=42)
marital_counts['GMM Cluster'] = gmm.fit_predict(X_scaled)
st.write(marital_counts[['Country', 'GMM Cluster']].head())

# PCA Visualization for GMM Clustering
st.subheader("PCA Visualization of GMM Clusters")
pca_df['GMM Cluster'] = marital_counts['GMM Cluster']
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='GMM Cluster', data=pca_df, palette='Set1')
plt.title('GMM Clustering Visualization using PCA')
st.pyplot(plt)

# DBSCAN Clustering
st.subheader("DBSCAN Clustering")
eps = st.slider("Select eps for DBSCAN", 0.1, 10.0, value=0.5)
min_samples = st.slider("Select min_samples for DBSCAN", 1, 10, value=5)
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
marital_counts['DBSCAN Cluster'] = dbscan.fit_predict(X_scaled)
st.write(marital_counts[['Country', 'DBSCAN Cluster']].head())

# PCA Visualization for DBSCAN Clustering
st.subheader("PCA Visualization of DBSCAN Clusters")
pca_df['DBSCAN Cluster'] = marital_counts['DBSCAN Cluster']
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='DBSCAN Cluster', data=pca_df, palette='Set1')
plt.title('DBSCAN Clustering Visualization using PCA')
st.pyplot(plt)