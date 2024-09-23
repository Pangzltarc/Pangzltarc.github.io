import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Title of the Streamlit App
st.title('Machine Learning Assignment - G6')

# File Uploader to load dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset loaded successfully!")
    st.dataframe(df.head())

    # Missing data visualization
    st.subheader("Missing Data Visualization")
    msno.matrix(df)
    st.pyplot(plt)

    # Selecting numeric columns for scaling
    X = df.select_dtypes(include=[np.number])
    
    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering Section
    st.subheader("Clustering Analysis")

     # Assuming each row represents an individual
    df['Count'] = 1  # This line should only be executed if df is defined
    
    # Pivot the data
    marital_counts = df.pivot_table(index='Country', columns='Marital Status', values='Count', aggfunc='sum', fill_value=0).reset_index()
    
    # Rename columns for clarity
    marital_counts.columns = ['Country', 'Divorced', 'Married', 'Single', 'Widowed', 'Separated']

    # K-Means Clustering
    if st.checkbox("Run K-Means Clustering"):
        n_clusters = st.slider("Select number of clusters (k)", 2, 10, value=4)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels_kmeans = kmeans.fit_predict(X_scaled)
        st.write("KMeans Clustering completed!")
        st.write(f"Silhouette Score: {silhouette_score(X_scaled, labels_kmeans)}")
        st.dataframe(pd.DataFrame({'Cluster': labels_kmeans}, index=df.index))

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

    # GMM Clustering
    if st.checkbox("Run GMM Clustering"):
        n_clusters_gmm = st.slider("Select number of clusters for GMM", 2, 10, value=4)
        gmm = GaussianMixture(n_components=n_clusters_gmm, random_state=42)
        labels_gmm = gmm.fit_predict(X_scaled)
        st.write("GMM Clustering completed!")
        st.write(f"Silhouette Score: {silhouette_score(X_scaled, labels_gmm)}")
        st.dataframe(pd.DataFrame({'Cluster': labels_gmm}, index=df.index))

    # DBSCAN Clustering
    if st.checkbox("Run DBSCAN Clustering"):
        eps = st.slider("Select eps for DBSCAN", 0.1, 10.0, value=0.5)
        min_samples = st.slider("Select min_samples for DBSCAN", 1, 10, value=5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels_dbscan = dbscan.fit_predict(X_scaled)
        st.write("DBSCAN Clustering completed!")
        st.write(f"Silhouette Score: {silhouette_score(X_scaled, labels_dbscan[labels_dbscan != -1]) if len(set(labels_dbscan)) > 1 else 'Not enough clusters'}")
        st.dataframe(pd.DataFrame({'Cluster': labels_dbscan}, index=df.index))

    # Hierarchical Clustering
    if st.checkbox("Run Hierarchical Clustering"):
        n_clusters_hierarchical = st.slider("Select number of clusters for Hierarchical Clustering", 2, 10, value=4)
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters_hierarchical)
        labels_hierarchical = hierarchical.fit_predict(X_scaled)
        st.write("Hierarchical Clustering completed!")
        st.write(f"Silhouette Score: {silhouette_score(X_scaled, labels_hierarchical)}")
        st.dataframe(pd.DataFrame({'Cluster': labels_hierarchical}, index=df.index))

    # Spectral Clustering
    if st.checkbox("Run Spectral Clustering"):
        n_clusters_spectral = st.slider("Select number of clusters for Spectral Clustering", 2, 10, value=4)
        spectral = SpectralClustering(n_clusters=n_clusters_spectral, random_state=42)
        labels_spectral = spectral.fit_predict(X_scaled)
        st.write("Spectral Clustering completed!")
        st.write(f"Silhouette Score: {silhouette_score(X_scaled, labels_spectral)}")
        st.dataframe(pd.DataFrame({'Cluster': labels_spectral}, index=df.index))

    # Visualization of the clusters
    st.subheader("Cluster Visualization")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    
    # Combine cluster labels into PCA DataFrame
    if 'labels_kmeans' in locals():
        pca_df['Cluster'] = labels_kmeans
    elif 'labels_gmm' in locals():
        pca_df['Cluster'] = labels_gmm
    elif 'labels_dbscan' in locals():
        pca_df['Cluster'] = labels_dbscan
    elif 'labels_hierarchical' in locals():
        pca_df['Cluster'] = labels_hierarchical
    elif 'labels_spectral' in locals():
        pca_df['Cluster'] = labels_spectral
    
    st.subheader("PCA Visualization of Clusters")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='Set1')
    plt.title('Clusters Visualization using PCA')
    st.pyplot(plt)

    # Geographical visualization (if applicable)
    if 'Country' in df.columns:  # Assuming 'Country' is a column in your data
        st.subheader("Geographical Visualization of Clusters")
        fig = px.choropleth(marital_counts, 
                            locations="Country", 
                            locationmode="country names",
                            color="Cluster", 
                            hover_name="Country", 
                            color_continuous_scale="Viridis")
        st.plotly_chart(fig)
