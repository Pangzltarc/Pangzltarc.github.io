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
    
    # K-Means Clustering
    if st.checkbox("Run K-Means Clustering"):
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X_scaled)
        labels_kmeans = kmeans.labels_
        st.write("KMeans Clustering completed!")
        st.write(f"Silhouette Score: {silhouette_score(X_scaled, labels_kmeans)}")
        st.write("Cluster Centers:")
        st.dataframe(kmeans.cluster_centers_)

    # Gaussian Mixture Model Clustering
    if st.checkbox("Run GMM Clustering"):
        gmm = GaussianMixture(n_components=3)
        gmm.fit(X_scaled)
        labels_gmm = gmm.predict(X_scaled)
        st.write("GMM Clustering completed!")
        st.write(f"Silhouette Score: {silhouette_score(X_scaled, labels_gmm)}")

    # DBSCAN Clustering
    if st.checkbox("Run DBSCAN Clustering"):
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels_dbscan = dbscan.fit_predict(X_scaled)
        st.write("DBSCAN Clustering completed!")
        st.write(f"Silhouette Score: {silhouette_score(X_scaled, labels_dbscan[labels_dbscan != -1]) if len(set(labels_dbscan)) > 1 else 'Not enough clusters'}")

    # Hierarchical Clustering
    if st.checkbox("Run Hierarchical Clustering"):
        hierarchical = AgglomerativeClustering(n_clusters=3)
        labels_hierarchical = hierarchical.fit_predict(X_scaled)
        st.write("Hierarchical Clustering completed!")
        st.write(f"Silhouette Score: {silhouette_score(X_scaled, labels_hierarchical)}")

    # Spectral Clustering
    if st.checkbox("Run Spectral Clustering"):
        spectral = SpectralClustering(n_clusters=3)
        labels_spectral = spectral.fit_predict(X_scaled)
        st.write("Spectral Clustering completed!")
        st.write(f"Silhouette Score: {silhouette_score(X_scaled, labels_spectral)}")

    # Visualization of the clusters (optional)
    st.subheader("Cluster Visualization")
    if st.checkbox("Show Cluster Visualization"):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df['Cluster'] = 'No Cluster'
        
        # Assigning cluster labels based on user-selected method
        if 'labels_kmeans' in locals():
            df['Cluster'] = labels_kmeans
        elif 'labels_gmm' in locals():
            df['Cluster'] = labels_gmm
        elif 'labels_dbscan' in locals():
            df['Cluster'] = labels_dbscan
        elif 'labels_hierarchical' in locals():
            df['Cluster'] = labels_hierarchical
        elif 'labels_spectral' in locals():
            df['Cluster'] = labels_spectral

        fig = px.scatter(df, x=X_pca[:, 0], y=X_pca[:, 1], color='Cluster')
        st.plotly_chart(fig)
