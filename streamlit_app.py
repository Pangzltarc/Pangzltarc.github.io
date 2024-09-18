import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Function to plot missing data
def plot_missing_data(df):
    st.subheader("Missing Data Visualization")
    msno.matrix(df)
    st.pyplot(plt)

# Function to perform K-Means clustering
def perform_kmeans(X_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    return kmeans, clusters

# Function to plot Elbow Method
def plot_elbow_method(X_scaled):
    st.subheader("Elbow Method")
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

# Function to plot PCA
def plot_pca(X_scaled, clusters):
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = clusters
    
    st.subheader("PCA Visualization of Clusters")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='Set1')
    plt.title('K-Mean Clusters Visualization using PCA')
    st.pyplot(plt)

# Function to plot geographical data
def plot_geographical_clusters(marital_counts):
    st.subheader("Geographical Visualization of Clusters")
    fig = px.choropleth(marital_counts, 
                        locations="Country", 
                        locationmode="country names",
                        color="Cluster", 
                        hover_name="Country", 
                        color_continuous_scale="Viridis")
    st.plotly_chart(fig)

# Title of the Streamlit App
st.title('Machine Learning Assignment-G6')

# File Uploader to load dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset loaded successfully!")
    st.dataframe(df.head())
    
    # Missing Data Visualization
    plot_missing_data(df)
    
    # Preprocessing
    df['Count'] = 1
    marital_counts = df.pivot_table(index='Country', columns='Marital Status', values='Count', aggfunc='sum', fill_value=0).reset_index()
    marital_counts.columns = ['Country', 'Divorced', 'Married', 'Single', 'Widowed', 'Separated']
    
    X = marital_counts[['Divorced', 'Married', 'Single', 'Widowed', 'Separated']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means Clustering
    st.subheader("K-Means Clustering")
    n_clusters = st.slider("Select number of clusters (k)", 2, 10, value=4)
    kmeans, clusters = perform_kmeans(X_scaled, n_clusters)
    
    marital_counts['Cluster'] = clusters
    st.write("Clustering Results:")
    st.write(marital_counts[['Country', 'Cluster']].head())
    
    # Elbow Method Visualization
    plot_elbow_method(X_scaled)
    
    # Silhouette Score
    silhouette_avg = silhouette_score(X_scaled, clusters)
    st.write(f'Silhouette Score for {n_clusters} clusters: {silhouette_avg}')
    
    # PCA Visualization
    plot_pca(X_scaled, clusters)
    
    # Geographical Visualization
    plot_geographical_clusters(marital_counts)
    


