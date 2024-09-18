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

    # Check actual columns in the DataFrame
    st.write("Columns in the DataFrame:")
    st.write(df.columns)

    # Ensure the columns you are trying to use exist in the DataFrame
    columns = ['Country', 'Age Group', 'Marital Status', 'Population_Count']  # Ensure these are correct
    marriage_data = df[columns].dropna()

    # Pivot the data to create a matrix for clustering
    pivot_data = marriage_data.pivot_table(index='Country', columns='Age Group', values='Population_Count', fill_value=0)

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot_data)

    # Check scaled data shape
    st.write("Scaled data shape:")
    st.write(scaled_data.shape)

    # Use the Elbow method to find the optimal number of clusters
    sse = []
    k_range = range(1, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        sse.append(kmeans.inertia_)

    # Plot the Elbow graph
    plt.figure(figsize=(10,6))
    plt.plot(k_range, sse, marker='o')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of squared distances (Inertia)')
    plt.grid(True)
    st.pyplot(plt)

    # Apply K-Means with the optimal number of clusters (e.g., k = 4)
    optimal_k = 4  # Replace with the optimal k from the Elbow graph
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(scaled_data)

    # Add cluster labels to the original data
    pivot_data['Cluster'] = kmeans.labels_
    st.write(pivot_data.head())

    # Calculate the Silhouette score
    silhouette_avg = silhouette_score(scaled_data, kmeans.labels_)
    st.write(f'Silhouette Score for {optimal_k} clusters: {silhouette_avg}')

    # Perform PCA for 2D visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # Create a DataFrame for the PCA data
    pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = kmeans.labels_

    # Plot the PCA result with clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='Set1')
    plt.title('K-Mean Clusters Visualization using PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    st.pyplot(plt)

    # Optional: Visualize Silhouette Score for each sample
    from sklearn.metrics import silhouette_samples
    import matplotlib.cm as cm

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(scaled_data, kmeans.labels_)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    y_lower = 10
    for i in range(optimal_k):
        ith_cluster_silhouette_values = sample_silhouette_values[kmeans.labels_ == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / optimal_k)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title("Silhouette plot for K-Means clustering")
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    st.pyplot(plt)


