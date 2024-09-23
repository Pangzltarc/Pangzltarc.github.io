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

    # Assuming 'X' contains the features you want to scale
    X = df.select_dtypes(include=[np.number])  # Select numeric columns
    
    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Now you can use X_scaled for clustering or further analysis
    # For example, you can add KMeans clustering
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_scaled)
    st.write("KMeans Clustering completed!")