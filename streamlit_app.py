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
st.title('Clustering Dashboard and Visualization')

# File Uploader to load dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
df = None  # Initialize df to ensure it's defined outside the if statement

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset loaded successfully!")
    st.dataframe(df.head())

    # Missing data visualization
    st.subheader("Missing Data Visualization")
    msno.matrix(df)
    st.pyplot(plt)

    # Check for required columns
    required_columns = ['Country', 'Marital Status']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Uploaded file must contain the following columns: {required_columns}")
        st.stop()

    # Clustering method selection
    clustering_method = st.selectbox(
        'Select a clustering method:',
        ('K-Means', 'Hierarchical Clustering', 'DBSCAN', 'GMM', 'Spectral Clustering')
    )

    # Display the selected clustering method
    st.write(f"You selected: {clustering_method}")

    # Continue with your clustering logic...







