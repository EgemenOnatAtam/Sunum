import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Load the clean and unclean datasets
@st.cache_data
def load_data(clean=True):
    if clean:
        data = pd.read_csv('clean_data.csv')
    else:
        data = pd.read_csv('unclean_data.csv')
    return data

st.title("K-Means Kümeleme Analizi Uygulaması")

# Sidebar for user input
clean_option = st.sidebar.checkbox('Temizlenmiş Veriyi Göster', value=True)
unclean_option = st.sidebar.checkbox('Temizlenmemiş Veriyi Göster', value=False)
perform_kmeans = st.sidebar.checkbox('K-Means Uygula', value=False)

# Initialize data
data = None

# Load and display the selected dataset
if clean_option:
    st.subheader("Temizlenmiş Veri Seti")
    data = load_data(clean=True)
    st.write(data)
elif unclean_option:
    st.subheader("Temizlenmemiş Veri Seti")
    data = load_data(clean=False)
    st.write(data)

# Ensure data is loaded before proceeding
if data is not None and perform_kmeans:
    st.subheader("K-Means Kümeleme Analizi Sonuçları")

    # Select numerical columns for K-Means
    X = data.select_dtypes(include=[np.number])

    # Finding optimal number of clusters
    n_clusters = 30
    cost = []
    for i in range(1, n_clusters):
        kmean = KMeans(i)
        kmean.fit(X)
        cost.append(kmean.inertia_)

    # Applying K-Means with the optimal number of clusters (for example, 6)
    kmean = KMeans(6)
    kmean.fit(X)
    labels = kmean.labels_

    # Adding cluster information to the dataset
    clusters = pd.concat([data, pd.DataFrame({'cluster': labels})], axis=1)

    # Cosine similarity and PCA for visualization
    dist = 1 - cosine_similarity(X)
    pca = PCA(2)
    pca.fit(dist)
    X_PCA = pca.transform(dist)

    # Visualize clusters using PCA components
    fig, ax = plt.subplots(figsize=(20, 13))
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
    for i in range(6):
        ax.scatter(X_PCA[labels == i, 0], X_PCA[labels == i, 1], s=15, c=colors[i], label=f'Cluster {i}')
    ax.legend()
    ax.set_title("Kümeleme Sonuçlarının Görselleştirilmesi")
    st.pyplot(fig)
