import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.graph_objs as go
import numpy as np

# Function to load data
@st.cache(allow_output_mutation=True)
def load_data(filename):
    return pd.read_csv(filename)

# Perform K-Means clustering
def apply_kmeans(df, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    numerical_data = df[['beer_servings', 'spirit_servings', 'wine_servings']]  # Assuming these are the columns
    y_kmeans = kmeans.fit_predict(numerical_data)
    return y_kmeans, kmeans

# Prepare visualization data
def prepare_visualization_data(df, y_kmeans):
    cluster_colors = {0: 'red', 1: 'green', 2: 'blue'}
    df['hover_text'] = df.apply(lambda row: f"{row['country']}<br>Cluster: {y_kmeans[row.name]}", axis=1)
    df['cluster_color'] = df.apply(lambda row: cluster_colors[y_kmeans[row.name]], axis=1)
    return df

# Function to create 3D scatter plot
def create_3d_scatter(df, highlight_country=None):
    traces = []
    for index, row in df.iterrows():
        if highlight_country and row['country'] == highlight_country:
            marker_size = 10
            marker_color = 'yellow'  # Highlight color
        else:
            marker_size = 5
            marker_color = row['cluster_color']

        trace = go.Scatter3d(
            x=[row['beer_servings']],
            y=[row['spirit_servings']],
            z=[row['wine_servings']],
            text=[row['hover_text']],
            name=row['country'],
            mode='markers',
            marker=dict(size=marker_size, color=marker_color),
            hovertemplate=(
                "Country: %{text}<br>"
                "Beer: %{x}<br>"
                "Spirit: %{y}<br>"
                "Wine: %{z}<br>"
            )
        )
        traces.append(trace)
    
    layout = go.Layout(
        title='3D Scatter Plot of Alcohol Consumption by Country',
        scene=dict(
            xaxis_title='Beer Servings',
            yaxis_title='Spirit Servings',
            zaxis_title='Wine Servings'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    fig = go.Figure(data=traces, layout=layout)
    return fig

# Streamlit application layout
st.title("Alcohol Consumption Clustering")

# Sidebar for controls
st.sidebar.title("Controls")
selected_dataset_name = st.sidebar.selectbox('Choose a dataset', ('drinks', 'drinks_without_3'))

# Load the data
data = load_data(f'{selected_dataset_name}.csv')

# Dropdown for selecting a country to highlight
country_list = data['country'].unique().tolist()
selected_country = st.sidebar.selectbox('Select a country to highlight', ['None'] + country_list)

# Perform K-Means clustering on the selected dataset
cluster_labels, kmeans_model = apply_kmeans(data)

# Prepare data for visualization
visualization_data = prepare_visualization_data(data, cluster_labels)

# If a country is selected, pass it to the create_3d_scatter function to highlight it
fig = create_3d_scatter(visualization_data, highlight_country=selected_country if selected_country != 'None' else None)

# Display dataset and plot vertically
st.write(f"Displaying dataset: {selected_dataset_name}")
st.dataframe(data)

st.write(f"3D Scatter Plot for {selected_dataset_name}")
st.plotly_chart(fig, use_container_width=True)

# Show cluster centers if interested
if st.sidebar.checkbox('Show cluster centers'):
    centers = pd.DataFrame(kmeans_model.cluster_centers_, columns=['beer_servings', 'spirit_servings', 'wine_servings'])
    st.sidebar.write(centers)
