import streamlit as st
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title('Clustering on Iris Dataset')
st.write('This app performs KMeans and Gaussian Mixture Model clustering on the Iris dataset and displays the results.')

# Load dataset
dataset = load_iris()

X = pd.DataFrame(dataset.data)
X.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
y = pd.DataFrame(dataset.target)
y.columns = ['Targets']

colormap = np.array(['red', 'lime', 'black'])

# Create the plots
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# REAL PLOT
ax[0].scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40)
ax[0].set_title('Real')

# K-PLOT
model = KMeans(n_clusters=3)
model.fit(X)
predY = np.choose(model.labels_, [0, 1, 2]).astype(np.int64)
ax[1].scatter(X.Petal_Length, X.Petal_Width, c=colormap[predY], s=40)
ax[1].set_title('KMeans')

# GMM PLOT
scaler = preprocessing.StandardScaler()
scaler.fit(X)
xsa = scaler.transform(X)
xs = pd.DataFrame(xsa, columns=X.columns)
gmm = GaussianMixture(n_components=3)
gmm.fit(xs)
y_cluster_gmm = gmm.predict(xs)
ax[2].scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm], s=40)
ax[2].set_title('GMM Classification')

st.pyplot(fig)
