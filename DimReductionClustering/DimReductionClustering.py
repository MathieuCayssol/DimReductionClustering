import numpy as np
from sklearn.base import BaseEstimator 
from sklearn.utils.validation import check_is_fitted
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import matplotlib.pyplot as plt


import umap.umap_ as umap
from DBCV.DBCV import DBCV

class DimReductionClustering(BaseEstimator):
    def __init__(self, n_neighbors=15,min_dist=0.1,n_components=2,metric="euclidean",epsilon=None,score_metric='DBCV', knn_topk=3, min_pts=4, random_state=42):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.metric = metric
        self.score_metric = score_metric
        self.knn_topk = knn_topk
        self.min_pts = min_pts
        self.random_state = random_state

    def elbow_numerical_search(self,X,k):
        """
        Perform a projection of points on a line. 
        Determine epsilon by taking y-coordonate of the point with the highest distance d((x,y),Proj(x,y)).
        """
        n_samples = X.shape[0]
        topk_nearest = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X)
        distances, indices = topk_nearest.kneighbors(X)
        Y = np.sort(distances[:,k])[::-1]
        X = np.array([i for i in range(n_samples)])

        plt.plot(X, Y,'-o')
        plt.show()
        
        vector_v = np.array([X[0]-X[n_samples-1],Y[0]-Y[n_samples-1]])
        W_matrix = np.transpose([X,Y])
        origin_v = np.array([X[n_samples-1],Y[n_samples-1]])
        projW_on_v = ((np.sum((W_matrix-origin_v)*vector_v, axis=1))[:,np.newaxis]*vector_v)/np.power(np.linalg.norm(vector_v),2)
        dist_compute = np.sqrt(np.power((projW_on_v+origin_v)[:,0]-X,2), np.power((projW_on_v+origin_v)[:,1]-Y,2))
        res = np.argmax(dist_compute)

        return Y[res]

    def fit(self, X):
        """
        Fit method (sklearn estimators interface)
        Dimension Reduction + Clustering
        n_neighbors=self.n_neighbors,min_dist=self.min_dist,n_components=self.n_components,metric=self.metric
        """
        if(X.shape[1] > 2):
            X_train = StandardScaler().fit_transform(X)
            self.reducer = umap.UMAP(n_neighbors=self.n_neighbors,min_dist=self.min_dist,n_components=self.n_components, random_state=self.random_state)
            self.X_low = self.reducer.fit_transform(X)
        else:
            self.X_low = StandardScaler().fit_transform(X)
        epsilon = self.elbow_numerical_search(X=self.X_low, k=self.knn_topk)
        print(f"Using elbow method, we found epsilon : {epsilon}")
        self.dbscan = DBSCAN(eps=epsilon, min_samples=self.min_pts)
        clustering = self.dbscan.fit(self.X_low)
        self.clustering_label = clustering.labels_
        return self

    def transform(self, X_test):
        """
        Transform method (sklearn estimators interface)
        """
        check_is_fitted(self, "dbscan")
        self.X_low_test = self.reducer.transform(X_test)
        return self.X_low_test

    def fit_transform(self, X_test):
        self.fit(X_test)
        return self.transform(X_test)

    def score(self, X=None):
        """
        Score metrics for the grid_search
        """
        check_is_fitted(self, "dbscan")
        if(self.score_metric == 'silhouette'):
            s = silhouette_score(X=self.X_low, labels=self.clustering_label, metric=self.metric)
        elif(self.score_metric == 'calinski_harabasz'):
            s = calinski_harabasz_score(X=self.X_low, labels=self.clustering_label)
        elif(self.score_metric == 'DBCV'):
            s = DBCV(X=self.X_low, labels = self.clustering_label)
        return s

    def display_plotly(self):
        check_is_fitted(self, "dbscan")
        if(self.n_components == 2):
            fig = px.scatter(x=self.X_low[:,0],y=self.X_low[:,1] , color = self.clustering_label, width=700, height=700)
            fig.show()
        elif(self.n_components == 3):
            fig = px.scatter_3d(x=self.X_low[:,0],y=self.X_low[:,1], z=self.X_low[:,2], color = self.clustering_label, width=700, height=700)
            fig.show()

    def get_params(self, deep=True):
        """Get model parameters"""
        get_param = dict()
        for attribute, value in self.__dict__.items():
            get_param.update({attribute: value})
        return get_param

    def set_params(self, **parameters):
        """Set model parameters"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
