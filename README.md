# Dimensionality Reduction + Clustering + Unsupervised Score Metrics

1. Introduction
2. Installation
3. Usage
4. Hyperparameters matters
5. BayesSearch example

# 1. Introduction

DimReductionClustering is a sklearn estimator allowing to reduce the dimension of your data and then to apply an unsupervised clustering algorithm. The quality of the cluster can be done according to different metrics. The steps of the pipeline are the following: 

- Perform a dimension reduction of the data using [UMAP](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html)
- Numerically find the best epsilon parameter for DBSCAN
- Perform a density based clustering methods : [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- Estimate cluster quality using [silhouette score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) or [DBCV](https://github.com/christopherjenness/DBCV)

# 2. Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install DimReductionClustering like below. 
Rerun this command to check for and install  updates .
```bash
!pip install umap-learn
!pip install git+https://github.com/christopherjenness/DBCV.git

!pip install git+https://github.com/MathieuCayssol/DimReductionClustering.git
```

# 3. Usage

Example on mnist data.


Import data
```
from sklearn.model_selection import train_test_split
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[1]))
X, X_test, Y, Y_test = train_test_split(x_train, y_train, stratify=y_train, test_size=0.9)
```

Fit the model (same interface as a sklearn estimators)
```
model = DimReductionClustering(n_components=2, min_dist=0.000001, score_metric='silhouette', knn_topk=8, min_pts=4)

```
Return the epsilon using elbow method :
<img src="/images/epsilon_elbow.png?raw=true" width="100" height="100">
