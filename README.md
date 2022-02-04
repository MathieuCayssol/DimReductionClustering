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


- Import the data
```
from sklearn.model_selection import train_test_split
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[1]))
X, X_test, Y, Y_test = train_test_split(x_train, y_train, stratify=y_train, test_size=0.9)
```


- Fit the model (same interface as a sklearn estimators)
```
model = DimReductionClustering(n_components=2, min_dist=0.000001, score_metric='silhouette', knn_topk=8, min_pts=4)
```

Return the epsilon using elbow method :

<img src="/images/epsilon_elbow.png?raw=true" width="400">

- Show the 2D plot :
```
model.display_plotly()
```

<img src="/images/minist.png?raw=true" width="400">

- Get the score (Silhouette coefficient here)

```
model.score()
```

# 4. Hyperparameters matters

## 4.1 UMAP (dim reduction)

- `n_neighbors`  (global/local tradeoff) (default:15 ; 2-1/4 of data)
    
    → low value (glue small chain, more local) 
    
    → high value (glue big chain, more global)
    
- `min_dist` (0 to 0.99) the minimum distance apart that points are allowed to be in the low dimensional representation. This means that low values of `min_dist` will result in clumpier embeddings. This can be useful if you are interested in clustering, or in finer topological structure. Larger values of `min_dist` will prevent UMAP from packing points together and will focus on the preservation of the broad topological structure instead.
- `n_components` low dimensional space. 2 or 3
- `metric` (’euclidian’ by default). For NLP, good idea to choose ‘cosine’ as infrequent/frequent words will have different magnitude.


## 4.1 DBSCAN (clustering)

- `min_pts` MinPts ≥ 3. Basic rule : = 2 * Dimension  (4 for 2D and 6 for 3D). Higher for noisy data.
    
- ~~`Epsilon`~~ The maximum distance between two samples for one to be considered as in the neighborhood of the other.

! There is no Epsilon in the implementation becauze it will be calculate using elbow method with KNN.

- `knn_topk` k-th Nearest Neighbors.
- k-distance graph with k nearest neighbor. Sort result by ascending order. Find elbow. and it will be the right epsilon distance. [Click here to know more about elbow method](https://www.ccs.neu.edu/home/vip/teach/DMcourse/2_cluster_EM_mixt/notes_slides/revisitofrevisitDBSCAN.pdf)


