# Dimensionality Reduction + Clustering + Unsupervised Score Metrics

1. Introduction
2. Installation
3. Usage
4. Hyperparameters matters
5. BayesSearch example

# 1. Introduction

DimReductionClustering is a sklearn estimator allowing to reduce the dimension of your data and then to apply an unsupervised clustering algorithm. The quality of the cluster can be done according to different metrics. The steps of the pipeline are the following: 

- Perform a dimension reduction of the data using (UMAP)[https://umap-learn.readthedocs.io/en/latest/how_umap_works.html]
- Numerically find the best epsilon parameter for DBSCAN
- Perform a density based clustering methods : (DBSCAN)[https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html]
- Estimate cluster quality using (silhouette score)[https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html] or (DBCV)[https://github.com/christopherjenness/DBCV]

# 2. Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install DimReductionClustering like below. 
Rerun this command to check for and install  updates .
```bash
!pip install umap-learn
!pip install git+https://github.com/christopherjenness/DBCV.git

!pip install git+https://github.com/MathieuCayssol/DimReductionClustering.git
```

# 3. Usage


