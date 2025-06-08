
# metta_ul:cluster:spectral-clustering

## Overview
This module implements the Spectral Clustering algorithm. It supports two methods for constructing the affinity matrix: RBF (Radial Basis Function) kernel and KNN binary graph. The algorithm computes the normalized graph Laplacian, performs eigen-decomposition to extract spectral embeddings, normalizes these embeddings, and clusters them using k-means. The algorithm is designed to cluster data that may not be linearly separable, leveraging the eigenstructure of the Laplacian.

## Function Definitions

### `spectral-clustering.square-norm`
Computes the square norm for each row (data point) in the dataset.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.
  - Type: `(NPArray ($N $D))`
#### Returns:
- A vector where each element is the sum of squares of the components of the corresponding data point.
  - Type: `(NPArray ($N 1)`
---

### `spectral-clustering.square-distance-matrix`
Computes the matrix of squared Euclidean distances between each pair of data points in the dataset.

#### Parameters:
- `$square-norm-X`: The vector of square norms computed from `$X`.
  - Type: `(NPArray ($N 1))`
- `$X`: The dataset.
  - Type: `(NPArray ($N $D))`

#### Returns:
- A square matrix where the entry at (i, j) is the squared Euclidean distance between data point *i* and *j*.
  - Type: `(NPArray ($N $N))`

---

### `spectral-clustering.rbf-affinity-matrix`
Generates the RBF (Radial Basis Function) affinity matrix from a squared distance matrix.

#### Parameters:
- `$sqr-distance-matrix-X`: The matrix of squared distances between data points.
  - Type: `(NPArray ($N $N))`
- `$rbf-kernel-sigma`: The sigma parameter for the RBF kernel.
  - Type: `Number`

#### Returns:
- A symmetric affinity matrix where each entry represents the similarity between a pair of data points based on the RBF kernel.
  - Type: `(NPArray ($N $N))`
---

### `spectral-clustering.compute-rbf-affinity-matrix`
Computes the complete RBF affinity matrix from the input dataset by combining square norm computation, distance matrix calculation, and RBF kernel application.

#### Parameters:
- `$X`: The dataset.
  - Type: `(NPArray ($N $D))`
- `$rbf-kernel-sigma`: The sigma parameter for the RBF kernel.
  - Type: `Number`

#### Returns:
- A symmetric RBF affinity matrix.
  - Type: `(NPArray ($N $N))`
---

### `spectral-clustering.compute-knn-binary-graph-affinity-matrix`
Computes a binary affinity matrix based on k-nearest neighbors (KNN). Each data point is connected to its k nearest neighbors with weight 1, and all other connections have weight 0.

#### Parameters:
- `$X`: The dataset.
  - Type: `(NPArray ($N $D))`
- `$n-neighbors`: The number of nearest neighbors to connect to each data point.
  - Type: `Number`

#### Returns:
- A binary affinity matrix where 1 indicates a connection between k-nearest neighbors and 0 otherwise.
  - Type: `(NPArray ($N $N))`
---

### `spectral-clustering.degree`
Computes the degree of each node (data point) based on the affinity matrix.

#### Parameters:
- `$W`: The affinity matrix.
  - Type: `(NPArray ($N $N))`
#### Returns:
- A vector where each element is the sum of the corresponding row in `$W`.
  - Type: `(NPArray ($N 1))`
---

### `spectral-clustering.inverse-degree-matrix`
Constructs the inverse degree matrix used for normalization.

#### Parameters:
- `$degree-W`: The vector of node degrees computed from `$W`.
  - Type: `(NPArray ($N 1))`
#### Returns:
- A diagonal matrix where each diagonal element is the inverse square root of the corresponding degree.
  - Type: `(NPArray ($N $N))`
---

### `spectral-clustering.normalized-laplacian`
Computes the normalized graph Laplacian from the affinity matrix and its inverse degree matrix.

#### Parameters:
- `$W`: The affinity matrix.
  - Type: `(NPArray ($N $N))`
- `$inverse-degree-matrix-W`: The inverse degree matrix computed from `$W`.
  - Type: `(NPArray ($N $N))`
#### Returns:
- The normalized Laplacian matrix defined as *I - D^{-1/2} W D^{-1/2}*, where *I* is the identity matrix.
  - Type: (NPArray ($N $N))
---

### `spectral-clustering.eigh`
Performs eigen-decomposition on a given matrix.

#### Parameters:
- `$X`: The matrix to decompose (typically the normalized Laplacian).
  - Type: `(NPArray ($N $N))`
#### Returns:
- A tuple containing the eigenvalues and eigenvectors of `$X`.
  - Type: `EighResult`
---

### `spectral-clustering.eigenvalues`
Extracts the eigenvalues from the result of the eigen-decomposition.

#### Parameters:
- `$eigh-X`: The tuple returned from `spectral-clustering.eigh`.
  - Type: `EighResult`

#### Returns:
- A vector containing the eigenvalues.
  - Type: `(NPArray ($N))`
---

### `spectral-clustering.eigenvectors`
Extracts the eigenvectors from the result of the eigen-decomposition.

#### Parameters:
- `$eigh-X`: The tuple returned from `spectral-clustering.eigh`.
  - Type: `EighResult`
#### Returns:
- A matrix whose columns correspond to the eigenvectors of `$X`.
  - Type: `(NPArray ($N $N))`
---

### `spectral-clustering.eigval-top-k-index`
Finds the indices corresponding to the smallest *k* eigenvalues (after sorting).

#### Parameters:
- `$eigval-L`: The vector of eigenvalues.
  - Type: `(NPArray ($N))`
- `$k`: The number of top eigenvalue indices to select.
  - Type: `Number`

#### Returns:
- A vector of indices for the top *k* eigenvalues.
  - Type: `(NPArray ($N))`
---

### `spectral-clustering.spectral-embeddings`
Computes the spectral embeddings by selecting the top *k* eigenvectors based on their eigenvalues.

#### Parameters:
- `$eigh-I`: The eigen-decomposition result of the normalized Laplacian.
  - Type: `EighResult`
- `$k`: The number of clusters (and hence dimensions for the embeddings).
  - Type: `Number`
#### Returns:
- A matrix of spectral embeddings extracted from the selected eigenvectors.
  - Type: `(NPArray ($N $D))`
---

### `spectral-clustering.row-normalize`
Normalizes each row of a matrix to have unit norm.

#### Parameters:
- `$X`: A matrix (such as the spectral embeddings).
  - Type: `(NPArray ($N $D))`

#### Returns:
- The row-normalized version of `$X`.
  - Type: `(NPArray ($N $D))`

---

### `spectral-clustering.cluster`
Clusters the spectral embeddings using the k-means algorithm.

#### Parameters:
- `$X`: The spectral embeddings matrix.
  - Type: `(NPArray ($N $D))`
- `$num-clusters`: The desired number of clusters.
  - Type: `Number`
- `$max-kmeans-iter`: The maximum number of iterations for the k-means algorithm.
  - Type: `Number`

#### Returns:
- The centroids obtained after clustering the row-normalized spectral embeddings with k-means.
  - Type: `(NPArray ($K $D))`
---

### `spectral-clustering.fit` (Full Version)
Performs the complete spectral clustering process on the dataset with configurable affinity computation method.

#### Parameters:
- `$X`: The dataset, represented as an array of data points.
  - Type: `(NPArray ($N $D))`
- `$num-clusters`: The desired number of clusters.
  - Type: `Number`
- `$affinity-mode`: The method for computing the affinity matrix. Supported values:
  - `"rbf-affinity-matrix"`: Uses RBF kernel for similarity computation
  - `"binary-knn-graph"`: Uses k-nearest neighbors binary graph
  - Type: `String`
- `$affinity-param`: Parameter for the chosen affinity method:
  - For RBF: the sigma parameter for the kernel
  - For KNN: the number of nearest neighbors
  - Type: `Number`
- `$max-kmeans-iter`: The maximum number of iterations for the k-means algorithm.
  - Type: `Number`

#### Returns:
- A tuple containing spectral embeddings and the final centroids computed from clustering the spectral embeddings.
  - Type: `((NPArray ($N $C)) (NPArray ($K $C)))`

---

### `spectral-clustering.fit` (Simplified Version)
Performs spectral clustering with default parameters (RBF affinity with sigma=0.1 and 10 k-means iterations).

#### Parameters:
- `$X`: The dataset, represented as an array of data points.
  - Type: `(NPArray ($N $D))`
- `$num-clusters`: The desired number of clusters.
  - Type: `Number`

#### Returns:
- A tuple containing spectral embeddings and the final centroids computed from clustering the spectral embeddings.
  - Type: `((NPArray ($N $C)) (NPArray ($K $C)))`

---

### `spectral-clustering.predict`
Predicts the cluster assignment for each data point in the dataset using the spectral embeddings and computed centroids.

#### Parameters:
- A tuple `($embeddings $centroids)`, where:
  1. `$embeddings`: The spectral embedding matrix.
  2. `$centroids`: The centroids obtained from the clustering step.
  - Type: `((NPArray ($N $C)) (NPArray ($K $C)))`
- `$num-clusters`: The number of clusters.
  - Type: `Number`

#### Returns:
- A vector of cluster labels, one for each data point, determined by assigning each point to the nearest centroid.
  - Type: `((NPArray ($N)))`

## Usage

### Using RBF Affinity Matrix
To perform spectral clustering on a dataset `S` with RBF kernel:
```metta
(=
    (embeddings-and-centroids)
    (spectral-clustering.fit S 3 "rbf-affinity-matrix" 0.1 100)
)
```
### Using KNN Binary Graph
To perform spectral clustering on a dataset `S` with KNN binary graph (connecting each point to its 5 nearest neighbors):
```metta
(=
    (embeddings-and-centroids)
    (spectral-clustering.fit S 3 "binary-knn-graph" 5 100)
)
```
### Using Default Parameters
To predict the cluster labels for the dataset `S`:
``` metta
(=
    (cluster-labels)
    (spectral-clustering.predict
        (embeddings-and-centroids)        
        3
    )    
)
```
