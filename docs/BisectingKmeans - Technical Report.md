# Bisecting K-means in MeTTa

**Author:** Ramin Barati, Amirhossein Nouranizadeh, Farhoud Mojahedzadeh  
**Date:** May 24, 2025  
**Version:** 1.0  

---

## Abstract
This report presents an implementation of Bisecting K-means clustering in the MeTTa language. Bisecting K-means is a hierarchical variant of the standard K-means algorithm that recursively divides clusters through binary splits. Our implementation constructs a complete divisive clustering solution, maintaining a hierarchical structure of the resulting partitions. Benchmarks show competitive clustering quality with runtimes of ~4.9s, offering a robust hierarchical approach that's conceptually simpler than agglomerative methods. This work demonstrates MeTTa's capability to express recursive algorithms with complex data structures while highlighting both strengths and areas for optimization in hierarchical clustering.

## Introduction
Bisecting K-means is a divisive hierarchical clustering approach that offers several advantages over standard K-means and agglomerative methods:

- Produces a hierarchical tree structure that represents clustering at multiple levels
- Generally more efficient than agglomerative approaches for large datasets
- Often creates more balanced clusters than standard K-means
- Follows a top-down approach that can be more intuitive for certain applications

Implementing Bisecting K-means in MeTTa:

- Demonstrates MeTTa's capability to implement recursive algorithms
- Showcases MeTTa's ability to represent hierarchical data structures
- Extends MeTTa's ML toolkit with a hierarchical clustering approach
- Targets open‑source users in AI and Data Science

MeTTa is a multi‑paradigm language for declarative and functional computations over knowledge metagraphs. See http://www.metta‑lang.dev for details.

## Algorithm Overview
Bisecting K-means operates by recursively splitting clusters until the desired number of clusters is reached. The core strategy follows these steps:

1. **Initialization**: Begin with all data points in a single cluster
2. **Iterative Bisection**:
   - Select the cluster with the largest Sum of Squared Errors (SSE)
   - Apply standard K-means with k=2 to divide this cluster into two subclusters
   - Add the two resulting clusters to the cluster list
   - Remove the original cluster from the list
3. **Termination**: Continue until the desired number of clusters is reached

The algorithm leverages the divisive hierarchical approach, which starts with all points in one cluster and recursively splits clusters. The key insight is that by always splitting the cluster with the largest SSE, we prioritize improving the clustering quality by focusing on the most heterogeneous clusters first.

## MeTTa Implementation Details

The MeTTa implementation organizes clusters as tuples of (indices, center, SSE, hierarchy) and uses recursion for the bisecting process:

1. **Cluster Representation**  
   Each cluster is represented as a tuple containing:
   ```metta
   ($indices $center $sse $hierarchy)
   ```
   - `$indices`: Array of indices for data points in the cluster
   - `$center`: Centroid of the cluster
   - `$sse`: Sum of squared errors within the cluster
   - `$hierarchy`: Hierarchical structure of subclusters

2. **Initial Cluster Creation**  
   Begins with all data points in a single cluster:
   ```metta
   (bisecting-kmeans.compute-initial-cluster $X)
   ```
   - **Time Complexity:** O(n·d) for computing the initial mean
   - **Space Complexity:** O(n) for storing indices

3. **Finding Cluster to Split**  
   Selects the cluster with the largest SSE:
   ```metta
   (bisecting-kmeans.find-max-cluster $clusters)
   ```
   - **Time Complexity:** O(k) where k is the current number of clusters
   - **Space Complexity:** O(1) for storing the max cluster

4. **Bisecting a Cluster**  
   Applies K-means with k=2 to split the selected cluster:
   ```metta
   (bisecting-kmeans.bisect-cluster $X $max-cluster $max-iter)
   ```
   - **Time Complexity:** O(t·n'·d) where t is K-means iterations and n' is points in the cluster
   - **Space Complexity:** O(n') for storing the new cluster assignments

5. **Recursive Bisecting Process**  
   Recursively applies the bisecting process until reaching the desired number of clusters:
   ```metta
   (bisecting-kmeans.recursive-bisecting-kmeans $X $clusters $max-num-clusters $max-iter $hierarchy)
   ```
   - **Time Complexity:** O(k·t·n·d) where k is the target number of clusters
   - **Space Complexity:** O(k·n) for storing all cluster hierarchies

6. **Complete Fitting Function**  
   Orchestrates the complete bisecting K-means algorithm:
   ```metta
   (bisecting-kmeans.fit $X $max-num-clusters $max-kmeans-iter)
   ```
   - **Overall Time Complexity:** O(k·t·n·d)
   - **Overall Space Complexity:** O(k·n)

7. **Prediction Function**  
   Assigns new data points to clusters based on closest centroids:
   ```metta
   (bisecting-kmeans.predict $X $hierarchy)
   ```
   - **Time Complexity:** O(n·k·d) for assignment calculation
   - **Space Complexity:** O(n) for assignment array

The implementation handles recursive clustering through functional decomposition, with the hierarchical structure explicitly maintained during the bisecting process.

## Comparison with scikit‑learn

scikit‑learn's `BisectingKMeans` (introduced in version 1.1) offers some optimizations compared to our MeTTa implementation:

| Feature                 | MeTTa Implementation                  | scikit‑learn                         |
|-------------------------|---------------------------------------|--------------------------------------|
| Cluster selection       | Largest SSE only                      | Largest SSE or random                |
| Bisection method        | Standard K-means                      | K-means or K-means++                 |
| Parallel execution      | None                                  | Optional parallel processing         |
| Hierarchy tracking      | Full hierarchy maintained             | Final clusters only                  |
| Memory efficiency       | Stores all intermediate clusters      | More memory-efficient implementation |
| Implementation          | Recursive with functional approach    | Iterative with optimized C/Cython    |

The MeTTa implementation provides explicit hierarchy tracking that scikit‑learn doesn't expose, while scikit‑learn offers better performance optimizations.

## Benchmark Setup
- **Datasets** (500 samples each, synthetic generation via scikit‑learn):
  - `noisy_circles`: Two concentric circles
  - `noisy_moons`: Two interleaving half‑circles
  - `varied`: Varied variance blobs
- **Environment**: CPU = 4 cores @ 3.0 GHz, Hyperon 0.2.2, NumPy 2.2.2, scikit‑learn 1.6.1

## Results

Performance comparison on noisy_circles dataset:

| Algorithm            | Runtime (s)  | Silhouette | Calinski‑Harabasz | Davies‑Bouldin | ARI    | NMI   | AMI   |
|----------------------|--------------|------------|-------------------|----------------|--------|-------|-------|
| Bisecting K-means    | 4.945        | 0.350      | 283.23            | 1.192          | -0.002 | 0.000 | -0.001|
| K-means              | 3.822        | 0.351      | 285.69            | 1.187          | -0.002 | 0.000 | -0.001|
| Spectral Clustering  | 1.124        | 0.114      | 0.008             | 240.59         | 1.000  | 1.000 | 1.000 |

The benchmark results reveal several insights:

1. **Runtime Performance**: Bisecting K-means shows approximately 30% longer runtime compared to standard K-means, which is expected due to the additional overhead of hierarchical operations and multiple K-means applications.

2. **Clustering Quality**: On datasets with complex non-convex shapes like `noisy_circles`, both Bisecting K-means and standard K-means struggle (with near-zero ARI, NMI, and AMI), while Spectral Clustering excels. This confirms the known limitation of K-means variants on non-convex cluster geometries.

3. **Intrinsic Metrics**: The Silhouette, Calinski-Harabasz, and Davies-Bouldin scores for Bisecting K-means are nearly identical to standard K-means, suggesting that the hierarchical approach doesn't overcome the fundamental geometric limitations of centroid-based clustering.

These results highlight that while Bisecting K-means provides hierarchical information not available in standard K-means, it doesn't necessarily improve clustering quality for datasets with complex geometries where boundary-based approaches like Spectral Clustering are more appropriate.

## Usage Example
```metta
(import! &self metta_ul) 
(import! &self bisecting-kmeans)

(let 
    $labels 
    (bisecting-kmeans.predict X $hierarchy)
    (println! $labels)
)
```

This snippet:
1. Imports required modules
2. Fits Bisecting K-means with 2 clusters and max 10 iterations for each K-means application
3. Predicts cluster assignments
4. Prints the resulting labels

## Limitations & Future Work
- **Non-convex Clusters**: As with standard K-means, Bisecting K-means struggles with non-convex cluster shapes, as shown in the benchmark results for `noisy_circles`.
- **Cluster Selection Strategy**: Currently only selects the cluster with largest SSE; could be extended with different selection criteria like random selection or cluster size.
- **Initialization Method**: Uses standard K-means for bisection; could benefit from K-means++ initialization.
- **Memory Efficiency**: Stores the complete hierarchy which could be memory-intensive for large datasets with many clusters.
- **Termination Criteria**: Currently only terminates based on number of clusters; could add additional criteria based on minimum SSE improvement.

Potential improvements:
- Implement parallel processing for the bisection step
- Add alternative cluster selection strategies
- Integrate with other clustering methods for the bisection step
- Develop memory-optimized representations of the hierarchy

## Conclusion
The MeTTa implementation of Bisecting K-means demonstrates the language's capability to express recursive algorithms and hierarchical data structures. While the algorithm doesn't outperform Spectral Clustering on complex geometries, it provides valuable hierarchical information not available in flat clustering methods. The functional and declarative approach in MeTTa makes the algorithm structure clear and maintainable, though with some performance trade-offs compared to optimized implementations.

Bisecting K-means represents an important middle ground between simple partitional methods like K-means and more complex techniques like Spectral Clustering, making it a valuable addition to the MeTTa machine learning toolkit, particularly for applications where hierarchical structure is important and clusters are relatively convex.

## References
1. Steinbach, M., Karypis, G., & Kumar, V. (2000). A comparison of document clustering techniques.
2. Savaresi, S. M., & Boley, D. L. (2004). A comparative analysis on the bisecting K-means and the PDDP clustering algorithms.
3. Pedregosa et al. (2011). scikit‑learn: Machine Learning in Python.
4. MeTTa Language Specification. http://www.metta‑lang.dev/spec
