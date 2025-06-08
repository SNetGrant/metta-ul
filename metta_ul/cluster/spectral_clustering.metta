!(import! &self metta_ul)
!(import! &self kmeans)

; import name is set to sc-np (spectral-clustering np) to avoid conflicts due to importing redundant modules
! (bind! sc-np (py-atom numpy))

(: X (-> (NPArray ($N $D))))

; Python code: np.sum(X**2, axis=1, keepdims=True)
; Compute squared norms for each point in X, which helps calculate pairwise distances.
(: spectral-clustering.square-norm (-> (NPArray ($N $D)) (NPArray ($N 1))))
(=
    (spectral-clustering.square-norm $X)
    (np.sum
        (np.power $X 2)
        1
        py.none
        py.none
        py.true
    )
)

; Python code: sq_norms + sq_norms.T - 2 * np.dot(X, X.T)
; Compute pairwise squared Euclidean distances using the identity:
; ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i*x_j.
(: spectral-clustering.square-distance-matrix (-> (NPArray ($N 1)) (NPArray ($N $D)) (NPArray ($N $N))))
(=
    (spectral-clustering.square-distance-matrix $square-norm-X $X)
    (np.sub
        (np.add
            $square-norm-X
            (np.transpose
                $square-norm-X
            )
        )
        (np.mul
            2
            (np.matmul
                $X
                (np.transpose
                    $X
                )
            )
        )
    )
)

; Python code: np.exp(-sq_dists / (2 * sigma**2))
; Apply the Gaussian (RBF) kernel to the squared distances, converting them into affinity values.
(: spectral-clustering.rbf-affinity-matrix (-> (NPArray ($N $N)) Number (NPArray ($N $N))))
(=
    (spectral-clustering.rbf-affinity-matrix $sqr-distance-matrix-X $rbf-kernel-sigma)
    (np.exp
        (np.div
            (np.mul
                -1
                $sqr-distance-matrix-X
            )
            (np.mul
                2
                (np.power
                    (np.array ($rbf-kernel-sigma))
                    2
                )
            )
        )
    )
)

(: spectral-clustering.compute-rbf-affinity-matrix (-> (NPArray ($N $D)) Number (NPArray ($N $N))))
(=
    (spectral-clustering.compute-rbf-affinity-matrix $X $rbf-kernel-sigma)
    (spectral-clustering.rbf-affinity-matrix
        (spectral-clustering.square-distance-matrix
            (spectral-clustering.square-norm $X)
            $X
        )
        $rbf-kernel-sigma
    )
)
; =================== KNN mode ===================
(: spectral-clustering.compute-knn-binary-graph-affinity-matrix (-> (NPArray ($N $D)) Number (NPArray ($N $N))))
(=
    (spectral-clustering.compute-knn-binary-graph-affinity-matrix $X $n-neighbors)
    (let
        $dist
        (spectral-clustering.square-distance-matrix
            (spectral-clustering.square-norm $X)
        $X
        )
        (let
            $N
            (np.shape $dist 0)
            (let
                $end-index
                (+ 1 $n-neighbors)
                (let
                    $knn-indices
                    (np.take (np.argsort $dist 1) (np.arange 1 $end-index) 1)
                    (let
                        $row-index
                        (np.reshape (np.arange $N) (np.array ($N 1)))
                        (let
                            $flat-index
                            (np.reshape
                                (np.add (np.mul $row-index $N) $knn-indices)
                                -1
                            )
                            (let
                                $mask
                                (np.isin (np.arange (* $N $N)) $flat-index)
                                (let
                                    $A-flat
                                    (np.where $mask 1.0 0.0)
                                    (let
                                        $A
                                        (np.reshape $A-flat (np.array ($N $N)))
                                        (np.add 0.00000001 (np.maximum $A (np.transpose $A)))
                                    )

                                )
                            )
                        )
                    )
                )
            )
        )
    )
)

; ================================================
; Python code: d = np.sum(W, axis=1)
; Compute the degree for each node by summing its affinities to all other nodes.
(: spectral-clustering.degree (-> (NPArray ($N $N)) (NPArray ($N 1))))
(=
    (spectral-clustering.degree $W)
    (np.sum $W 1)
)

; Python code: np.diag(1.0 / np.sqrt(d))
; Create the diagonal matrix D^{-1/2} used to normalize the Laplacian.
(:spectral-clustering.inverse-degree-matrix (-> (NPArray ($N 1)) (NPArray ($N $N))))
(=
    (spectral-clustering.inverse-degree-matrix $degree-W)
    (np.diag
        (np.div
            1.0
            (np.sqrt
                $degree-W
            )
        )
    )
)

; Python code: np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
; Compute the symmetric normalized Laplacian: L_sym = I - D^{-1/2} * W * D^{-1/2}.
(: spectral-clustering.normalized-laplacian (-> (NPArray ($N $N)) (NPArray ($N $N)) (NPArray ($N $N))))
(=
    (spectral-clustering.normalized-laplacian $W $inverse-degree-matrix-W)
    (np.sub
        (np.eye (np.shape $W 0))
        (np.matmul
            $inverse-degree-matrix-W
            (np.matmul
                $W
                $inverse-degree-matrix-W
            )
        )
    )
)

; Python code: np.linalg.eigh(L_sym)
; Compute eigenvalues and eigenvectors of the symmetric Laplacian matrix for spectral embedding.
; TODO: py-dot had to be used because np.linalg.eigh in numme.py,
; resulted into fatal errors when combined with slice
(: spectral-clustering.eigh (-> (NPArray ($N $N)) EighResult))
(=
    (spectral-clustering.eigh $X)
    ((py-dot sc-np linalg.eigh) $X)
)

; Python code: np.linalg.eigh(L_sym)[0]
; Extract eigenvalues of the Laplacian, which reveal important graph structure for clustering.
(: spectral-clustering.eigenvalues (-> EighResult (NPArray ($N))))
(=
    (spectral-clustering.eigenvalues $eigh-X)
    ((py-dot $eigh-X __getitem__) 0)
)

; Python code: np.linalg.eigh(L_sym)[1]
; Extract eigenvectors of the Laplacian, used as the spectral embedding of the data.
(: spectral-clustering.eigenvectors (-> EighResult (NPArray ($N $N))))
(=
    (spectral-clustering.eigenvectors $eigh-X)
    ((py-dot $eigh-X __getitem__) 1)
)

; Python code: np.argsort(eigenvalues)
; Get indices that would sort the eigenvalues in ascending order to select the smallest ones.
(: spectral-clustering.eigval-top-k-index (-> (NPArray ($N)) Number (NPArray ($N))))
(=
    (spectral-clustering.eigval-top-k-index $eigval-X $k)
    ((py-dot (np.argsort $eigval-X) __getitem__) (np.arange $k))
)

; Python code: eigenvectors[:, idx[:k]]
; Select the first k eigenvectors corresponding to the smallest eigenvalues for clustering.
(: spectral-clustering.spectral-embeddings (-> EighResult Number (NPArray ($N $D))))
(=
    (spectral-clustering.spectral-embeddings $eigh-I $k)
    (np.take
        (spectral-clustering.eigenvectors $eigh-I)
        ((py-dot (np.argsort (spectral-clustering.eigenvalues $eigh-I)) __getitem__) (np.arange $k))
        1
    )
)

; Python code: U / np.linalg.norm(U, axis=1, keepdims=True)
; Normalize each row of the spectral embedding to unit length for stability in k-means clustering.
(: spectral-clustering.row-normalize (-> (NPArray ($N $D)) (NPArray ($N $D))))
(=
    (spectral-clustering.row-normalize $X)
    (np.div
        $X
        (np.linalg.norm
            $X
            py.none
            1
            py.true
        )
    )
)

; Python code: kmeans(X=U_norm, k=k, max_iter=max_iter)
; Apply k-means to the normalized spectral embedding to partition data into k clusters.
(: spectral-clustering.cluster (-> (NPArray ($N $D)) Number Number (NPArray ($K $D))))
(=
    (spectral-clustering.cluster $X $num-clusters $max-kmeans-iter)
    (kmeans.fit
        $X
        $num-clusters
        $max-kmeans-iter
        0.0001
    )
)

; Fit spectral clustering algorithm and return spectral embedding and cluster centroids.
(: spectral-clustering.fit (-> (NPArray ($N $D)) Number String Number Number ((NPArray ($N $C)) (NPArray ($K $C)))))
(=
    (spectral-clustering.fit $X $num-clusters $affinity-mode $affinity-param $max-kmeans-iter)
    (let
        $affinity-matrix
        (case
            $affinity-mode
            (
                ("binary-knn-graph" (spectral-clustering.compute-knn-binary-graph-affinity-matrix $X $affinity-param))
                ("rbf-affinity-matrix" (spectral-clustering.compute-rbf-affinity-matrix $X $affinity-param))
            )
        )
        (let
            $embeddings
            (spectral-clustering.row-normalize
                (spectral-clustering.spectral-embeddings
                    (spectral-clustering.eigh
                        (spectral-clustering.normalized-laplacian
                            $affinity-matrix
                            (spectral-clustering.inverse-degree-matrix
                                (spectral-clustering.degree
                                    $affinity-matrix
                                )
                            )
                        )
                    )
                    $num-clusters
                )
            )
            ($embeddings (spectral-clustering.cluster $embeddings $num-clusters $max-kmeans-iter))
        )
    )
)

; Fit spectral clustering algorithm and return spectral embedding and cluster centroids.
(: spectral-clustering.fit (-> (NPArray ($N $D)) Number ((NPArray ($N $C)) (NPArray ($K $C)))))
(=
    (spectral-clustering.fit $X $num-clusters)
    (spectral-clustering.fit $X $num-clusters "rbf-affinity-matrix" 0.1 10)
)

; Predict the cluster assignments of data given spectral embeddings, centroids and number of clusters.
(: spectral-clustering.predict (-> ((NPArray ($N $C)) (NPArray ($K $C))) Number ((NPArray ($N)))))
(=
    (spectral-clustering.predict ($embeddings $centroids) $num-clusters)
    (np.argmax
        (np.transpose
            (kmeans.assign
                $embeddings
                $centroids
            )
        )
        1
    )
)