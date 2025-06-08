import math

from hyperon import MeTTa, Atom
import numpy as np
from sklearn.metrics import adjusted_rand_score


def test_compute_affinity(metta: MeTTa):
    metta.run(
        """
        ! (import! &self metta_ul:cluster:spectral_clustering)
        
        (: X-1D-Single (-> (NPArray (1 1))))
        (=
            (X-1D-Single)
            (np.array ((1)))
        )
        
        (: X-1D-Multiple (-> (NPArray (2 1))))
        (=
            (X-1D-Multiple)
            (np.array ((0) (1)))
        )
        
        (: X-2D-Multiple (-> (NPArray (3 2))))
        (=
            (X-2D-Multiple)
            (np.array ((0 0) (0 1) (1 0)))
        )
        
        (: X-2D-Random (-> (NPArray (10 3))))
        (=
            (X-2D-Random)
            (np.random.rand 10 3)
        )
        """
    )
    result: Atom = metta.run(
        """
        ! (spectral-clustering.rbf-affinity-matrix 
                (spectral-clustering.square-distance-matrix 
                    (spectral-clustering.square-norm 
                        (X-1D-Single) 
                    ) 
                    (X-1D-Single)
                )  
                1.0
            )
        """
    )[0][0]
    W: int = result.get_object().value
    assert W.shape == (1, 1), "Expected a 1x1 matrix for a single sample."
    assert np.allclose(
        W, np.array([[1.0]])
    ), "Affinity for a single sample should be 1."

    result: Atom = metta.run(
        """
        ! (spectral-clustering.rbf-affinity-matrix 
                (spectral-clustering.square-distance-matrix 
                    (spectral-clustering.square-norm 
                        (X-1D-Multiple) 
                    ) 
                    (X-1D-Multiple)
                )  
                1.0
            )
        """
    )[0][0]
    W: int = result.get_object().value
    expected_W = np.array([[1.0, math.exp(-0.5)], [math.exp(-0.5), 1.0]])
    assert np.allclose(W, expected_W), "Affinity matrix for two samples is incorrect."

    result: Atom = metta.run(
        """
        ! (spectral-clustering.rbf-affinity-matrix 
                (spectral-clustering.square-distance-matrix 
                    (spectral-clustering.square-norm 
                        (X-2D-Multiple) 
                    ) 
                    (X-2D-Multiple)
                )  
                1.0
            )
        """
    )[0][0]
    W: int = result.get_object().value
    expected_W = np.array(
        [
            [1.0, math.exp(-0.5), math.exp(-0.5)],
            [math.exp(-0.5), 1.0, math.exp(-1.0)],
            [math.exp(-0.5), math.exp(-1.0), 1.0],
        ]
    )
    assert np.allclose(
        W, expected_W
    ), "Affinity matrix for three 2D samples is incorrect."

    result: Atom = metta.run(
        """
        ! (spectral-clustering.rbf-affinity-matrix 
                (spectral-clustering.square-distance-matrix 
                    (spectral-clustering.square-norm 
                        (X-2D-Random)
                    ) 
                    (X-2D-Random)
                )  
                1.0
            )
        """
    )[0][0]
    W: int = result.get_object().value
    assert np.allclose(W, W.T), "Affinity matrix should be symmetric."

    result: Atom = metta.run(
        """        
        ! (spectral-clustering.rbf-affinity-matrix 
                (spectral-clustering.square-distance-matrix 
                    (spectral-clustering.square-norm 
                        (X-1D-Multiple) 
                    ) 
                    (X-1D-Multiple)
                )  
                1.0
            )
        """
    )[0][0]
    W_with_pos_sigma: int = result.get_object().value

    result: Atom = metta.run(
        """        
        ! (spectral-clustering.rbf-affinity-matrix 
                (spectral-clustering.square-distance-matrix 
                    (spectral-clustering.square-norm 
                        (X-1D-Multiple) 
                    ) 
                    (X-1D-Multiple)
                )  
                -1.0
            )
        """
    )[0][0]
    W_with_neg_sigma: int = result.get_object().value
    assert np.allclose(
        W_with_pos_sigma, W_with_neg_sigma
    ), "Affinity matrix should be identical for sigma and -sigma."


def test_compute_normalized_laplacian(metta: MeTTa):
    metta.run(
        """
        !(import! &self metta_ul:cluster:spectral_clustering)
        
        (=
            (I)
            (np.array ((1 0 0) (0 1 0) (0 0 1)))
        )
        
        (=
            (W-2)
            (np.array ((1 1) (1 1)))
        )
        
        (=
            (W-3)
            (np.array ((1 1 1) (1 1 1) (1 1 1)))
        )
        (=
            (X-2D-Random)
            (np.random.rand 5 2)
        )
        """
    )
    result: Atom = metta.run(
        """        
        ! (spectral-clustering.normalized-laplacian 
            (I) 
            (spectral-clustering.inverse-degree-matrix 
                (spectral-clustering.degree 
                    (I)
                )
            )
        )
        """
    )[0][0]
    L_norm: int = result.get_object().value
    expected = np.zeros((3, 3))
    assert np.allclose(
        L_norm, expected
    ), "L_norm must be a zero matrix when W is the identity matrix."

    result: Atom = metta.run(
        """        
        ! (spectral-clustering.normalized-laplacian 
            (W-2) 
            (spectral-clustering.inverse-degree-matrix 
                (spectral-clustering.degree 
                    (W-2)
                )
            )
        )
        """
    )[0][0]
    L_norm: int = result.get_object().value
    expected = np.array([[0.5, -0.5], [-0.5, 0.5]])
    assert np.allclose(
        L_norm, expected
    ), "L_norm for a 2x2 complete graph is incorrect."

    result: Atom = metta.run(
        """        
        ! (spectral-clustering.normalized-laplacian 
            (W-3) 
            (spectral-clustering.inverse-degree-matrix 
                (spectral-clustering.degree 
                    (W-3)
                )
            )
        )
        """
    )[0][0]
    L_norm: int = result.get_object().value
    expected = np.array(
        [
            [1 - 1 / 3, -1 / 3, -1 / 3],
            [-1 / 3, 1 - 1 / 3, -1 / 3],
            [-1 / 3, -1 / 3, 1 - 1 / 3],
        ]
    )
    assert np.allclose(
        L_norm, expected
    ), "L_norm for a 3x3 complete graph is incorrect."

    result: Atom = metta.run(
        """ 
        (=
            (W-X-2D-Random)
            (spectral-clustering.rbf-affinity-matrix 
                (spectral-clustering.square-distance-matrix 
                    (spectral-clustering.square-norm 
                        (X-2D-Random)
                    ) 
                    (X-2D-Random)
                )  
                1.0
            )
        )
        
        ! (spectral-clustering.normalized-laplacian
            (W-X-2D-Random)
            (spectral-clustering.inverse-degree-matrix 
                (spectral-clustering.degree 
                    (W-X-2D-Random)
                )
            )
        )
        """
    )[0][0]
    L_norm: int = result.get_object().value
    assert np.allclose(L_norm, L_norm.T), "L_norm should be symmetric."


def test_spectral_embedding(metta: MeTTa):
    metta.run(
        """        
        !(import! &self metta_ul:cluster:spectral_clustering)
        
        (=
            (I)
            (np.array ((1 0 0) (0 1 0) (0 0 1)))
        )
        """
    )
    result: Atom = metta.run(
        """
        (=
            (eigh-I)
            (spectral-clustering.eigh (I))
        )
        
        ! (spectral-clustering.spectral-embeddings (eigh-I) 2)
        """
    )[0][0]
    U = result.get_object().value

    L = np.eye(3)
    k = 2
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    expected_U = eigenvectors[:, idx[:k]]
    assert np.allclose(
        np.abs(U), np.abs(expected_U)
    ), "Eigenvectors from spectral_embedding do not match expected values for a identity matrix."

    # Test 1: Diagonal matrix test
    result: Atom = metta.run(
        """
        (=
            (L1)
            (np.array ((0 0 0) (0 1 0) (0 0 2)))
        )
        (=
            (eigh-L1)
            (spectral-clustering.eigh (L1))
        )

        ! (spectral-clustering.spectral-embeddings (eigh-L1) 2)
        """
    )[0][0]
    U = result.get_object().value
    # Check the shape of the output
    assert U.shape == (
        3,
        2,
    ), "Expected output shape (3,2) for a 3x3 diagonal matrix with k=2."

    # Compute the expected eigen-decomposition
    L = np.diag([0, 1, 2])
    k = 2
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    expected_U = eigenvectors[:, idx[:k]]
    assert np.allclose(
        np.abs(U), np.abs(expected_U)
    ), "Eigenvectors from spectral_embedding do not match expected values for a diagonal matrix."

    # Test 2: Small symmetric matrix test
    result: Atom = metta.run(
        """        
        (=            
            (L2)         
            (np.array ((0.5 -0.5) (-0.5 0.5)))
        )
        (=            
            (eigh-L2)         
            (spectral-clustering.eigh (L2))
        )
        
        ! (spectral-clustering.spectral-embeddings (eigh-L2) 2)                       
        """
    )[0][0]
    U = result.get_object().value
    assert U.shape == (2, 2), "Expected output shape (2,2) for a 2x2 matrix with k=2."

    # Verify that each column satisfies the eigenvalue equation: L_sym * u = lambda * u
    L = np.array([[0.5, -0.5], [-0.5, 0.5]])
    k = 2
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[idx]
    for i in range(k):
        lam = sorted_eigenvalues[i]
        # Check that L_sym @ U[:, i] ≈ lam * U[:, i]
        assert np.allclose(
            L @ U[:, i], lam * U[:, i]
        ), f"Eigenvector {i} does not satisfy the eigenvalue equation."

    # Test 3: Edge case with k = 0
    result: Atom = metta.run(
        """        
        (=            
            (L3)         
            (np.array ((0 0) (0 0)))
        )
        (=            
            (eigh-L3)         
            (spectral-clustering.eigh (L3))
        )

        ! (spectral-clustering.spectral-embeddings (eigh-L2) 0)                       
        """
    )[0][0]
    U = result.get_object().value

    assert U.shape == (2, 0), "Expected an output with shape (n_samples, 0) when k=0."


def test_row_normalize(metta: MeTTa):
    metta.run(
        """        
        !(import! &self metta_ul:cluster:spectral_clustering)

        (=
            (A)
            (np.array ((3 4) (5 12)))
        )
        (=
            (B)
            (np.array ((3) (-4) (5)))
        )
        (=
            (C)
            (np.array ((0 0) (1 2)))
        )
        (=
            (D)
            (np.array ((1 1 1) (1 1 1) (1 1 1)))
        )
        """
    )
    result: Atom = metta.run(
        """
        ! (spectral-clustering.row-normalize (A))
        """
    )[0][0]
    A_norm = result.get_object().value
    expected = np.array([[3 / 5, 4 / 5], [5 / 13, 12 / 13]])
    assert A_norm.shape == expected.shape, "Output shape must match input shape."
    assert np.allclose(A_norm, expected), "Row normalization failed on standard input."

    result: Atom = metta.run(
        """
        ! (spectral-clustering.row-normalize (B))
        """
    )[0][0]
    B_norm = result.get_object().value
    expected = np.array([[1.0], [-1.0], [1.0]])
    assert B_norm.shape == expected.shape, "Output shape must match input shape."
    assert np.allclose(B_norm, expected), "Row normalization failed on standard input."

    with np.errstate(divide="ignore", invalid="ignore"):
        result: Atom = metta.run(
            """
            ! (spectral-clustering.row-normalize (C))
            """
        )[0][0]
    C_norm = result.get_object().value
    # For the zero row, dividing by zero should result in nan values.
    assert np.isnan(
        C_norm[0]
    ).all(), "Row normalization should yield nan for zero vector."
    # For the nonzero row:
    expected_nonzero = np.array([1, 2]) / np.linalg.norm([1, 2])
    assert np.allclose(
        C_norm[1], expected_nonzero
    ), "Row normalization failed on nonzero row with a zero row present."

    result: Atom = metta.run(
        """
        ! (spectral-clustering.row-normalize (D))
        """
    )[0][0]
    D_norm = result.get_object().value
    norm_value = np.sqrt(3)
    expected = np.ones((3, 3)) / norm_value
    assert (
        D_norm.shape == expected.shape
    ), "Output shape must match input shape for ones matrix."
    assert np.allclose(D_norm, expected), "Row normalization failed for ones matrix."


def test_spectral_clustering_cluster(metta: MeTTa):
    metta.run(
        """        
        !(import! &self metta_ul:cluster:spectral_clustering)                
        """
    )
    result: Atom = metta.run(
        """  
        (= 
            (X1)
            (np.array ((1 0 0) (0 1 0) (0 0 1)))
        )
        (=
            (K1)
            3
        )      
        (=
            (embeddings1)
            (spectral-clustering.spectral-embeddings
                (spectral-clustering.eigh
                    (spectral-clustering.normalized-laplacian
                        (spectral-clustering.compute-rbf-affinity-matrix 
                            (X1)
                            0.1
                        )
                        (spectral-clustering.inverse-degree-matrix
                            (spectral-clustering.degree
                                (spectral-clustering.compute-rbf-affinity-matrix 
                                    (X1)
                                    0.1
                                )
                            )
                        )
                    )
                )
                (K1)
            )
        )        
        
        ! (np.argmax
            (np.transpose
                (kmeans.assign
                    (embeddings1)
                    (spectral-clustering.cluster (embeddings1) (K1) 100)
                )
            )
            1
        )
        """
    )[0][0]

    labels = result.get_object().value
    ground_truth = np.array([0, 1, 2])
    ari = adjusted_rand_score(ground_truth, labels)
    assert ari == 1.0, f"Expected ARI of 1.0, but got {ari}"

    result: Atom = metta.run(
        """        
        (= 
            (X2)
            (np.array ((0 0) (0.1 0) (1.0 1.0) (1.1 1.0)))
        )
        (=
            (K2)
            2
        )    
                
        (=
            (embeddings2)
            (spectral-clustering.spectral-embeddings
                (spectral-clustering.eigh
                    (spectral-clustering.normalized-laplacian
                        (spectral-clustering.compute-rbf-affinity-matrix 
                            (X2)
                            0.1
                        )
                        (spectral-clustering.inverse-degree-matrix
                            (spectral-clustering.degree
                                (spectral-clustering.compute-rbf-affinity-matrix 
                                    (X2)
                                    0.1
                                )
                            )
                        )
                    )
                )
                (K2)
            )
        )        

        ! (np.argmax
                (np.transpose
                    (kmeans.assign 
                        (embeddings2) 
                        (spectral-clustering.cluster (embeddings2) (K2) 100) 
                    )
                )
                1            
        )              
        """
    )[0][0]
    labels = result.get_object().value
    ground_truth = np.array([0, 0, 1, 1])
    ari = adjusted_rand_score(ground_truth, labels)
    assert ari == 1.0, f"Expected ARI of 1.0, but got {ari}"


def test_spectral_clustering_fit_and_predict(metta: MeTTa):
    metta.run(
        """
        ! (import! &self metta_ul:cluster:spectral_clustering)
        """
    )

    result: Atom = metta.run(
        """
        (= 
            (X)
            (np.array ((0.0 0.0) (0.1 0) (1.0 1.0) (1.1 1.0)))
        )
        (: fit-outputs (-> ((NPArray (4 2)) (NPArray (2 2)))))
        (=
            (fit-outputs)
            (spectral-clustering.fit (X) 2)
        )
        
        ! (spectral-clustering.predict (fit-outputs) 2)
        """
    )[0][0]
    cluster_labels = result.get_object().value
    # 1. Assert that the first two elements are the same and the last two elements are the same
    assert (
        cluster_labels[0] == cluster_labels[1]
    ), "The first two elements are not the same!"
    assert (
        cluster_labels[-2] == cluster_labels[-1]
    ), "The last two elements are not the same!"

    # 2. Assert that the value in the first two elements is distinct from the value in the last two elements
    assert (
        cluster_labels[0] != cluster_labels[-2]
    ), "The first two and last two elements should be distinct!"

    # 3. Assert that the unique values in the array are 0 and 1
    unique_values = np.unique(cluster_labels)
    expected_values = np.array([0, 1])
    # Ensure unique_values match the expected values
    assert np.array_equal(
        unique_values, expected_values
    ), "The unique values in the array are not [0, 1]!"

def test_spectral_clustering_fit_and_predict_knn_graph_mode(metta: MeTTa):
    metta.run(
        """
        ! (import! &self metta_ul:cluster:spectral_clustering)
        """
    )
    result: Atom = metta.run(
        """
        (= 
            (X)
            (np.array ((0.0 0.0) (0.1 0) (1.0 1.0) (1.1 1.0)))
        )
                        
        (: fit-outputs (-> ((NPArray (4 2)) (NPArray (2 2)))))
        (=
            (fit-outputs)
            (spectral-clustering.fit (X) 2 "binary-knn-graph" 1 10)
        )
        ;! (fit-outputs)
        ! (spectral-clustering.predict (fit-outputs) 2)
        ;! (np.add 0.0000001 (spectral-clustering.compute-knn-binary-graph-affinity-matrix (X) 1))
        """
    )[0][0]

    cluster_labels = result.get_object().value
    assert (
            cluster_labels[0] == cluster_labels[1]
    ), f"The first two elements are not the same! {cluster_labels}"
    assert (
            cluster_labels[-2] == cluster_labels[-1]
    ), f"The last two elements are not the same! {cluster_labels}"

    assert (
            cluster_labels[0] != cluster_labels[-2]
    ), f"The first two and last two elements should be distinct! {cluster_labels}"

    unique_values = np.unique(cluster_labels)
    expected_values = np.array([0, 1])

    assert np.array_equal(
        unique_values, expected_values
    ), f"The unique values in the array are not [0, 1]! {cluster_labels}"
    pass

# def test_real_data(metta: MeTTa):
#     metta.run(
#         """
#         ! (import! &self metta_ul:cluster:spectral_clustering)
#         ! (ul-import sklearn.datasets as dts)
#         """
#     )
#     result: Atom = metta.run(
#         """
#         (= (get-cons $n) (match &self (Cons $n $y) $y))
#         (Cons seed 30)
#         (Cons random_state 170)
#         (Cons n_samples 1000)
#         (Param default (Cons n_clusters 3))
#         (=
#             (data)
#             (dts.make_circles (n_samples (get-cons n_samples)) (factor 0.5) (noise 0.05) (random_state (get-cons seed)))
#         )
#         (=
#             (get-X ($X $y))
#             $X
#         )
#         (=
#             (get-y ($X $y))
#             $y
#         )
#         ! (data)
#         """
#     )[0][0]
#     X = result.get_children()[0].get_object().content
#     y_true = result.get_children()[1].get_object().content
#     result: Atom = metta.run(
#         """
#         (=
#             (fit-outputs)
#             (spectral-clustering.fit (get-X (data)) 2)
#         )
#
#         ! (spectral-clustering.predict (fit-outputs) 2)
#         """
#     )[0][0]
#     y_pred = result.get_object().content
#     ari = adjusted_rand_score(y_true, y_pred)
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#
#     # Create a scatter plot using X (2D data) and color points by cluster labels (y_pred)
#     plt.figure(figsize=(10, 8))
#     sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_pred, palette='viridis', s=50, alpha=0.8)
#     plt.title('Spectral Clustering Results')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.legend(title='Cluster')
#     plt.tight_layout()
#     plt.show()
