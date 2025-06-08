! (import! &self metta_ul:cluster:spectral_clustering)
! (ul-import math as ma)
! (ul-import sklearn.metrics as metrics)

;; Setting the random seed
! (np.random.seed 1)

;; ========================================================================
;; ======================= Defining test variables ========================
;; ========================================================================

(=
    (X-1D-Single)
    (np.array ((1)))
)

(=
    (X-1D-Multiple)
    (np.array ((0) (1)))
)

(=
    (X-2D-Multiple)
    (np.array ((0 0) (0 1) (1 0)))
)

(=
    (X-2D-Random-1)
    (np.random.rand 10 3)
)

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
    (X-2D-Random-2)
    (np.random.rand 5 2)
)

(=
    (L1)
    (np.array ((0 0 0) (0 1 0) (0 0 2)))
)

(=
    (L2)
    (np.array ((0.5 -0.5) (-0.5 0.5)))
)

(=
    (L3)
    (np.array ((0 0) (0 0)))
)

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

(=
    (X1)
    (np.array ((1 0 0) (0 1 0) (0 0 1)))
)

(=
    (K1)
    3
)

(=
    (X2)
    (np.array ((0 0) (0.1 0) (1.0 1.0) (1.1 1.0)))
)

(=
    (K2)
    2
)

(=
    (X-Fit)
    (np.array ((0.0 0.0) (0.1 0) (1.0 1.0) (1.1 1.0)))
)

;; ========================================================================
;; ============= Test spectral-clustering.rbf-affinity-matrix =============
;; ========================================================================

(=
    (spectral-clustering.rbf-affinity-matrix.test1)
    (let
        $W
        (spectral-clustering.compute-rbf-affinity-matrix
            (X-1D-Single)
            1.0
        )

        (assertEqual
            (
                (np.assertAllClose (np.shape $W 0) 1)
                (np.assertAllClose (np.shape $W 1) 1)
                (np.assertAllClose $W (np.array ((1.0))))
            )
            (() () ())
        )
    )
)
(Test spectral-clustering.rbf-affinity-matrix.test1)

(=
    (spectral-clustering.rbf-affinity-matrix.test2)
    (let
        $W
        (spectral-clustering.compute-rbf-affinity-matrix
            (X-1D-Multiple)
            1.0
        )

        (np.assertAllClose
            $W
            (np.array
                (
                    (1.0 (ma.exp -0.5))
                    ((ma.exp -0.5) 1.0)
                )
            )
        )
    )
)
(Test spectral-clustering.rbf-affinity-matrix.test2)


(=
    (spectral-clustering.rbf-affinity-matrix.test3)
    (let
        $W
        (spectral-clustering.compute-rbf-affinity-matrix
            (X-2D-Multiple)
            1.0
        )

        (np.assertAllClose
            $W
            (np.array
                (
                    (1.0 (ma.exp -0.5) (ma.exp -0.5))
                    ((ma.exp -0.5) 1.0 (ma.exp -1.0))
                    ((ma.exp -0.5) (ma.exp -1.0) 1.0)
                )
            )
        )
    )
)
(Test spectral-clustering.rbf-affinity-matrix.test3)

(=
    (spectral-clustering.rbf-affinity-matrix.test4)
    (let
        $W
        (spectral-clustering.compute-rbf-affinity-matrix
            (X-2D-Multiple)
            1.0
        )

        (np.assertAllClose
            $W
            (np.array
                (
                    (1.0 (ma.exp -0.5) (ma.exp -0.5))
                    ((ma.exp -0.5) 1.0 (ma.exp -1.0))
                    ((ma.exp -0.5) (ma.exp -1.0) 1.0)
                )
            )
        )
    )
)
(Test spectral-clustering.rbf-affinity-matrix.test4)

(=
    (spectral-clustering.rbf-affinity-matrix.test5)
    (let
        $W
        (spectral-clustering.compute-rbf-affinity-matrix
            (X-2D-Random-1)
            1.0
        )

        (np.assertAllClose
            $W
            (np.transpose $W)
        )
    )
)
(Test spectral-clustering.rbf-affinity-matrix.test5)

;; ========================================================================
;; ========== Test spectral-clustering.normalized-laplacian ===============
;; ========================================================================
(=
    (spectral-clustering.normalized-laplacian.test1)
    (let
        $L-norm
        (spectral-clustering.normalized-laplacian
            (I)
            (spectral-clustering.inverse-degree-matrix
                (spectral-clustering.degree
                    (I)
                )
            )
        )

        (np.assertAllClose
            $L-norm
            (np.array
                (
                    (0.0 0.0 0.0)
                    (0.0 0.0 0.0)
                    (0.0 0.0 0.0)
                )
            )
        )
    )
)
(Test spectral-clustering.normalized-laplacian.test1)

(=
    (spectral-clustering.normalized-laplacian.test2)
    (let
        $L-norm
        (spectral-clustering.normalized-laplacian
            (W-2)
            (spectral-clustering.inverse-degree-matrix
                (spectral-clustering.degree
                    (W-2)
                )
            )
        )

        (np.assertAllClose
            $L-norm
            (np.array
                (
                    (0.5 -0.5)
                    (-0.5 0.5)
                )
            )
        )
    )
)
(Test spectral-clustering.normalized-laplacian.test2)

(=
    (spectral-clustering.normalized-laplacian.test3)
    (let
        $L-norm
        (spectral-clustering.normalized-laplacian
            (W-3)
            (spectral-clustering.inverse-degree-matrix
                (spectral-clustering.degree
                    (W-3)
                )
            )
        )

        (np.assertAllClose
            $L-norm
            (np.array
                (
                    ((- 1 (/ 1.0 3)) (/ -1.0 3) (/ -1.0 3))
                    ((/ -1.0 3) (- 1 (/ 1.0 3)) (/ -1.0 3))
                    ((/ -1.0 3) (/ -1.0 3) (- 1 (/ 1.0 3)))
                )
            )
        )
    )
)
(Test spectral-clustering.normalized-laplacian.test3)

(=
    (spectral-clustering.normalized-laplacian.test4)
    (let
        $W-X-2D-Random
        (spectral-clustering.compute-rbf-affinity-matrix
            (X-2D-Random-2)
            1.0
        )
        (let
            $L-norm
            (spectral-clustering.normalized-laplacian
                $W-X-2D-Random
                (spectral-clustering.inverse-degree-matrix
                    (spectral-clustering.degree
                        $W-X-2D-Random
                    )
                )
            )

            (np.assertAllClose
                $L-norm
                (np.transpose $L-norm)
            )

        )
    )
)
(Test spectral-clustering.normalized-laplacian.test4)

;; ========================================================================
;; ========== Test spectral-clustering.spectral-embeddings ================
;; ========================================================================
(=
    (spectral-clustering.spectral-embeddings.test1)
    (let
        $eigh-I
        (spectral-clustering.eigh (I))

        (np.assertAllClose
            (np.abs (spectral-clustering.spectral-embeddings $eigh-I 2))
            (np.abs
                (np.array
                    (
                        (1.0 0.0)
                        (0.0 1.0)
                        (0.0 0.0)
                    )
                )
            )
        )
    )
)
(Test spectral-clustering.spectral-embeddings.test1)

(=
    (spectral-clustering.spectral-embeddings.test2)
    (let
        $eigh-L1
        (spectral-clustering.eigh (L1))
        (let
            $U-L1
            (spectral-clustering.spectral-embeddings $eigh-L1 2)
            (assertEqual
                (
                    (np.assertAllClose (np.shape $U-L1 0) 3)
                    (np.assertAllClose (np.shape $U-L1 1) 2)
                    (np.assertAllClose
                        (np.abs (spectral-clustering.spectral-embeddings $eigh-L1 2))
                        (np.abs
                            (np.array
                                (
                                    (1.0 0.0)
                                    (0.0 1.0)
                                    (0.0 0.0)
                                )
                            )
                        )
                    )
                )
                (() () ())
            )
        )
    )
)
(Test spectral-clustering.spectral-embeddings.test2)

(=
    (spectral-clustering.spectral-embeddings.test3)
    (let
        $eigh-L2
        (spectral-clustering.eigh (L2))
        (let
            $U-L2
            (spectral-clustering.spectral-embeddings $eigh-L2 2)
            (let
                $Lambda-L2
                (np.diag
                    (np.take
                        (np.argsort
                            (spectral-clustering.eigenvalues $eigh-L2)
                        )
                        (np.arange 2)
                    )
                )
                (assertEqual
                    (
                        (np.assertAllClose (np.shape $U-L2 0) 2)
                        (np.assertAllClose (np.shape $U-L2 1) 2)
                        (np.assertAllClose
                            (np.matmul (L2) $U-L2)
                            (np.matmul $U-L2 $Lambda-L2)
                        )
                    )
                    (() () ())
                )
            )
        )
    )
)
(Test spectral-clustering.spectral-embeddings.test3)

(=
    (spectral-clustering.spectral-embeddings.test5)
    (let
        $eigh-L3
        (spectral-clustering.eigh (L3))
        (let
            $U-L3
            (spectral-clustering.spectral-embeddings $eigh-L3 0)

            (assertEqual
                (
                    (np.assertAllClose (np.shape $U-L3 0) 2)
                    (np.assertAllClose (np.shape $U-L3 1) 0)
                )
                (() ())

            )
        )
    )
)
(Test spectral-clustering.spectral-embeddings.test5)

;; ========================================================================
;; ============== Test spectral-clustering.row-normalize ==================
;; ========================================================================
(=
    (spectral-clustering.row-normalize.test1)
    (let
        $A-norm
        (spectral-clustering.row-normalize (A))

        (assertEqual
            (
                (np.assertAllClose (np.shape $A-norm 0) 2)
                (np.assertAllClose (np.shape $A-norm 1) 2)
                (np.assertAllClose
                    $A-norm
                    (np.array
                        (((/ 3.0 5) (/ 4.0 5)) ((/ 5.0 13) (/ 12.0 13)))
                    )
                )
            )
            (() () ())
        )

    )
)
(Test spectral-clustering.row-normalize.test1)

(=
    (spectral-clustering.row-normalize.test2)
    (let
        $B-norm
        (spectral-clustering.row-normalize (B))

        (assertEqual
            (
                (np.assertAllClose (np.shape $B-norm 0) 3)
                (np.assertAllClose (np.shape $B-norm 1) 1)
                (np.assertAllClose
                    $B-norm
                    (np.array
                        ((1.0) (-1.0) (1.0))
                    )
                )
            )
            (() () ())
        )

    )
)
(Test spectral-clustering.row-normalize.test2)

(=
    (spectral-clustering.row-normalize.test3)
    (let
        $C-norm
        (spectral-clustering.row-normalize (C))

        (assertEqual
            (
                (np.assertAllClose ((py-dot (np.slice (np.isnan $C-norm) [0]) all)) True)
                (np.assertAllClose
                    (np.slice $C-norm [1])
                    (np.array (0.4472135954999579 0.8944271909999159))
                )
            )
            (() ())
        )

    )
)
(Test spectral-clustering.row-normalize.test3)

(=
    (spectral-clustering.row-normalize.test4)
    (let
        $D-norm
        (spectral-clustering.row-normalize (D))

        (assertEqual
            (
                (np.assertAllClose (np.shape $D-norm 0) 3)
                (np.assertAllClose (np.shape $D-norm 1) 3)
                (np.assertAllClose
                    $D-norm
                    (np.array
                        (
                            (0.57735 0.57735 0.57735)
                            (0.57735 0.57735 0.57735)
                            (0.57735 0.57735 0.57735)
                        )
                    )
                )
            )
            (() () ())
        )

    )
)
(Test spectral-clustering.row-normalize.test4)

;; ========================================================================
;; ================== Test spectral-clustering.cluster ====================
;; ========================================================================
(=
    (spectral-clustering.cluster.test1)
    (let
        $W-X1
        (spectral-clustering.compute-rbf-affinity-matrix
            (X1)
            0.1
        )
        (let
            $U-X1
            (spectral-clustering.spectral-embeddings
                (spectral-clustering.eigh
                    (spectral-clustering.normalized-laplacian
                        $W-X1
                        (spectral-clustering.inverse-degree-matrix
                            (spectral-clustering.degree
                                $W-X1
                            )
                        )
                    )
                )
                (K1)
            )
            (let
                $labels-X1
                (np.argmax
                    (np.transpose
                        (kmeans.assign
                            $U-X1
                            (spectral-clustering.cluster $U-X1 (K1) 10)
                        )
                    )
                    1
                )
                (np.assertAllClose
                    (metrics.adjusted_rand_score $labels-X1 (np.array (0 1 2)))
                    1.0
                )
            )
        )
    )
)
(Test spectral-clustering.cluster.test1)

(=
    (spectral-clustering.cluster.test2)
    (let
        $W-X2
        (spectral-clustering.compute-rbf-affinity-matrix
            (X2)
            0.1
        )
        (let
            $U-X2
            (spectral-clustering.spectral-embeddings
                (spectral-clustering.eigh
                    (spectral-clustering.normalized-laplacian
                        $W-X2
                        (spectral-clustering.inverse-degree-matrix
                            (spectral-clustering.degree
                                $W-X2
                            )
                        )
                    )
                )
                (K2)
            )
            (let
                $labels-X2
                (np.argmax
                    (np.transpose
                        (kmeans.assign
                            $U-X2
                            (spectral-clustering.cluster $U-X2 (K2) 10)
                        )
                    )
                    1
                )
                (np.assertAllClose
                    (metrics.adjusted_rand_score $labels-X2 (np.array (0 0 1 1)))
                    1.0
                )
            )
        )
    )
)
(Test spectral-clustering.cluster.test2)

;; ========================================================================
;; == Test spectral-clustering.fit === Test spectral-clustering.predict ===
;; ========================================================================
(=
    (spectral-clustering.fit-predict.test1)
    (let
        $fit-outputs
        (spectral-clustering.fit (X-Fit) 2)
        (let
            $pred
            (spectral-clustering.predict $fit-outputs 2)
            (assertEqual
                (
                    (np.assertAllClose (np.slice $pred [0]) (np.slice $pred [1]))
                    (np.assertAllClose (np.slice $pred [-2]) (np.slice $pred [-1]))
                    (np.assertAllClose (np.unique $pred) (np.array (0 1)))
                )
                (() () ())
            )
        )
    )
)
(Test spectral-clustering.fit-predict.test1)

(=
    (spectral-clustering.fit-predict.test2)
    (let
        $fit-outputs
        (spectral-clustering.fit (X-Fit) 2 "binary-knn-graph" 1 100)
        (let
            $pred
            (spectral-clustering.predict $fit-outputs 2)
            (assertEqual
                (
                    (np.assertAllClose (np.slice $pred [0]) (np.slice $pred [1]))
                    (np.assertAllClose (np.slice $pred [-2]) (np.slice $pred [-1]))
                    (np.assertAllClose (np.unique $pred) (np.array (0 1)))
                )
                (() () ())
            )
        )
    )
)
(Test spectral-clustering.fit-predict.test2)
