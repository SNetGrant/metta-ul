;fix steps from test_kmeans

! (import! &self metta_ul)
! (import! &self metta_ul:cluster:spectral_clustering)

! (ul-import pandas as pd)
! (ul-import numpy as unp)
! (ul-from sklearn.preprocessing import StandardScaler)
! (ul-from sklearn.preprocessing import OneHotEncoder)
! (ul-from sklearn.impute import SimpleImputer)
! (ul-from sklearn.manifold import TSNE)
! (ul-import seaborn as usns)


! (bind! &df (pd.read_csv "tests/housing.csv" (nrows 100) ))

! (import.df &df House)

(= (values ($k $v)) $v)

(: df-values (-> Expression Expression))
(= (df-values $expr)
   (if (== $expr ()) ()
       (let* (($head (car-atom $expr))
              ($tail (cdr-atom $expr))
              ($head-new (values $head))
              ($tail-new (df-values $tail))
             )
         (cons-atom $head-new $tail-new)
       )
   )
)

(= (df-indexed-values $id $expr) (let $vals (df-values $expr) (cons-atom $id $vals)))


! (bind! &data (np.array (collapse (match &self (House $id $props)  (df-indexed-values $id $props)))))



(= (pre $values)
    (let* (
        ($data (np.slice $values [:,1:]))
        ($imputer (SimpleImputer (strategy "median")))
        ($numerical-numpy (ul-slice $data [:,:-1]))
        ($categorical-numpy (ul-slice $data [:,-1:]))
        ($imputed-numerical-numpy (SimpleImputer.fit_transform $imputer $numerical-numpy))
        ($one-encoder (OneHotEncoder (handle_unknown "ignore") (sparse_output (py-atom False))))
        ($encoded-numpy (OneHotEncoder.fit_transform $one-encoder $categorical-numpy))
        ($X (unp.concatenate (py-tuple ($imputed-numerical-numpy $encoded-numpy)) (axis 1)))
        ($scaler (StandardScaler))
        ($X-scaled (StandardScaler.fit_transform $scaler $X))
    ) $X-scaled)
)

! (bind! &X-scaled (pre &data))


(= (fit-predict $ds)  
  (let* (
    ($model (spectral-clustering.fit $ds 3))
    ($labels (spectral-clustering.predict $model 3))
  ) $labels )
)

(= (reduce-dimension $ds)  
  (let* (
    ($tsne (TSNE (n_components 2) (random_state 42) (perplexity 30) (max_iter 1000)))
    ($res (TSNE.fit_transform $tsne $ds))
  ) $res )
)


! (bind! &labels (fit-predict &X-scaled))

! (cons-labels &data &labels ClusterLabel)

! (match &self (ClusterLabel $index $label) ($index $label))

! (match &self (= (spectral-clustering.compute-affinity-matrix $X $rbf-kernel-sigma) $Expression) (remove-atom &self (= (spectral-clustering.compute-affinity-matrix $X $rbf-kernel-sigma) $Expression)))
(=
    (spectral-clustering.compute-affinity-matrix $X $n-neighbors)
    (let*
        (
            ($dist (spectral-clustering.square-distance-matrix (spectral-clustering.square-norm $X) $X))
            ($N (np.shape $dist 0))
            ($end-index (+ 1 $n-neighbors))
            ($knn-indices (np.take (np.argsort $dist 1) (np.arange 1 $end-index) 1))
            ($row-index (np.reshape (np.arange $N) (np.array ($N 1))))
            ($flat-index (np.reshape (np.add (np.mul $row-index $N) $knn-indices) -1))
            ($mask (np.isin (np.arange (* $N $N)) $flat-index))
            ($A-flat (np.where $mask 1.0 0.0))
            ($A (np.reshape $A-flat (np.array ($N $N))))
        )
        (np.add 0.00000001 (np.maximum $A (np.transpose $A)))
    )
)


! (bind! &labels2 (fit-predict &X-scaled))

! (cons-labels &data &labels2 ClusterLabel2)

! (match &self (ClusterLabel2 $index $label) ($index $label))