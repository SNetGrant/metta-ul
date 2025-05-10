! (import! &self metta_ul)
! (import! &self metta_ul:cluster:kmeans)
! (import! &self metta_ul:cluster:gmm)
! (import! &self metta_ul:cluster:agglomerative)
! (import! &self metta_ul:cluster:spectral_clustering)
! (import! &self metta_ul:cluster:bisecting_kmeans)
! (ul-import sklearn.datasets as dts)
! (ul-from sklearn.preprocessing import StandardScaler)
! (ul-import matplotlib.pyplot as plt)
! (ul-import sklearn.metrics as metrics)
! (ul-from numpy.random import RandomState)


;time functions
! (ul-import datetime.datetime.now as now)
! (ul-from time import perf_counter)

(= (time.sub $t1 $t2) ((py-dot $t1 __sub__)$t2))
(= (time.fmt $t) ((py-dot $t strftime) "%Y/%m/%d %H:%M:%S" ))
(= (timedelta.fmt $td) ((py-dot $td total_seconds)))

(= (timeit (quote $f)) (
  let* (
    ($t1 (perf_counter))
    ($res $f)
    ($t2 (perf_counter))
    ($t (time.sub $t2 $t1))
  )
  ($res $t)
))

(= (get-cons $n) (match &self (Cons $n $y) $y))


(=  (get-param $n $k) 
    (unify &self (Param $n (Cons $k $r)) $r (match &self (Param default (Cons $k $r)) $r))
)

(= (has-tile $state) (
  if (== (get-state $state) False)
  (change-state! $state True)
  ($state)
))

(= (inc $counter) (change-state! $counter (+ (get-state $counter) 1)))

(: r-cycle (-> Expression Expression Number Expression))
(= (r-cycle $d $t $s)
        (if (== $s 0) ()
            (let* (($expr (if (== $t ()) $d $t))
                    ($head (car-atom $expr))
                    ($tail (cdr-atom $expr))
                    ($tail-new (r-cycle $d $tail (- $s 1)))
                    )
                (cons-atom $head $tail-new)
            )
        )
)

(: cycle (-> Expression Number Expression))
(= (cycle $d $s) (r-cycle $d () $s))

(= (count $x) (let* (
      ($count (new-state 0))
      ($_ 
          (collapse (match &self $x 
              (change-state! $count (+ (get-state $count) 1))))
      )
    )
    (get-state $count)
))

!(bind! &colors ("#377eb8" "#ff7f00" "#4daf4a" "#f781bf" "#a65628" "#984ea3" "#999999" "#e41a1c" "#dede00"))
(= (make-color $labels) 
    ;(np.take (np.array (cycle &colors (+ ((py-atom max) $labels) 1))) $labels)
    (let* (
      ($max ((py-atom int)((py-atom max) $labels)))
      ($cl (np.array (cycle &colors (+ $max 1))))
      ($c2 (np.take $cl $labels))
      ($res (np.append $c2 (py-list ("#000000"))))
    ) $c2)
)

(= (make-aniso) 
  (let* (
    (($X $Y) (dts.make_blobs (n_samples (get-cons n_samples)) (random_state (get-cons random_state))))
    ($transformation (np.array ((0.6 -0.6) (-0.4 0.8))))
    ($X-aniso (np.dot $X $transformation))
  )
  ($X-aniso $Y))
)

(Cons seed 30)
(Cons random_state 170)
(Cons n_samples 500)
(Param default (Cons n_clusters 3))

(ToyData noisy_circles
  (dts.make_circles (n_samples (get-cons n_samples)) (factor 0.5) (noise 0.05) (random_state (get-cons seed))))
(Param noisy_circles (Cons n_clusters 2)) 

(ToyData noisy_moon (dts.make_moons (n_samples (get-cons n_samples)) (noise 0.05) (random_state (get-cons seed))))
(Param noisy_moon (Cons n_clusters 2))

(ToyData blobs (dts.make_blobs (n_samples (get-cons n_samples)) (random_state (get-cons seed))))

(ToyData varied (dts.make_blobs (n_samples (get-cons n_samples)) (cluster_std (py-list (1.0 2.5 0.5))) (random_state (get-cons random_state))))

(ToyData aniso (make-aniso))

(ToyData no-structure ((RandomState.rand (RandomState (get-cons random_state)) (get-cons n_samples) 2) ()))
(= (normalize $x) (StandardScaler.fit_transform (StandardScaler) $x))

(= (cal-intrinsic-metrics $data $labels)
  (let* (
    ($sil (metrics.silhouette_score $data $labels))
    ($chi (metrics.calinski_harabasz_score $data $labels))
    ($dbi (metrics.davies_bouldin_score $data $labels))
  )
  ($sil $chi $dbi)
  )
)

(= (cal-extrinsic-metrics $ty $labels)
  (if (== $ty ()) (0 0 0)
    (let* 
    (
      ($ari (metrics.adjusted_rand_score $ty $labels))
      ($nmi (metrics.normalized_mutual_info_score $ty $labels))
      ($ami (metrics.adjusted_mutual_info_score $ty $labels))
    )
    ($ari $nmi $ami)
    )
  )
)

(= (fit-predict $dn $ds)  
  (let* (
    (() (println! ("start kmeans on " $dn "data type" (get-type $ds))))
    (($model $t1) (timeit (quote (kmeans $ds (get-param $dn n_clusters) 100))))
    (($labels $t2) (timeit (quote (kmeans.predict $ds $model))))
    (() (println! ("kmeans done on " $dn take (+ $t1 $t2))))
  ) ($labels "Kmeans" (+ $t1 $t2)))
)


(= (fit-predict $dn $ds) 
  (let* (
    (() (println! ("start GMM on " $dn "data type" (get-type $ds))))
    (($model $t1) (timeit (quote (gmm $ds (get-param $dn n_clusters)))))
    (($labels $t2) (timeit (quote (gmm.predict $ds $model))))
    (() (println! ("GMM done on " $dn take (+ $t1 $t2))))
  ) ($labels "GMM" (+ $t1 $t2)))
)

(= (fit-predict $dn $ds) 
  (let* (
    (() (println! ("start bisecting on " $dn "data type" (get-type $ds) (get-param $dn n_clusters))))
    (($model $t1) (timeit (quote (bisecting-kmeans.fit $ds (get-param $dn n_clusters) 10))))
    (($labels $t2) (timeit (quote (bisecting-kmeans.predict $ds $model))))
    (() (println! ("bisecting done on " $dn take (+ $t1 $t2))))
  ) ($labels "bisecting" (+ $t1 $t2)))
)

(= (fit-predict $dn $ds) 
  (let* (
    (() (println! ("start spectral-clustering on " $dn "data type" (get-type $ds))))
    (($model $t1) (timeit (quote (spectral-clustering.fit $ds (get-param $dn n_clusters)))))
    (() (println! ("fitted" $dn)))
    (($labels $t2) (timeit (quote (spectral-clustering.predict $model (get-param $dn n_clusters)))))
    (() (println! ("spectral-clustering done on " $dn take (+ $t1 $t2))))
  ) ($labels "SpectralClustering" (+ $t1 $t2)))
)

; (= (fit-predict $dn $ds) 
;   (let* (
;     (() (println! ("start Agglomerative on " $dn "data type" (get-type $ds) "n_cluster" (get-param $dn n_clusters))))
;     (($labels $t) (timeit (quote (agglomerative.fit-predict $ds (get-param $dn n_clusters) "single"))))
;     (() (println! ("Agglomerative done on " $dn $labels)))
;   ) ($labels "Agglomerative" $t))
; )


!(bind! &tdc (count (ToyData $s $y)))
!(bind! &agn (let* (
      ($count (new-state 0))
      ($_ 
          (collapse (match &self (= (fit-predict $a $b) $y) 
              (change-state! $count (+ (get-state $count) 1))))
      )
    )
    (get-state $count)
))

(= (plot-it $data $labels $algn $plt-idx) (
  let* (
    ($splt (plt.subplot &tdc &agn (get-state $plt-idx)))
    ($t (if (<= (get-state $plt-idx) &agn) (plt.title $algn (size 18)) $algn))
    ($plt (plt.scatter (np.slice $data [:,0]) (np.slice $data [:,1]) (s 10) (color (make-color $labels))))
    ($xl (plt.xlim -2.5 2.5))
    ($yl (plt.ylim -2.5 2.5))
    ($yt (plt.xticks (py-list ())))
    ($xt (plt.yticks (py-list ())))
    ($count (inc $plt-idx))
  )
  ()
))


!(plt.figure (figsize (py-tuple ((+ 3 (*  9 2)) 13))))

!(plt.subplots_adjust (left 0.02) (right 0.98) (top 0.95) (wspace 0.05) (hspace 0.01))

!(collapse(let* (
  ($plt-idx  (new-state 1))
  (() (println! ("datasets count:" &tdc "alg count" &agn)))
  (($dn ($X $ty)) (match &self (ToyData $name $dataset) ($name $dataset)))
  ($norm (normalize $X))
  (($labels $algn $dur) (fit-predict $dn $norm))
  (($sil $chi $dbi) (cal-intrinsic-metrics $norm $labels))
  (($ari $nmi $ami) (cal-extrinsic-metrics $ty $labels))
  (() (println! ("fit predict" $algn "on" $dn "take" $dur )))
  (() (println! (Intrinsic Metrics ":"  Silhouette $sil Calinski-Harabasz $chi Davies-Bouldin $dbi)))
  (() (println! (Extrinsic Metrics ":"  Adjusted Rand Index $ari Normalized Mutual Information $nmi Adjusted Mutual Information $ami)))
  (() (plot-it $norm $labels $algn $plt-idx))
) ()))
!(plt.savefig "out.png")
