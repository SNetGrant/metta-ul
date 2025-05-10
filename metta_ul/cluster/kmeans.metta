!(import! &self metta_ul)

(: kmeans.update (-> (NPArray ($n $d)) (NPArray ($k $n)) (NPArray ($k $d))))
(=
    (kmeans.update $X $assignments)
    (np.div
        (np.matmul $assignments $X)
        (np.sum 
            $assignments
            1
            py.none
            py.none
            True
        )
    )
)

(: kmeans.assign (-> (NPArray ($n $d)) (NPArray ($k $d)) (NPArray ($k $n))))
(=
    (kmeans.assign $X $centroids)
    (np.transpose
        (np.one_hot
            (np.argmin
                (np.linalg.norm
                    (np.sub
                        (np.expand_dims $X 0)
                        (np.expand_dims $centroids 1)
                    )
                    py.none
                    2
                )
                0
            )
            (np.shape $centroids 0)
        )
    )
)


(: kmeans.recursion (-> (NPArray ($n $d)) (NPArray ($k $d)) Number (NPArray ($k $d))))
(= 
    (kmeans.recursion $X $centroids $max-iter)
    (if (> $max-iter 0)
        (kmeans.recursion
            $X
            (kmeans.update
                $X
                (kmeans.assign $X $centroids)
            )
            (- $max-iter 1)
        )
        $centroids
    )
)
(: kmeans (-> (NPArray ($n $d)) Number (NPArray ($k $d))))
(=
    (kmeans $X $k)
    (kmeans $X $k 100)
)


(: kmeans (-> (NPArray ($n $d)) Number Number (NPArray ($k $d))))
(=
    (kmeans $X $k $max-iter)
    (kmeans.recursion $X (np.choose $X $k) $max-iter)
)

(=
    (kmeans.fit $X $k $max-iter) 
    (kmeans $X $k $max-iter)
)

(=
  (kmeans.predict $X $centers)
    (np.argmax (kmeans.assign $X $centers) 0)
)



