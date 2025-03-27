!(import! &self metta_ul:numme)

! (bind! pyNone (py-atom "None"))
! (bind! pyTrue (py-atom "True"))
! (bind! pyFalse (py-atom "False"))

(=
    (kmeans.update $X $assignments)
    (np.div
        (np.matmul $assignments $X)
        (np.sum 
            $assignments
            1
            pyNone
            pyNone
            pyTrue
        )
    )
)

(=
    (kmeans.assign $X $centroids $k)
    (np.transpose
        (np.one_hot
            (np.argmin
                (np.linalg.norm
                    (np.sub
                        (np.expand_dims $X 0)
                        (np.expand_dims $centroids 1)
                    )
                    pyNone
                    2
                )
                0
            )
            $k
        )
    )
)

(= 
    (kmeans.recursion $X $centroids $k $max-iter)
    (if (> $max-iter 0)
        (kmeans.recursion
            $X
            (kmeans.update
                $X
                (kmeans.assign $X $centroids $k)
            )
            $k
            (- $max-iter 1)
        )
        $centroids
    )
)

(=
    (kmeans $X $k)
    (kmeans $X $k 100)
)

(=
    (kmeans $X $k $max-iter)
    (kmeans.recursion $X (np.choose $X $k) $k $max-iter)
)

(=
  (kmeans.fit $X $k)
  (let* (($centers (kmeans $X $k)) ($lables (np.argmax (kmeans.assign $X $centers $k) 0))) (KmeansModel $X $centers $lables))
)

(=
  (labels (KmeansModel $A $B $C))
  $C
)


