! (import! &self metta_ul)
! (import! &self metta_ul:cluster:kmeans)

! (bind! &data (np.array ((1 0 0) (0 1 0) (0 0 1))))

! (bind! &cn (kmeans &data 3))

! (assertEqual (get-type $cn) (get-type (py-list )))
