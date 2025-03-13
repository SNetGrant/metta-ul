! (import! &self metta_ul:pdm)
! (import! &self metta_ul:skl)
! (import! &self metta_ul:cluster:kmeans)

! (bind! &df (pdm.read_csv "tests/housing.csv" (usecols (py-list ("longitude" "latitude")))))

! (bind! &norm (pdm.values &df))

! (bind! &n (skl.preprocessing.normalize &norm))

! (bind! &cn (kmeans &norm 3))

! (assertEqual (get-type $cn) (get-type (py-list )))
