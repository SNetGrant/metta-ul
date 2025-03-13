! (import! &self metta_ul:pdm)
! (import! &self metta_ul:skl)

! (bind! &df (pdm.read_csv "tests/housing.csv" (usecols (py-list ("longitude" "latitude")))))

! (assertEqual (get-type &df) PDDataFrame)

! (bind! &norm (pdm.values &df))

! (assertEqual (get-type &norm) (NPArray (20640 2)))

! (bind! &n (skl.preprocessing.normalize &norm))

! (assertEqual (get-type &n) (NPArray (20640 2)))

