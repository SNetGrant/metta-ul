! (import! &self metta_ul:pdm)
! (import! &self metta_ul:skl)

! (bind! &df (pdm.read_csv "tests/housing.csv" (usecols (py-list ("longitude" "latitude")))))

! (bind! &n (skl.preprocessing.normalize (pdm.values &df)))

! (assertEqual (get-type &n) (NPArray (20640 2)))
