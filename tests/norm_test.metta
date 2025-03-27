! (import! &self metta_ul)

! (bind! &df (pdm.read_csv "tests/housing.csv" (usecols (py-list ("longitude" "latitude")))))

! (bind! &n (skl.preprocessing.normalize (pdm.values &df)))

! (assertEqual (get-type &n) (NPArray (20640 2)))
