! (import! &self metta_ul)
! (import! &self metta_ul:cluster:kmeans)

! (bind! &df (pdm.read_csv "tests/housing.csv" (usecols (py-list ("longitude" "latitude")))))

! (bind! &norm (pdm.values &df))

! (bind! &n (skl.preprocessing.normalize &norm))

! (bind! &model (kmeans.fit &norm 3))

! (bind! &plot (sns.scatterplot (data &df) (x "longitude")  (y "latitude") (hue (labels &model))))

