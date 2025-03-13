! (import! &self metta_ul:pdm)
! (import! &self metta_ul:skl)
! (import! &self metta_ul:sns)

! (bind! &df (pdm.read_csv "tests/housing.csv" (usecols (py-list ("longitude" "latitude" "median_house_value")))))

! (sns.scatterplot (data &df) (x "longitude")  (y "latitude") (hue "median_house_value"))
