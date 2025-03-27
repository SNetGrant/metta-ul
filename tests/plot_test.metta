! (import! &self metta_ul)

! (bind! &df (pdm.read_csv "tests/housing.csv" (usecols (py-list ("longitude" "latitude" "median_house_value")))))

! (bind! &plot (sns.scatterplot (data &df) (x "longitude")  (y "latitude") (hue "median_house_value")))

! (assertEqual (get-type $plot) (get-type Plot))
