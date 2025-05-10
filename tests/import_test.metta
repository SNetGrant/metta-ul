 ! (import! &self metta_ul)
! (import! &self metta_ul:cluster:kmeans)
! (import! &self metta_ul:helper)
! (ul-from pandas import DataFrame)


! (assertEqual (kv-values ((k1 1) (k2 2))) (1 2))

! (ul-import pandas as pd)

! (bind! &df (pd.read_csv "tests/housing.csv" (nrows 100) (usecols (py-list ("longitude" "latitude")))))

! (import.df &df House)

;! (match &self (House 0 $expr) (assertEqual (get-type $expr) ((longitude Number) (latitude Number))))

;! (assertEqualToResult () ())
! (bind! &data (make-df (collapse (match &self (House $id $expr) ($id $expr)))))

! (bind! &model (kmeans (DataFrame.values &data) 3))
! (bind! &labels (kmeans.predict (DataFrame.values &data) &model))

! (cons-labels (DataFrame.values (DataFrame.index &data)) &labels ClusterLabelHouse)



; ! (bind! &res (match &self (ClusterLabelHouse $id $label) $id))
; ! (assertEqual (get-type &res) Number)

; !(match &self (,(House $id $expr) (ClusterLabelHouse $id 0)) (assertEqual (get-type (kv-values $expr)) ((Number Number))))

; !(println!  "Done")
