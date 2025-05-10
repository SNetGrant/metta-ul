! (import! &self metta_ul)
;! (ul-import pandas as pdm)
! (ul-from pandas import DataFrame)
; ! (ul-from sklearn.preprocessing import normalize)

; ! (bind! &df (pd.read_csv "tests/housing.csv" (usecols (py-list ("longitude" "latitude")))))

; ! (assertEqual (get-type &df) PDDataFrame)

; ! (bind! &norm (pdm.values &df))

; ! (assertEqual (get-type &norm) (NPArray (20640 2)))

; ! (bind! &n (normalize &norm))

; ! (assertEqual (get-type &n) (NPArray (20640 2)))

(House 1 ((lat 1) (lang 2)))
(House 2 ((lat 1) (lang 2)))
!(bind! &df (make-df (collapse (match &self (House $x $expr) ($x $expr)))))
!(bind! &res (DataFrame.values (make-df (collapse (match &self (House $x $expr) ($x $expr))))))
!(assertEqual &res (np.array ((1 2) (1 2))))