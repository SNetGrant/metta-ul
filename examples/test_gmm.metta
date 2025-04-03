! (import! &self metta_ul)
! (import! &self metta_ul:cluster:gmm)

! (bind! &wine (skl.datasets.load_wine))
! (bind! &data  (np.slice &wine [:,:2]))

! (assertEqual (get-type &data) (NPArray (178 2)))


! (bind! &model (gmm &data 2))


! (bind! &labels (gmm.predict &data &model))


! (sns.scatterplot (x (np.slice &data [:,0]))  (y (np.slice &data [:,1])) (hue &labels) (alpha 0.6))

