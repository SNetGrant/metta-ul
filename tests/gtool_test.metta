! (import! &self metta_ul)

!(ul-import math as ma)

!(assertEqual (ma.sqrt 4) 2)

!(ul-import max)
!(assertEqual (get-metatype max) Grounded)
!(assertEqual (max (py-list (1 2 3))) 3)

!(assertEqual (get-metatype RandomState) Symbol)

!(ul-from numpy.random import RandomState)

!(assertEqual (get-metatype RandomState) Grounded)


!(ul-import sklearn.datasets as ds)

!(assertEqual (get-metatype ds.make_blobs) Grounded)

!(assertEqual (ul-slice (py-list (1 2 3)) [0]) 1)
!(assertEqual (ul-slice (py-tuple (1 2 3)) [0]) 1)

!(ul-from numpy import ndarray)
!(assertEqual (get-metatype ndarray.shape) Grounded)
!(bind! &x (ndarray.shape (np.array (1,2,3))))
!(assertEqual &x (1))

