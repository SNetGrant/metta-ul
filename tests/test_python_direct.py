from hyperon import MeTTa, Atom
import numpy as np


def test_kmeans(metta: MeTTa):
    metta.run(
        '''
        ! (import! &self metta_ul:cluster:kmeans)

        (=
            (X)
            ((1 0 0) (0 1 0) (0 0 1))
        )
        '''
    )

    result: Atom = metta.run('! (kmeans (np.array (X)) 3)')[0][0]

    centroids_true = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    centroids: np.ndarray = result.get_object().value

    assert np.allclose(centroids.sum(axis=0), [1, 1, 1])
    assert np.allclose(centroids.sum(axis=1), [1, 1, 1])

    assert centroids[0].tolist() in centroids_true
    assert centroids[1].tolist() in centroids_true
    assert centroids[2].tolist() in centroids_true


def test_py_dot_kmeans(metta: MeTTa):
    metta.run(
        '''
        ! (import! &self metta_ul:cluster:py_dot_kmeans)

        (=
            (X)
            ((1 0 0) (0 1 0) (0 0 1))
        )
        '''
    )

    result: Atom = metta.run('! (kmeans (py-list (X)) 3 5)')[0][0]

    centroids_true = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    centroids: np.ndarray = np.asarray(result.get_object().value)

    assert np.allclose(centroids.sum(axis=0), [1, 1, 1])
    assert np.allclose(centroids.sum(axis=1), [1, 1, 1])

    assert centroids[0].tolist() in centroids_true
    assert centroids[1].tolist() in centroids_true
    assert centroids[2].tolist() in centroids_true
