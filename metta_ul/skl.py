from hyperon.atoms import G
from hyperon.ext import register_atoms

from sklearn.preprocessing import normalize
from sklearn.datasets import load_wine

from .numme import PatternOperation
from .numme import wrapnpop


def _load_wine_data():
    return load_wine().data


@register_atoms
def skl_atoms():

    skl_normalize = G(
        PatternOperation(
            "skl.preprocessing.normalize", wrapnpop(normalize), unwrap=False
        )
    )

    skl_windata = G(
        PatternOperation(
            "skl.datasets.load_wine", wrapnpop(_load_wine_data), unwrap=False
        )
    )

    return {
        r"skl\.preprocessing\.normalize": skl_normalize,
        r"skl\.datasets\.load_wine": skl_windata,
    }
