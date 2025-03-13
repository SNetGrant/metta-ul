from hyperon.atoms import G
from hyperon.ext import register_atoms

from sklearn.preprocessing import normalize
from numme import PatternOperation
from numme import wrapnpop


@register_atoms
def slk_atoms():

    skl_normalize = G(PatternOperation(
        "skl.preprocessing.normalize", wrapnpop(normalize), unwrap=False))

    return {
        r"skl\.preprocessing\.normalize": skl_normalize,
    }
