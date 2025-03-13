from hyperon.atoms import G, Atoms
from hyperon.ext import register_atoms

import seaborn as sns
from numme import PatternOperation
from pdm import unwrap_args


def wrapnpop(func):
    def wrapper(*args):
        a, k = unwrap_args(args)
        func(*a, **k)
        return [Atoms.UNIT]
    return wrapper


@register_atoms
def slk_atoms():

    snsScatterplot = G(PatternOperation(
        "sns.scatterplot", wrapnpop(sns.scatterplot), unwrap=False))

    return {
        r"sns\.scatterplot": snsScatterplot,
    }
