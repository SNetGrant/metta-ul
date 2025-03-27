from hyperon.atoms import (
    G,
    Atoms,
    MatchableObject,
    GroundedAtom,
    ExpressionAtom,
    VariableAtom,
    OperationObject,
    S,
    AtomType,
    NoReduceError,
    get_string_value,
)
from hyperon.ext import register_atoms
import matplotlib.pyplot as plt
import seaborn as sns
from .pdm import unwrap_args


class PlotValue(MatchableObject):

    def __eq__(self, other):
        return isinstance(other, PlotValue) and self.content == other.content

    def match_(self, other):
        bindings = {}
        if isinstance(other, GroundedAtom):
            other = other.get_object()
        # Match by equality with another TensorValue
        if isinstance(other, PlotValue):
            return [{}] if other == self else []

        if isinstance(other, ExpressionAtom):
            ch = other.get_children()
            # TODO: constructors and operations
            for i in range(len(ch)):
                res = self.content[i]
                typ = _plot_atom_type(res)
                res = PlotValue(res)
                if isinstance(ch[i], VariableAtom):
                    bindings[ch[i].get_name()] = G(res, typ)
                elif isinstance(ch[i], ExpressionAtom):
                    bind_add = res.match_(ch[i])
                    if bind_add == []:
                        return []
                    bindings.update(bind_add[0])
        return [] if len(bindings) == 0 else [bindings]


class PatternValue(MatchableObject):

    def match_(self, other):
        if isinstance(other, GroundedAtom):
            other = other.get_object().content
        if not isinstance(other, PatternValue):
            return other.match_(self)
        # TODO: match to patterns
        return []


class PatternOperation(OperationObject):

    def __init__(self, name, op, unwrap=False, rec=False):
        super().__init__(name, op, unwrap)
        self.rec = rec

    def execute(self, *PlotValue, res_typ=AtomType.UNDEFINED):
        if self.rec:
            PlotValue = PlotValue[0].get_children()
            PlotValue = [
                self.execute(arg)[0] if isinstance(
                    arg, ExpressionAtom) else arg
                for arg in PlotValue
            ]
        # If there is a variable or PatternValue in arguments, create PatternValue
        # instead of executing the operation
        for arg in PlotValue:
            if (
                isinstance(arg, GroundedAtom)
                and isinstance(arg.get_object(), PatternValue)
                or isinstance(arg, VariableAtom)
            ):
                return [G(PatternValue([self, PlotValue]))]
        return super().execute(*PlotValue, res_typ=res_typ)


def _plot_atom_type(df):
    return S("PlotValue")


def wrapnpop(func):
    def wrapper(*args):
        a, k = unwrap_args(args)
        res = func(*a, **k)
        plt.show()
        fig = res.get_figure()
        fig.savefig("out.png")
        typ = _plot_atom_type(res)
        return [G(PlotValue(res), typ)]

    return wrapper


@register_atoms
def sns_atoms():

    snsScatterplot = G(
        PatternOperation("sns.scatterplot", wrapnpop(
            sns.scatterplot), unwrap=False)
    )

    return {
        r"sns\.scatterplot": snsScatterplot,
    }
