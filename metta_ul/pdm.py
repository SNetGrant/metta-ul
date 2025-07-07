from hyperon.atoms import (
    OperationObject,
    AtomType,
    GroundedAtom,
    ExpressionAtom,
    VariableAtom,
    ValueAtom,
    S,
    G,
    E,
    Atoms,
    OperationAtom,
)
from hyperon.ext import register_atoms
from .grounding_tools import unwrap_args
from .numme import _np_atom_type, NumpyValue
import pandas as pd
import numpy as np


def parse_value(value):
    """
    Returns:
        - float if the input is a numeric string or number
        - string if the input is a non-numeric string
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value



def import_as_atom(metta):
    def wrapper(*args):
        df = args[0].get_object().value
        aname = args[1].get_name()
        r = []
        for i, row in enumerate(df.itertuples(index=False)):
            a = E(
                S(aname),
                ValueAtom(i),
                E(*[E(S(str(k)), ValueAtom(parse_value(v))) for k, v in row._asdict().items()]),
            )
            metta.space().add_atom(a)
            r.append(a)
        return [Atoms.UNIT]

    return wrapper


def makeLabels(metta):
    def wrapper(data, labels, name):

        r = []
        for i, val in enumerate(data.get_object().value):
            print(val)
            a = E(
                S(name.get_name()),
                ValueAtom(int(val[0])),
                ValueAtom(int(labels.get_object().value[i])),
            )
            metta.space().add_atom(a)
            r.append(a)
        return [Atoms.UNIT]

    return wrapper


class DFOperation(OperationObject):
    def __init__(self):
        super().__init__("make-df", pd.DataFrame, True)

    def execute(self, *args, res_typ=AtomType.UNDEFINED):
        l = list()
        c = set()
        i = list()
        for expr in args[0].get_children():
            for expr_i in expr.get_children():
                if isinstance(expr_i, GroundedAtom):
                    i.append(expr_i.get_object().value)
                else:
                    w = list()
                    for col in expr_i.get_children():
                        if isinstance(col, GroundedAtom):
                            w.append(col.get_object().value)
                        if isinstance(col, ExpressionAtom):
                            cd = col.get_children()
                            c.add(cd[0].get_name())
                            w.append(cd[1].get_object().value)
                    l.append(w)
        df = pd.DataFrame(l, i, list(c))
        return [ValueAtom(df)]


@register_atoms(pass_metta=True)
def pdme_atoms(run_context):

    return {
        r"make-df": G(DFOperation()),
        r"import\.df": OperationAtom(
            "import-df", import_as_atom(run_context), unwrap=False
        ),
        r"cons-labels": OperationAtom(
            "cons-labels", makeLabels(run_context), unwrap=False
        ),
    }
