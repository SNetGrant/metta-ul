from hyperon.atoms import *
from hyperon.ext import register_atoms

import numpy as np


class NumpyValue(MatchableObject):

    def __eq__(self, other):
        return (
            isinstance(other, NumpyValue)
            and (self.content.shape == other.content.shape)
            and (self.content == other.content).all()
        )

    def match_(self, other):
        sh = self.content.shape
        bindings = {}
        if isinstance(other, GroundedAtom):
            other = other.get_object()
        # Match by equality with another NumpyValue
        if isinstance(other, NumpyValue):
            return [{}] if other == self else []
        # if isinstance(other, PatternValue):
        #     other = other.to_expr()
        if isinstance(other, ExpressionAtom):
            ch = other.get_children()
            # TODO: constructors and operations
            if len(ch) != sh[0]:
                return []
            for i in range(len(ch)):
                res = self.content[i]
                typ = _np_atom_type(res)
                res = NumpyValue(res)
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

    def execute(self, *args, res_typ=AtomType.UNDEFINED):
        if self.rec:
            args = args[0].get_children()
            args = [
                self.execute(arg)[0] if isinstance(
                    arg, ExpressionAtom) else arg
                for arg in args
            ]
        # If there is a variable or PatternValue in arguments, create PatternValue
        # instead of executing the operation
        for arg in args:
            if (
                isinstance(arg, GroundedAtom)
                and isinstance(arg.get_object(), PatternValue)
                or isinstance(arg, VariableAtom)
            ):
                return [G(PatternValue([self, args]))]
        return super().execute(*args, res_typ=res_typ)


def _np_atom_type(npobj):
    if not isinstance(npobj, np.ndarray):
        return AtomType.UNDEFINED
    return E(S("NPArray"), E(*[ValueAtom(s, "Number") for s in npobj.shape]))


def wrapnpop(func):
    def wrapper(*args):
        a = [arg.get_object().value if isinstance(arg, GroundedAtom)
             else arg.get_name() for arg in args]
        res = func(*a)
        typ = _np_atom_type(res)
        return [G(NumpyValue(res), typ)]

    return wrapper


def _slice(*args):
    if args[0] is None:
        return None
    arr = args[0]
    slice_str = parse_to_slice(args[1])
    return arr[slice_str]


def parse_to_slice(input_str):
    """
    Parses a string representing numpy indexing and returns the corresponding NumPy array slice.
    Supports multi-dimensional indexing with missing values like "[:, :2]".
    """
    # Remove brackets and spaces
    input_str = input_str.strip("[]").replace(" ", "")

    # Split by commas to handle multiple dimensions
    parts = input_str.split(",")
    result = []

    for part in parts:
        if ":" in part:
            # Handle cases like ":", ":2", "1:", "1:3:2"
            slice_parts = part.split(":")
            # Convert non-empty values to int
            slice_values = [int(x) if x else None for x in slice_parts]
            result.append(slice(*slice_values))  # Convert to slice object
        else:
            # Handle single integer indices
            result.append(int(part))

    # Convert result to tuple if multi-dimensional
    return tuple(result) if len(result) > 1 else result[0]


@ register_atoms
def numme_atoms():

    # FIXME: we don't add types for operations, because numpy operations types
    # are too loose
    nmVectorAtom = G(
        PatternOperation(
            "np.vector", wrapnpop(lambda *args: np.array(args)), unwrap=False
        )
    )
    nmArrayAtom = G(
        PatternOperation(
            "np.array", wrapnpop(lambda *args: np.array(args, dtype='float')), unwrap=False, rec=True
        )
    )
    nmAddAtom = G(PatternOperation("np.add", wrapnpop(np.add), unwrap=False))
    nmSubAtom = G(PatternOperation(
        "np.sub", wrapnpop(np.subtract), unwrap=False))
    nmMulAtom = G(PatternOperation(
        "np.mul", wrapnpop(np.multiply), unwrap=False))
    nmDivAtom = G(PatternOperation(
        "np.div", wrapnpop(np.divide), unwrap=False))
    nmMMulAtom = G(PatternOperation(
        "np.matmul", wrapnpop(np.matmul), unwrap=False))
    nmArgmin = G(PatternOperation(
        "np.argmin", wrapnpop(np.argmin), unwrap=False))
    nmTranspose = G(
        PatternOperation("np.transpose", wrapnpop(np.transpose), unwrap=False)
    )
    nmNorm = G(
        PatternOperation("np.linalg.norm", wrapnpop(
            np.linalg.norm), unwrap=False)
    )
    nmSum = G(PatternOperation("np.sum", wrapnpop(np.sum), unwrap=False))
    nmOneHot = G(
        PatternOperation(
            "np.one_hot", wrapnpop(lambda labels, k: np.eye(k)[labels]), unwrap=False
        )
    )
    nmExpandDims = G(
        PatternOperation("np.expand_dims", wrapnpop(
            np.expand_dims), unwrap=False)
    )
    nmChoose = G(
        PatternOperation(
            "np.choose",
            wrapnpop(lambda x, k: x[np.random.choice(
                x.shape[0], k, replace=False)]),
            unwrap=False,
        )
    )
    nmLogDet = G(
        PatternOperation(
            "np.linalg.slogabsdet",
            wrapnpop(lambda x: np.linalg.slogdet(x).logabsdet),
            unwrap=False,
        )
    )
    nmInv = G(PatternOperation("np.linalg.inv",
                               wrapnpop(np.linalg.inv), unwrap=False))
    nmEinsum = G(PatternOperation(
        "np.einsum", wrapnpop(np.einsum), unwrap=False))
    nmExp = G(PatternOperation("np.exp", wrapnpop(np.exp), unwrap=False))
    nmLog = G(PatternOperation("np.log", wrapnpop(np.log), unwrap=False))
    nmCov = G(PatternOperation("np.cov", wrapnpop(np.cov), unwrap=False))
    nmRepeat = G(PatternOperation(
        "np.repeat", wrapnpop(np.repeat), unwrap=False))
    nmEye = G(PatternOperation("np.eye", wrapnpop(np.eye), unwrap=False))
    nmOnes = G(PatternOperation("np.ones", wrapnpop(np.ones), unwrap=False))
    nmPower = G(PatternOperation("np.power", wrapnpop(np.power), unwrap=False))
    nmRandomRand = G(
        PatternOperation("np.random.rand", wrapnpop(
            np.random.rand), unwrap=False)
    )
    nmDiag = G(PatternOperation("np.diag", wrapnpop(np.diag), unwrap=False))
    nmSqrt = G(PatternOperation("np.sqrt", wrapnpop(np.sqrt), unwrap=False))
    nmArgsort = G(PatternOperation(
        "np.argsort", wrapnpop(np.argsort), unwrap=False))
    nmArange = G(PatternOperation(
        "np.arange", wrapnpop(np.arange), unwrap=False))
    nmTake = G(PatternOperation("np.take", wrapnpop(np.take), unwrap=False))

    nmSlice = G(PatternOperation("np.slice", wrapnpop(_slice), unwrap=False))
    nmArgmax = G(PatternOperation(
        "np.argmax", wrapnpop(np.argmax), unwrap=False))
    nmIx_ = G(PatternOperation("np.ix_", wrapnpop(
        np.ix_), unwrap=False, rec=False))
    nmMin = G(PatternOperation("np.min", wrapnpop(np.min), unwrap=False))
    nmMax = G(PatternOperation("np.max", wrapnpop(np.max), unwrap=False))
    nmMean = G(PatternOperation("np.mean", wrapnpop(np.mean), unwrap=False))
    nmSqueeze = G(PatternOperation(
        "np.squeeze", wrapnpop(np.squeeze), unwrap=False))
    nmRandomSeed = G(PatternOperation("np.random.seed",
                     wrapnpop(np.random.seed), unwrap=False))
    nmShape = G(PatternOperation("np.shape", wrapnpop(
        lambda _x, _i: _x.shape[_i]), unwrap=False))

    return {
        "np.vector": nmVectorAtom,
        "np.array": nmArrayAtom,
        "np.add": nmAddAtom,
        "np.sub": nmSubAtom,
        "np.mul": nmMulAtom,
        "np.matmul": nmMMulAtom,
        "np.div": nmDivAtom,
        "np.argmin": nmArgmin,
        "np.transpose": nmTranspose,
        "np.linalg.norm": nmNorm,
        "np.sum": nmSum,
        "np.one_hot": nmOneHot,
        "np.expand_dims": nmExpandDims,
        "np.choose": nmChoose,
        "np.linalg.slogabsdet": nmLogDet,
        "np.linalg.inv": nmInv,
        "np.einsum": nmEinsum,
        "np.exp": nmExp,
        "np.log": nmLog,
        "np.cov": nmCov,
        "np.repeat": nmRepeat,
        "np.eye": nmEye,
        "np.ones": nmOnes,
        "np.power": nmPower,
        "np.random.rand": nmRandomRand,
        "np.diag": nmDiag,
        "np.sqrt": nmSqrt,
        "np.argsort": nmArgsort,
        "np.arange": nmArange,
        "np.take": nmTake,
        "np.argmax": nmArgmax,
        "np.slice": nmSlice,
        "np.ix_": nmIx_,
        "np.mean": nmMean,
        "np.min": nmMin,
        "np.max": nmMax,
        "np.squeeze": nmSqueeze,
        "np.random.seed": nmRandomSeed,
        "np.shape": nmShape,
    }
