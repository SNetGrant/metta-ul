from hyperon.atoms import *
from hyperon.ext import register_atoms
import numpy as np

from .array_like_tools import parse_to_slice


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
                self.execute(arg)[0] if isinstance(arg, ExpressionAtom) else arg
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


def _np_atom_value(npobj, typ):
    return G(NumpyValue(npobj), typ)


def wrapnpop(func):
    def wrapper(*args):
        a = [
            arg.get_object().value if isinstance(arg, GroundedAtom) else arg.get_name()
            for arg in args
        ]
        res = func(*a)
        typ = _np_atom_type(res)
        return [_np_atom_value(res, typ)]

    return wrapper


def _slice(*args):
    if args[0] is None:
        return None
    arr = args[0]
    slice_str = parse_to_slice(args[1])
    return arr[slice_str]


def _assert_isNone(actual: GroundedAtom):
    try:
        if actual.get_object().value is None:
            return [E()]
        return [
            E(
                S("Error"),
                E(S("py.assertIsNone"), actual, None),
                S(
                    f"\nExpected None\nGot: {actual}\nMissed results: None\nExcessive results: {actual}"
                ),
            )
        ]
    except Exception as e:
        return [E(S("Error"), E(S("py.assertIsNone"), actual, None), S(str(e)))]


def _assert_allclose(
    actual: GroundedAtom,
    expected: GroundedAtom,
    rtol=ValueAtom(1e-05),
    atol=ValueAtom(1e-08),
    equal_nan=ValueAtom(False),
):
    try:
        if np.allclose(
            actual.get_object().value,
            expected.get_object().value,
            rtol.get_object().value,
            atol.get_object().value,
            equal_nan.get_object().value,
        ):
            return [E()]
        return [
            E(
                S("Error"),
                E(S("np.assertAllEqual"), actual, expected),
                S(
                    f"\nExpected [{expected}]\nGot: [{actual}]\nMissed results: {expected}\nExcessive results: {actual}"
                ),
            )
        ]
    except Exception as e:
        return [E(S("Error"), E(S("np.assertAllEqual"), actual, expected), S(str(e)))]


@register_atoms
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
            "np.array", wrapnpop(lambda *args: np.array(args)), unwrap=False, rec=True
        )
    )
    nmAddAtom = G(PatternOperation("np.add", wrapnpop(np.add), unwrap=False))
    nmSubAtom = G(PatternOperation("np.sub", wrapnpop(np.subtract), unwrap=False))
    nmMulAtom = G(PatternOperation("np.mul", wrapnpop(np.multiply), unwrap=False))
    nmDivAtom = G(PatternOperation("np.div", wrapnpop(np.divide), unwrap=False))
    nmMMulAtom = G(PatternOperation("np.matmul", wrapnpop(np.matmul), unwrap=False))
    nmArgmin = G(PatternOperation("np.argmin", wrapnpop(np.argmin), unwrap=False))
    nmTranspose = G(
        PatternOperation("np.transpose", wrapnpop(np.transpose), unwrap=False)
    )
    nmNorm = G(
        PatternOperation("np.linalg.norm", wrapnpop(np.linalg.norm), unwrap=False)
    )
    nmSum = G(PatternOperation("np.sum", wrapnpop(np.sum), unwrap=False))
    nmOneHot = G(
        PatternOperation(
            "np.one_hot", wrapnpop(lambda labels, k: np.eye(k)[labels]), unwrap=False
        )
    )
    nmExpandDims = G(
        PatternOperation("np.expand_dims", wrapnpop(np.expand_dims), unwrap=False)
    )
    nmChoose = G(
        PatternOperation(
            "np.choose",
            wrapnpop(lambda x, k: x[np.random.choice(x.shape[0], k, replace=False)]),
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
    nmEigh = G(
        PatternOperation("np.linalg.eigh", wrapnpop(np.linalg.eigh), unwrap=False)
    )

    nmInv = G(PatternOperation("np.linalg.inv", wrapnpop(np.linalg.inv), unwrap=False))
    nmEinsum = G(PatternOperation("np.einsum", wrapnpop(np.einsum), unwrap=False))
    nmExp = G(PatternOperation("np.exp", wrapnpop(np.exp), unwrap=False))
    nmLog = G(PatternOperation("np.log", wrapnpop(np.log), unwrap=False))
    nmCov = G(PatternOperation("np.cov", wrapnpop(np.cov), unwrap=False))
    nmRepeat = G(PatternOperation("np.repeat", wrapnpop(np.repeat), unwrap=False))
    nmEye = G(PatternOperation("np.eye", wrapnpop(np.eye), unwrap=False))
    nmOnes = G(PatternOperation("np.ones", wrapnpop(np.ones), unwrap=False))
    nmZeros = G(PatternOperation("np.zeros", wrapnpop(np.zeros), unwrap=False))
    nmPower = G(PatternOperation("np.power", wrapnpop(np.power), unwrap=False))
    nmRandomRand = G(
        PatternOperation("np.random.rand", wrapnpop(np.random.rand), unwrap=False)
    )
    nmDiag = G(PatternOperation("np.diag", wrapnpop(np.diag), unwrap=False))
    nmSqrt = G(PatternOperation("np.sqrt", wrapnpop(np.sqrt), unwrap=False))
    nmArgsort = G(PatternOperation("np.argsort", wrapnpop(np.argsort), unwrap=False))
    nmArange = G(PatternOperation("np.arange", wrapnpop(np.arange), unwrap=False))
    nmTake = G(PatternOperation("np.take", wrapnpop(np.take), unwrap=False))

    nmSlice = G(PatternOperation("np.slice", wrapnpop(_slice), unwrap=False))
    nmArgmax = G(PatternOperation("np.argmax", wrapnpop(np.argmax), unwrap=False))
    nmMaximum = G(PatternOperation("np.maximum", wrapnpop(np.maximum), unwrap=False))
    nmIx_ = G(PatternOperation("np.ix_", wrapnpop(np.ix_), unwrap=False, rec=False))
    nmMin = G(PatternOperation("np.min", wrapnpop(np.min), unwrap=False))
    nmMax = G(PatternOperation("np.max", wrapnpop(np.max), unwrap=False))
    nmMean = G(PatternOperation("np.mean", wrapnpop(np.mean), unwrap=False))
    nmSqueeze = G(PatternOperation("np.squeeze", wrapnpop(np.squeeze), unwrap=False))
    nmRandomSeed = G(
        PatternOperation("np.random.seed", wrapnpop(np.random.seed), unwrap=False)
    )
    nmShape = G(
        PatternOperation(
            "np.shape", wrapnpop(lambda _x, _i: _x.shape[_i]), unwrap=False
        )
    )
    nmWhere = G(PatternOperation("np.where", wrapnpop(np.where), unwrap=False))
    nmIsin = G(PatternOperation("np.isin", wrapnpop(np.isin), unwrap=False))
    nmEqual = G(PatternOperation("np.equal", wrapnpop(np.equal), unwrap=False))
    nmAppend = G(PatternOperation("np.append", wrapnpop(np.append)))
    nmPut = G(PatternOperation("np.put", wrapnpop(np.put), unwrap=False))
    nmConcat = G(PatternOperation("np.concat", wrapnpop(np.concat), unwrap=False))
    nmAllClose = G(PatternOperation("np.allclose", wrapnpop(np.allclose), unwrap=False))
    nmAssertAllClose = G(
        PatternOperation("np.assertAllClose", _assert_allclose, unwrap=False)
    )
    nmAbs = G(PatternOperation("np.abs", wrapnpop(np.abs), unwrap=False))
    nmIsnan = G(PatternOperation("np.isnan", wrapnpop(np.isnan), unwrap=False))
    nmUnique = G(PatternOperation("np.unique", wrapnpop(np.unique), unwrap=False))
    nmAssertIsNone = G(
        PatternOperation("py.assertIsNone", _assert_isNone, unwrap=False)
    )
    nmSort = G(PatternOperation("np.sort", wrapnpop(np.sort), unwrap=False))
    nmReshape = G(PatternOperation("np.reshape", wrapnpop(np.reshape), unwrap=False))

    pyNone = ValueAtom(None)
    pyTrue = ValueAtom(True)
    pyPINF = ValueAtom(float("inf"))
    nmTrue = G(PatternOperation("np.True_", wrapnpop(np.True_), unwrap=False))

    atomPi = ValueAtom(np.pi)

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
        "np.linalg.eigh": nmEigh,
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
        "np.where": nmWhere,
        "np.equal": nmEqual,
        "np.append": nmAppend,
        "np.put": nmPut,
        "np.concat": nmConcat,
        "np.allclose": nmAllClose,
        "np.assertAllClose": nmAssertAllClose,
        "py.none": pyNone,
        "py.true": pyTrue,
        "py.pinf": pyPINF,
        "math.pi": atomPi,
        "np.abs": nmAbs,
        "np.isnan": nmIsnan,
        "np.unique": nmUnique,
        "np.True_": nmTrue,
        "py.assertIsNone": nmAssertIsNone,
        "np.sort": nmSort,
        "np.reshape": nmReshape,
        "np.zeros": nmZeros,
        "np.isin": nmIsin,
        "np.maximum": nmMaximum
    }