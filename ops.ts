// Based loosely on AutoGrad's numpy_jvps.py
// tslint:disable-next-line:max-line-length
// https://github.com/HIPS/autograd/blob/e99d1276653a54114aa8835bef8f831c82c8d3e3/autograd/numpy/numpy_jvps.py
// Forward OPs only use NDArrayMath.
// Backwards OPs must be defined in terms of forward OPs in order to support
// higher order gradients.
import { $, allEqual } from "./propel";
import * as backprop from "./backprop";
import { Array1D, Array2D, Array3D, Array4D }
  from "./deeplearnjs/src/math/ndarray";
import { NDArrayMath } from "./deeplearnjs/src/math/math";
import { NDArrayMathCPU } from "./deeplearnjs/src/math/math_cpu";
import { NDArray } from "./deeplearnjs/src/math/ndarray";
import { Shape, Tensor, DLTensor, TensorLike } from "./tensor";
import { assert } from "./util";

const cpuMath: NDArrayMathCPU = new NDArrayMathCPU();

type FWFunc = (math: NDArrayMath, ...args: TensorLike[]) => NDArray;
type BWFunc = (grad: Tensor, ans: Tensor, ...args: TensorLike[]) => Tensor;
type OpFunc = (...args: TensorLike[]) => Tensor;

let nextOpId = 1;

interface OpInfo {
  name: string;
  opFunc: OpFunc;
  fwFunc: FWFunc;
  bwFuncs: BWFunc[];
}

const ops = {}; // name -> OpInfo

function defFW(name: string, fwFunc: FWFunc): OpFunc {
  const opFunc: OpFunc = (...args: TensorLike[]): Tensor => {
    // Gather ids of args that are tensors. null for non-Tensor args.
    const inputIds = args.map((t) => (t as Tensor).id);
    const math = cpuMath; // TODO decide if we should use GPU math here.
    const ndarrayAns = fwFunc(math, ...args);
    const ans = new DLTensor(ndarrayAns);
    backprop.recordOp({
      name,
      oid: nextOpId++,
      inputIds,
      outputIds: [ans.id],
      ans,
      inputs: args,
    });
    return ans;
  };
  ops[name] = {
    name,
    opFunc,
    fwFunc,
    bwFuncs: null,
  };
  return opFunc;
}

export function getBackwardFuncs(name: string): BWFunc[] {
  return ops[name].bwFuncs;
}

function defBW(name: string, ...bwFuncs: Array<null | BWFunc>) {
  ops[name].bwFuncs = bwFuncs.map((f) => {
    if (f == null) {
      return (g, ans, ...args) => ans.zerosLike();
    } else {
      return f;
    }
  });
}

// TODO Identity for now.
function broadcast(x, target) {
  return x;
}

function C(x: TensorLike): NDArray {
  return (Tensor.convert(x) as DLTensor).ndarray;
}

export let mul = defFW("mul", (m, x, y) => m.multiply(C(x), C(y)));
defBW("mul",
  (g, ans, x, y) => mul(g, y),
  (g, ans, x, y) => mul(g, x));

export let exp = defFW("exp", (m, x) => m.exp(C(x)));
defBW("exp", (g, ans, x) => mul(ans, g));

export let neg = defFW("neg", (m, x) => m.neg(C(x)));
defBW("neg", (g, ans, x) => neg(g));

export let add = defFW("add", (m, x, y) => {
  return m.add(C(x), C(y));
});
defBW("add", (g, ans, x, y) => broadcast(g, ans),
  (g, ans, x, y) => broadcast(g, ans));

export let sub = defFW("sub", (m, x, y) => m.sub(C(x), C(y)));
defBW("sub", (g, ans, x, y) => broadcast(g, ans),
  (g, ans, x, y) => broadcast(neg(g), ans));

export let div = defFW("div", (m, x, y) => m.divide(C(x), C(y)));
defBW("div",
  (g, ans, x, y) => div(g, y),
  (g, ans, x, y) => div(neg(mul(g, x)), mul(y, y)));

export let reshape = defFW("reshape", (m, x, newShape: Shape) => {
  return C(x).reshape(newShape);
});
defBW("reshape",
  (g, ans, x, newShape) => reshape(g, Tensor.convert(x).shape),
  null);

function concatFW(m, axis: number, ...tensors: TensorLike[]): NDArray {
  const tensors_ = tensors.map(t => Tensor.convert(t) as DLTensor);
  const ndarrays = tensors_.map(t => t.ndarray);
  const shapes = tensors_.map(t => t.shape);
  assert(allEqual(...shapes), "shapes not all equal");
  const rank = shapes[0].length;
  const r = ndarrays.reduce((a, b) => {
    if (rank == 0) {
      return m.concat1D(a.as1D(), b.as1D());
    } else if (rank == 1) {
      return m.concat1D(a.as1D(), b.as1D());
    } else if (rank == 2) {
      return m.concat2D(a as Array2D, b as Array2D, axis);
    } else if (rank == 3) {
      return m.concat3D(a as Array3D, b as Array3D, axis);
    } else if (rank == 4) {
      return m.concat4D(a as Array4D, b as Array4D, axis);
    } else {
      assert(false, 'Unsupported Tensor rank.');
    }
  });
  return r;
}

export let concat = defFW("concat", concatFW);
defBW("concat",
  null,
  (g, ans, axis, ...tensors: TensorLike[]) => {
    // TODO return g.slice(axis))
    return Tensor.convert(0);
  });
