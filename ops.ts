// Based loosely on AutoGrad's numpy_jvps.py
// https://github.com/HIPS/autograd/blob/e99d1276653a54114aa8835bef8f831c82c8d3e3/autograd/numpy/numpy_jvps.py
// Forward OPs only use NDArrayMath.
// Backwards OPs must be defined in terms of forward OPs in order to support
// higher order gradients.
import { NDArrayMath } from './deeplearnjs/src/math/math';
import { NDArray } from './deeplearnjs/src/math/ndarray';
import { NDArrayMathCPU } from './deeplearnjs/src/math/math_cpu';
import { Tensor, TensorLike } from "./tensor";
import * as backprop from "./backprop";

let cpuMath: NDArrayMathCPU = new NDArrayMathCPU();

type FWFunc = (math: NDArrayMath, ...args: NDArray[]) => NDArray;
type BWFunc = (grad: Tensor, ans: Tensor, ...args: Tensor[]) => Tensor;
type OpFunc = (...args: TensorLike[]) => Tensor;

let nextOpId: number = 1;

interface OpInfo {
  name: string;
  fwFunc: FWFunc;
  bwFuncs: BWFunc[];
}

let ops = {}; // name -> OpInfo

function defFW(name: string, fwFunc: FWFunc): OpFunc {
  let opFunc: OpFunc = (...args: TensorLike[]): Tensor => {
    let tArgs = args.map(a => Tensor.convert(a));
    let inputIds = tArgs.map(t => t.id);
    let ndArgs = tArgs.map(t => t.ndarray);
    let math = cpuMath; // TODO decide if we should use GPU math here.
    let ndarrayAns = fwFunc(math, ...ndArgs);
    let ans = new Tensor(ndarrayAns);
    backprop.recordOp({
      name: name,
      oid: nextOpId++,
      inputIds: inputIds,
      outputIds: [ans.id],
      ans: ans,
      inputs: tArgs,
    });
    return ans;
  };

  ops[name] = {
    name: name,
    fwFunc: fwFunc,
    bwFuncs: null
  };

  return opFunc;
}

export function getBackwardFuncs(name: string): BWFunc[] {
  return ops[name].bwFuncs;
}

function defBW(name: string, ...bwFuncs: BWFunc[]) {
  ops[name].bwFuncs = bwFuncs;
}

// TODO Identity for now.
function broadcast(x, target) {
  return x;
}

export let mul = defFW("mul", (m, x, y) => m.multiply(x, y));
defBW("mul",
  (g, ans, x, y) => mul(g, y),
  (g, ans, x, y) => mul(g, x))

export let exp = defFW("exp", (m, x) => m.exp(x));
defBW("exp", (g, ans, x) => mul(ans, g));

export let neg = defFW("neg", (m, x) => m.neg(x));
defBW("neg", (g, ans, x) => neg(g));

export let add = defFW("add", (m, x, y) => m.add(x, y));
defBW("add", (g, ans, x, y) => broadcast(g, ans),
  (g, ans, x, y) => broadcast(g, ans));

export let sub = defFW("sub", (m, x, y) => m.sub(x, y));
defBW("sub", (g, ans, x, y) => broadcast(g, ans),
  (g, ans, x, y) => broadcast(neg(g), ans));

export let div = defFW("div", (m, x, y) => m.divide(x, y));
defBW("div",
  (g, ans, x, y) => div(g, y),
  (g, ans, x, y) => div(neg(mul(g, x)), mul(y, y)));
