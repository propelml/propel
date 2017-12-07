// Backprop ops.
// Based loosely on AutoGrad's numpy_jvps.py
// tslint:disable-next-line:max-line-length
// https://github.com/HIPS/autograd/blob/e99d1276653a54114aa8835bef8f831c82c8d3e3/autograd/numpy/numpy_jvps.py
import * as backprop from "./backprop";
import { basicOps } from "./basic";
import { ChainableTensor, convertChainable } from "./chainable_tensor";
import { BasicTensor, TensorLike } from "./types";

type FWFunc = (...args: BasicTensor[]) => BasicTensor;
type BWFunc = (grad: ChainableTensor, ans: ChainableTensor, ...args:
  TensorLike[]) => ChainableTensor;
type OpFunc = (...args: TensorLike[]) => ChainableTensor;

let nextOpId = 1;

interface OpInfo {
  name: string;
  opFunc: OpFunc;
  fwFunc: FWFunc;
  bwFuncs: BWFunc[];
}

const ops = {}; // name -> OpInfo

// Define a forward op.
function defFW(name: string, fwFunc: FWFunc): OpFunc {
  const opFunc: OpFunc = (...args: TensorLike[]): ChainableTensor => {
    const cargs = args.map((a) => convertChainable(a));
    // Gather ids of args that are tensors. null for non-Tensor args.
    const inputIds = cargs.map((t) => t.id);
    const bargs = cargs.map((t) => t.basic);
    const basicAnswer = fwFunc(...bargs);
    const ans = new ChainableTensor(basicAnswer);
    backprop.recordOp({
      ans,
      inputIds,
      inputs: args,
      name,
      oid: nextOpId++,
      outputIds: [ans.id],
    });
    return ans;
  };
  ops[name] = {
    bwFuncs: null,
    fwFunc,
    name,
    opFunc,
  };
  return opFunc;
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

export function getBackwardFuncs(name: string): BWFunc[] {
  return ops[name].bwFuncs;
}

// TODO Identity for now.
function broadcast(x, target) {
  return x;
}

export let add = defFW("add", ( x, y) => basicOps.add(x, y));
defBW("add",
  (g, ans, x, y) => broadcast(g, ans),
  (g, ans, x, y) => broadcast(g, ans));

export let sub = defFW("sub", (x, y) => basicOps.sub(x, y));
defBW("sub",
  (g, ans, x, y) => broadcast(g, ans),
  (g, ans, x, y) => broadcast(neg(g), ans));

export let mul = defFW("mul", (x, y) => basicOps.mul(x, y));
defBW("mul",
  (g, ans, x, y) => mul(g, y),
  (g, ans, x, y) => mul(g, x));

export let div = defFW("div", (x, y) => basicOps.div(x, y));
defBW("div",
  (g, ans, x, y) => div(g, y),
  (g, ans, x, y) => div(neg(mul(g, x)), mul(y, y)));

export let neg = defFW("neg", (x) => basicOps.neg(x));
defBW("neg", (g, ans, x) => neg(g));

export let exp = defFW("exp", (x) => basicOps.exp(x));
defBW("exp", (g, ans, x) => mul(ans, g));

export let square = defFW("square", (x) => basicOps.square(x));
defBW("square", (g, ans, x) => mul(g, mul(2, x)));

export let sinh = defFW("sinh", (x) => basicOps.sinh(x));
defBW("sinh", (g, ans, x) => mul(g, cosh(x)));

export let cosh = defFW("cosh", (x) => basicOps.cosh(x));
defBW("cosh", (g, ans, x) => mul(g, sinh(x)));

export let tanh = defFW("tanh", (x) => basicOps.tanh(x));
defBW("tanh", (g, ans, x) => div(g, square(cosh(x))));

export let transpose = defFW("transpose", (x, perm) => {
  return basicOps.transpose(x, perm);
});
defBW("transpose", (g, ans, x, perm) => {
  return transpose(g, convertChainable(perm, "int32"));
});
