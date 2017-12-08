// Backprop ops.
// Based loosely on AutoGrad's numpy_jvps.py
// tslint:disable-next-line:max-line-length
// https://github.com/HIPS/autograd/blob/e99d1276653a54114aa8835bef8f831c82c8d3e3/autograd/numpy/numpy_jvps.py
import * as backprop from "./backprop";
import { basicOps } from "./basic";
import { ChainableTensor, convertChainable } from "./chainable_tensor";
import { BasicTensor, TensorLike } from "./types";
import { assert } from "./util";

type FWFunc = (...args) => BasicTensor;
type BWFunc = (grad: ChainableTensor, ...savedArgs) => ChainableTensor;
type OpFunc = (...args) => ChainableTensor;

let nextOpId = 1;

interface OpInfo {
  name: string;
  opFunc: OpFunc;
  bwFuncs: BWFunc[];
}

const ops = {}; // name -> OpInfo

// Define a forward op.
function defFW(name: string, fwFunc: FWFunc): OpFunc {
  const opFunc: OpFunc = (...args): ChainableTensor => {
    // We no longer automatically convert the op args to tensors.
    // It's up to the caller.

    const cTensors: ChainableTensor[] = [];

    // Gather ids of args that are tensors. null for non-Tensor args.
    const inputIds = args.map((t) => {
      if ((t as ChainableTensor).id) {
        cTensors.push(t);
        return (t as ChainableTensor).id;
      } else {
        return null;
      }
    });

    // Convert any ChainableTensor args to basic ones.
    const bargs = args.map((t) => {
      if ((t as ChainableTensor).basic) {
        return (t as ChainableTensor).basic;
      } else {
        return t;
      }
    });

    // Call the forward function, and wrap the resulting BasicTensor in a
    // ChainableTensor.
    const basicAnswer: BasicTensor = fwFunc(...bargs);
    const ans = new ChainableTensor(basicAnswer);
    cTensors.push(ans);

    const savedForBackward =
      convertSavedBasicsToChainables(globalSavedForBackward, cTensors);
    globalSavedForBackward = null;

    backprop.recordOp({
      inputIds,
      name,
      oid: nextOpId++,
      outputIds: [ans.id],
      savedForBackward,
    });
    return ans;
  };
  ops[name] = {
    bwFuncs: null,
    name,
    opFunc,
  };
  return opFunc;
}

function convertSavedBasicsToChainables(saved: any[], cTensors:
                                        ChainableTensor[]) {
  if (!saved) return null;
  return saved.map((t) => {
    if ((t as BasicTensor).getData) {
      const b = t as BasicTensor;
      for (const ct of cTensors) {
        if (ct.basic === b) return ct;
      }
      throw new Error("Couldn't find corresponding ChainableTensor.");
    } else {
      // Not a tensor. Just pass it through.
      return t;
    }
  });
}

function defBW(name: string, ...bwFuncs: Array<null | BWFunc>) {
  ops[name].bwFuncs = bwFuncs.map((f) => {
    if (f == null) {
      return (g, ...args) => g.zerosLike();
    } else {
      return f;
    }
  });
}

let globalSavedForBackward = null;
function saveForBackward(...args) {
  // JavaScript is single threaded. Therefore we don't to worry about multiple
  // call stacks.
  assert(globalSavedForBackward === null);
  globalSavedForBackward = args;
}

export function getBackwardFuncs(name: string): BWFunc[] {
  return ops[name].bwFuncs;
}

export const add = defFW("add", (x, y) => basicOps.add(x, y));
defBW("add",
  (g) => g,
  (g) => g);

export const sub = defFW("sub", (x, y) => basicOps.sub(x, y));
defBW("sub",
  (g) => g,
  (g) => neg(g));

export const mul = defFW("mul", (x, y) => {
  saveForBackward(x, y);
  return basicOps.mul(x, y);
});
defBW("mul",
  (g, x, y) => mul(g, y),
  (g, x, y) => mul(g, x));

export const div = defFW("div", (x, y) => {
  saveForBackward(x, y);
  return basicOps.div(x, y);
});
defBW("div",
  (g, x, y) => div(g, y),
  (g, x, y) => div(neg(mul(g, x)), mul(y, y)));

export const matmul = defFW("matmul",
  (a, b, transposeA = false, transposeB = false) => {
    saveForBackward(a, b, transposeA, transposeB);
    return basicOps.matmul(a, b, transposeA, transposeB);
  });
// y = a * b
// da = dy * bT
// db = aT * dy
defBW("matmul",
  (g, a, b, tA, tB) => matmul(g, b, tA, !tB),
  (g, a, b, tA, tB) => matmul(a, g, !tA, tB));

export const neg = defFW("neg", (x) => basicOps.neg(x));
defBW("neg", (g) => neg(g));

export const exp = defFW("exp", (x) => {
  const ans = basicOps.exp(x);
  saveForBackward(ans);
  return ans;
});
defBW("exp", (g, ans) => mul(ans, g));

export let log = defFW("log", (x) => {
  saveForBackward(x);
  return basicOps.log(x);
});
defBW("log", (g, x) => div(g, x));

export const square = defFW("square", (x) => {
  saveForBackward(x);
  return basicOps.square(x);
});
defBW("square", (g, x) => mul(g, mul(x, convertChainable(2, x.dtype))));

export const sinh = defFW("sinh", (x) => {
  saveForBackward(x);
  return basicOps.sinh(x);
});
defBW("sinh", (g, x) => mul(g, cosh(x)));

export let cosh = defFW("cosh", (x) => {
  saveForBackward(x);
  return basicOps.cosh(x);
});
defBW("cosh", (g, x) => mul(g, sinh(x)));

export let tanh = defFW("tanh", (x) => {
  saveForBackward(x);
  return basicOps.tanh(x);
});
defBW("tanh", (g, x) => div(g, square(cosh(x))));

export let transpose = defFW("transpose", (x, perm) => {
  saveForBackward(perm);
  return basicOps.transpose(x, perm);
});
defBW("transpose", (g, perm) => transpose(g, perm));

export let reverse = defFW("reverse", (x, dims) => {
  saveForBackward(dims);
  return basicOps.reverse(x, dims);
});
defBW("reverse", (g, dims) => reverse(g, dims));

export let reduceSum = defFW("reduceSum", (x, axes, keepDims) => {
  saveForBackward(x);
  return basicOps.reduceSum(x, axes, keepDims);
});
defBW("reduceSum", (g, x) => mul(g, x.onesLike()));

export let reduceMax = defFW("reduceMax", (x, axes, keepDims) => {
  return basicOps.reduceMax(x, axes, keepDims);
});
defBW("reduceMax", (g, axes, keepDims) => {
  throw new Error("Not Implemented.");
});

export let equal = defFW("equal", (x, y) => {
  return basicOps.equal(x, y);
});
defBW("equal", null, null); // equal is not differentiable.

export let reshape = defFW("reshape", (x, newShape) => {
  saveForBackward(x.shape);
  return basicOps.reshape(x, newShape);
});
defBW("reshape", (g, origShape) => reshape(g, origShape));

export let reduceLogSumExp = defFW("reduceLogSumExp",
  (x, axes: number[], keepDims = false) => {
    const m = basicOps.reduceMax(x, axes, true);
    const e = basicOps.exp(basicOps.sub(x, m));
    const s = basicOps.reduceSum(e, axes, true);
    const sLog = basicOps.log(s);
    const ans =  basicOps.add(m, sLog);
    saveForBackward(ans, x);
    return ans;
  });
defBW("reduceLogSumExp", (g, ans, x) => {
  return mul(g, exp(sub(x, ans)));
});

export const softmax = defFW("softmax", (x) => {
  assert(x.shape.length === 2);
  const ans = basicOps.softmax(x);
  saveForBackward(ans);
  return ans;
});
defBW("softmax", (g, ans) => {
  return g.sub(g.mul(ans).reduceSum([1]).reshape([-1, 1])).mul(ans);
});

export const logSoftmax = defFW("logSoftmax", (x) => {
  assert(x.shape.length === 2);
  const ans = basicOps.logSoftmax(x);
  saveForBackward(ans);
  return ans;
});
defBW("logSoftmax", (g, ans) => {
  const softmax = ans.exp();
  return g.sub(g.reduceSum([1], true).mul(softmax));
});
