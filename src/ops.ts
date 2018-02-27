/*!
   Copyright 2018 Propel http://propel.site/.  All rights reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */
// Backprop ops.
// Based loosely on AutoGrad's numpy_jvps.py
// tslint:disable-next-line:max-line-length
// https://github.com/HIPS/autograd/blob/e99d1276653a54114aa8835bef8f831c82c8d3e3/autograd/numpy/numpy_jvps.py
import { bo } from "./backend";
import * as backprop from "./backprop";
import { convert, Tensor } from "./tensor";
import * as types from "./types";
import { assert, bcastGradientArgs, shapesEqual } from "./util";

// FWFunc defines a "primative" op (using autograd nomenclature). It should
// never use Tensors, only BasicTensors. These forward pass functions
// are defined in ops.ts.
type FWFunc = (...args) => types.BasicTensor;

// BWFunc is a backwards pass function which receives the gradient and any
// objects passed to saveForBackward(). Backwards pass functions are defined
// alongside their foward pass counterparts in ops.ts. Unlike FWFunc, BWFunc
// should use Tensors.
export type BWFunc = (grad: Tensor, ...savedArgs) => Tensor;

// OpFunc is returned from defFW and is what is the external interface to
// backprop ops. These are called a lot in api.ts and tensor.ts
// which together define the public API. OpFuncs might even be exposed directly
// to users. An OpFunc will record a TapeEntry when it is called.
type OpFunc = (...args) => Tensor;

let nextOpId = 1;

interface OpInfo {
  name: string;
  opFunc: OpFunc;
  bwFuncs: BWFunc[];
}

const ops: { [name: string]: OpInfo } = {};

// Define a forward op.
function defFW(name: string, fwFunc: FWFunc): OpFunc {
  const opFunc: OpFunc = (...args): Tensor => {
    // We no longer automatically convert the op args to tensors.
    // It's up to the caller.

    const cTensors: Tensor[] = [];

    // Gather ids of args that are tensors. null for non-Tensor args.
    const inputIds = args.map((t) => {
      if ((t as Tensor).id) {
        cTensors.push(t);
        return (t as Tensor).id;
      } else {
        return null;
      }
    });

    // An array of tuples [shape, dtype] for each argument.
    // Non-tensor arguments are null.
    const inputShapeDTypes: types.ShapeDTypeList = args.map((t) => {
      if ((t as Tensor).shape) {
        const ct = t as Tensor;
        const st: types.ShapeDType = [ct.shape, ct.dtype];
        return st;
      } else {
        return null;
      }
    });

    // Convert any Tensor args to basic ones.
    const bargs = args.map((t) => {
      if ((t as Tensor).basic) {
        return (t as Tensor).basic;
      } else {
        return t;
      }
    });

    // Call the forward function, and wrap the resulting BasicTensor in a
    // Tensor.
    const basicAnswer: types.BasicTensor = fwFunc(...bargs);
    const ans = new Tensor(basicAnswer);
    cTensors.push(ans);

    const savedForBackward =
      convertSavedBasicsTos(globalSavedForBackward, cTensors);
    globalSavedForBackward = null;

    backprop.recordOp({
      name,
      oid: nextOpId++,
      inputIds,
      inputShapeDTypes,
      outputIds: [ans.id],
      savedForBackward,
    });
    return ans;
  };
  ops[name] = {
    name,
    opFunc,
    bwFuncs: null,
  };
  return opFunc;
}

function convertSavedBasicsTos(saved: any[], cTensors: Tensor[]) {
  if (!saved) return null;
  return saved.map((t) => {
    if ((t as types.BasicTensor).dataSync) {
      const b = t as types.BasicTensor;
      for (const ct of cTensors) {
        if (ct.basic === b) return ct;
      }
      throw new Error("Couldn't find corresponding Tensor.");
    } else {
      // Not a tensor. Just pass it through.
      return t;
    }
  });
}

function defBW(name: string, ...bwFuncs: Array<null | BWFunc>) {
  ops[name].bwFuncs = bwFuncs;
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

export function ones(shape: types.Shape, opts: types.TensorOpts) {
  return fill(convert(1, opts), shape);
}

export function zeros(shape: types.Shape, opts: types.TensorOpts) {
  return fill(convert(0, opts), shape);
}

function addGrad(firstArg: boolean) {
  return (g: Tensor, sx: types.Shape, sy: types.Shape) => {
    // If sx and sy are the same (no broadcasting) just return g.
    if (shapesEqual(sx, sy)) return g;
    // Broadcast.
    const [rx, ry] = bcastGradientArgs(sx, sy);
    if (firstArg) {
      return g.reduceSum(rx).reshape(sx);
    } else {
      return g.reduceSum(ry).reshape(sy);
    }
  };
}

export const add = defFW("add", (x, y) => {
  saveForBackward(x.shape, y.shape);
  return bo.add(x, y);
});
defBW("add",
  (g, sx, sy) => addGrad(true)(g, sx, sy),
  (g, sx, sy) => addGrad(false)(g, sx, sy));

export const sub = defFW("sub", (x, y) => {
  saveForBackward(x.shape, y.shape);
  return bo.sub(x, y);
});
defBW("sub",
  (g, sx, sy) => addGrad(true)(g, sx, sy),
  (g, sx, sy) => addGrad(false)(g, sx, sy).neg());

function mulDivGrad(firstArg: boolean, isMul: boolean) {
  return (g: Tensor, x: Tensor, y: Tensor) => {
    if (isMul) {
      g = firstArg ? g.mul(y) : g.mul(x);
    } else {
      g = firstArg ? g.div(y) : g.mul(x).neg().div(y.square());
    }
    // If sx and sy are the same (no broadcasting) just return g.
    if (shapesEqual(x.shape, y.shape)) return g;
    // Broadcast.
    const [rx, ry] = bcastGradientArgs(x.shape, y.shape);
    if (firstArg) {
      return g.reduceSum(rx).reshape(x.shape);
    } else {
      return g.reduceSum(ry).reshape(y.shape);
    }
  };
}

export const mul = defFW("mul", (x, y) => {
  saveForBackward(x, y);
  return bo.mul(x, y);
});
defBW("mul",
  (g, x, y) => mulDivGrad(true, true)(g, x, y),
  (g, x, y) => mulDivGrad(false, true)(g, x, y));

export const div = defFW("div", (x, y) => {
  saveForBackward(x, y);
  return bo.div(x, y);
});
defBW("div",
  (g, x, y) => mulDivGrad(true, false)(g, x, y),
  (g, x, y) => mulDivGrad(false, false)(g, x, y));

export const matmul = defFW("matmul",
  (a, b, transposeA = false, transposeB = false) => {
    saveForBackward(a, b, transposeA, transposeB);
    return bo.matmul(a, b, transposeA, transposeB);
  });
// y = a * b
// da = dy * bT
// db = aT * dy
defBW("matmul",
  (g, a, b, tA, tB) => matmul(g, b, tA, !tB),
  (g, a, b, tA, tB) => matmul(a, g, !tA, tB));

export const neg = defFW("neg", (x) => bo.neg(x));
defBW("neg", (g) => neg(g));

export const exp = defFW("exp", (x) => {
  const ans = bo.exp(x);
  saveForBackward(ans);
  return ans;
});
defBW("exp", (g, ans) => mul(ans, g));

export let log = defFW("log", (x) => {
  saveForBackward(x);
  return bo.log(x);
});
defBW("log", (g, x) => div(g, x));

export const fill = defFW("fill", (value, shape) => {
  saveForBackward(value);
  return bo.fill(value, shape);
});
defBW("fill", (g, value) => {
  throw new Error("Not Implemented: backward pass of fill.");
});

export const square = defFW("square", (x) => {
  saveForBackward(x);
  return bo.square(x);
});
defBW("square", (g, x) => {
  const two = convert(2, x);
  return g.mul(x.mul(two));
});

export const sinh = defFW("sinh", (x) => {
  saveForBackward(x);
  return bo.sinh(x);
});
defBW("sinh", (g, x) => mul(g, cosh(x)));

export let cosh = defFW("cosh", (x) => {
  saveForBackward(x);
  return bo.cosh(x);
});
defBW("cosh", (g, x) => mul(g, sinh(x)));

export let tanh = defFW("tanh", (x) => {
  saveForBackward(x);
  return bo.tanh(x);
});
defBW("tanh", (g, x) => div(g, square(cosh(x))));

export let relu = defFW("relu", (x) => {
  saveForBackward(x);
  return bo.relu(x);
});
defBW("relu", (g, x) => reluGrad(g, x));

export let reluGrad = defFW("reluGrad", (grad, features) => {
  saveForBackward(features);
  return bo.reluGrad(grad, features);
});
defBW("reluGrad",
  (g, features) => reluGrad(g, features),
  null);

export let sigmoid = defFW("sigmoid", (x) => {
  const ans = bo.sigmoid(x);
  saveForBackward(ans);
  return ans;
});
defBW("sigmoid", (g, ans) => {
  // s(x) * (1 - s(x))
  return g.mul(ans.sub(ans.square()));
});

export let abs = defFW("abs", (x) => {
  saveForBackward(x);
  return bo.abs(x);
});
defBW("abs", (g, x) => {
  const z = x.zerosLike();
  const cond = x.greater(z);
  return cond.select(g, g.neg());
});

export let transpose = defFW("transpose", (x, perm) => {
  saveForBackward(perm);
  return bo.transpose(x, perm);
});
defBW("transpose", (g, perm) => transpose(g, perm));

export let reverse = defFW("reverse", (x, dims) => {
  saveForBackward(dims);
  return bo.reverse(x, dims);
});
defBW("reverse", (g, dims) => reverse(g, dims));

export let argmax = defFW("argmax", (x, axis: number) => {
  return bo.argmax(x, axis);
});
defBW("argmax", null);  // Not differentiable.

export let argmin = defFW("argmin", (x, axis: number) => {
  return bo.argmin(x, axis);
});
defBW("argmin", null);  // Not differentiable.

export let reduceSum = defFW("reduceSum", (x, axes, keepDims) => {
  saveForBackward(x.shape, x.dtype, axes);
  return bo.reduceSum(x, axes, keepDims);
});
defBW("reduceSum", (g, xs, xd, axes) => {
  const gs = xs.slice(); // copy
  for (const a of axes) {
    const i = a < 0 ? xs.length + a : a;
    gs[i] = 1;
  }
  return g.reshape(gs).mul(ones(xs, xd));
});

export let reduceMean = defFW("reduceMean", (x, axes, keepDims) => {
  saveForBackward(axes, x.shape, x.dtype);
  return bo.reduceMean(x, axes, keepDims);
});
defBW("reduceMean", (g, axes, shape, dtype) => {
  let n = 1;
  const gs = shape.slice(); // copy
  for (const i of axes) {
    const j = i < 0 ? shape.length + i : i;
    n *= shape[j];
    gs[j] = 1;
  }
  const a = convert(1 / n, {dtype: "float32", device: g.device});
  return g.reshape(gs).mul(fill(a, shape));
});

export let reduceMax = defFW("reduceMax", (x, axes, keepDims) => {
  return bo.reduceMax(x, axes, keepDims);
});
defBW("reduceMax", (g, axes, keepDims) => {
  throw new Error("Not Implemented.");
});

export let equal = defFW("equal", (x, y) => bo.equal(x, y));
defBW("equal", null, null); // Not differentiable.

export let greater = defFW("greater", (x, y) => bo.greater(x, y));
defBW("greater", null, null); // Not differentiable.

export let greaterEqual = defFW("greaterEqual",
  (x, y) => bo.greaterEqual(x, y));
defBW("greaterEqual", null, null); // Not differentiable.

export let less = defFW("less", (x, y) => bo.less(x, y));
defBW("less", null, null); // Not differentiable.

export let lessEqual = defFW("lessEqual", (x, y) => bo.lessEqual(x, y));
defBW("lessEqual", null, null); // Not differentiable.

export let select = defFW("select", (cond, x, y) => {
  saveForBackward(cond);
  return bo.select(cond, x, y);
});
defBW("select", null,
  (g, cond) => cond.cast(g.dtype).mul(g),
  (g, cond) => cond.cast(g.dtype).neg().add(1).mul(g));

export const sign = defFW("sign", (x) => bo.sign(x));
defBW("sign", null); // Not differentiable.

export let slice = defFW("slice", (x, begin, size) => {
  saveForBackward(x.shape, begin, size);
  return bo.slice(x, begin, size);
});
defBW("slice", (g, sx, begin, size) => {
  throw new Error("Not Implemented.");
});

export const concat = defFW("concat",
  (axis: number, ...inputs: types.BasicTensor[]) => {
    return bo.concat(axis, inputs);
  });
defBW("concat", (g) => {
  throw new Error("Not Implemented.");
});

export let reshape = defFW("reshape", (x, newShape) => {
  saveForBackward(x.shape);
  return bo.reshape(x, newShape);
});
defBW("reshape", (g, origShape) => g.reshape(origShape));

export let reduceLogSumExp = defFW("reduceLogSumExp",
  (x, axes: number[], keepDims = false) => {
    const m = bo.reduceMax(x, axes, true);
    const e = bo.exp(bo.sub(x, m));
    const s = bo.reduceSum(e, axes, true);
    const sLog = bo.log(s);
    const ans =  bo.add(m, sLog);
    saveForBackward(ans, x);
    return ans;
  });
defBW("reduceLogSumExp", (g, ans, x) => {
  return g.mul(exp(x.sub(ans)));
});

export const softmax = defFW("softmax", (x) => {
  assert(x.shape.length === 2);
  const ans = bo.softmax(x);
  saveForBackward(ans);
  return ans;
});
defBW("softmax", (g, ans) => {
  return g.sub(g.mul(ans).reduceSum([1]).reshape([-1, 1])).mul(ans);
});

export const logSoftmax = defFW("logSoftmax", (x) => {
  assert(x.shape.length === 2);
  const ans = bo.logSoftmax(x);
  saveForBackward(ans);
  return ans;
});
defBW("logSoftmax", (g, ans) => {
  const softmax = ans.exp();
  return g.sub(g.reduceSum([1], true).mul(softmax));
});

export const cast = defFW("cast", (x, dtype) => {
  saveForBackward(x.dtype);
  return bo.cast(x, dtype);
});
defBW("cast", (g, dtype) => {
  return g.cast(dtype);
});

export const oneHot = defFW("oneHot",
  (x, depth: number, onValue: number, offValue: number) => {
    return bo.oneHot(x, depth, onValue, offValue);
  });
defBW("oneHot", null);

export const setDiag = defFW("setDiag", (input, diag) => {
  return bo.setDiag(input, diag);
});
defBW("setDiag", (g) => {
  // TODO
  // return bo.setDiag(g, zeros);
  throw new Error("Not Implemented.");
}, (g) => {
  // TODO
  // return bo.getDiag(g);
  throw new Error("Not Implemented.");
});
