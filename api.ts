import { bo } from "./backend";
import * as backprop from "./backprop";
export { backend } from "./backend";
import { convert, Tensor } from "./tensor";
export { Tensor } from "./tensor";
import * as ops from "./ops";
import * as types from "./types";
import { assert, assertShapesEqual } from "./util";

export { grad, multigrad, multigradAndVal, gradAndVal, gradParams, ParamsFn }
  from "./backprop";
import { gradParams, ParamsFn }
  from "./backprop";

// Turns a javascript array of numbers
// into a tensor. Like this:
//
//    let tensor = $([[1, 2, 3],
//                    [4, 5, 6]]);
//    console.log(tensor.square());
//
// If a tensor is given to $, it simply returns it.
export function $(t: types.TensorLike, dtype?: types.DType): Tensor {
  return convert(t, dtype);
}

// Returns the identity matrix of a given size.
export function eye(size: number, dtype: types.DType = "float32"): Tensor {
  const t = bo.eye(size, dtype);
  return new Tensor(t);
}

// Returns num evenly spaced samples, calculated over the interval
// [start, stop].
export const linspace = (start: number, stop: number, num = 50): Tensor => {
  const t = bo.linspace(start, stop, num);
  return new Tensor(t);
};

// Return evenly spaced numbers over a specified interval.
export const arange = function(...args: number[]): Tensor {
  let start: number, limit: number, delta: number;
  switch (args.length) {
    case 1:
      start = 0;
      limit = args[0];
      delta = 1;
      break;

    case 2:
      start = args[0];
      limit = args[1];
      delta = 1;
      break;

    case 3:
      start = args[0];
      limit = args[1];
      delta = args[2];
      break;

    default:
      throw new Error("Bad number of arguments.");
  }
  const t = bo.arange(start, limit, delta);
  return new Tensor(t);
};

export function randn(shape: number[]): Tensor {
  const t = bo.randn(shape);
  return new Tensor(t);
}

// fill returns a new tensor of the given shape, filled with constant values
// specified by the `value` argument. `value` must be a scalar tensor.
export function fill(value: types.TensorLike, shape: types.Shape): Tensor {
  return ops.fill($(value), shape);
}

// Return a new tensor of given shape and dtype, filled with zeros.
export function zeros(shape: types.Shape,
                      dtype: types.DType = "float32"): Tensor {
  return ops.zeros(shape, dtype);
}

// Return a new tensor of given shape and dtype, filled with ones.
export function ones(shape: types.Shape,
                     dtype: types.DType = "float32"): Tensor {
  return ops.ones(shape, dtype);
}

export const matmul = (x, y) => $(x).matmul(y);

export interface ArgsSGD {
  callback: (step: number, loss: number, params: Params) => void;
  lossFn: ParamsFn;
  params?: Params;
  learningRate: number;
  momentum: number;
  steps: number;
}

// Stochastic gradient descent with momentum.
export function sgd(args: ArgsSGD): Params {
  const m = args.momentum;
  assert(0 <= m && m <= 1.0);
  let params = args.params ? args.params : new Params();
  // Get gradient of objective using autograd.
  const gradFn = gradParams(args.lossFn);
  // TODO Design note. The name "Params" doesn't fit well with what velocity
  // is. Maybe "Params" should be named more generically, like "NamedTensors".
  // But I prefer to have a more nuanced name than "NamedTensors".  "Params"
  // works for now.
  const velocity = new Params();
  const updated = new Params();
  // Training loop.
  for (let step = 1; step < args.steps; ++step) {
    // Forward/Backward pass
    const [grads, loss] = gradFn(params);
    assert(loss.rank === 0);
    assert(grads instanceof Params);
    // Update each param tensor.
    grads.forEach((g, name) => {
      const p = params.get(name);
      let v = velocity.zeros(name, p.shape);
      if (step > 1) {
        assertShapesEqual(p.shape, g.shape);
        assertShapesEqual(p.shape, v.shape);
      }
      // m (momentum) is usually 0.9, so we're saying use 90% v (velocity) and
      // 10% from g (grad).
      v = velocity.set(name, v.mul(m).sub(g.mul(1 - m)));
      // p += v * lr
      updated.set(name, p.add(v.mul(args.learningRate)));
    });
    params = updated;

    const lossVal = loss.getData()[0];
    if (args.callback) args.callback(step, lossVal, params);
  }
  return updated;
}

// A collection of named Tensors. Used with sgd().
// Iterate over it like this:
//
//    params.forEach((tensor, name) => {
//      console.log("name", tensor);
//    });
//
export class Params {
  // Note TS doesn't allow extending Map:
  // https://github.com/Microsoft/TypeScript/issues/10853
  store = new Map<string, Tensor>();

  has(name: string): boolean {
    return this.store.has(name);
  }

  get(name: string): Tensor {
    return this.store.get(name);
  }

  set(name: string, t: Tensor): Tensor {
    this.store.set(name, t);
    return t;
  }

  forEach(cb): void {
    this.store.forEach(cb);
  }

  // If the given name does not exist in the parameters object, this
  // initializes a new random normal tensor. If the name does exist
  // in the parameters object, this just returns that stored tensor.
  randn(name: string, shape: types.Shape, scale = 0.1): Tensor {
    if (this.has(name)) {
      return this.get(name);
    }
    // Initialize.
    const t = randn(shape).mul(scale);
    this.set(name, t);
    return t;
  }

  // If the given name does not exist in the parameters object, this
  // initializes a new tensor with zero values. If the name does exist
  // in the parameters object, this just returns that stored tensor.
  zeros(name: string, shape: types.Shape, dtype:
        types.DType = "float32"): Tensor {
    if (this.has(name)) {
      return this.get(name);
    }
    // Initialize.
    const t = zeros(shape);
    this.set(name, t);
    return t;
  }
}
