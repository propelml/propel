import * as backprop from "./backprop";
import { basicOps } from "./basic";
export { backend } from "./basic";
import { ChainableTensor, convertChainable } from "./chainable_tensor";
import * as types from "./types";

export type Tensor = ChainableTensor;

export function $(t: types.TensorLike): Tensor {
  return convertChainable(t);
}

export const grad = backprop.grad;
export const multigrad = backprop.multigrad;

export function eye(size: number, dtype: types.DType = "float32"): Tensor {
  const t = basicOps.eye(size, dtype);
  return new ChainableTensor(t);
}

// Returns num evenly spaced samples, calculated over the interval
// [start, stop].
export const linspace = (start: number, stop: number, num = 50): Tensor => {
  const t = basicOps.linspace(start, stop, num);
  return new ChainableTensor(t);
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
  const t = basicOps.arange(start, limit, delta);
  return new ChainableTensor(t);
};

export const square = (x) => $(x).square();
export const sinh = (x) => $(x).sinh();
export const cosh = (x) => $(x).cosh();
export const tanh = (x) => $(x).tanh();

export function randn(...shape: number[]): Tensor {
  const t = basicOps.randn(shape);
  return new ChainableTensor(t);
}

export const matmul = (x, y) => $(x).matmul(y);
