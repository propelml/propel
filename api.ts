import * as types from "./types";
import { convertChainable, ChainableTensor } from "./chainable_tensor";
import { basicOps } from "./basic";
import * as backprop from "./backprop";

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

// Return evenly spaced numbers over a specified interval.
//
// Returns num evenly spaced samples, calculated over the interval
// [start, stop].
export const linspace = (start, stop, num = 50): Tensor => {
  const a = [];
  const n = num - 1;
  const d = (stop - start) / n;
  for (let i = 0; i <= n; ++i) {
    a.push(start + i * d);
  }
  return $(a);
};

export const arange = function(start, stop, step = 1): Tensor {
  const a = [];
  for (let i = start; i < stop; i += step) {
    a.push(i);
  }
  return $(a);
};

export const square = (x) => $(x).square();
export const sinh = (x) => $(x).sinh();
export const cosh = (x) => $(x).cosh();
export const tanh = (x) => $(x).tanh();
