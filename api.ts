import * as backprop from "./backprop";
import { basicOps } from "./basic";
export { backend } from "./basic";
import { ChainableTensor as Tensor, convertChainable }
  from "./chainable_tensor";
export { ChainableTensor as Tensor } from "./chainable_tensor";
import * as ops from "./ops";
import * as types from "./types";

// Turns a javascript array of numbers
// into a tensor. Like this:
//
//    let tensor = $([[1, 2, 3],
//                    [4, 5, 6]]);
//    console.log(tensor.square());
//
// If a tensor is given to $, it simply returns it.
export function $(t: types.TensorLike): Tensor {
  return convertChainable(t);
}

// grad(f) returns a gradient function. If f is a function that maps
// R^n to R^m, then the gradient function maps R^n to R^n.
// When evaluated at a point, it gives the slope in each dimension of
// the function f. For example:
//
//   let f = (x) => $(x).square();
//
// Then grad(f) is 2*x (being the derivative of x^2).
//
//   g = grad(f);
//   g(10) // is 2 * 10
export const grad = backprop.grad;
export const multigrad = backprop.multigrad;

// Returns the identity matrix of a given size.
export function eye(size: number, dtype: types.DType = "float32"): Tensor {
  const t = basicOps.eye(size, dtype);
  return new Tensor(t);
}

// Returns num evenly spaced samples, calculated over the interval
// [start, stop].
export const linspace = (start: number, stop: number, num = 50): Tensor => {
  const t = basicOps.linspace(start, stop, num);
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
  const t = basicOps.arange(start, limit, delta);
  return new Tensor(t);
};

export const square = (x) => $(x).square();
export const sinh = (x) => $(x).sinh();
export const cosh = (x) => $(x).cosh();
export const tanh = (x) => $(x).tanh();

export function randn(...shape: number[]): Tensor {
  const t = basicOps.randn(shape);
  return new Tensor(t);
}

// fill returns a new tensor of the given shape, filled with constant values
// specified by the `value` argument. `value` must be a scalar tensor.
export function fill(value: types.TensorLike, shape: types.Shape): Tensor {
  return ops.fill($(value), shape);
}

export const matmul = (x, y) => $(x).matmul(y);
