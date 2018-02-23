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
import { bo } from "./backend";
import * as ops from "./ops";
import { convert, Tensor } from "./tensor";
import * as types from "./types";
export { DType, TensorLike } from "./types";
export { params, Params } from "./params";
export { dataset } from "./dataset";
export { print, experiment } from "./experiment";

/** Turns a javascript array of numbers into a tensor. Like this:
 *
 *    import { tensor } from "propel";
 *    tensor([[1, 2, 3], [4, 5, 6]]).square();
 *
 * If a tensor is given to tensor, it simply returns it.
 *
 * @arg t - An array of numbers, representing a Tensor. Or a Tensor in which
 *          case tensor() acts as the identity function.
 * @arg args - An object like this { dtype: "int32", device: "GPU:0" }
 */
export function tensor(t: types.TensorLike, args?: types.TensorOpts): Tensor {
  return convert(t, args);
}

// For backwards compatibility with the old API.
// TODO: remove this after re-running gendoc and updating the default notebook.
export const T = tensor;

/** Returns a list of available device names.
 *
 *    import { listDevices } from "propel";
 *    listDevices();
 */
export function listDevices(): string[] {
  return bo.listDevices();
}

export { Tensor } from "./tensor";

export { grad, multigrad, multigradAndVal, gradAndVal, gradParams, ParamsFn }
  from "./backprop";

/** Returns the identity matrix of a given size.
 *
 *    import { eye } from "propel";
 *    eye(5);
 */
export function eye(size: number,
                    opts: types.TensorOpts = {dtype: "float32"}): Tensor {
  const matrix = zeros([size, size], opts);
  const diag = ones([size], opts);
  return matrix.setDiag(diag);
}

/** Returns num evenly spaced samples, calculated over the interval
 * [start, stop].
 *
 *    import { linspace } from "propel"
 *    linspace(-1, 1, 5);
 */
export function linspace(start: number, stop: number, num = 50): Tensor {
  const t = bo.linspace(start, stop, num);
  return new Tensor(t);
}

/** Return evenly spaced numbers over a specified interval.
 *  This as a similar interface as arange in numpy.
 *
 *    import { range } from "propel"
 *    range(10);
 */
export function range(...args: number[]): Tensor {
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
  const t = bo.range(start, limit, delta);
  return new Tensor(t);
}

/** Produces a new tensor with random values, drawn from the standard normal
 * distribution.
 *
 *    import { range, randn, plot } from "propel"
 *    n = 1000
 *    plot(range(n), randn([n]))
 */
export function randn(shape: number[],
                      opts: types.TensorOpts = {dtype: "float32"}): Tensor {
  let t = bo.randn(shape);
  if (opts.device && opts.device !== "CPU:0") {
    t = bo.copyToDevice(t, opts.device);
  }
  return new Tensor(t);
}

/** fill returns a new tensor of the given shape, filled with constant values
 * specified by the `value` argument. `value` must be a scalar tensor.
 *
 *    import { fill } from "propel"
 *    fill(31337, [2, 2])
 */
export function fill(value: types.TensorLike, shape: types.Shape): Tensor {
  if (!(shape instanceof Array)) {
    throw new Error("Fill takes a shape as an argument");
  }
  return ops.fill(tensor(value), shape);
}

/** Return a new tensor of given shape and dtype, filled with zeros.
 *
 *    import { zeros } from "propel"
 *    zeros([5, 2])
 */
export function zeros(shape: types.Shape,
                      opts: types.TensorOpts = {dtype: "float32"}): Tensor {
  if (!(shape instanceof Array)) {
    throw new Error("Zeros takes a shape as an argument");
  }
  return ops.zeros(shape, opts);
}

/** Return a new tensor of given shape and dtype, filled with ones.
 *
 *    import { ones } from "propel"
 *    ones([2, 3])
 */
export function ones(shape: types.Shape,
                     opts: types.TensorOpts = {dtype: "float32"}): Tensor {
  if (!(shape instanceof Array)) {
    throw new Error("Ones takes a shape as an argument");
  }
  return ops.ones(shape, opts);
}

export { backend } from "./backend";

export { plot, imshow } from "./matplotlib";
