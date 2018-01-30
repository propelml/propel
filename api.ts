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
import { gradParams, ParamsFn } from "./backprop";
import * as ops from "./ops";
import { convert, gc, Tensor } from "./tensor";
import * as types from "./types";
export { DType, TensorLike } from "./types";
import { assert, assertShapesEqual } from "./util";

/** Turns a javascript array of numbers into a tensor. Like this:
 *
 *    import { T } from "propel";
 *    T([[1, 2, 3], [4, 5, 6]]).square();
 *
 * If a tensor is given to T, it simply returns it.
 *
 * @arg t - An array of numbers, representing a Tensor. Or a Tensor in which
 *          case T() acts as the identity function.
 * @arg args - An object like this { dtype: "int32", device: "GPU:0" }
 */
export function T(t: types.TensorLike, args?: types.TensorOpts): Tensor {
  return convert(t, args);
}

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
 *    import { range, randn } from "propel"
 *    n = 1000
 *    plot(range(n), randn([n]))
 */
export function randn(shape: number[]): Tensor {
  const t = bo.randn(shape);
  return new Tensor(t);
}

/** fill returns a new tensor of the given shape, filled with constant values
 * specified by the `value` argument. `value` must be a scalar tensor.
 *
 *    import { fill } from "propel"
 *    fill(31337, [2, 2])
 */
export function fill(value: types.TensorLike, shape: types.Shape): Tensor {
  return ops.fill(T(value), shape);
}

/** Return a new tensor of given shape and dtype, filled with zeros.
 *
 *    import { zeros } from "propel"
 *    zeros([5, 2])
 */
export function zeros(shape: types.Shape,
                      opts: types.TensorOpts = {dtype: "float32"}): Tensor {
  return ops.zeros(shape, opts);
}

/** Return a new tensor of given shape and dtype, filled with ones.
 *
 *    import { ones } from "propel"
 *    ones([2, 3])
 */
export function ones(shape: types.Shape,
                     opts: types.TensorOpts = {dtype: "float32"}): Tensor {
  return ops.ones(shape, opts);
}

/** A collection of named Tensors. Used with OptimizerSGD.
 * Iterate over it like this:
 *
 *    import { Params } from "propel";
 *    let params = new Params()
 *    params.randn("A", [2]);
 *    params.zeros("B", [2, 2]);
 *    params.forEach((tensor, name) => {
 *      console.log(name);
 *      console.log(tensor);
 *    });
 */
export class Params {
  // Note TS doesn't allow extending Map:
  // https://github.com/Microsoft/TypeScript/issues/10853
  private store = new Map<string, Tensor>();

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

  /** If the given name does not exist in the parameters object, this
   * initializes a new random normal tensor. If the name does exist
   * in the parameters object, this just returns that stored tensor.
   */
  randn(name: string, shape: types.Shape,
        { device = "CPU:0", scale = 0.1 } = {}): Tensor {
    if (this.has(name)) {
      return this.get(name);
    }
    // Initialize.
    let t = randn(shape).mul(scale);
    if (device && device !== "CPU:0") {
      t = new Tensor(bo.copyToDevice(t.basic, device));
    }
    this.set(name, t);
    return t;
  }

  /** If the given name does not exist in the parameters object, this
   * initializes a new tensor with zero values. If the name does exist
   * in the parameters object, this just returns that stored tensor.
   */
  zeros(name: string, shape: types.Shape, dtype: types.DType = "float32",
        device = "CPU:0"): Tensor {
    if (this.has(name)) {
      return this.get(name);
    }
    // Initialize.
    let t = zeros(shape);
    if (device && device !== "CPU:0") {
      t = new Tensor(bo.copyToDevice(t.basic, device));
    }
    this.set(name, t);
    return t;
  }
}

/** Stochastic gradient descent with momentum. */
export class OptimizerSGD {
  steps: number;
  params: Params;

  // TODO access to grads.
  // grads?: Params;

  // TODO Design note. The name "Params" doesn't fit well with what velocity
  // is. Maybe "Params" should be named more generically, like
  // "NamedTensors".  But I prefer to have a more nuanced name than
  // "NamedTensors".  "Params" works for now.
  velocity: Params;

  constructor() {
    this.steps = 0;
    this.params = new Params();
    this.velocity = new Params();
  }

  step(learningRate: number, momentum: number, lossFn: ParamsFn): number {
    const m = momentum;
    assert(0 <= m && m <= 1.0);
    let lossValue;

    gc((keep) => {
      // Get gradient of objective using autograd.
      // TODO it's possible that calling gradParams every step is killing the
      // possibility of a good optimization in backprop. Re-evaluate later.
      const gradFn = gradParams(lossFn);
      // Forward/Backward pass
      const [grads, loss] = gradFn(this.params);
      assert(loss.rank === 0);
      assert(grads instanceof Params);

      // TODO allow access to grads.
      // this.grads = grads;

      // Update each param tensor.
      grads.forEach((g, name) => {
        const p = this.params.get(name);
        const v = this.velocity.zeros(name, p.shape);
        if (this.steps > 0) {
          assertShapesEqual(p.shape, g.shape);
          assertShapesEqual(p.shape, v.shape);
        }

        // v = m * v - (1 - m) * g
        // m (momentum) is usually 0.9, so we're saying use 90% v (velocity) and
        // 10% from g (grad).
        v.assign(g.mul(1 - m).sub(v.mul(m)).neg());
        keep(v);
        assert(g.device === v.device);
        // p += v * lr
        p.assign(p.add(v.mul(learningRate)));
        keep(p);
      });

      this.steps++;
      lossValue = loss.cpu().getData()[0];
    });

    return lossValue;
  }
}

export { backend } from "./backend";

export { plot, imshow } from "./matplotlib";
