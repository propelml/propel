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
import { range } from "./api";
import { bo, convertBasic } from "./backend";
import * as format from "./format";
import * as ops from "./ops";
import * as types from "./types";
import { allFinite, assert, assertShapesEqual } from "./util";

export function convert(t: types.TensorLike,
                        opts?: types.TensorOpts): Tensor {
  if (t instanceof Tensor) return t;
  return new Tensor(convertBasic(t, opts));
}

/** Tensor wraps a BasicTensor object. This is the main public
 * interface to tensor operatiors. Each instance has a unique id for use in
 * backprop.  Nothing about Tensors is backend specific.
 * Tensor might be renamed to BoxedTensor in the near future. To
 * external users this class is called just Tensor. We use a more specific name
 * internally so as not to confuse it with the many other tensor classes in
 * Propel.
 */
export class Tensor implements types.BasicTensor {
  /* TODO The basic property should be private. Probably better would be if
   * Tensor was an interface.
   * Users should not access the basic tensor.
   * But it is mutable, and thus tensors are mutable.
   */
  basic: null | types.BasicTensor;

  private static nextId = 1;
  readonly id: number;

  constructor(t: types.BasicTensor) {
    this.basic = t;
    this.id = Tensor.nextId++;
    track(this);
  }

  /** Manually collect garbage. This is required in some form or another
   * when using the DL/WebGL backend, see also gc().
   */
  dispose(): void {
    this.basic.dispose();
    this.basic = null;
    untrack(this);
  }

  /** In-place replacement of a tensor.
   * The argument passed to assign is destroyed and cannot be used after this
   * call. (FIXME this behavior is very aggressive.)
   */
  assign(t: Tensor): void {
    // Do we want to relax any of these constraints?
    // assert(t.device === this.device);
    assertShapesEqual(t.shape, this.shape);
    assert(t.dtype === this.dtype);

    this.dispose();
    this.basic = t.basic;
    // It would be nice to not forcably destroy the argument here, but
    // that would require reference counting basic.
    t.basic = null;
  }

  /** Returns an iterator over the values of the tensor.
   *
   *    import { range } from "propel";
   *    for (let i in range(10)) {
   *      console.log(i)
   *    }
   */
  [Symbol.iterator]() {
    const d = this.basic.getData();
    let i = 0;
    return {
      next: () => {
        if (i < d.length) {
          return { value: d[i++], done: false };
        } else {
          return { value: null, done: true };
        }
      }
    };
  }

  // This is similar convert() - it turns TensorLike objects into Tensors - but
  // the function further ensures that the returned tensor is on the same
  // devices as this tensor.
  private colocate(t: types.TensorLike, dtype?: types.DType): Tensor {
    if (t instanceof Tensor) {
      if (t.device === this.device) return t;
      // TODO Warning! This might be an unnecessary copy. Maybe we should
      // notify the user? Maybe this shouldn't be allowed even.
      // For now we stay silent.
      return new Tensor(bo.copyToDevice(t.basic, this.device));
    }
    return new Tensor(convertBasic(t, {dtype, device: this.device}));
  }

  getData(): types.TypedArray {
    return this.basic.getData();
  }

  get rank(): number {
    return this.shape.length;
  }

  get device(): string {
    return bo.getDevice(this.basic);
  }

  get dtype(): types.DType {
    return this.basic.dtype;
  }

  get shape(): types.Shape {
    return this.basic.shape;
  }

  toString(): string {
    return format.toString(this.shape, this.getData());
  }

  /** Copies the tensor to the specified device (usually "CPU:0" or "GPU:0").
   * If device is unspecified, it makse a copy on the same device.
   */
  copy(device?: string): Tensor {
    if (!device) device = this.device;
    const r = bo.copyToDevice(this.basic, device);
    return new Tensor(r);
  }

  gpu(): Tensor {
    if (this.device === "GPU:0") return this;
    // TODO: support different GPUs.
    const r = bo.copyToDevice(this.basic, "GPU:0");
    return new Tensor(r);
  }

  cpu(): Tensor {
    if (this.device === "CPU:0") return this;
    // TODO: support different CPUs? Is that even possible?
    const r = bo.copyToDevice(this.basic, "CPU:0");
    return new Tensor(r);
  }

  cast(dtype: types.DType): Tensor {
    return ops.cast(this, dtype);
  }

  add(x: types.TensorLike): Tensor {
    return ops.add(this, this.colocate(x));
  }

  sub(x: types.TensorLike): Tensor {
    return ops.sub(this, this.colocate(x));
  }

  mul(x: types.TensorLike): Tensor {
    return ops.mul(this, this.colocate(x));
  }

  div(x: types.TensorLike): Tensor {
    return ops.div(this, this.colocate(x));
  }

  matmul(x: types.TensorLike): Tensor {
    return ops.matmul(this, this.colocate(x));
  }

  onesLike(): Tensor {
    const b = bo.onesLike(this.basic);
    return new Tensor(b);
  }

  zerosLike(): Tensor {
    const b = bo.zerosLike(this.basic);
    return new Tensor(b);
  }

  /** Negates each element of the tensor.
   *
   *    import { linspace, plot } from "propel";
   *    x = linspace(-5, 5, 200);
   *    plot(x, x.neg())
   */
  neg(): Tensor {
    return ops.neg(this);
  }

  /** Exponentiates each element of the tensor.
   *
   *    import { linspace, plot } from "propel";
   *    x = linspace(-5, 5, 200);
   *    plot(x, x.exp())
   */
  exp(): Tensor {
    return ops.exp(this);
  }

  /** Applies the natural logarithm to each element of the tensor.
   *
   *    import { linspace, plot } from "propel";
   *    x = linspace(0.001, 5, 200);
   *    plot(x, x.log())
   */
  log(): Tensor {
    return ops.log(this);
  }

  /** Squares each element of the tensor.
   *
   *    import { linspace, plot } from "propel";
   *    x = linspace(-5, 5, 200);
   *    plot(x, x.square())
   */
  square() {
    return ops.square(this);
  }

  /** Applies the hyperbolic sine function component-wise.
   *
   *    import { linspace, plot } from "propel";
   *    x = linspace(-5, 5, 200);
   *    plot(x, x.sinh())
   */
  sinh() {
    return ops.sinh(this);
  }

  /** Applies the hyperbolic cosine function component-wise.
   *
   *    import { linspace, plot } from "propel";
   *    x = linspace(-5, 5, 200);
   *    plot(x, x.cosh())
   */
  cosh() {
    return ops.cosh(this);
  }

  /** Applies the hyperbolic tangent function component-wise.
   * Where tanh(x) is defined to be (1 - exp(-2x)) / (1 + exp(-2x)).
   *
   *    import { linspace, plot } from "propel";
   *    x = linspace(-10, 10, 200);
   *    plot(x, x.tanh())
   */
  tanh(): Tensor {
    return ops.tanh(this);
  }

  /** Applies the rectified linear unit function component-wise.
   * Where relu(x) is defined to be max(x, 0).
   *
   *    import { linspace, plot } from "propel";
   *    x = linspace(-10, 10, 200);
   *    plot(x, x.relu())
   */
  relu(): Tensor {
    return ops.relu(this);
  }

  /** Applies the sigmoid function component-wise to the tensor.
   * The sigmoid function is defined as 1 / (1 + exp(-x)).
   *
   *    import { linspace, plot } from "propel";
   *    x = linspace(-10, 10, 200);
   *    plot(x, x.sigmoid())
   */
  sigmoid(): Tensor {
    return ops.sigmoid(this);
  }

  /** Applies absolute value component-wise to the tensor.
   *
   *    import { linspace, plot } from "propel";
   *    x = linspace(-10, 10, 200);
   *    plot(x, x.abs())
   */
  abs(): Tensor {
    return ops.abs(this);
  }

  transpose(perm?: types.TensorLike): Tensor {
    if (perm === undefined) {
      perm = range(this.rank).reverse();
    }
    perm = this.colocate(perm, "int32");
    return ops.transpose(this, perm);
  }

  /** Reverses specific dimensions of a tensor. */
  reverse(dims?: number[]): Tensor {
    if (!dims) dims = [-1];
    // Convert dims to 1D tensor of booleans.
    const ta = new Uint8Array(this.rank);
    for (const dim of dims) {
      assert(-this.rank <= dim && dim < this.rank);
      const i = dim >= 0 ? dim : this.rank + dim;
      ta[i] = 1;
    }

    const dimsT = this.colocate(ta, "bool");
    return ops.reverse(this, dimsT);
  }

  /** Returns the index with the largest value across an axis of a tensor.
   * @param axis defaults to 0.
   */
  argmax(axis?: number): Tensor {
    if (axis === undefined) axis = 0;
    return ops.argmax(this, axis);
  }

  /** Returns the index with the smallest value across an axis of a tensor.
   * @param axis defaults to 0.
   */
  argmin(axis?: number): Tensor {
    if (axis === undefined) axis = 0;
    return ops.argmin(this, axis);
  }

  /** Sum the tensor over the given axes.
   * @param axes defaults to all
   */
  reduceSum(axes?: number[], keepDims = false): Tensor {
    if (!axes) axes = rangeJS(this.rank);
    return ops.reduceSum(this, axes, keepDims);
  }

  /** Mean the tensor over the given axes.
   * @param axes defaults to all.
   */
  reduceMean(axes?: number[], keepDims = false): Tensor {
    if (!axes) axes = rangeJS(this.rank);
    return ops.reduceMean(this, axes, keepDims);
  }

  /** Take the maximum value over the given axes.
   * @param axes defaults to all.
   */
  reduceMax(axes?: number[], keepDims = false): Tensor {
    if (!axes) axes = rangeJS(this.rank);
    return ops.reduceMax(this, axes, keepDims);
  }

  reduceLogSumExp(axes?: number[], keepDims = false): Tensor {
    if (!axes) axes = rangeJS(this.rank);
    return ops.reduceLogSumExp(this, axes, keepDims);
  }

  /** Element-wise comparison. Returns a tensor with dtype == "bool". */
  equal(x: types.TensorLike): Tensor {
    return ops.equal(this, this.colocate(x));
  }

  /** Returns a boolean tensor with the truth value of (this > x)
   * element-wise.
   */
  greater(x: types.TensorLike): Tensor {
    return ops.greater(this, this.colocate(x, this.dtype));
  }

  /** Returns a boolean tensor with the truth value of (this >= x)
   * element-wise.
   */
  greaterEqual(x: types.TensorLike): Tensor {
    return ops.greaterEqual(this, this.colocate(x, this.dtype));
  }

  /** Returns a boolean tensor with the truth value of (this < x)
   * element-wise.
   */
  less(x: types.TensorLike): Tensor {
    return ops.less(this, this.colocate(x, this.dtype));
  }

  /** Returns a boolean tensor with the truth value of (this <= x)
   * element-wise.
   */
  lessEqual(x: types.TensorLike): Tensor {
    return ops.lessEqual(this, this.colocate(x, this.dtype));
  }

  /** Selects elements from `t` or `f`, depending on the condition (this).
   * this should be a boolean Tensor.
   */
  select(t: types.TensorLike, f: types.TensorLike): Tensor {
    const tT = this.colocate(t);
    const fT = this.colocate(f, tT.dtype);
    return ops.select(this, tT, fT);
  }

  /** Returns an element-wise indication of the sign of a number.
   * -1 for negative values and 1 for positive values.
   */
  sign(): Tensor {
    return ops.sign(this);
  }

  /** Return a slice from 'input'.
   * The output tensor is a tensor with dimensions described by 'size' whose
   * values are extracted from 'input' starting at the offsets in 'begin'.
   * begin[i] specifies the offset into the ith dimension of 'input' to slice
   * from.  size[i] specifies the number of elements of the ith dimension of
   * 'input' to slice. If size[i] is -1, all remaining elements in dimension
   * are included in the slice -- this is equivalent to setting
   *   size[i] = input.shape[i] - begin[i]
   */
  slice(begin: number[], size: number[]): Tensor {
    assert(allFinite(begin));
    assert(allFinite(size));
    return ops.slice(this, begin, size);
  }

  /** Reshapes the tensor without changing its data. O(1).
   *
   *    import { range } from "propel";
   *    n = 5
   *    m = 2
   *    t = range(m*n)
   *    console.log(t.reshape([n, m]))
   *    console.log(t.reshape([m, n]))
   */
  reshape(newShape: types.Shape): Tensor {
    return ops.reshape(this, newShape);
  }

  /** Return a copy of the tensor collapsed into one dimension.
   *
   *    import { T } from "propel";
   *    T([[1, 2], [3, 4]]).flatten();
   */
  flatten(): Tensor {
    return this.reshape([-1]);
  }

  /** Remove single-dimensional axes from the shape of a tensor.
   *
   *    import { T } from "propel";
   *    T([[[2, 3, 4]]]).squeeze();
   */
  squeeze(): Tensor {
    const newShape = this.shape.filter((d) => d > 1);
    return this.reshape(newShape);
  }

  /** Returns the softmax activations of a tensor. */
  softmax(axis = -1): Tensor {
    return softmaxHelper(this, axis, ops.softmax);
  }

  /** Numerically stable log(softmax(x)). */
  logSoftmax(axis = -1): Tensor {
    return softmaxHelper(this, axis, ops.logSoftmax);
  }

  /** Dot product of two tensors. For 2D tensors it is equivalent to matrix
   * multiplication. For 1D tensors to inner product of vectors (without
   * complex conjugation). Currently higher order tensors are not supported.
   */
  dot(x: types.TensorLike): Tensor {
    const xx = this.colocate(x);
    let left, right;
    let lShape, rShape;
    if (this.rank === 0) {
      left = this.reshape([1, 1]);
      lShape = [];
    } else if (this.rank === 1) {
      assert(this.shape[0] === xx.shape[0]);
      left = this.reshape([1, this.shape[0]]);
      lShape = [];
    } else if (this.rank === 2) {
      left = this;
      lShape = [this.shape[0]];
    } else {
      left = null;
    }

    if (xx.rank === 0) {
      right = xx.reshape([1, 1]);
      rShape = [];
    } else if (xx.rank === 1) {
      assert(this.shape[this.rank - 1] === xx.shape[0]);
      right = xx.reshape([xx.shape[0], 1]);
      rShape = [];
    } else if (xx.rank === 2) {
      right = xx;
      rShape = [xx.shape[xx.rank - 1]];
    } else {
      right = null;
    }

    if (!left || !right) {
      throw new Error("dot with tensors of rank greater " +
        "than 2 is not yet implemented.");
    }
    const outShape = lShape.concat(rShape);
    return left.matmul(right).reshape(outShape);
  }

  /** Build a one-hot tensor from labels.
   *
   *    import { T } from "propel";
   *    let labels = T([1, 3, 0], {dtype: "int32"});
   *    labels.oneHot(5);
   */
  oneHot(depth: number, onValue = 1.0, offValue = 0.0): Tensor {
    if (this.dtype === "float32") {
      throw new Error("Must use integer type with oneHot.");
    }
    return ops.oneHot(this, depth, onValue, offValue);
  }

  /** Computes softmax cross entropy on logits.
   * This is known as softmax_cross_entropy_with_logits in TF.
   * @param labels A batch_size x num_classes matrix. The caller must ensure
   *               that each batch of labels represents a valid probability
   *               distribution. Often labels is one-hot along axis 1.
   */
  softmaxCE(labels: types.TensorLike): Tensor {
    const logits = this;
    const labelsT = this.colocate(labels).cast("float32");
    assert(labelsT.rank === 2);
    assert(logits.rank === 2);
    const logQ = logits.logSoftmax();
    const pLogQ = labelsT.mul(logQ);
    return pLogQ.reduceSum([1]).neg();
  }

  /** Sets the diagonal part of a matrix.
   *
   *    import { zeros } from "propel";
   *    zeros([4, 3]).setDiag([1, 2, 3]);
   */
  setDiag(diag: types.TensorLike): Tensor {
    return ops.setDiag(this, this.colocate(diag, this.dtype));
  }
}

/** Like range() but outputs a javascript array of numbers. */
function rangeJS(limit: number): number[] {
  const r = new Array(limit);
  for (let i = 0; i < limit; i++) {
    r[i] = i;
  }
  return r;
}

/** The softmax and logSoftmax ops can only accept 2D tensors of the form
 * [batch_size, num_classes]. This function reshapes higher rank tensors to
 * that.
 */
function softmaxHelper(t: Tensor, axis: number, op): Tensor {
  if (axis !== -1) {
    throw new Error("Softmax along a non-last axis is not yet supported.");
  }
  if (t.rank === 2) {
    return op(t, axis);
  }
  const origShape = t.shape;
  const numClasses = t.shape[t.rank - 1];
  const result = op(t.reshape([-1, numClasses]));
  return result.reshape(origShape);
}

/* Manual memory management.
 *  - WebGL textures are not garbage collected. Therefore we need to destory
 *    them manually.
 *  - Although TensorFlow tensors are properly GCed we can help by destroying
 *    tensors we know will not be used again.
 *  - Interally we will use this during backprop and each SGD step.
 *  - Ideally this would not be exposed to the public API. But given the
 *    state of WebGL it doesn't seem likely that we can completely abstract
 *    away memory management.
 */

export type GCScopeFn = (keep: (t: Tensor) => void) => void;

export function gc(fn: GCScopeFn) {
  const s = new GCScope();
  scopes.push(s);
  const keep = (t: Tensor): void => { s.keep(t); };
  try {
    fn(keep);
  } finally {
    assert(s === scopes.pop());
    s.clean();
  }
}

function track(t: Tensor) {
  if (scopes.length > 0) {
    scopes[scopes.length - 1].track(t);
  }
}

function untrack(t: Tensor) {
  for (const s of scopes) {
    s.untrack(t);
  }
}

class GCScope {
  private keeping = new Set<Tensor>();
  private tensors = new Set<Tensor>();

  track(t: Tensor): void {
    this.tensors.add(t);
  }

  untrack(t: Tensor): void {
    this.tensors.delete(t);
    this.keeping.delete(t);
  }

  keep(t: Tensor): void {
    this.keeping.add(t);
  }

  clean(): void {
    this.tensors.forEach(t => {
      // If we're not keeping it, nor have we already
      // disposed of it (which happens with assign)
      // then we dispose.
      if (!this.keeping.has(t) && t.basic != null) {
        t.dispose();
      }
    });
    this.tensors.clear();
    this.keeping.clear();
  }
}

const scopes: GCScope[] = [];
