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
import { bo, convertStorage } from "./backend";
import * as format from "./format";
import * as layers from "./layers";
import * as ops from "./ops";
import { Params } from "./params";
import { allFinite, assertShapesEqual } from "./tensor_util";
import * as types from "./types";
import { assert, IS_NODE } from "./util";

export function convert(t: types.TensorLike,
                        opts?: types.TensorOpts): Tensor {
  if (t instanceof Tensor) return t;
  return new Tensor(convertStorage(t, opts));
}

/** Tensor wraps a Storage object. This is the main public
 * interface to tensor operatiors. Each instance has a unique id for use in
 * backprop.  Nothing about Tensors is backend specific.
 * Tensor might be renamed to BoxedTensor in the near future. To
 * external users this class is called just Tensor. We use a more specific name
 * internally so as not to confuse it with the many other tensor classes in
 * Propel.
 */
export class Tensor implements types.Storage {
  /* TODO The storage property should be private. Probably better would be if
   * Tensor was an interface.
   * Users should not access the storage object.
   * But it is mutable, and thus tensors are mutable.
   */
  storage: null | types.Storage;

  private static nextId = 1;
  private _id: number;

  constructor(t: types.Storage) {
    this.storage = t;
    this._id = Tensor.nextId++;
    track(this);
  }

  /** Manually collect garbage. This is required in some form or another
   * when using the DL/WebGL backend, see also gc().
   */
  dispose(): void {
    this.storage.dispose();
    this.storage = null;
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
    this.storage = t.storage;
    this._id = t.id;
    // It would be nice to not forcably destroy the argument here, but
    // that would require reference counting storage.
    t.storage = null;
  }

  /** Returns an iterator over the values of the tensor.
   *
   *    import { range } from "propel";
   *    for (let i in range(10)) {
   *      console.log(i)
   *    }
   */
  [Symbol.iterator]() {
    const d = this.storage.dataSync();
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
      return new Tensor(bo.copyToDevice(t.storage, this.device));
    }
    return new Tensor(convertStorage(t, {dtype, device: this.device}));
  }

  /** Returns a TypedArray containing the actual data of the tensor.
   * This function is often a bottleneck in the browser, prefer to use
   * `Tensor.data()` instead.
   *
   *    import * as pr from "propel";
   *    const t = pr.range(10).reshape([2, 5]);
   *    t.dataSync();
   */
  dataSync(): types.TypedArray {
    return this.storage.dataSync();
  }

  /** Returns a promise to a TypedArray of the actual data.
   *
   *    import * as pr from "propel";
   *    const t = pr.range(10).reshape([2, 5]);
   *    await t.data();
   */
  data(): Promise<types.TypedArray> {
    return this.storage.data();
  }

  get id(): number {
    return this._id;
  }

  get rank(): number {
    return this.shape.length;
  }

  get device(): string {
    return bo.getDevice(this.storage);
  }

  get dtype(): types.DType {
    return this.storage.dtype;
  }

  get shape(): types.Shape {
    return this.storage.shape;
  }

  toString(): string {
    return format.toString(this.shape, this.dataSync());
  }

  /** Copies the tensor to the specified device (usually "CPU:0" or "GPU:0").
   * If device is unspecified, it makse a copy on the same device.
   */
  copy(device?: string): Tensor {
    if (!device) device = this.device;
    const r = bo.copyToDevice(this.storage, device);
    return new Tensor(r);
  }

  gpu(): Tensor {
    if (this.device === "GPU:0") return this;
    // TODO: support different GPUs.
    const r = bo.copyToDevice(this.storage, "GPU:0");
    return new Tensor(r);
  }

  cpu(): Tensor {
    if (this.device === "CPU:0") return this;
    // TODO: support different CPUs? Is that even possible?
    const r = bo.copyToDevice(this.storage, "CPU:0");
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
    const b = bo.onesLike(this.storage);
    return new Tensor(b);
  }

  zerosLike(): Tensor {
    const b = bo.zerosLike(this.storage);
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

  /** Raises each element component-wise to the provided exponent.
   *
   *    import { linspace, plot } from "propel";
   *    x = linspace(-5, 5, 200);
   *    plot(x, x.pow(3))
   */
  pow(exponent: types.TensorLike): Tensor {
    exponent = this.colocate(exponent);
    return ops.pow(this, exponent);
  }

  /** Applies the square root to each element of the input.
   *
   *    import * as pr from "propel";
   *    x = pr.linspace(0, 10, 200);
   *    pr.plot(x, x.sqrt())
   */
  sqrt(): Tensor {
    return ops.sqrt(this);
  }

  /** Applies the sine (sinusoid) to each element of the input.
   *
   *    import * as pr from "propel";
   *    x = pr.linspace(-4*Math.PI, 4*Math.PI, 200);
   *    pr.plot(x, x.sin())
   */
  sin(): Tensor {
    return ops.sin(this);
  }

  /** Applies the cosine function to each element of the input.
   *
   *    import * as pr from "propel";
   *    x = pr.linspace(-4*Math.PI, 4*Math.PI, 200);
   *    pr.plot(x, x.cos())
   */
  cos(): Tensor {
    return ops.cos(this);
  }

  /** Applies the tangent function to each element of the input.
   *
   *    import * as pr from "propel";
   *    x = pr.linspace(-Math.PI/4, Math.PI/4, 200);
   *    pr.plot(x, x.tan())
   */
  tan(): Tensor {
    return ops.tan(this);
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

  /** Prevents from backproping thru the tensor. Returned tensor uses the same
   * storage.
   */
  stopGradient(): Tensor {
    return new Tensor(this.storage);
  }

  /** Calculates mean and variance. */
  moments(axes?: number[], keepDims = false):
          { mean: Tensor, variance: Tensor } {
    const x = this.cast("float32");
    let mean = x.reduceMean(axes, true);
    const sqDiff = x.sub(mean.stopGradient()).square();
    let variance = sqDiff.reduceMean(axes, true);
    if (!keepDims) {
      mean = mean.squeeze();
      variance = variance.squeeze();
    }
    return { mean, variance };
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
   *
   * If the first argument is a number instead of an array of numbers, slice
   * operates on the first axis only:
   *
   *    import * as pr from "propel";
   *    t = pr.tensor([
   *      [1, 2, 3, 4],
   *      [5, 6, 7, 8],
   *      [9, 10, 11, 12],
   *    ]);
   *    t.slice(1);
   *
   * The second argument says how long the slice should be. So if you wanted
   * just the first element of the first axis:
   *
   *    t.slice(0, 1);
   *
   * The most general form takes two arrays as arguments, where the first
   * argument says where to start, and the second say how large the slice
   * should be:
   *
   *    t.slice([0, 1], [2, 3]);
   *
   */
  slice(begin: number | number[], size?: number | number[]): Tensor {
    let begin_: number[];
    if (typeof begin === "number") {
      begin_ = [begin, ...new Array(this.rank - 1).fill(0)];
    } else if (begin.length < this.rank) {
      begin_ = begin.concat(new Array(this.rank - begin.length).fill(0));
    } else {
      begin_ = begin;
    }

    let size_: number[];
    if (size == null) {
      size_ = new Array(this.rank).fill(-1);
    } else if (typeof size === "number") {
      size_ = [size, ...new Array(this.rank - 1).fill(-1)];
    } else if (size.length < this.rank) {
      size_ = size.concat(new Array(this.rank - size.length).fill(-1));
    } else {
      size_ = size;
    }

    assert(allFinite(begin_));
    assert(allFinite(size_));
    return ops.slice(this, begin_, size_);
  }

  /** Concatenates tensors along the specified axis:
   *
   *    import * as pr from "propel";
   *    t = pr.tensor([[1, 2], [3, 4]]);
   *    t.concat(0, [[5, 6], [7, 8]]);
   */
  concat(axis: number, t: types.TensorLike, ...rest: types.TensorLike[])
      : Tensor {
    const tensors = [this, this.colocate(t)];
    for (const arg of rest) {
      assert(arg instanceof Tensor);
      tensors.push(this.colocate(arg as Tensor));
    }
    return ops.concat(axis, ...tensors);
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
   *    import { tensor } from "propel";
   *    tensor([[1, 2], [3, 4]]).flatten();
   */
  flatten(): Tensor {
    return this.reshape([-1]);
  }

  /** Remove single-dimensional axes from the shape of a tensor.
   *
   *    import { tensor } from "propel";
   *    tensor([[[2, 3, 4]]]).squeeze();
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
   *    import { tensor } from "propel";
   *    let labels = tensor([1, 3, 0], {dtype: "int32"});
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

  /** Computes the average softmax cross entropy on logits.
   * This is very similiar to softmaxCE() except it averages the results
   * and allows label indicies instead of just one-hot labels.
   * Normally the logits, the input to this function, comes out of a
   * linear layer which has not activation function performed on it.
   * Ideally the inputs to this function are zero centered.
   *
   *    import * as pr from "propel";
   *    let logits = pr.randn([3, 10]);
   *    logits.softmaxLoss([9, 0, 1]);
   */
  softmaxLoss(labels: types.TensorLike): Tensor {
    const logits = this;
    assert(logits.rank === 2);
    const numLabels = logits.shape[1];
    let labelsT = this.colocate(labels);
    if (labelsT.rank === 1) {
      // Assume labels represent indicies.
      labelsT = labelsT.cast("int32").oneHot(numLabels);
    }
    return this.softmaxCE(labelsT).reduceMean();
  }

  /** Sets the diagonal part of a matrix.
   *
   *    import { zeros } from "propel";
   *    zeros([4, 3]).setDiag([1, 2, 3]);
   */
  setDiag(diag: types.TensorLike): Tensor {
    return ops.setDiag(this, this.colocate(diag, this.dtype));
  }

  /** To rescale a tensor from inScale to outScale.
   * This is often used rescale images from [0, 255] integer
   * range to [-1, 1] range. This function will automatically cast tensors to
   * float32.
   *
   *    import { tensor } from "propel";
   *    let pretendImage = tensor([0, 127, 255], "int32");
   *    pretendImage.rescale([0, 255], [-1, 1]);
   */
  rescale(inScale: [number, number], outScale: [number, number]): Tensor {
    assert(inScale[0] < inScale[1]);
    assert(outScale[0] < outScale[1]);
    return this.cast("float32")
               .sub(inScale[0])
               .div(inScale[1] - inScale[0])
               .mul(outScale[1] - outScale[0])
               .add(outScale[0]);
  }

  /** Max pool.
   * Input should be shaped [batch, height, width, channels]
   * and by default the pooling is 2x2 with stride 2.
   *
   *    import * as pr from "propel"
   *    img = pr.range(4*4).reshape([1, 4, 4, 1]);
   *    img.maxPool({ size: 2, stride: 2 });
   */
  maxPool(opts?: types.PoolOpts): Tensor {
    const defaults: types.PoolOpts = {
      size: 2,
      stride: 2,
      padding: "valid",
    };
    assert(this.rank === 4);
    return ops.maxPool(this, Object.assign(defaults, opts));
  }

  /** Returns x*w+b for the input tensor x.
   * Where w ("weights") and b ("bias") looked up in params.
   * If the params object doesn't contain these parameters, they are
   * initialized.
   *
   * If the input tensor has shape [d0, d1, d2, ... ] then it will be reshaped
   * to [d0, d1 * d2 * ...] before applying the matmul. That means you can use
   * 4D image tensors with this function without having to reshape it.
   *
   *    import { params, zeros } from "propel";
   *    params = params();
   *    inputs = zeros([2, 5]);
   *    outputs = inputs.linear("L1", params, 10);
   *    params.has("L1/weights") && params.has("L1/bias");
   */
  linear(name: string, params: Params, outDim: number,
         opts?: layers.LinearOpts): Tensor {
    return layers.linear(this, params.scope(name), outDim, opts);
  }

  /** Convolutional Layer */
  conv2d(name: string, params: Params, outChans: number,
         opts?: layers.ConvOpts): Tensor {
    return layers.conv2d(this, params.scope(name), outChans, opts);
  }

  /** Batch Normalization. input must be rank 4. */
  batchNorm(name: string, params: Params,
            opts?: layers.BatchNormOpts): Tensor {
    return layers.batchNorm(this, params.scope(name), opts);
  }
}

if (IS_NODE) {
  // This is currently node.js specific.
  const s = require("util").inspect.custom;
  Tensor.prototype[s] = function(depth: number, opts: {}) {
    return this.toString();
  };

  process.on("unhandledRejection", (error) => {
    throw error;
  });
}

export type NamedTensors = { [name: string]: Tensor };

export interface LinearOpts {
  useBias?: boolean;
  scale?: number;
  // TODO custom initializers.
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
      if (!this.keeping.has(t) && t.storage != null) {
        t.dispose();
      }
    });
    this.tensors.clear();
    this.keeping.clear();
  }
}

const scopes: GCScope[] = [];
