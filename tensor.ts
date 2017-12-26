import { arange } from "./api";
import { bo, convertBasic } from "./backend";
import * as format from "./format";
import * as ops from "./ops";
import * as types from "./types";
import { allFinite, assert } from "./util";

export function convert(t: types.TensorLike, dtype?: types.DType): Tensor {
  if (t instanceof Tensor) return t;
  return new Tensor(convertBasic(t, dtype));
}
const $ = convert;

// Tensor wraps a BasicTensor object. This is the main public
// interface to tensor operatiors. Each instance has a unique id for use in
// backprop.  Nothing about Tensors is backend specific.
// Tensor might be renamed to BoxedTensor in the near future. To
// external users this class is called just Tensor. We use a more specific name
// internally so as not to confuse it with the many other tensor classes in
// Propel.
export class Tensor implements types.BasicTensor {
  readonly dtype: types.DType;
  readonly shape: types.Shape;
  readonly basic: types.BasicTensor;
  private static nextId = 1;
  readonly id: number;

  constructor(t: types.BasicTensor) {
    this.shape = t.shape;
    this.dtype = t.dtype;
    this.basic = t;
    this.id = Tensor.nextId++;
  }

  getData(): types.TypedArray {
    return this.basic.getData();
  }

  get rank(): number {
    return this.shape.length;
  }

  toString(): string {
    return format.toString(this.shape, this.getData());
  }

  cast(dtype: types.DType): Tensor {
    return ops.cast(this, dtype);
  }

  add(x: types.TensorLike): Tensor {
    return ops.add(this, $(x));
  }

  sub(x: types.TensorLike): Tensor {
    return ops.sub(this, $(x));
  }

  mul(x: types.TensorLike): Tensor {
    return ops.mul(this, $(x));
  }

  div(x: types.TensorLike): Tensor {
    return ops.div(this, $(x));
  }

  matmul(x: types.TensorLike): Tensor {
    return ops.matmul(this, $(x));
  }

  neg(): Tensor {
    return ops.neg(this);
  }

  exp(): Tensor {
    return ops.exp(this);
  }

  log(): Tensor {
    return ops.log(this);
  }

  onesLike(): Tensor {
    const b = bo.onesLike(this.basic);
    return new Tensor(b);
  }

  zerosLike(): Tensor {
    const b = bo.zerosLike(this.basic);
    return new Tensor(b);
  }

  square = () => ops.square(this);
  sinh = () => ops.sinh(this);
  cosh = () => ops.cosh(this);
  tanh = () => ops.tanh(this);
  relu = () => ops.relu(this);
  sigmoid = () => ops.sigmoid(this);
  abs = () => ops.abs(this);

  transpose(perm?: types.TensorLike): Tensor {
    if (perm === undefined) {
      perm = arange(this.rank).reverse();
    }
    perm = $(perm, "int32");
    return ops.transpose(this, perm);
  }

  // Reverses specific dimensions of a tensor.
  reverse(dims?: number[]): Tensor {
    if (!dims) dims = [-1];
    // Convert dims to 1D tensor of booleans.
    const ta = new Uint8Array(this.rank);
    for (const dim of dims) {
      assert(-this.rank <= dim && dim < this.rank);
      const i = dim >= 0 ? dim : this.rank + dim;
      ta[i] = 1;
    }

    const dimsT = $(ta, "bool");
    return ops.reverse(this, dimsT);
  }

  // Returns the index with the largest value across an axis of a tensor.
  // axis defaults to 0.
  argmax(axis?: number): Tensor {
    if (axis === undefined) axis = 0;
    return ops.argmax(this, axis);
  }

  // Returns the index with the smallest value across an axis of a tensor.
  // axis defaults to 0.
  argmin(axis?: number): Tensor {
    if (axis === undefined) axis = 0;
    return ops.argmin(this, axis);
  }

  // Sum the tensor over the given axes.
  // axes defaults to all.
  reduceSum(axes?: number[], keepDims = false): Tensor {
    if (!axes) axes = rangeJS(this.rank);
    return ops.reduceSum(this, axes, keepDims);
  }

  // Mean the tensor over the given axes.
  // axes defaults to all.
  reduceMean(axes?: number[], keepDims = false): Tensor {
    if (!axes) axes = rangeJS(this.rank);
    return ops.reduceMean(this, axes, keepDims);
  }

  // Take the maximum value over the given axes.
  // axes defaults to all.
  reduceMax(axes?: number[], keepDims = false): Tensor {
    if (!axes) axes = rangeJS(this.rank);
    return ops.reduceMax(this, axes, keepDims);
  }

  reduceLogSumExp(axes?: number[], keepDims = false): Tensor {
    if (!axes) axes = rangeJS(this.rank);
    return ops.reduceLogSumExp(this, axes, keepDims);
  }

  // Element-wise comparison. Returns a tensor with dtype == "bool".
  equal(x: types.TensorLike): Tensor {
    return ops.equal(this, $(x));
  }

  // Returns a boolean tensor with the truth value of (this > x) element-wise.
  greater(x: types.TensorLike): Tensor {
    return ops.greater(this, $(x, this.dtype));
  }

  // Returns a boolean tensor with the truth value of (this >= x) element-wise.
  greaterEqual(x: types.TensorLike): Tensor {
    return ops.greaterEqual(this, $(x, this.dtype));
  }

  // Returns a boolean tensor with the truth value of (this < x) element-wise.
  less(x: types.TensorLike): Tensor {
    return ops.less(this, $(x, this.dtype));
  }

  // Returns a boolean tensor with the truth value of (this <= x) element-wise.
  lessEqual(x: types.TensorLike): Tensor {
    return ops.lessEqual(this, $(x, this.dtype));
  }

  // Selects elements from `t` or `f`, depending on the condition (this).
  // this should be a boolean Tensor.
  select(t: types.TensorLike, f: types.TensorLike): Tensor {
    const tT = $(t);
    const fT = $(f, tT.dtype);
    return ops.select(this, tT, fT);
  }

  // Returns an element-wise indication of the sign of a number.
  // -1 for negative values and 1 for positive values.
  sign = () => ops.sign(this);

  // Return a slice from 'input'.
  // The output tensor is a tensor with dimensions described by 'size' whose
  // values are extracted from 'input' starting at the offsets in 'begin'.
  // begin[i] specifies the offset into the ith dimension of 'input' to slice
  // from.  size[i] specifies the number of elements of the ith dimension of
  // 'input' to slice. If size[i] is -1, all remaining elements in dimension
  // are included in the slice -- this is equivalent to setting
  //   size[i] = input.shape[i] - begin[i]
  slice(begin: number[], size: number[]): Tensor {
    assert(allFinite(begin));
    assert(allFinite(size));
    return ops.slice(this, begin, size);
  }

  // Reshapes the tensor without changing its data.
  reshape(newShape: types.Shape): Tensor {
    return ops.reshape(this, newShape);
  }

  // Return a copy of the tensor collapsed into one dimension.
  flatten(): Tensor {
    return this.reshape([-1]);
  }

  // Remove single-dimensional axes from the shape of a tensor.
  squeeze(): Tensor {
    const newShape = this.shape.filter((d) => d > 1);
    return this.reshape(newShape);
  }

  // Returns the softmax activations of a tensor.
  softmax(axis = -1): Tensor {
    return softmaxHelper(this, axis, ops.softmax);
  }

  // Numerically stable log(softmax(x)).
  logSoftmax(axis = -1): Tensor {
    return softmaxHelper(this, axis, ops.logSoftmax);
  }

  // Dot product of two tensors. For 2D tensors it is equivalent to matrix
  // multiplication. For 1D tensors to inner product of vectors (without
  // complex conjugation). Currently higher order tensors are not supported.
  dot(x: types.TensorLike): Tensor {
    const xx = $(x);
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

  oneHot(depth: number, onValue = 1.0, offValue = 0.0): Tensor {
    if (this.dtype === "float32") {
      throw new Error("Must use integer type with oneHot.");
    }
    return ops.oneHot(this, depth, onValue, offValue);
  }

  // Computes softmax cross entropy on logits.
  // This is known as softmax_cross_entropy_with_logits in TF.
  // @param labels A batch_size x num_classes matrix. The caller must ensure
  //               that each batch of labels represents a valid probability
  //               distribution. Often labels is one-hot along axis 1.
  softmaxCE(labels: types.TensorLike): Tensor {
    const logits = this;
    const labelsT = $(labels);
    assert(labelsT.rank === 2);
    assert(logits.rank === 2);
    const logQ = logits.logSoftmax();
    const pLogQ = labelsT.mul(logQ);
    return pLogQ.reduceSum([1]).neg();
  }
}

// Like arange() but outputs a javascript array of numbers.
function rangeJS(limit: number): number[] {
  const r = new Array(limit);
  for (let i = 0; i < limit; i++) {
    r[i] = i;
  }
  return r;
}

// The softmax and logSoftmax ops can only accept 2D tensors of the form
// [batch_size, num_classes]. This function reshapes higher rank tensors to
// that.
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
