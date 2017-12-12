import { arange } from "./api";
import { basicOps, convertBasic } from "./basic";
import * as ops from "./ops";
import * as types from "./types";
import { assert } from "./util";

export function convertChainable(t: types.TensorLike,
                                 dtype?: types.DType): ChainableTensor {
  if (t instanceof ChainableTensor) return t;
  return new ChainableTensor(convertBasic(t, dtype));
}
const $ = convertChainable;

// ChainableTensor wraps a BasicTensor object. This is the main public
// interface to tensor operatiors. Each instance has a unique id for use in
// backprop.  Nothing about ChainableTensors is backend specific.
// ChainableTensor might be renamed to BoxedTensor in the near future. To
// external users this class is called just Tensor. We use a more specific name
// internally so as not to confuse it with the many other tensor classes in
// Propel.
export class ChainableTensor implements types.BasicTensor {
  readonly dtype: types.DType;
  readonly shape: types.Shape;
  readonly basic: types.BasicTensor;
  private static nextId = 1;
  readonly id: number;

  constructor(t: types.BasicTensor) {
    this.shape = t.shape;
    this.dtype = t.dtype;
    this.basic = t;
    this.id = ChainableTensor.nextId++;
  }

  getData(): types.TypedArray {
    return this.basic.getData();
  }

  get rank(): number {
    return this.shape.length;
  }

  toString(): string {
    // TODO This should pretty print the tensor.
    return `[${this.getData()}]`;
  }

  add(x: types.TensorLike): ChainableTensor {
    return ops.add(this, $(x));
  }

  sub(x: types.TensorLike): ChainableTensor {
    return ops.sub(this, $(x));
  }

  mul(x: types.TensorLike): ChainableTensor {
    return ops.mul(this, $(x));
  }

  div(x: types.TensorLike): ChainableTensor {
    return ops.div(this, $(x));
  }

  matmul(x: types.TensorLike): ChainableTensor {
    return ops.matmul(this, $(x));
  }

  neg(): ChainableTensor {
    return ops.neg(this);
  }

  exp(): ChainableTensor {
    return ops.exp(this);
  }

  log(): ChainableTensor {
    return ops.log(this);
  }

  onesLike(): ChainableTensor {
    const b = basicOps.onesLike(this.basic);
    return new ChainableTensor(b);
  }

  zerosLike(): ChainableTensor {
    const b = basicOps.zerosLike(this.basic);
    return new ChainableTensor(b);
  }

  square = () => ops.square(this);
  sinh = () => ops.sinh(this);
  cosh = () => ops.cosh(this);
  tanh = () => ops.tanh(this);

  transpose(perm?: types.TensorLike): ChainableTensor {
    if (perm === undefined) {
      perm = arange(this.rank).reverse();
    }
    perm = $(perm, "int32");
    return ops.transpose(this, perm);
  }

  // Reverses specific dimensions of a tensor.
  reverse(dims?: number[]): ChainableTensor {
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
  argmax(axis?: number): ChainableTensor {
    if (axis === undefined) axis = 0;
    return ops.argmax(this, axis);
  }

  // Returns the index with the smallest value across an axis of a tensor.
  // axis defaults to 0.
  argmin(axis?: number): ChainableTensor {
    if (axis === undefined) axis = 0;
    return ops.argmin(this, axis);
  }

  // Sum the tensor over the given axes.
  // axes defaults to all.
  reduceSum(axes?: number[], keepDims = false): ChainableTensor {
    if (!axes) axes = rangeJS(this.rank);
    return ops.reduceSum(this, axes, keepDims);
  }

  // Take the maximum value over the given axes.
  // axes defaults to all.
  reduceMax(axes?: number[], keepDims = false): ChainableTensor {
    if (!axes) axes = rangeJS(this.rank);
    return ops.reduceMax(this, axes, keepDims);
  }

  reduceLogSumExp(axes?: number[], keepDims = false): ChainableTensor {
    if (!axes) axes = rangeJS(this.rank);
    return ops.reduceLogSumExp(this, axes, keepDims);
  }

  // Element-wise comparison. Returns a tensor with dtype == "bool".
  equal(x: types.TensorLike): ChainableTensor {
    return ops.equal(this, $(x));
  }

  // Return a slice from 'input'.
  // The output tensor is a tensor with dimensions described by 'size' whose
  // values are extracted from 'input' starting at the offsets in 'begin'.
  // begin[i] specifies the offset into the ith dimension of 'input' to slice
  // from.  size[i] specifies the number of elements of the ith dimension of
  // 'input' to slice. If size[i] is -1, all remaining elements in dimension
  // are included in the slice -- this is equivalent to setting
  //   size[i] = input.shape[i] - begin[i]
  slice(begin: number[], size: number[]): ChainableTensor {
    return ops.slice(this, begin, size);
  }

  // Reshapes the tensor without changing its data.
  reshape(newShape: types.Shape): ChainableTensor {
    return ops.reshape(this, newShape);
  }

  // Return a copy of the tensor collapsed into one dimension.
  flatten(): ChainableTensor {
    return this.reshape([-1]);
  }

  // Remove single-dimensional axes from the shape of a tensor.
  squeeze(): ChainableTensor {
    const newShape = this.shape.filter((d) => d > 1);
    return this.reshape(newShape);
  }

  // Returns the softmax activations of a tensor.
  softmax(axis = -1): ChainableTensor {
    return softmaxHelper(this, axis, ops.softmax);
  }

  // Numerically stable log(softmax(x)).
  logSoftmax(axis = -1): ChainableTensor {
    return softmaxHelper(this, axis, ops.logSoftmax);
  }

  // Dot product of two tensors. For 2D tensors it is equivalent to matrix
  // multiplication. For 1D tensors to inner product of vectors (without
  // complex conjugation). Currently higher order tensors are not supported.
  dot(x: types.TensorLike): ChainableTensor {
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
function softmaxHelper(t: ChainableTensor, axis: number, op): ChainableTensor {
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
