import { arange } from "./api";
import { basicOps, convertBasic } from "./basic";
import * as ops from "./ops";
import * as types from "./types";
import { assert } from "./util";

export function convertChainable(t: types.TensorLike,
  dtype: types.DType = "float32"): ChainableTensor {
  if (t instanceof ChainableTensor) return t;
  return new ChainableTensor(convertBasic(t, dtype));
}
const $ = convertChainable;

// ChainableTensor wraps a BasicTensor object. This is the main public
// interface to tensor operatiors. Each instance has a unique id for use in
// backprop.  Nothing about ChainableTensors is backend specific.
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

  reduceSum(axes?: number[], keepDims = false): ChainableTensor {
    if (!axes) axes = rangeJS(this.rank);
    return ops.reduceSum(this, axes, keepDims);
  }

  reduceMax(axes?: number[], keepDims = false): ChainableTensor {
    if (!axes) axes = rangeJS(this.rank);
    return ops.reduceMax(this, axes, keepDims);
  }

  equal(x: types.TensorLike): ChainableTensor {
    return ops.equal(this, $(x));
  }

  reshape(newShape: types.Shape): ChainableTensor {
    return ops.reshape(this, newShape);
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
