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
    return ops.add(this, x);
  }

  sub(x: types.TensorLike): ChainableTensor {
    return ops.sub(this, x);
  }

  mul(x: types.TensorLike): ChainableTensor {
    return ops.mul(this, x);
  }

  div(x: types.TensorLike): ChainableTensor {
    return ops.div(this, x);
  }

  neg(): ChainableTensor {
    return ops.neg(this);
  }

  exp(): ChainableTensor {
    return ops.exp(this);
  }

  onesLike(): ChainableTensor {
    const b = basicOps.onesLike(this.basic);
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
    perm = convertChainable(perm, "int32");
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

    const dimsT = convertChainable(ta, "bool");
    return ops.reverse(this, dimsT);
  }
}
