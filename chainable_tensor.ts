import * as types from "./types";
import * as ops from "./ops";
import { basicOps, convertBasic } from "./basic";

export function convertChainable(t: types.TensorLike): ChainableTensor {
  if (t instanceof ChainableTensor) return t;
  return new ChainableTensor(convertBasic(t));
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
}
