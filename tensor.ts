import { NDArray } from './deeplearnjs/src/math/ndarray'
export { NDArray } from './deeplearnjs/src/math/ndarray'
import * as ops from './ops';
import { RegularArray, inferShape, flatten } from './deeplearnjs/src/util';
import { NDArrayMathCPU } from './deeplearnjs/src/math/math_cpu';
import { NDArrayMathGPU } from './deeplearnjs/src/math/math_gpu';
import { NDArrayMath } from './deeplearnjs/src/math/math';
import { assert } from './util';

export type TensorLike = boolean | number | RegularArray<boolean> |
   RegularArray<number> | NDArray | Tensor;
export type Shape = number[];

let cpuMath: NDArrayMathCPU = new NDArrayMathCPU();
let gpuMath: NDArrayMathGPU = null;

export class Tensor {
  private static nextId: number = 1;
  math: NDArrayMath = cpuMath;
  id: number;
  shape: Shape;
  ndarray: NDArray; // TODO private
  dtype: 'float32' | 'uint8';

  static convert(x: TensorLike): Tensor {
    if (x instanceof Tensor) {
      return x;
    } else {
      return new Tensor(x);
    }
  }

  constructor(x: TensorLike) {
    if (x instanceof Array) {
      // Argument is a JS array like [[1, 2], [3, 4]].
      let shape = inferShape(x);
      let data = flatten(x) as Array<number>;
      this.ndarray = NDArray.make(shape, { values: new Float32Array(data) });
      this.shape = shape;
    } else if (typeof x == "number") {
      // Scalar
      this.ndarray = NDArray.make([], { values: new Float32Array([x]) });
      this.shape = [];
    } else if (x instanceof NDArray) {
      this.ndarray = x;
      if (x.inGPU()) {
        this.math = Tensor.gpuMath();
      }
      this.shape = x.shape;
    }

    this.dtype = 'float32'; // TODO Support other dtypes.

    this.id = Tensor.nextId;
    Tensor.nextId++;
  }

  // Lazily initialize.
  private static gpuMath(): NDArrayMathGPU {
    if (!gpuMath) {
      gpuMath = new NDArrayMathGPU();
    }
    return gpuMath
  }

  // Returns a copy of the Tensor that is stored on the GPU.
  gpu(): Tensor {
    Tensor.gpuMath();

    let ndarray = NDArray.like(this.ndarray);
    assert(null != ndarray.getTexture()); // Upload to GPU.

    let t = new Tensor(ndarray);
    assert(t.math == gpuMath);

    return t;
  }

  inGPU(): boolean {
    return this.ndarray.inGPU();
  }

  toNumber(): number {
    let values = this.ndarray.getValues();
    if (values.length != 1) {
      throw new Error("toNumber() can only be used on scalar tensors.");
    }
    return values[0];
  }

  zerosLike(): Tensor {
    let zeros = NDArray.zerosLike(this.ndarray);
    return new Tensor(zeros);
  }

  onesLike(): Tensor {
    let ones = NDArray.zerosLike(this.ndarray);
    ones.fill(1.0);
    return new Tensor(ones);
  }

  toString(): string {
    // TODO This should pretty print the tensor.
    return `[${this.ndarray.getValues()}]`
  }

  exp(): Tensor {
    return ops.exp(this);
  }

  neg(): Tensor {
    return ops.neg(this);
  }

  add(x: TensorLike): Tensor {
    return ops.add(this, x);
  }

  sub(x: TensorLike): Tensor {
    return ops.sub(this, x);
  }

  div(x: TensorLike): Tensor {
    return ops.div(this, x);
  }

  mul(x: TensorLike): Tensor {
    return ops.mul(this, x);
  }

  reshape(newshape: Shape): Tensor {
    return ops.reshape(this, newshape);
  }
}
