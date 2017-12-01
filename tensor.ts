import { NDArray } from "./deeplearnjs/src/math/ndarray";
export { NDArray } from "./deeplearnjs/src/math/ndarray";
import { expandShapeToKeepDim } from "./deeplearnjs/src/math/axis_util";
import { NDArrayMath } from "./deeplearnjs/src/math/math";
import { NDArrayMathCPU } from "./deeplearnjs/src/math/math_cpu";
import { NDArrayMathGPU } from "./deeplearnjs/src/math/math_gpu";
import { flatten, inferShape, RegularArray, TypedArray }
  from "./deeplearnjs/src/util";
import * as ops from "./ops";
import { assert } from "./util";

export type TensorLike = boolean | number | RegularArray<boolean> |
  RegularArray<number> | TypedArray | Tensor;
export type Shape = number[];

const cpuMath: NDArrayMathCPU = new NDArrayMathCPU();
let gpuMath: NDArrayMathGPU = null;

export class Tensor {
  private static nextId = 1;
  math: NDArrayMath = cpuMath;
  id: number;
  shape: Shape;
  ndarray: NDArray; // TODO private
  dtype: "float32" | "uint8";

  static convert(x: TensorLike): Tensor {
    if (x instanceof Tensor) {
      return x;
    } else {
      return new Tensor(x);
    }
  }

  constructor(x: TensorLike | NDArray) {
    if (x instanceof Array) {
      // Argument is a JS array like [[1, 2], [3, 4]].
      const shape = inferShape(x);
      if (shape.length == 1 && shape[0] == 0) {
        // Special case the empty tensor...
        this.shape = [];
        this.ndarray = null;
      } else {
        const data = flatten(x) as number[];
        this.ndarray = NDArray.make(shape, { values: new Float32Array(data) });
        this.shape = shape;
      }
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

    this.dtype = "float32"; // TODO Support other dtypes.

    this.id = Tensor.nextId;
    Tensor.nextId++;
  }

  getValues(): TypedArray {
    return this.ndarray.getValues();
  }

  // Lazily initialize.
  private static gpuMath(): NDArrayMathGPU {
    if (!gpuMath) {
      gpuMath = new NDArrayMathGPU();
    }
    return gpuMath;
  }

  // Returns a copy of the Tensor that is stored on the GPU.
  gpu(): Tensor {
    Tensor.gpuMath();

    const ndarray = NDArray.like(this.ndarray);
    assert(null != ndarray.getTexture()); // Upload to GPU.

    const t = new Tensor(ndarray);
    assert(t.math == gpuMath);

    return t;
  }

  inGPU(): boolean {
    return this.ndarray.inGPU();
  }

  toNumber(): number {
    const values = this.ndarray.getValues();
    if (values.length != 1) {
      throw new Error("toNumber() can only be used on scalar tensors.");
    }
    return values[0];
  }

  get(...locs: number[]): number {
    return this.ndarray.get(...locs);
  }

  zerosLike(): Tensor {
    const zeros = NDArray.zerosLike(this.ndarray);
    return new Tensor(zeros);
  }

  onesLike(): Tensor {
    const ones = NDArray.zerosLike(this.ndarray);
    ones.fill(1.0);
    return new Tensor(ones);
  }

  toString(): string {
    // TODO This should pretty print the tensor.
    return `[${this.ndarray.getValues()}]`;
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

  reshape(newShape: Shape): Tensor {
    return ops.reshape(this, newShape);
  }

  expandDims(axis: number): Tensor {
    const newShape = expandShapeToKeepDim(this.shape, [axis]);
    return ops.reshape(this, newShape);
  }

  // Messy. TF's reduce_all could be used here for example. We will revisit
  // this op after the bindings are in place.
  equals(t: TensorLike): boolean {
    const a = this.ndarray;
    const b = Tensor.convert(t).ndarray;
    if (a === null) {
      return b === null;
    } else if (b === null) {
      return false;
    }
    const r = this.math.equalStrict(a, b);
    const v = r.getValues();
    for (let i = 0; i < v.length; ++i) {
      if (v[i] === 0) return false;
    }
    return true;
  }
}
