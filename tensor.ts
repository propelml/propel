import { NDArray } from "./deeplearnjs/src/math/ndarray";
export { NDArray } from "./deeplearnjs/src/math/ndarray";
import { expandShapeToKeepDim } from "./deeplearnjs/src/math/axis_util";
import { NDArrayMath } from "./deeplearnjs/src/math/math";
import { NDArrayMathCPU } from "./deeplearnjs/src/math/math_cpu";
import { NDArrayMathGPU } from "./deeplearnjs/src/math/math_gpu";
import { flatten, inferShape, RegularArray, TypedArray }
  from "./deeplearnjs/src/util";
import * as ops from "./ops";
import * as tf from "./tf";
import { arange } from "./propel";
import { assert, shapesEqual } from "./util";

export type TensorLike = boolean | number | RegularArray<boolean> |
  RegularArray<number> | TypedArray | Tensor;
export type Shape = number[];

const cpuMath: NDArrayMathCPU = new NDArrayMathCPU();
let gpuMath: NDArrayMathGPU = null;

export abstract class Tensor {
  private static nextId = 1;
  id: number;
  shape: Shape;
  dtype: "float32" | "uint8";

  static convert(x: TensorLike): Tensor {
    if (x instanceof Tensor) {
      return x;
    } else if (tf.binding) {
      return new TFTensor(x);
    } else {
      return new DLTensor(x);
    }
  }

  constructor() {
    this.dtype = "float32";
    this.id = Tensor.nextId;
    Tensor.nextId++;
  }

  abstract getValues(): TypedArray;

  // Returns a copy of the Tensor that is stored on the GPU.
  abstract gpu(): Tensor;

  abstract inGPU(): boolean;

  abstract toNumber(): number;

  abstract get(...locs: number[]): number;

  abstract zerosLike(): Tensor;

  abstract onesLike(): Tensor;

  abstract toString(): string;

  abstract exp(): Tensor;

  abstract neg(): Tensor;

  abstract add(x: TensorLike): Tensor;

  abstract sub(x: TensorLike): Tensor;

  abstract div(x: TensorLike): Tensor;

  abstract mul(x: TensorLike): Tensor;

  abstract reshape(newShape: Shape): Tensor;

  abstract expandDims(axis: number): Tensor;

  abstract equals(t: TensorLike): boolean;
}

export class DLTensor extends Tensor {
  math: NDArrayMath = cpuMath;
  ndarray: NDArray; // TODO private

  constructor(x: TensorLike | NDArray) {
    super();
    if (x instanceof Tensor) {
      this.ndarray = (x as DLTensor).ndarray;
      this.shape = x.shape;
    } else if (x instanceof Array) {
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
    } else if (typeof x == "boolean") {
      throw new Error("Not Implemented");
    } else if (x instanceof NDArray) {
      this.ndarray = x;
      if (x.inGPU()) {
        this.math = DLTensor.gpuMath();
      }
      this.shape = x.shape;
    } else {
      // TypedArray
      this.shape = [x.length];
      this.ndarray = NDArray.make(this.shape, { values: x });
    }

    this.dtype = "float32"; // TODO Support other dtypes.
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
    DLTensor.gpuMath();

    const ndarray = NDArray.like(this.ndarray);
    assert(null != ndarray.getTexture()); // Upload to GPU.

    const t = new DLTensor(ndarray);
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
    return new DLTensor(zeros);
  }

  onesLike(): Tensor {
    const ones = NDArray.zerosLike(this.ndarray);
    ones.fill(1.0);
    return new DLTensor(ones);
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
    const b = (Tensor.convert(t) as DLTensor).ndarray;
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

export class TFTensor extends Tensor {
  // handle is a binding.Tensor. It's called handle so we don't have too many
  // things named Tensor, and because it's essentially as JS version of
  // TFE_TensorHandle.
  handle;
  private values: TypedArray;

  static convert(x: TensorLike): TFTensor {
    if (x instanceof TFTensor) {
      return x;
    } else {
      return new TFTensor(x);
    }
  }

  constructor(x: TensorLike | any, shape_ = null) {
    super();

    if (x instanceof Tensor) {
      this.values = x.getValues();
      this.shape = x.shape;
    } else if (x instanceof tf.binding.Tensor) {
      this.handle = x;
      this.shape = x.shape;
    } else if (x instanceof Array) {
      // Argument is a JS array like [[1, 2], [3, 4]].
      assert(shape_ == null);
      const shape = inferShape(x);
      if (shape.length == 1 && shape[0] == 0) {
        // Special case the empty tensor...
        this.shape = [];
        this.values = new Float32Array([]);
      } else {
        const data = flatten(x) as number[];
        this.values = new Float32Array(data);
        this.shape = shape;
      }
    } else if (typeof x == "number") {
      // Scalar
      assert(shape_ == null);
      this.values = new Float32Array([x]);
      this.shape = [];
    } else if (typeof x == "boolean") {
      throw new Error("Not Implemented.");
    } else {
      // TypedArray
      this.values = x;
      this.shape = shape_ ? shape_ : [x.length];
    }

    if (!this.handle) {
      this.handle = new tf.binding.Tensor(this.values, this.shape);
    }
    this.dtype = "float32"; // TODO Support other dtypes.
  }

  getValues(): TypedArray {
    if (!this.values) {
      this.values = new Float32Array(this.handle.asArrayBuffer());
    }
    return this.values;
  }

  // Returns a copy of the Tensor that is stored on the GPU.
  gpu(): Tensor {
    throw new Error("Not Implemented.");
  }

  inGPU(): boolean {
    throw new Error("Not Implemented.");
  }

  toNumber(): number {
    return this.values[0];
  }

  get(...locs: number[]): number {
    throw new Error("Not Implemented.");
  }

  zerosLike(): Tensor {
    throw new Error("Not Implemented.");
  }

  onesLike(): Tensor {
    throw new Error("Not Implemented.");
  }

  toString(): string {
    throw new Error("Not Implemented.");
  }

  exp(): Tensor {
    throw new Error("Not Implemented.");
  }

  neg(): Tensor {
    throw new Error("Not Implemented.");
  }

  add(x: TensorLike): Tensor {
    throw new Error("Not Implemented.");
  }

  sub(x: TensorLike): Tensor {
    throw new Error("Not Implemented.");
  }

  div(x: TensorLike): Tensor {
    throw new Error("Not Implemented.");
  }

  mul(x: TensorLike): Tensor {
    const xx = TFTensor.convert(x);
    const r = tf.execute0("Mul", [this.handle, xx.handle], [
      ["T", tf.binding.ATTR_TYPE, tf.binding.TF_FLOAT],
    ]);
    return new TFTensor(r);
  }

  reshape(newShape: Shape): Tensor {
    throw new Error("Not Implemented.");
  }

  expandDims(axis: number): Tensor {
    throw new Error("Not Implemented.");
  }

  equals(t: TensorLike): boolean {
    const tt = TFTensor.convert(t);

    if (!shapesEqual(this.shape, tt.shape)) {
      return false;
    }

    if (this.shape.length == 0) {
      assert(tt.shape.length == 0);
      return this.toNumber() === tt.toNumber();
    }

    const r = tf.execute0("Equal", [this.handle, tt.handle], [
      ["T", tf.binding.ATTR_TYPE, tf.binding.TF_FLOAT],
    ]);
    assert(r.dtype == tf.binding.TF_BOOL);

    const idx = arange(0, this.shape.length) as TFTensor;

    const r2 = tf.execute0("All", [r, idx.handle], [
      ["Tidx", tf.binding.ATTR_TYPE, tf.binding.TF_INT32],
      ["keep_dims", tf.binding.ATTR_BOOL, false],
    ]);
    assert(r2.dtype == tf.binding.TF_BOOL);
    const out = new Uint8Array(r2.asArrayBuffer());
    assert(out.length == 1);
    return Boolean(out[0]);
  }
}
