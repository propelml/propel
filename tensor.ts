import { NDArray } from "./deeplearnjs/src/math/ndarray";
export { NDArray } from "./deeplearnjs/src/math/ndarray";
import { expandShapeToKeepDim } from "./deeplearnjs/src/math/axis_util";
import { NDArrayMath } from "./deeplearnjs/src/math/math";
import { NDArrayMathCPU } from "./deeplearnjs/src/math/math_cpu";
import { NDArrayMathGPU } from "./deeplearnjs/src/math/math_gpu";
import { flatten, inferShape, RegularArray } from "./deeplearnjs/src/util";
import * as ops from "./ops";
import { assert } from "./util";

export type TensorLike = boolean | number | RegularArray<boolean> |
  RegularArray<number> | NDArray | Tensor;
export type Shape = number[];

const cpuMath: NDArrayMathCPU = new NDArrayMathCPU();
let gpuMath: NDArrayMathGPU = null;

export class Tensor {
  private static nextId: number = 1;
  public math: NDArrayMath = cpuMath;
  public id: number;
  public shape: Shape;
  public ndarray: NDArray; // TODO private
  public dtype: "float32" | "uint8";

  public static convert(x: TensorLike): Tensor {
    if (x instanceof Tensor) {
      return x;
    } else {
      return new Tensor(x);
    }
  }

  constructor(x: TensorLike) {
    if (x instanceof Array) {
      // Argument is a JS array like [[1, 2], [3, 4]].
      const shape = inferShape(x);
      const data = flatten(x) as number[];
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

    this.dtype = "float32"; // TODO Support other dtypes.

    this.id = Tensor.nextId;
    Tensor.nextId++;
  }

  // Lazily initialize.
  private static gpuMath(): NDArrayMathGPU {
    if (!gpuMath) {
      gpuMath = new NDArrayMathGPU();
    }
    return gpuMath;
  }

  // Returns a copy of the Tensor that is stored on the GPU.
  public gpu(): Tensor {
    Tensor.gpuMath();

    const ndarray = NDArray.like(this.ndarray);
    assert(null != ndarray.getTexture()); // Upload to GPU.

    const t = new Tensor(ndarray);
    assert(t.math == gpuMath);

    return t;
  }

  public inGPU(): boolean {
    return this.ndarray.inGPU();
  }

  public toNumber(): number {
    const values = this.ndarray.getValues();
    if (values.length != 1) {
      throw new Error("toNumber() can only be used on scalar tensors.");
    }
    return values[0];
  }

  public get(...locs: number[]): number {
    return this.ndarray.get(...locs);
  }

  public zerosLike(): Tensor {
    const zeros = NDArray.zerosLike(this.ndarray);
    return new Tensor(zeros);
  }

  public onesLike(): Tensor {
    const ones = NDArray.zerosLike(this.ndarray);
    ones.fill(1.0);
    return new Tensor(ones);
  }

  public toString(): string {
    // TODO This should pretty print the tensor.
    return `[${this.ndarray.getValues()}]`;
  }

  public exp(): Tensor {
    return ops.exp(this);
  }

  public neg(): Tensor {
    return ops.neg(this);
  }

  public add(x: TensorLike): Tensor {
    return ops.add(this, x);
  }

  public sub(x: TensorLike): Tensor {
    return ops.sub(this, x);
  }

  public div(x: TensorLike): Tensor {
    return ops.div(this, x);
  }

  public mul(x: TensorLike): Tensor {
    return ops.mul(this, x);
  }

  public reshape(newShape: Shape): Tensor {
    return ops.reshape(this, newShape);
  }

  public expandDims(axis: number): Tensor {
    const newShape = expandShapeToKeepDim(this.shape, [axis]);
    return ops.reshape(this, newShape);
  }
}
